import logging
from dataclasses import asdict
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import Decoder, Encoder, init_decoder_weights, init_encoder_weights
from sepsis_osc.ldm.checkpoint_utils import load_checkpoint, save_checkpoint
from sepsis_osc.ldm.commons import binary_logits, ordinal_logits, smooth_labels
from sepsis_osc.ldm.data_loading import get_data_sets, prepare_batches
from sepsis_osc.ldm.gru import GRUPredictor
from sepsis_osc.ldm.helper_structs import (
    AuxLosses,
    LoadingConfig,
    LossesConfig,
    LRConfig,
    ModelConfig,
    SaveConfig,
    TrainingConfig,
)
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel, update_betas
from sepsis_osc.ldm.logging_utils import flatten_dict, log_train_metrics, log_val_metrics
from sepsis_osc.ldm.lookup import LatentLookup, as_3d_indices
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE, jax_random_seed
from sepsis_osc.utils.jax_config import EPS, setup_jax, typechecker
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.utils import timing

INT_INF = jnp.astype(jnp.inf, jnp.int32)


@jaxtyped(typechecker=typechecker)
def sofa_ps_from_logits(logits: Float[Array, "time n_classes_minus_1"]) -> Float[Array, "time n_classes"]:
    p_gt = jax.nn.sigmoid(logits)  # (T, K)
    p0 = 1.0 - p_gt[..., 0:1]  # P(S=0) shape (T,1)
    p_middle = p_gt[..., :-1] - p_gt[..., 1:]  # P(S=k) for 1..K-1  -> (T,K-1)
    plast = p_gt[..., -1:]  # P(S=K) shape (T,1)
    probs = jnp.concatenate([p0, p_middle, plast], axis=-1)  # (T,S)
    # numeric safety
    probs = jnp.clip(probs, EPS, 1.0 - EPS)
    return probs / probs.sum(axis=-1, keepdims=True)


@jaxtyped(typechecker=typechecker)
def sofa_event_prob_all_times(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    mask: Bool[Array, "time time"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, ""]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]  # (S, S)
    soft_I = jax.nn.sigmoid((diff - delta) / tau)  # (S, S)

    # All pairwise combinations of times
    p1 = sofa_probs[:, None, :, None]  # (T, 1, S, 1)
    p2 = sofa_probs[None, :, None, :]  # (1, T, 1, S)
    joint = p1 * p2  # (T, T, S, S)

    # Probability for each pair
    p_events = jnp.sum(joint * soft_I, axis=(2, 3)) * mask  # (T, T)

    # Soft OR across all pairs
    return 1.0 - jnp.prod(1.0 - p_events)


@jaxtyped(typechecker=typechecker)
def sofa_event_prob_consecutive_times(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, ""]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]  # (S, S)
    soft_I = jax.nn.sigmoid((diff - delta) / tau)  # (S, S)

    p1 = sofa_probs[:-1, :, None]
    p2 = sofa_probs[1:, None, :]
    joint = p1 * p2  # (T-1, S, S)

    # Probability for each pair
    p_events = jnp.sum(joint * soft_I, axis=(1, 2))

    # Soft OR across
    return 1.0 - jnp.prod(1.0 - p_events)


@jaxtyped(typechecker=typechecker)
def sofa_event_prob_any_future(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, " time"]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]
    soft_I = jax.nn.sigmoid((diff - delta) / tau)

    # all pairs
    p1 = sofa_probs[:, None, :, None]  # (T, 1, S, 1)
    p2 = sofa_probs[None, :, None, :]  # (1, T, 1, S)
    joint = p1 * p2  # (T, T, S, S)
    p_events = jnp.sum(joint * soft_I, axis=(2, 3))  # (T, T)

    return jnp.sum(jax.nn.softmax(jnp.tril(p_events, k=-1), axis=1), axis=-1)  # (T,)


@jaxtyped(typechecker=typechecker)
def sofa_increase_probs(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, " time-1"]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)  # (T, S)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]  # (S, S)
    soft_I = jax.nn.sigmoid((diff - delta) / tau)  # (S, S)

    # joint (T-1, S, S)
    p1 = sofa_probs[:-1, :, None]
    p2 = sofa_probs[1:, None, :]
    joint = p1 * p2

    # probability of increase at each step
    return jnp.sum(joint * soft_I, axis=(1, 2))  # (T-1,)


@jaxtyped(typechecker=typechecker)
def organ_failure_increase_probs(
    sofa_frac: Float[Array, " time"],
    tau: Float[Array, "1"],
    delta: float = 0.1,
) -> Float[Array, " time-1"]:
    diffs = sofa_frac[1:] - sofa_frac[:-1]  # (T-1,)
    return jax.nn.sigmoid((diffs - delta) / tau)  # (T-1,)


@jaxtyped(typechecker=typechecker)
def ordinal_loss(
    predicted_sofa: Float[Array, " time"],
    true_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    sofa_dist: Float[Array, " n_classes"],
) -> Float[Array, " time"]:
    # CORN (Cumulative Ordinal Regression)
    n_classes = thresholds.shape[-1] + 1

    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)

    targets = jnp.arange(n_classes - 1)  # [0, 1, ..., 22]
    true_sofa = true_sofa.squeeze()
    true_ord_targets = true_sofa[:, None] > targets[None, :]  # [B, 23]
    loss_per_sample = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_ord_targets), axis=-1)

    # return loss_per_sample
    # NOTE
    class_weights = jnp.log(1.0 + 1.0 / (sofa_dist + EPS))

    weights_per_sample = 1 + class_weights[true_sofa.astype(jnp.int32)]  # [B]
    return weights_per_sample * loss_per_sample


@jaxtyped(typechecker=typechecker)
def calc_directional_loss(
    pred_sofa: Int[Array, " time"] | Float[Array, " time"], true_sofa: Float[Array, " time"]
) -> Float[Array, " time-1"]:
    tdelta = true_sofa[1:] - true_sofa[:-1]
    pdelta = pred_sofa[1:] - pred_sofa[:-1]
    return jax.nn.softplus(-pdelta * tdelta)


@jaxtyped(typechecker=typechecker)
def calc_trend_loss(
    pred_sofa: Int[Array, " time"] | Float[Array, " time"], true_sofa: Float[Array, " time"]
) -> Float[Array, ""]:
    ttrend = true_sofa[-1] - true_sofa[0]
    ptrend = pred_sofa[-1] - pred_sofa[0]
    return jax.nn.softplus(-ptrend * ttrend)


@jaxtyped(typechecker=typechecker)
def calc_tc_loss(z_seq: Float[Array, "batch time zlatent_dim"]) -> Float[Array, ""]:
    z_flat = z_seq.reshape(-1, z_seq.shape[-1])
    z_centered = z_flat - jnp.mean(z_flat, axis=0, keepdims=True)

    cov = (z_centered.T @ z_centered) / z_flat.shape[0]

    var = jnp.diag(cov)
    std = jnp.sqrt(var + EPS)
    corr = cov / (std[:, None] * std[None, :])  # correlation matrix

    off_diag = corr - jnp.diag(jnp.diag(corr))
    return jnp.sum(off_diag**2) / (z_flat.shape[-1] * (z_flat.shape[-1] - 1) + EPS)


def constrain_z(z: Float[Array, "batch time zlatent_dim"]) -> Float[Array, "batch time zlatent_dim"]:
    z = jnp.clip(z, 0, 1)
    beta, sigma = jnp.split(z, 2, axis=-1)
    beta = beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma = sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]
    return jnp.concatenate([beta, sigma], axis=-1)


def bound_z(z: Float[Array, "batch time zlatent_dim"]) -> Float[Array, "batch time zlatent_dim"]:
    beta, sigma = jnp.split(z, 2, axis=-1)
    beta = beta.clip(BETA_SPACE[0], BETA_SPACE[1])
    sigma = sigma.clip(SIGMA_SPACE[0], SIGMA_SPACE[1])
    return jnp.concatenate([beta, sigma], axis=-1)


def cosine_annealing(base_val: float, num_steps: int, num_dead_steps: int, current_step: jnp.int32) -> Float[Array, ""]:
    step = jnp.clip(current_step - num_dead_steps, 0, num_steps)
    cosine = 0.5 * (1 + jnp.cos(jnp.pi * step / num_steps))
    return base_val + (1.0 - base_val) * (1.0 - cosine)


def uncertainty_scale(loss: Float[Array, "*"], log_sigma: Float[Array, ""]) -> Float[Array, "*"]:
    return loss
    sigma_sq = jnp.exp(2 * log_sigma)
    return ((loss / (2 * sigma_sq)) + log_sigma).squeeze()


@jaxtyped(typechecker=typechecker)
def get_latent_seq(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    *,
    key: jnp.ndarray,
) -> tuple[Float[Array, "batch time zlatent_dim"], Float[Array, "batch time vlatent_dim"]]:
    batch_size, T, _input_dim = x.shape

    def make_keys(base_key: jnp.ndarray) -> jnp.ndarray:
        return jr.split(base_key, 4)

    sample_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(batch_size))
    drop_keys = jax.vmap(make_keys)(sample_keys)

    # --------- Prediction
    zbeta0, zsigma0, vsofa0, vinf0, h_0 = jax.vmap(model.encoder)(x[:, 0], dropout_keys=drop_keys)

    # NOTE alpha is only predicted for t=0, and stays constant throughout the prediction horizon for each patient
    zv0 = jnp.concatenate([zbeta0, zsigma0, vsofa0, vinf0], axis=-1)

    def predict_next(
        carry: tuple[Float[Array, " latent_dim"], Float[Array, " gru_h_dim"]], _: Float[Array, " z_dim"]
    ) -> tuple[tuple[Float[Array, " latent_dim"], Float[Array, " gru_h_dim"]], Float[Array, " * z_dim"]]:
        zv_prev, h_prev = carry
        dzv_next, h_next = jax.vmap(model.predictor)(zv_prev, h_prev)
        zv_next = zv_prev + dzv_next
        new_carry = (zv_next, h_next)
        return new_carry, zv_next

    carry_init = (zv0, h_0)  # z0 = (batch, zv_dim)

    (_, _), zv_preds = jax.lax.scan(predict_next, carry_init, xs=None, length=T - 1)

    # build full time sequence
    zv_seq = jax.nn.sigmoid(
        jnp.transpose(jnp.concatenate([zv0[None, ...], zv_preds], axis=0), axes=(1, 0, 2))
    )  # (T, batch, zv_dim)

    return (
        constrain_z(zv_seq[..., : model.predictor.z_proj_out.out_features]),
        zv_seq[..., model.predictor.z_proj_out.out_features :],  # (batch, T, zv_dim)
    )


# === Loss Function ===
@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  # (donate="all")
def loss(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    true_concepts: Float[Array, "batch time 3"],
    step: Int[Array, ""] = INT_INF,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    aux = AuxLosses.empty()
    sofa_true, infection_true, sepsis_true = true_concepts[..., 0], true_concepts[..., 1], true_concepts[..., 2]
    # NOTE
    # aux.anneal_threshs = cosine_annealing(
    #     0.0, int(params.anneal_threshs_iter * params.steps_per_epoch), params.steps_per_epoch * 10, step
    # )
    aux.anneal_threshs = cosine_annealing(
        base_val=0.0,
        num_steps=int(100 * params.steps_per_epoch),
        num_dead_steps=params.steps_per_epoch * 100,
        current_step=step,
    )
    # ordinal_thresholds = model.ordinal_thresholds(aux.anneal_threshs)
    ordinal_thresholds = model.ordinal_thresholds(jnp.array(0.0))

    # aux.thresh_loss = jnp.mean(
    #     jax.nn.relu((1 / 24) - jax.nn.softmax(model.learned_deltas))
    # )  # punish deviations from uniform

    z_seq, v_seq = get_latent_seq(model, x, key=key)  # (batch, T, z_dim)

    vsofa, vinf = v_seq[..., 0], v_seq[..., 1]
    aux.alpha, aux.beta, aux.sigma = jnp.ones_like(z_seq[..., 0]) * model.alpha, z_seq[..., 0], z_seq[..., 1]

    pred_metrics = jax.vmap(lookup_func, in_axes=(1, None))(
        jnp.concat([aux.alpha[..., None], z_seq], axis=-1), model.lookup_temperature
    ).swapaxes(0, 1)  # (T, batch, 2) -> (batch, T, 2)
    # NOTE
    sofa_pred = pred_metrics[..., 0]  # ** model.sofa_exp  # (batch, T, 1)
    infection_pred = pred_metrics[..., 1]  # ** model.inf_exp  # (batch, T, 1)

    aux.matching_loss = jnp.mean(((vsofa - sofa_pred) ** 2) + ((vinf - infection_pred) ** 2))

    # --------- Difference Loss
    # try to hvae uniformly distributed metrics
    # aux.diff_loss = (sofa_pred.var() - 1 / 12) ** 2

    # --------- Acceleration Loss
    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    # aux.accel_loss = jnp.mean((z_seq[:, 2:] - 2 * z_seq[:, 1:-1] + z_seq[:, :-2]) ** 2)

    # --------- Recon Loss
    # TODO x_recon_loss_t
    x_recon = jax.vmap(jax.vmap(model.decoder))(jnp.concat([z_seq, v_seq], axis=-1))
    aux.recon_loss = jnp.mean((x - x_recon) ** 2)

    # --------- Total correlation loss (one per batch)
    # aux.tc_loss = calc_tc_loss(z_seq)

    # --------- Concept losses

    # aux.sofa_loss_t = jax.vmap(ordinal_loss, in_axes=(0, 0, None, None, None))(
    #     sofa_pred, sofa_true, ordinal_thresholds, model.label_temperature, model.sofa_dist
    # )
    aux.sofa_loss_t = (sofa_pred - (sofa_true / 24.0)) ** 2
    sofa_score_pred = jnp.sum(sofa_pred[..., None] > ordinal_thresholds, axis=-1)

    # sofa_increase_p = jax.vmap(sofa_increase_probs, in_axes=(0, None, None, None, None))(
    #     sofa_pred,
    #     ordinal_thresholds,
    #     model.label_temperature,
    #     model.delta_temperature,
    #     0.0,
    # )
    sofa_increase_p = jax.vmap(organ_failure_increase_probs, in_axes=(0, None, None))(
        sofa_pred, model.delta_temperature, 0.0
    )
    # sofa_increase_p = jax.vmap(sofa_event_prob_any_future, in_axes=(0, None, None, None, None))(
    #     sofa_pred,
    #     ordinal_thresholds,
    #     model.label_temperature,
    #     model.delta_temperature,
    #     0.0,
    # )
    # aux.sofa_d2_p_loss = optax.sigmoid_focal_loss(
    #     binary_logits(sofa_increase_p), (jnp.max(sofa_true[None, :] - sofa_true[:, None], axis=1) > 1.0).astype(jnp.int32), alpha=0.75, gamma=1.0
    # )
    # aux.sofa_d2_p_loss = optax.sigmoid_focal_loss(
    #     binary_logits(sofa_increase_p), (jnp.diff(sofa_true, axis=-1) > 1.0).astype(jnp.int32), alpha=0.75, gamma=2.0
    # )
    # 
    aux.sofa_d2_p_loss = optax.sigmoid_focal_loss(
        binary_logits(sofa_increase_p), (jnp.diff(sofa_true, axis=-1) > 0.0).astype(jnp.int32), alpha=0.99, gamma=1.0
    )
    # aux.sofa_d2_p = sofa_increase_p.max(axis=-1)
    aux.sofa_d2_p = 1.0 - jnp.prod(1.0 - sofa_increase_p, axis=-1)

    aux.infection_p_loss_t = jax.vmap(optax.sigmoid_focal_loss, in_axes=(0, 0, None, None))(
        binary_logits(infection_pred), infection_true, params.w_inf_alpha, params.w_inf_gamma
    )
    # aux.susp_inf_p = infection_pred.max(axis=-1)  #
    aux.susp_inf_p = 1.0 - jnp.prod(1.0 - infection_pred, axis=-1)

    aux.sep3_p = model.get_sepsis_probs(aux.sofa_d2_p, aux.susp_inf_p, model.betas)
    aux.sep3_p_loss = optax.sigmoid_focal_loss(
        binary_logits(aux.sep3_p), (sepsis_true == 1.0).any(axis=-1), alpha=0.99, gamma=1.0
    )
    # --------- Directional Loss
    # aux.directional_loss = jnp.mean(jax.vmap(calc_directional_loss)(sofa_pred, sofa_true))
    # aux.trend_loss = jnp.mean(jax.vmap(calc_trend_loss)(sofa_pred, sofa_true))

    # --------- Total Loss
    aux.hists_sofa_metric = jax.lax.stop_gradient(pred_metrics[..., 0].reshape(-1))
    aux.hists_sofa_score = jax.lax.stop_gradient(sofa_score_pred)
    aux.hists_inf_prob = jax.lax.stop_gradient(infection_pred)
    aux.total_loss = (
        aux.recon_loss * params.w_recon
        # + uncertainty_scale(jnp.mean(aux.sofa_d2_p_loss) * params.w_sofa_d2, model.sofa_lsigma)
        + uncertainty_scale(jnp.mean(aux.infection_p_loss_t) * params.w_inf, model.inf_lsigma)
        # SOFA only for first time point
        # + uncertainty_scale(jnp.mean(aux.sofa_loss_t) * params.w_sofa_classification, model.sep3_lsigma)
        # + jnp.mean(aux.sofa_loss_t * jnp.array([1.0, 1.0, 0.5, 0.25, 0.125, 0.0625])[None, :])* params.w_sofa_classification
        # + jnp.mean(aux.sep3_p_loss) * params.w_sep3 * aux.anneal_threshs
        # + aux.matching_loss * params.w_matching
        # + aux.directional_loss * params.w_sofa_direction
        # + aux.trend_loss * params.w_sofa_trend
        # + aux.tc_loss * params.w_tc
        # + aux.accel_loss * params.w_accel
        # + aux.diff_loss * params.w_diff
        # + aux.thresh_loss * params.w_thresh
    )
    return aux.total_loss, aux


@eqx.filter_jit
def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    model_params: PyTree,
    model_static: PyTree,
    opt_state: optax.OptState,
    update: Callable,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[PyTree, PyTree, tuple[Array, AuxLosses]]:
    model = eqx.combine(model_params, model_static)
    (total_loss, aux_losses), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        model, x_batch, true_c_batch, opt_state[1][0].count, key=key, lookup_func=lookup_func, params=loss_params
    )

    updates, opt_state = update(grads, opt_state, model_params)
    model_params = eqx.apply_updates(model_params, updates)
    model = eqx.combine(model_params, model_static)

    return model_params, opt_state, (total_loss, aux_losses)


@timing
@eqx.filter_jit
def process_train_epoch(
    model: LatentDynamicsModel,
    opt_state: optax.OptState,
    x_data: Float[Array, "ntbatches batch time input_dim"],
    y_data: Float[Array, "ntbatches batch time 3"],
    update: Callable,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[PyTree, PyTree, AuxLosses, jnp.ndarray]:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    key, _ = jr.split(key)
    step_losses = []

    def scan_step(
        carry: tuple[PyTree, optax.OptState, jnp.ndarray],
        batch: tuple[Float[Array, "batch time input_dim"], Float[Array, "bath time 3"]],
    ) -> tuple[tuple[PyTree, optax.OptState, jnp.ndarray], AuxLosses]:
        model_params, opt_state, key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        model_params, opt_state, (_, aux_losses) = step_model(
            batch_x,
            batch_true_c,
            model_params,
            model_static,
            opt_state,
            update,
            key=batch_key,
            lookup_func=lookup_func,
            loss_params=loss_params,
        )

        return (
            model_params,
            opt_state,
            key,
        ), aux_losses

    carry = (
        model_params,
        opt_state,
        key,
    )
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    model_params, opt_state, key = carry

    return (
        model_params,
        opt_state,
        step_losses,
        key,
    )


@timing
@eqx.filter_jit
def process_val_epoch(
    model: LatentDynamicsModel,
    x_data: Float[Array, "nvbatches batch t input_dim"],
    y_data: Float[Array, "nvbatches batch t 3"],
    step: Int[Array, ""],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[jnp.ndarray, AuxLosses]:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    key, _ = jr.split(key)

    def scan_step(
        carry: tuple[PyTree, jnp.ndarray],
        batch: tuple[Float[Array, "batch time input_dim"], Float[Array, "bath time 3"]],
    ) -> tuple[tuple[PyTree, jnp.ndarray], AuxLosses]:
        model_flat, key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        model_full = eqx.combine(model_flat, model_static)

        _, aux_losses = loss(
            model_full, batch_x, batch_true_c, key=batch_key, lookup_func=lookup_func, params=loss_params, step=step
        )

        return (model_flat, key), aux_losses

    carry = (model_params, key)
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    model_params, key = carry

    return (key, step_losses)


def custom_warmup_cosine(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    steps_per_cycle: list[int],
    end_value: float,
    peak_decay: float = 0.5,
) -> Callable:
    cycle_boundaries = jnp.array([warmup_steps + sum(steps_per_cycle[:i]) for i in range(len(steps_per_cycle))])
    cycle_lengths = jnp.array(steps_per_cycle)
    num_cycles = len(steps_per_cycle)

    def schedule(step: Int[Array, ""]) -> Float[Array, ""]:
        step = jnp.asarray(step)

        # --- Warmup ---
        def in_warmup_fn(step: Int[Array, ""]) -> Float[Array, ""]:
            frac = step / jnp.maximum(warmup_steps, 1)
            return (init_value + frac * (peak_value - init_value)).astype(jnp.float32)

        # --- Cosine Decay Cycles ---
        def in_decay_fn(step: Int[Array, ""]) -> Float[Array, ""]:
            # Which cycle are we in?
            rel_step = step - warmup_steps
            cycle_idx = jnp.sum(rel_step >= cycle_boundaries - warmup_steps) - 1
            cycle_idx = jnp.clip(cycle_idx, 0, num_cycles - 1)

            cycle_start = cycle_boundaries[cycle_idx]
            cycle_len = cycle_lengths[cycle_idx]
            step_in_cycle = step - cycle_start

            cycle_frac = jnp.clip(step_in_cycle / jnp.maximum(cycle_len, 1), 0.0, 1.0)
            peak = peak_value * (peak_decay**cycle_idx)
            return end_value + 0.5 * (peak - end_value) * (1 + jnp.cos(jnp.pi * cycle_frac))

        # Select warmup vs decay
        return jax.lax.cond(step < warmup_steps, in_warmup_fn, in_decay_fn, step)

    return schedule


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info")
    dtype = jnp.float32
    logger = logging.getLogger(__name__)

    model_conf = ModelConfig(
        z_latent_dim=2,
        v_latent_dim=2,
        input_dim=52,
        enc_hidden=512,
        dec_hidden=128,
        predictor_z_hidden=4,
        predictor_v_hidden=32,
        dropout_rate=0.2,
    )
    train_conf = TrainingConfig(batch_size=1024, window_len=6, epochs=100000, perc_train_set=1.0, validate_every=5)
    lr_conf = LRConfig(init=0.0, peak=5e-3, peak_decay=0.5, end=5e-6, warmup_epochs=100, enc_wd=1e-3, grad_norm=0.9)
    loss_conf = LossesConfig(
        w_recon=1e-3,
        w_tc=0,
        w_accel=0.0,
        w_diff=0.0,
        w_sofa_direction=0.0,
        w_sofa_trend=0.0,
        w_sofa_classification=0.0,
        w_sofa_d2=0.0,
        w_inf=50.0,
        w_inf_alpha=0.99,
        w_inf_gamma=1.0,
        w_sep3=0.0,
        w_thresh=0.0,
        w_matching=0.0,
        anneal_threshs_iter=5 * 1 / train_conf.perc_train_set,
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=100, perform=True)

    # === Data ===
    train_x, train_y, val_x, val_y, test_x, test_y = get_data_sets(window_len=train_conf.window_len, dtype=jnp.float32)
    logger.error(f"{jnp.unique(train_y[..., 0])}")
    logger.error(f"{jnp.unique(train_y[..., 1])}")
    logger.error(f"{jnp.unique(train_y[..., 2])}")
    logger.info(f"Train shape      - X {train_y.shape}, Y {train_x.shape}")
    logger.info(f"Validation shape - X {val_y.shape},  Y {val_x.shape}")
    logger.info(f"Test shape       - X {test_y.shape},  Y {test_x.shape}")

    train_x, train_y = test_x, test_y
    db_str = "Daisy2"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    sim_storage.close()

    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = DNMConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params), proto_metric=DNMMetrics, threshold=jnp.inf)
    metrics_3d = metrics_3d.to_jax()
    lookup_table = LatentLookup(
        metrics=metrics_3d.reshape((-1, 1)),
        indices=indices_3d.reshape((-1, 3)),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=jnp.float32,
    )
    steps_per_epoch = (train_x.shape[0] // train_conf.batch_size) * train_conf.batch_size
    schedule = custom_warmup_cosine(
        init_value=lr_conf.init,
        peak_value=lr_conf.peak,
        warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
        steps_per_cycle=[steps_per_epoch * n for n in [300]],  # cycle durations
        end_value=lr_conf.end,
        peak_decay=lr_conf.peak_decay,
    )

    # === Initialization ===
    key = jr.PRNGKey(jax_random_seed)
    key_enc, key_dec, key_predictor = jr.split(key, 3)
    metrics = jnp.sort(lookup_table.relevant_metrics[..., 0])
    # TODO dont do this for the windowed
    sofa_dist, _ = jnp.histogram(train_y[..., 0], bins=25, density=False)
    sofa_dist = sofa_dist / jnp.sum(sofa_dist)
    deltas = metrics[jnp.round(jnp.cumsum(sofa_dist) * len(metrics)).astype(dtype=jnp.int32)]
    deltas = jnp.ones_like(deltas)  # NOTE
    deltas = deltas / jnp.sum(deltas)
    deltas = jnp.asarray(deltas)

    encoder = Encoder(
        key_enc,
        model_conf.input_dim,
        model_conf.z_latent_dim,
        model_conf.enc_hidden,
        model_conf.predictor_v_hidden + model_conf.predictor_z_hidden,
        dropout_rate=model_conf.dropout_rate,
        dtype=dtype,
    )
    decoder = Decoder(
        key_dec,
        model_conf.input_dim,
        model_conf.z_latent_dim + model_conf.v_latent_dim,
        model_conf.dec_hidden,
        dtype=dtype,
    )
    key_enc_weights, key_dec_weights = jr.split(key, 2)
    encoder = init_encoder_weights(encoder, key_enc_weights)
    decoder = init_decoder_weights(decoder, key_dec_weights)
    gru = GRUPredictor(
        key=key_predictor,
        z_dim=model_conf.z_latent_dim,
        v_dim=model_conf.v_latent_dim,
        z_hidden_dim=model_conf.predictor_z_hidden,
        v_hidden_dim=model_conf.predictor_v_hidden,
        dtype=dtype,
    )

    model = LatentDynamicsModel(
        encoder=encoder,
        predictor=gru,
        decoder=decoder,
        alpha=-0.28,
        ordinal_deltas=deltas,
        sofa_dist=sofa_dist[:-1],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(lr_conf.grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
    )

    hyper_enc = {
        "input_dim": encoder.input_dim,
        "latent_dim": encoder.latent_dim,
        "enc_hidden": encoder.enc_hidden,
        "pred_hidden": encoder.pred_hidden,
        "dropout_rate": encoder.dropout_rate,
    }
    hyper_dec = {
        "input_dim": decoder.input_dim,
        "latent_dim": decoder.latent_dim,
        "dec_hidden": decoder.dec_hidden,
    }
    hyper_pred = {
        "z_dim": model_conf.z_latent_dim,
        "hidden_z_dim": gru.z_hidden_dim,
        "v_dim": model_conf.v_latent_dim,
        "hidden_v_dim": gru.v_hidden_dim,
    }
    params_model, static_model = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params_model)
    if load_conf.from_dir:
        try:
            model, opt_state = load_checkpoint(load_conf.from_dir + "/checkpoints", load_conf.epoch, optimizer)

            logger.info(f"Resuming training from epoch {load_conf.epoch + 1}")
        except FileNotFoundError as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting training from scratch.")
            load_conf.epoch = 0
            load_conf.from_dir = ""

    # === Training Loop ===
    writer = SummaryWriter()
    hyper_params = flatten_dict(
        {
            "model": asdict(model_conf),
            "losses": asdict(loss_conf),
            "train": asdict(train_conf),
            "lr": asdict(lr_conf),
        }
    )
    writer.add_hparams(hyper_params, metric_dict={}, run_name=".")

    shuffle_val_key, key = jr.split(key, 2)
    val_x, val_y, nval_batches = prepare_batches(val_x, val_y, train_conf.batch_size, key=shuffle_val_key)
    for epoch in range(load_conf.epoch if load_conf.from_dir else 0, train_conf.epochs):
        shuffle_key, key = jr.split(key)
        x_shuffled, y_shuffled, ntrain_batches = prepare_batches(
            train_x,
            train_y,
            key=shuffle_key,
            batch_size=train_conf.batch_size,
            perc=train_conf.perc_train_set,
            pos_fraction=-1,
        )

        loss_conf.steps_per_epoch = ntrain_batches
        params_model, opt_state, train_metrics, key = process_train_epoch(
            model,
            opt_state=opt_state,
            x_data=x_shuffled,
            y_data=y_shuffled,
            update=optimizer.update,
            key=key,
            lookup_func=lookup_table.soft_get_local,
            loss_params=loss_conf,
        )
        model = eqx.combine(params_model, static_model)

        log_msg = log_train_metrics(train_metrics, model.params_to_dict(), epoch, writer)
        writer.add_scalar(
            "lr/learning_rate", np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))), epoch
        )
        logger.info(log_msg)

        if epoch % train_conf.validate_every == 0:
            val_model = eqx.nn.inference_mode(model, value=True)
            model = update_betas(
                model,
                train_metrics.sofa_d2_p.flatten(),
                train_metrics.susp_inf_p.flatten(),
                jnp.asarray((y_shuffled[..., -1] == 1.0).any(axis=-1).flatten(), dtype=jnp.float32),
            )
            logger.error(model.betas)
            key, _ = jr.split(key)
            key, val_metrics = process_val_epoch(
                val_model,
                x_data=val_x,
                y_data=val_y,
                step=opt_state[1][0].count,
                key=key,
                lookup_func=lookup_table.hard_get_fsq,
                loss_params=loss_conf,
            )
            log_msg = log_val_metrics(val_metrics, val_y, lookup_table, epoch, writer)
            logger.warning(log_msg)
            del val_metrics
            model = eqx.nn.inference_mode(val_model, value=False)

        del x_shuffled, y_shuffled
        del train_metrics

        # --- Save checkpoint ---
        if (epoch + 1) % save_conf.save_every == 0 and save_conf.perform:
            save_dir = writer.get_logdir() if not load_conf.from_dir else load_conf.from_dir
            save_checkpoint(
                save_dir + "/checkpoints",
                epoch,
                model,
                opt_state,
                hyper_enc,
                hyper_dec,
                hyper_pred,
            )
    writer.close()
