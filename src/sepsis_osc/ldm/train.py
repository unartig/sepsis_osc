import logging
from dataclasses import asdict
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, Bool, PyTree, jaxtyped
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import Decoder, Encoder, init_decoder_weights, init_encoder_weights
from sepsis_osc.ldm.data_loading import get_data_sets, prepare_batches
from sepsis_osc.ldm.gru import GRUPredictor
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel
from sepsis_osc.ldm.lookup import LatentLookup
from sepsis_osc.ldm.model_utils import (
    AuxLosses,
    ConceptLossConfig,
    LoadingConfig,
    LossesConfig,
    LRConfig,
    ModelConfig,
    SaveConfig,
    TrainingConfig,
    as_3d_indices,
    flatten_dict,
    load_checkpoint,
    log_train_metrics,
    log_val_metrics,
    save_checkpoint,
)
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.utils import timing

ALPHA_SPACE = (-1.0, 1.0, 0.04)
BETA_SPACE = (0.2, 0.5, 0.02)
SIGMA_SPACE = (0.0, 1.5, 0.04)

EPS = 1e-10


@jaxtyped(typechecker=typechecker)
def sofa_ps_from_logits(logits: Float[Array, "time n_classes_minus_1"]) -> Float[Array, "time n_classes"]:
    p_gt = jax.nn.sigmoid(logits)  # (T, K)
    p0 = 1.0 - p_gt[..., 0:1]  # P(S=0) shape (T,1)
    p_middle = p_gt[..., :-1] - p_gt[..., 1:]  # P(S=k) for 1..K-1  -> (T,K-1)
    plast = p_gt[..., -1:]  # P(S=K) shape (T,1)
    probs = jnp.concatenate([p0, p_middle, plast], axis=-1)  # (T,S)
    # numeric safety
    probs = jnp.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


@jaxtyped(typechecker=typechecker)
def sofa_event_prob(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    mask: Bool[Array, "time time"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, ""]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    T, S = sofa_probs.shape
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
    p_any = 1.0 - jnp.prod(1.0 - p_events)
    return p_any


@jaxtyped(typechecker=typechecker)
def ordinal_logits(
    z: Float[Array, "*"], thresholds: Float[Array, " n_classes_minus_1"], temperature: Float[Array, "1"]
) -> Float[Array, "* n_classes_minus_1"]:
    logits = (z.squeeze()[:, None] - thresholds) / (temperature + EPS)
    return logits


@jaxtyped(typechecker=typechecker)
def ordinal_loss(
    predicted_sofa: Float[Array, " batch"],
    true_sofa: Float[Array, " batch"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    sofa_dist: Float[Array, " n_classes"],
) -> Float[Array, ""]:
    # https://arxiv.org/abs/1901.07884
    # CORN (Cumulative Ordinal Regression)
    n_classes = thresholds.shape[-1] + 1

    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)

    targets = jnp.arange(n_classes - 1)  # [0, 1, ..., 22]
    true_sofa = true_sofa.squeeze()
    true_ord_targets = true_sofa[:, None] > targets[None, :]  # [B, 23]
    loss_per_sample = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_ord_targets), axis=-1)

    class_weights = jnp.log(1.0 + 1.0 / (sofa_dist + EPS))

    weights_per_sample = class_weights[true_sofa.astype(jnp.int32)]  # [B]
    weighted_loss = jnp.mean(weights_per_sample * loss_per_sample)

    return weighted_loss


@jaxtyped(typechecker=typechecker)
def binary_logits(probs: Float[Array, " batch"]) -> Float[Array, " batch"]:
    probs = jnp.clip(probs, EPS, 1 - EPS)
    return jnp.log(probs) - jnp.log1p(-probs)


@jaxtyped(typechecker=typechecker)
def binary_loss(
    predicted_infection: Float[Array, " batch"], true_infection: Float[Array, " batch"]
) -> Float[Array, ""]:
    logits = binary_logits(predicted_infection)
    loss_per_sample = optax.sigmoid_binary_cross_entropy(logits, true_infection > 0.0)
    weights_per_sample = 1 # + true_infection
    weighted_loss = loss_per_sample * weights_per_sample
    return jnp.mean(weighted_loss, axis=-1)


@jaxtyped(typechecker=typechecker)
def calc_directional_loss(pred_sofa: Int[Array, " time"], true_sofa: Float[Array, " time"]):
    tdelta = true_sofa[1:] - true_sofa[:-1]
    pdelta = pred_sofa[1:] - pred_sofa[:-1]
    return jnp.clip(-pdelta * tdelta, 0.0, jnp.inf)


@jaxtyped(typechecker=typechecker)
def calc_tc_loss(z_seq: Float[Array, "batch time latent_dim"]) -> Float[Array, ""]:
    z_flat = z_seq.reshape(-1, z_seq.shape[-1])
    z_centered = z_flat - jnp.mean(z_flat, axis=0, keepdims=True)

    cov = (z_centered.T @ z_centered) / z_flat.shape[0]

    var = jnp.diag(cov)
    std = jnp.sqrt(var + EPS)
    corr = cov / (std[:, None] * std[None, :])  # correlation matrix

    off_diag = corr - jnp.diag(jnp.diag(corr))
    tc_loss = jnp.sum(off_diag**2) / (z_flat.shape[-1] * (z_flat.shape[-1] - 1) + EPS)

    return tc_loss


def constrain_z(z: Float[Array, "batch time latent_dim"]) -> Float[Array, "batch time latent_dim"]:
    # TODO clip or no clip?
    z = jnp.clip(z, 0, 1)
    alpha, beta, sigma = jnp.split(z, 3, axis=-1)
    alpha = alpha * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0]
    beta = beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma = sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]
    return jnp.concatenate([alpha, beta, sigma], axis=-1)


def bound_z(z: Float[Array, "batch time latent_dim"]) -> Float[Array, "batch time latent_dim"]:
    alpha, beta, sigma = jnp.split(z, 3, axis=-1)
    alpha = alpha.clip(ALPHA_SPACE[0], ALPHA_SPACE[1])
    beta = beta.clip(BETA_SPACE[0], BETA_SPACE[1])
    sigma = sigma.clip(SIGMA_SPACE[0], SIGMA_SPACE[1])
    return jnp.concatenate([alpha, beta, sigma], axis=-1)


def cosine_annealing(base_val: float, num_steps: int, current_step: jnp.int32) -> Float[Array, ""]:
    step = jnp.clip(current_step, 0, num_steps)
    cosine = 0.5 * (1 + jnp.cos(jnp.pi * step / num_steps))
    return base_val + (1.0 - base_val) * (1.0 - cosine)


def uncertainty_scale(loss: Float[Array, "*"], log_sigma: Float[Array, ""]) -> Float[Array, "*"]:
    sigma_sq = jnp.exp(2 * log_sigma)
    return (loss / (2 * sigma_sq)) + log_sigma


@jaxtyped(typechecker=typechecker)
def get_latent_seq(
    model: LatentDynamicsModel, x: Float[Array, "batch time input_dim"], *, key: jnp.ndarray
) -> Float[Array, "batch time latent_dim"]:
    batch_size, T, input_dim = x.shape

    def make_keys(base_key):
        return jr.split(base_key, 4)

    sample_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(batch_size))
    drop_keys = jax.vmap(make_keys)(sample_keys)

    # --------- Prediction
    alpha0, beta0, sigma0, h_init = jax.vmap(model.encoder)(x[:, 0], dropout_keys=drop_keys)

    # NOTE alpha is only predicted for t=0, and stays constant throughout the prediction horizon for each patient
    z0 = jnp.concatenate([beta0, sigma0], axis=-1)

    def predict_next(carry, _):
        z_prev, h_prev = carry
        h_prev = h_prev.at[..., -1].set(alpha0.squeeze())  # information about alpha is carried in hidden state
        dz_next, h_next = jax.vmap(model.predictor)(z_prev, h_prev)
        z_next = z_prev + dz_next
        new_carry = (z_next, h_next)
        return new_carry, z_next

    carry_init = (z0, h_init)  # z0 = (batch, z_dim)
    (carry_final, h_final), z_preds = jax.lax.scan(predict_next, carry_init, xs=None, length=T - 1)

    # build full time sequence
    z_seq = jnp.concatenate([z0[None, ...], z_preds], axis=0)  # (T, batch, z_dim)
    # insert alpha0
    z_seq = jnp.concatenate([alpha0[None, ...].repeat(z_seq.shape[0], axis=0), z_seq], axis=-1)
    z_seq = constrain_z(jnp.transpose(z_seq, (1, 0, 2)))  # (batch, T, z_dim)
    return z_seq


# === Loss Function ===
@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  # (donate="all")
def loss(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    true_concepts: Float[Array, "batch time 3"],
    step: Int[Array, ""] = jnp.astype(jnp.inf, jnp.int32),
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    aux = AuxLosses.empty()
    aux.anneal_concept = cosine_annealing(0.1, int(params.anneal_concept_iter * params.steps_per_epoch), step)
    aux.anneal_recon = cosine_annealing(0.001, int(params.anneal_recon_iter * params.steps_per_epoch), step)
    aux.anneal_threshs = cosine_annealing(0.0, int(params.anneal_threshs_iter * params.steps_per_epoch), step)

    ordinal_thresholds = model.ordinal_thresholds(aux.anneal_threshs)

    aux.thresh_loss = jnp.mean(
        jax.nn.relu((1 / 24) - jax.nn.softmax(model.learned_deltas))
    )  # punish deviations from uniform

    z_seq = get_latent_seq(model, x, key=key)  # (batch, T, z_dim)
    aux.alpha, aux.beta, aux.sigma = z_seq[..., 0].mean(), z_seq[..., 1].mean(), z_seq[..., 2].mean()

    pred_metrics = jax.vmap(lookup_func, in_axes=(1, None))(z_seq, model.lookup_temperature).swapaxes(
        0, 1
    )  # (T, batch, 2) -> (batch, T, 2)
    sofa_pred = pred_metrics[..., 0] #  ** model.sofa_exp  # (batch, T, 1)
    infection_pred = pred_metrics[..., 1] ** model.inf_exp  # (batch, T, 1)

    # --------- Difference Loss
    # try to hvae uniformly distributed metrics
    aux.diff_loss = (sofa_pred.var() - 1 / 12) ** 2

    # --------- Acceleration Loss
    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    aux.accel_loss = jnp.mean((z_seq[:, 2:] - 2 * z_seq[:, 1:-1] + z_seq[:, :-2]) ** 2)

    # --------- Recon Loss
    # TODO x_recon_loss_t
    x_recon = jax.vmap(jax.vmap(model.decoder))(z_seq)
    aux.recon_loss = jnp.mean((x - x_recon) ** 2)

    # --------- Total correlation loss (one per batch)
    aux.tc_loss = calc_tc_loss(z_seq)

    # --------- Concept losses
    sofa_true, infection_true, sepsis_true = true_concepts[..., 0], true_concepts[..., 1], true_concepts[..., 2]

    aux.sofa_loss_t = jax.vmap(ordinal_loss, in_axes=(0, 0, None, None, None))(
        sofa_pred, sofa_true, ordinal_thresholds, model.label_temperature, model.sofa_dist
    ).T
    sofa_score_pred = jnp.sum(sofa_pred[..., None] > ordinal_thresholds, axis=-1)

    aux.infection_loss_t = jax.vmap(binary_loss)(infection_pred, infection_true)
    infection_err = infection_pred - (infection_true > 0)

    mask =jnp.triu(jnp.ones((x.shape[1], x.shape[1])), k=1).astype(bool) 
    aux.sofa_d2_p = jax.vmap(sofa_event_prob, in_axes=(0, None, None, None, None, None))(
        sofa_pred, ordinal_thresholds, model.label_temperature, mask, jnp.array([0.5]), 1.5
        # sofa_pred, ordinal_thresholds, model.label_temperature, mask, model.delta_temperature, 1.5
    )
    aux.susp_inf_p = jax.nn.logsumexp(infection_pred, axis=-1)
    aux.sep3_p = aux.sofa_d2_p * aux.susp_inf_p
    aux.sep3_loss_t = jnp.mean(optax.sigmoid_binary_cross_entropy(aux.sep3_p, sepsis_true.any(axis=-1)))
    aux.concept_loss = jnp.mean(
        aux.sofa_loss_t * params.concept.w_sofa
        + aux.infection_loss_t * params.concept.w_inf
    )

    # --------- Directional Loss
    aux.directional_loss = jnp.mean(jax.vmap(calc_directional_loss)(sofa_score_pred, sofa_true))

    # --------- Total Loss
    aux.hists_sofa_metric = jax.lax.stop_gradient(pred_metrics[..., 0].reshape(-1))
    aux.hists_sofa_score = jax.lax.stop_gradient(sofa_score_pred)
    aux.hists_inf_error = jax.lax.stop_gradient(infection_err)
    aux.total_loss = (
        (aux.recon_loss * params.w_recon * aux.anneal_recon)
        + (aux.concept_loss * params.w_concept * aux.anneal_concept)
        + aux.tc_loss * params.w_tc
        + aux.accel_loss * params.w_accel
        + aux.directional_loss * params.w_direction
        + aux.diff_loss * params.w_diff
        + aux.thresh_loss * params.w_thresh
        + jnp.mean(aux.sep3_loss_t) * params.concept.w_sep3
    )
    return aux.total_loss, aux


@eqx.filter_jit
def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    model_params,
    model_static,
    opt_state: optax.OptState,
    update: Callable,
    *,
    key,
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

    def scan_step(carry, batch):
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

    def scan_step(carry, batch):
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
):
    cycle_boundaries = jnp.array([warmup_steps + sum(steps_per_cycle[:i]) for i in range(len(steps_per_cycle))])
    cycle_lengths = jnp.array(steps_per_cycle)
    num_cycles = len(steps_per_cycle)

    def schedule(step):
        step = jnp.asarray(step)

        # --- Warmup ---
        def in_warmup_fn(step):
            frac = step / jnp.maximum(warmup_steps, 1)
            return (init_value + frac * (peak_value - init_value)).astype(jnp.float64)

        # --- Cosine Decay Cycles ---
        def in_decay_fn(step):
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
    # jax.profiler.start_trace("tmp/profile-data")
    setup_jax(simulation=False)
    setup_logging("info")
    dtype = jnp.float32
    logger = logging.getLogger(__name__)

    model_conf = ModelConfig(
        latent_dim=3, input_dim=52, enc_hidden=512, dec_hidden=64, predictor_hidden=5, dropout_rate=0.5
    )
    train_conf = TrainingConfig(batch_size=1024, window_len=6, epochs=9000, perc_train_set=0.222, validate_every=5)
    lr_conf = LRConfig(init=0.0, peak=5e-3, peak_decay=0.5, end=1e-12, warmup_epochs=3, enc_wd=1e-4, grad_norm=0.7)
    loss_conf = LossesConfig(
        w_concept=10.0,
        w_recon=0.01,
        w_tc=0.0,
        w_accel=0.0,
        w_diff=0.0,
        w_direction=5.0,
        w_thresh=0.0,
        anneal_concept_iter=100 * 1 / train_conf.perc_train_set,
        anneal_recon_iter=20 * 1 / train_conf.perc_train_set,
        anneal_threshs_iter=50 * 1 / train_conf.perc_train_set,
        concept=ConceptLossConfig(w_sofa=1.0, w_inf=1.0, w_sep3=20.0),
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=10, perform=True)

    # === Data ===
    train_x, train_y, val_x, val_y, test_x, test_y = get_data_sets(window_len=train_conf.window_len, dtype=jnp.float32)
    logger.error(f"{jnp.unique(train_y[..., 0])}")
    logger.error(f"{jnp.unique(train_y[..., 1])}")
    logger.error(f"{jnp.unique(train_y[..., 2])}")
    logger.info(f"Train shape      - X {train_y.shape}, Y {train_x.shape}")
    logger.info(f"Validation shape - X {val_y.shape},  Y {val_x.shape}")
    logger.info(f"Test shape       - X {test_y.shape},  Y {test_x.shape}")

    db_str = "Daisy"
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
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params), proto_metric=DNMMetrics)
    metrics_3d = metrics_3d.to_jax()
    lookup_table = LatentLookup(
        metrics=metrics_3d.reshape((-1, 1)),
        indices=indices_3d.reshape((-1, 3)),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=dtype,
    )
    steps_per_epoch = (train_x.shape[0] // train_conf.batch_size) * train_conf.batch_size
    schedule = custom_warmup_cosine(
        init_value=lr_conf.init,
        peak_value=lr_conf.peak,
        warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
        steps_per_cycle=[steps_per_epoch * n for n in [1000]],  # cycle durations
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
    # deltas = jnp.ones_like(deltas)  # NOTE
    deltas = deltas / jnp.sum(deltas)
    deltas = jnp.asarray(deltas)

    encoder = Encoder(
        key_enc,
        model_conf.input_dim,
        model_conf.latent_dim,
        model_conf.enc_hidden,
        model_conf.predictor_hidden,
        dropout_rate=model_conf.dropout_rate,
        dtype=dtype,
    )
    decoder = Decoder(key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden, dtype=dtype)
    key_enc_weights, key_dec_weights = jr.split(key, 2)
    encoder = init_encoder_weights(encoder, key_enc_weights)
    decoder = init_decoder_weights(decoder, key_dec_weights)
    gru = GRUPredictor(
        key=key_predictor,
        dim=model_conf.latent_dim - 1,
        hidden_dim=model_conf.predictor_hidden,
        dtype=dtype,
    )

    model = LatentDynamicsModel(
        encoder=encoder,
        predictor=gru,
        decoder=decoder,
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
    hyper_pred = {"dim": model_conf.latent_dim - 1, "hidden_dim": gru.hidden_dim}
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
    hyper_params = flatten_dict({
        "model": asdict(model_conf),
        "losses": asdict(loss_conf),
        "train": asdict(train_conf),
        "lr": asdict(lr_conf),
    })
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
        logger.info(log_msg)

        val_model = eqx.nn.inference_mode(model, value=True)
        if epoch % train_conf.validate_every == 0:
            key, _ = jr.split(key)
            key, val_metrics = process_val_epoch(
                val_model,
                x_data=val_x,
                y_data=val_y,
                step=opt_state[1][0].count,
                key=key,
                lookup_func=lookup_table.hard_get_local,
                loss_params=loss_conf,
            )
            log_msg = log_val_metrics(val_metrics, val_y, epoch, writer)
            logger.warning(log_msg)
        model = eqx.nn.inference_mode(val_model, value=False)

        writer.add_scalar(
            "lr/learning_rate", np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))), epoch
        )

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
