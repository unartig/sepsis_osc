import gc
import logging
from collections.abc import Callable
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree, jaxtyped
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import (
    Decoder,
    Encoder,
    init_decoder_weights,
    init_encoder_weights,
)
from sepsis_osc.ldm.calibration_model import CalibrationModel
from sepsis_osc.ldm.checkpoint_utils import load_checkpoint, save_checkpoint
from sepsis_osc.ldm.commons import (
    approx_similarity_loss,
    binary_logits,
    custom_warmup_cosine,
    mahalanobis_similarity_loss,
    ordinal_logits,
    prob_increase,
    soft_entropy,
    uniform_cdf_loss,
)
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
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel
from sepsis_osc.ldm.logging_utils import (
    flatten_dict,
    log_train_metrics,
    log_val_metrics,
)
from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed
from sepsis_osc.utils.jax_config import EPS, setup_jax, typechecker
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.utils import timing

INT_INF = jnp.astype(jnp.inf, jnp.int32)


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
    return jax.nn.softplus(-pdelta * tdelta) * jax.nn.relu(tdelta)


@jaxtyped(typechecker=typechecker)
def calc_full_directional_loss(
    pred_sofa: Float[Array, " time"],
    true_sofa: Float[Array, " time"],
    reward_weight: float = 0.5,
    penalty_weight: float = 1.0,
) -> Float[Array, ""]:
    T = pred_sofa.shape[0]
    dp = pred_sofa[None, :] - pred_sofa[:, None]
    dt = true_sofa[None, :] - true_sofa[:, None]
    mask = jnp.triu(jnp.ones((T, T)), k=1)
    valid = jnp.abs(dt) > EPS
    weight = jnp.abs(dt) + 1

    alignment = dp * dt
    penalty = (
        penalty_weight * jax.nn.softplus(-alignment)
        # - reward_weight * jax.nn.softplus(alignment)
    )
    return (penalty * mask * valid * weight).sum()


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
    # sigma_sq = jnp.exp(2 * log_sigma)
    # return ((loss / (2 * sigma_sq)) + log_sigma).squeeze()


@jaxtyped(typechecker=typechecker)
def get_latent_seq(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    *,
    key: jnp.ndarray,
) -> tuple[Float[Array, "batch time zlatent_dim"], Float[Array, "batch time vlatent_dim"]]:
    batch_size, T, _input_dim = x.shape

    sample_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(batch_size))
    drop_keys = jax.vmap(model.encoder.make_keys)(sample_keys)

    # --------- Prediction
    zbeta0, zsigma0, vinf0, h_0 = jax.vmap(model.encoder)(x[:, 0], dropout_keys=drop_keys)

    # NOTE alpha is only predicted for t=0, and stays constant throughout the prediction horizon for each patient
    zv0 = jnp.concatenate([zbeta0, zsigma0, vinf0], axis=-1)

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
def loss(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    true_concepts: Float[Array, "batch time 3"],
    step: Int[Array, ""] = INT_INF,
    *,
    key: jnp.ndarray,
    calibrate_probs_func: Callable,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    aux = AuxLosses.empty()
    sofa_true, infection_true, sepsis_true = true_concepts[..., 0], true_concepts[..., 1], true_concepts[..., 2]
    ordinal_thresholds = model.ordinal_thresholds(jnp.array(0.0))


    z_seq, infection_pred = get_latent_seq(model, x, key=key)  # (batch, T, z_dim)

    aux.alpha, aux.beta, aux.sigma = jnp.ones_like(z_seq[..., 0]) * model.alpha, z_seq[..., 0], z_seq[..., 1]

    sofa_pred = jax.vmap(lookup_func, in_axes=(1, None, None))(
        jnp.concat([aux.alpha[..., None], z_seq], axis=-1), model.lookup_temperature, 3
    ).swapaxes(0, 1)  # (T, batch, 2) -> (batch, T, 2)
    # NOTE

    # --------- Matching Loss
    # aux.matching_loss = jnp.mean(((vsofa - sofa_pred) ** 2) + ((vinf - infection_pred) ** 2))

    # --------- Difference Loss
    # aux.diff_loss = (sofa_pred.var() - 1 / 12) ** 2

    # --------- Acceleration Loss
    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    # aux.velocity_loss = -1 * jnp.mean((z_seq[:, 1:] - z_seq[:, :-1]) ** 2)
    # aux.acceleration_loss = jnp.mean((z_seq[:, 2:] - 2 * z_seq[:, 1:-1] + z_seq[:, :-2]) ** 2)


    # --------- Recon Loss
    x_recon = jax.vmap(jax.vmap(model.decoder))(
        jnp.concat([z_seq, infection_pred, sofa_pred[..., None]], axis=-1)
    )
    aux.recon_loss = jnp.mean((x - x_recon) ** 2)
    sample_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(x.shape[0]))
    sample_keys = jax.vmap(lambda k: jr.split(k, x.shape[1] - 1))(sample_keys)
    drop_keys = jax.vmap(jax.vmap(model.encoder.make_keys))(sample_keys)
    beta_enc, sigma_enc, _, _ = jax.vmap(jax.vmap(model.encoder))(x[:, 1:], dropout_keys=drop_keys)
    enc_seq = constrain_z(jax.nn.sigmoid(jnp.concat([beta_enc, sigma_enc], axis=-1)))
    aux.sequence_loss = jnp.mean((z_seq[:, 1:] - enc_seq) ** 2)

    # --------- Total correlation loss (one per batch)
    # aux.tc_loss = calc_tc_loss(z_seq)

    # --------- Concept losses
    class_weights = jnp.log(1.0 + 1.0 / (model.sofa_dist + EPS))
    weights_per_sample = class_weights[sofa_true.astype(jnp.int32)]  # [B]
    aux.sofa_loss_t = (sofa_pred - (sofa_true / 24.0)) ** 2 * weights_per_sample
    aux.sofa_d2_p = jax.vmap(prob_increase, in_axes=(0, None, None))(
        sofa_pred, model.d_diff, model.d_scale
    )

    aux.sofa_d2_p_loss = optax.sigmoid_binary_cross_entropy(
        binary_logits(aux.sofa_d2_p),
        (jnp.diff(sofa_true / 24.0, axis=-1) > 0.0).any(axis=-1).astype(jnp.float32),
    )
    aux.infection_p_loss_t = jax.vmap(optax.sigmoid_binary_cross_entropy)(
        binary_logits(infection_pred),
        infection_true,
    )
    aux.susp_inf_p = 1.0 - jnp.prod(1.0 - infection_pred.squeeze(), axis=-1)

    aux.sep3_p = calibrate_probs_func(jax.lax.stop_gradient(aux.sofa_d2_p), jax.lax.stop_gradient(aux.susp_inf_p))
    aux.sep3_p_loss = optax.sigmoid_focal_loss(
        binary_logits(aux.sep3_p), (sepsis_true == 1.0).any(axis=-1), alpha=0.99, gamma=1.0
    )

    # --------- Directional Loss
    aux.directional_loss = jnp.mean(jax.vmap(calc_full_directional_loss)(sofa_pred, sofa_true / 24.0))

    # --------- Spreading Loss
    aux.spreading_loss = mahalanobis_similarity_loss(x[:, 0], z_seq[:, 0], model.input_sim_scale).sum()

    # --------- Total Loss
    sofa_score_pred = jnp.sum(sofa_pred[..., None] > ordinal_thresholds, axis=-1)
    aux.hists_sofa_score = jax.lax.stop_gradient(sofa_score_pred)
    aux.hists_inf_prob = jax.lax.stop_gradient(infection_pred)
    aux.total_loss = (
        uncertainty_scale(aux.recon_loss * params.w_recon, model.recon_lsigma)
        + uncertainty_scale(aux.sequence_loss * params.w_sequence, model.seq_lsigma)
        + jnp.mean(aux.sofa_d2_p_loss) * params.w_sofa_d2
        # + uncertainty_scale(jnp.mean(aux.sofa_d2_p_loss) * params.w_sofa_d2, model.sofa_d2_lsigma)
        # + uncertainty_scale(jnp.mean(aux.infection_p_loss_t) * params.w_inf, model.inf_lsigma)
        + uncertainty_scale(jnp.mean(aux.infection_p_loss_t) * params.w_inf, model.inf_lsigma)
        # SOFA only for first time point
        + uncertainty_scale(jnp.mean(aux.sofa_loss_t) * params.w_sofa_classification, model.sofa_class_lsigma)
        # + uncertainty_scale(jnp.mean(aux.sofa_loss_t[:, 0:2]) * params.w_sofa_classification, model.sofa_lsigma)
        # + jnp.mean(aux.sofa_loss_t * jnp.array([1.0, 1.0, 0.5, 0.25, 0.125, 0.0625])[None, :])
        # * params.w_sofa_classification
        # * aux.anneal_threshs
        # + uncertainty_scale(aux.directional_loss * params.w_sofa_direction * aux.anneal_threshs, model.sep3_lsigma)
        + uncertainty_scale(aux.directional_loss * params.w_sofa_direction, model.sofa_dir_lsigma)
        # + jnp.mean(aux.sep3_p_loss) * params.w_sep3  #  * aux.anneal_threshs
        # + aux.matching_loss * params.w_matching
        # + aux.trend_loss * params.w_sofa_trend
        # + aux.tc_loss * params.w_tc
        # + aux.acceleration_loss * params.w_acceleration
        # + uncertainty_scale(aux.velocity_loss * params.w_velocity, model.velo_lsigma)
        + uncertainty_scale(aux.spreading_loss * params.w_spreading, model.spread_lsigma)
        # + aux.diff_loss * params.w_diff
        # + aux.thresh_loss * params.w_thresh
    )
    return aux.total_loss, aux


def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    model_params: PyTree,
    model_static: PyTree,
    opt_state: optax.OptState,
    update: Callable,
    *,
    key: jnp.ndarray,
    calibrate_probs_func: Callable,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[PyTree, PyTree, tuple[Array, AuxLosses]]:
    model = eqx.combine(model_params, model_static)
    (total_loss, aux_losses), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        model,
        calibrate_probs_func=calibrate_probs_func,
        x=x_batch,
        true_concepts=true_c_batch,
        step=opt_state[1][0].count,
        key=key,
        lookup_func=lookup_func,
        params=loss_params,
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
    calibrate_probs_func: Callable,
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
            x_batch=batch_x,
            true_c_batch=batch_true_c,
            model_params=model_params,
            model_static=model_static,
            opt_state=opt_state,
            update=update,
            key=batch_key,
            calibrate_probs_func=calibrate_probs_func,
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
    calibrate_probs_func: Callable,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[AuxLosses]:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    def scan_step(
        carry: tuple[PyTree, jnp.ndarray, int],
        batch: tuple[Float[Array, "batch time input_dim"], Float[Array, "bath time 3"]],
    ) -> tuple[tuple[PyTree, jnp.ndarray, int], AuxLosses]:
        model_flat, key, i = carry
        batch_x, batch_true_c = batch

        batch_key = jr.fold_in(key, i)

        model_full = eqx.combine(model_flat, model_static)

        _, aux_losses = loss(
            model=model_full,
            x=batch_x,
            true_concepts=batch_true_c,
            key=batch_key,
            calibrate_probs_func=calibrate_probs_func,
            lookup_func=lookup_func,
            params=loss_params,
            step=step,
        )

        return (model_flat, key, i + 1), aux_losses

    carry = (model_params, key, int(0.0))
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    model_params, key, _ = carry

    return step_losses


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info")
    dtype = jnp.float32
    logger = logging.getLogger(__name__)

    model_conf = ModelConfig(
        z_latent_dim=2,
        v_latent_dim=1,
        input_dim=52,
        enc_hidden=128,
        dec_hidden=64,
        predictor_z_hidden=4,
        predictor_v_hidden=2,
        dropout_rate=0.3,
    )
    train_conf = TrainingConfig(
        batch_size=512, window_len=1 + 6, epochs=int(10e3), perc_train_set=1.0, validate_every=5
    )
    lr_conf = LRConfig(init=0.0, peak=5e-3, peak_decay=0.5, end=1e-8, warmup_epochs=100, enc_wd=1e-3, grad_norm=1.0)
    loss_conf = LossesConfig(
        w_recon=1e-1,
        w_sequence=1.0,
        w_tc=0.0,
        w_acceleration=0.0,
        w_velocity=0.0,
        w_diff=0.0,
        w_sofa_direction=30.0,
        w_sofa_trend=0.0,
        w_sofa_classification=100.0,
        w_sofa_d2=10.0,
        w_inf=0.0,
        w_inf_alpha=0.0,
        w_inf_gamma=0.0,
        w_sep3=0.0,
        w_thresh=0.0,
        w_matching=0.0,
        w_spreading= 2.0,
        anneal_threshs_iter=5 * 1 / train_conf.perc_train_set,
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=100, perform=True)

    # === Data ===
    train_x, train_y, val_x, val_y, test_x, test_y = get_data_sets(window_len=train_conf.window_len, dtype=jnp.float32)
    logger.error(f"{jnp.unique(train_y[..., 0])}")
    logger.error(f"{jnp.unique(train_y[..., 1])}")
    logger.error(f"{jnp.unique(train_y[..., 2])}")
    logger.info(f"Train shape      - Y {train_y.shape}, X {train_x.shape}")
    logger.info(f"Validation shape - Y {val_y.shape},  X {val_x.shape}")
    logger.info(f"Test shape       - Y {test_y.shape},  X {test_x.shape}")

    db_str = "Daisy2"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    sim_storage.close()

    b, s = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
    a = np.ones_like(b) * ALPHA
    indices_3d = jnp.concatenate([a[..., np.newaxis], b[..., np.newaxis], s[..., np.newaxis]], axis=-1)[np.newaxis, ...]

    spacing_3d = jnp.array([0.0, BETA_SPACE[2], SIGMA_SPACE[2]])
    params = DNMConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
    metrics_3d = metrics_3d.to_jax().reshape([1, *metrics_3d.shape["r_1"]])
    lookup_table = LatentLookup(
        metrics=metrics_3d.reshape((-1, 1)),
        indices=indices_3d.reshape((-1, 3)),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=jnp.bfloat16,
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
    metrics = jnp.sort(lookup_table.relevant_metrics)
    # TODO dont do this for the windowed
    sofa_dist, _ = jnp.histogram(train_y[..., 0], bins=24, range=(0, 23), density=True)
    sofa_dist = sofa_dist / jnp.sum(sofa_dist)
    deltas = metrics[jnp.round(jnp.cumsum(sofa_dist) * len(metrics)).astype(dtype=jnp.int32)]
    deltas = jnp.ones_like(deltas)  # NOTE
    deltas = deltas / jnp.sum(deltas)
    deltas = jnp.asarray(deltas)

    encoder = Encoder(
        key_enc,
        model_conf.input_dim,
        model_conf.enc_hidden,
        model_conf.predictor_z_hidden + model_conf.predictor_v_hidden,
        dropout_rate=model_conf.dropout_rate,
        dtype=dtype,
    )
    decoder = Decoder(
        key_dec,
        model_conf.input_dim,
        model_conf.z_latent_dim,
        model_conf.v_latent_dim,
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
        alpha=ALPHA,
        ordinal_deltas=deltas,
        sofa_dist=sofa_dist[:-1],
    )
    hyper_enc, hyper_dec, hyper_pred = model.hypers_dict()
    calibration_model = CalibrationModel()

    optimizer = optax.chain(
        optax.clip_by_global_norm(lr_conf.grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
    )
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

    logger.info(
        f"Instatiated model with {model.n_params} parameters. Encoder {model.encoder.n_params}, GRU {model.predictor.n_params}, Decoder {model.decoder.n_params}"
    )
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

    shuffle_val_key = jr.fold_in(key, jax_random_seed)
    val_x, val_y, nval_batches = prepare_batches(val_x, val_y, train_conf.batch_size, key=shuffle_val_key)
    for epoch in range(load_conf.epoch if load_conf.from_dir else 0, train_conf.epochs):
        shuffle_key = jr.fold_in(key, epoch)
        train_key = jr.fold_in(key, epoch + 1)
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
            calibrate_probs_func=calibration_model.get_sepsis_probs,
            # lookup_func=lookup_table.soft_get_local,
            lookup_func=lookup_table.soft_get_global if epoch < 50 else lookup_table.soft_get_local,
            loss_params=loss_conf,
        )
        model = eqx.combine(params_model, static_model)

        log_msg = log_train_metrics(train_metrics, model.params_to_dict(), epoch, writer)
        writer.add_scalar(
            "lr/learning_rate", np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))), epoch
        )
        logger.info(log_msg)

        if epoch % train_conf.validate_every == 0:
            calibration_model = calibration_model.calibrate(
                p_sofa=train_metrics.sofa_d2_p.flatten(),
                p_inf=train_metrics.susp_inf_p.flatten(),
                p_sepsis=jnp.asarray((y_shuffled[..., -1] == 1.0).any(axis=-1).flatten(), dtype=jnp.float32),
            )
            logger.error(f"betas: {calibration_model.betas}, a: {calibration_model.a} b: {calibration_model.b}")
            val_model = eqx.nn.inference_mode(model, value=True)
            val_key = jr.fold_in(key, epoch + 2)
            val_metrics = process_val_epoch(
                val_model,
                x_data=val_x,
                y_data=val_y,
                step=opt_state[1][0].count,
                key=val_key,
                calibrate_probs_func=calibration_model.get_sepsis_probs,
                lookup_func=lookup_table.hard_get_fsq,
                loss_params=loss_conf,
            )
            log_msg = log_val_metrics(val_metrics, val_y, lookup_table, epoch, writer)
            logger.warning(log_msg)
            model = eqx.nn.inference_mode(val_model, value=False)
            del val_metrics, val_model

        del x_shuffled, y_shuffled
        del train_metrics
        gc.collect()

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
