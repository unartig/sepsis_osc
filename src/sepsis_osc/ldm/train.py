import gc
import logging
from collections.abc import Callable
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree, jaxtyped
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import (
    CoordinateEncoder,
    Decoder,
    InfectionPredictor,
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
    prob_increase,
)
from sepsis_osc.ldm.data_loading import get_data_sets, prepare_batches
from sepsis_osc.ldm.early_stopping import EarlyStopping
from sepsis_osc.ldm.gru import GRUPredictor, init_gru_weights
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
) -> tuple[Float[Array, "batch time latent_dim"], Float[Array, "batch time latent_dim"], Float[Array, "batch time"]]:
    B, T, _input_dim = x.shape

    batch_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(B))
    inf_keys = jax.vmap(model.inf_predictor.make_keys)(batch_keys)
    batch_time_keys = jax.vmap(jax.vmap(jr.fold_in, in_axes=(None, 0)), in_axes=(0, None))(batch_keys, jnp.arange(T))
    enc_keys = jax.vmap(jax.vmap(model.encoder.make_keys))(batch_time_keys)

    # --------- Prediction
    zbeta, zsigma, h = jax.vmap(jax.vmap(model.encoder))(x, dropout_keys=enc_keys)
    inf0 = jax.vmap(model.inf_predictor)(x[:, 0], dropout_keys=inf_keys)

    def predict_next(
        carry: tuple[Float[Array, " latent_dim"], Float[Array, " gru_h_dim"]], _: Float[Array, " z_dim"]
    ) -> tuple[tuple[Float[Array, " latent_dim"], Float[Array, " gru_h_dim"]], Float[Array, " * z_dim"]]:
        z_prev, h_prev = carry
        dz, h_next = jax.vmap(model.predictor)(z_prev, h_prev)
        z_next = z_prev + dz
        new_carry = (z_next, h_next)
        return new_carry, z_next

    zbeta0, zsigma0, h0 = zbeta[:, 0], zsigma[:, 0], h[:, 0]
    z0 = jnp.concatenate([zbeta0, zsigma0], axis=-1)
    carry_init = (z0, h0)  # z0 = (batch, z_dim)
    (_, _), zv_preds = jax.lax.scan(predict_next, carry_init, xs=None, length=T - 1)

    # build full time sequence
    enc_seq = jax.nn.sigmoid(jnp.concatenate([zbeta, zsigma], axis=-1))  # (T, batch, z_dim)
    z_seq = jax.nn.sigmoid(
        jnp.transpose(jnp.concatenate([z0[None, ...], zv_preds], axis=0), (1, 0, 2))
    )  # (T, batch, z_dim)

    return (
        constrain_z(z_seq),  # (batch, T, z_dim)
        constrain_z(enc_seq),
        (jnp.ones_like(z_seq[..., 0]) * inf0),  # (batch, T)
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
    ordinal_thresholds = model.ordinal_thresholds

    d2_anneal = cosine_annealing(0.0, 200, 200, step)

    z_seq, enc_seq, infection_pred = get_latent_seq(model, x, key=key)  # (batch, T, z_dim)

    aux.alpha, aux.beta, aux.sigma = jnp.ones_like(z_seq[..., 0]) * model.alpha, z_seq[..., 0], z_seq[..., 1]

    # [0,1] -> [1,0] -> [12.5, 0] -> [13, 1] -> [26, 2] -> [27, 3]
    sofa_pred = jax.vmap(lookup_func, in_axes=(1, None, None))(
        jnp.concat([aux.alpha[..., None], z_seq], axis=-1), model.lookup_temperature, model.kernel_size
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
        jnp.concat(
            [
                z_seq,
                jax.lax.stop_gradient(infection_pred[..., None]),
                jax.lax.stop_gradient(sofa_pred[..., None]),
            ],
            axis=-1,
        )
    )
    aux.recon_loss = jnp.mean((x - x_recon) ** 2)

    # jax.debug.print("ORG{x}\nREC{y}", x=x[0, 0], y=x_recon[0, 0])
    # --------- Sequence Loss
    aux.sequence_loss = jnp.mean((z_seq[:, 1:] - jax.lax.stop_gradient(enc_seq[:, 1:])) ** 2)

    # --------- Total correlation loss (one per batch)
    # aux.tc_loss = calc_tc_loss(z_seq)

    # --------- Concept losses
    class_weights = jnp.log(1.0 + 1.0 / (model.sofa_dist + EPS))
    weights_per_sample = class_weights[sofa_true.astype(jnp.int32)]  # [B]
    aux.sofa_loss_t = (sofa_pred - (sofa_true / 24.0)) ** 2 * weights_per_sample
    aux.sofa_d2_p = jax.vmap(prob_increase, in_axes=(0, None, None))(sofa_pred, model.d_diff, model.d_scale)

    aux.sofa_d2_p_loss = optax.sigmoid_binary_cross_entropy(
        binary_logits(aux.sofa_d2_p), (jnp.diff(sofa_true / 24.0, axis=-1) > 0.0).any(axis=-1).astype(jnp.float32)
    )
    aux.infection_p_loss_t = jax.vmap(optax.sigmoid_binary_cross_entropy)(
        binary_logits(infection_pred.mean(axis=-1)),
        infection_true.max(axis=-1),
    )
    aux.susp_inf_p = 1.0 - jnp.prod(1.0 - infection_pred.squeeze(), axis=-1)

    aux.sep3_p = jax.lax.stop_gradient(aux.sofa_d2_p) * jax.lax.stop_gradient(aux.susp_inf_p)
    # aux.sep3_p_loss = optax.sigmoid_focal_loss(
    #     binary_logits(aux.sep3_p), (sepsis_true == 1.0).any(axis=-1), alpha=0.99, gamma=1.0
    # )

    # --------- Directional Loss
    aux.sofa_directional_loss = jnp.mean(jax.vmap(calc_full_directional_loss)(sofa_pred, sofa_true / 24.0))

    # --------- Spreading Loss
    # cov = jnp.cov(z_seq[:, 0], rowvar=False)
    zcov = jnp.cov(z_seq.reshape(-1, z_seq.shape[-1]), rowvar=False)
    zcov_loss = -jnp.log(jnp.linalg.det(zcov + EPS))
    aux.spreading_loss = zcov_loss

    # scov = jnp.cov(sofa_pred.reshape((-1, sofa_pred.shape[-1])), rowvar=False)
    # scov_loss = -jnp.log(jnp.linalg.det(scov + EPS))
    # aux.spreading_loss += scov_loss

    # --------- Distribution Loss
    # pred_sofa_hist, _ = jnp.histogram(
    #     sofa_pred.reshape(-1, z_seq.shape[-1]), range=(0.0, 1.0), bins=model.sofa_dist.size, density=True
    # )
    # pred_sofa_cdf = jnp.cumsum(pred_sofa_hist, axis=0)
    # true_sofa_cdf = jnp.cumsum(model.sofa_dist, axis=0)
    # aux.distr_loss = jnp.mean(jnp.abs(pred_sofa_cdf - true_sofa_cdf))

    # --------- Total Loss
    sofa_score_pred = jnp.sum(sofa_pred[..., None] > ordinal_thresholds, axis=-1)
    aux.hists_sofa_score = jax.lax.stop_gradient(sofa_score_pred)
    aux.hists_inf_prob = jax.lax.stop_gradient(infection_pred)
    aux.total_loss = (
        uncertainty_scale(aux.recon_loss * params.w_recon, model.recon_lsigma)
        + uncertainty_scale(aux.sequence_loss * params.w_sequence, model.seq_lsigma)
        + uncertainty_scale(jnp.mean(aux.sofa_d2_p_loss) * params.w_sofa_d2, model.sofa_d2_lsigma) * d2_anneal
        + uncertainty_scale(jnp.mean(aux.infection_p_loss_t) * params.w_inf, model.inf_lsigma)
        + uncertainty_scale(jnp.mean(aux.sofa_loss_t) * params.w_sofa_classification, model.sofa_class_lsigma)
        + uncertainty_scale(aux.sofa_directional_loss * params.w_sofa_direction, model.sofa_dir_lsigma)
        + uncertainty_scale(aux.spreading_loss * params.w_spreading, model.spread_lsigma)
        + uncertainty_scale(aux.tc_loss * params.w_tc, model.tc_lsigma)
        + uncertainty_scale(aux.distr_loss * params.w_distr, model.distr_lsigma)
    )
    return aux.total_loss, aux


@timing
@eqx.filter_jit
def process_train_epoch(
    model: LatentDynamicsModel,
    opt_state: optax.OptState,
    x_data: Float[Array, "ntbatches batch time input_dim"],
    y_data: Float[Array, "ntbatches batch time 3"],
    update: Callable,
    param_filter: PyTree,
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
        _model_params, opt_state, key = carry
        batch_x, batch_true_c = batch
        _model = eqx.combine(_model_params, model_static)

        (_, aux_losses), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            _model,
            x=batch_x,
            true_concepts=batch_true_c,
            step=opt_state[1][0].count,
            key=key,
            calibrate_probs_func=calibrate_probs_func,
            lookup_func=lookup_func,
            params=loss_params,
        )

        updates, opt_state = update(grads, opt_state, _model_params)
        updates = eqx.filter(updates, param_filter)
        model_params = eqx.apply_updates(_model_params, updates)

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
            lookup_func=lookup_func,
            calibrate_probs_func=calibrate_probs_func,
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
        input_dim=52,
        enc_hidden=64,
        inf_pred_hidden=32,
        dec_hidden=32,
        predictor_z_hidden=4,
        dropout_rate=0.3,
    )
    train_conf = TrainingConfig(
        batch_size=1024, window_len=1 + 6, epochs=int(10e3), perc_train_set=1.0, validate_every=5
    )
    lr_conf = LRConfig(init=0.0, peak=5e-3, peak_decay=0.5, end=1e-8, warmup_epochs=10, enc_wd=1e-3, grad_norm=1.0)
    loss_conf = LossesConfig(
        w_recon=1.0,
        w_sequence=1e-1,
        w_tc=0.0,
        w_acceleration=0.0,
        w_velocity=0.0,
        w_diff=0.0,
        w_sofa_direction=10.0,
        w_sofa_classification=10.0,
        w_sofa_d2=1e-3,
        w_inf=1.0,
        w_sep3=0.0,
        w_thresh=0.0,
        w_matching=0.0,
        w_spreading=1e-6,
        w_distr=0.0,
        anneal_threshs_iter=5 * 1 / train_conf.perc_train_set,
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=100, perform=True)

    # === Data ===
    train_x, train_y, val_x, val_y, test_x, test_y = get_data_sets(
        window_len=train_conf.window_len, swapaxes_y=(0, 2, 1), dtype=jnp.float32
    )
    # tmp_x, tmp_y = train_x, train_y
    # train_x, train_y = test_x, test_y
    # test_x, test_y = tmp_x, tmp_y
    # del tmp_x, tmp_y
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
        steps_per_cycle=[steps_per_epoch * n for n in [10]],  # cycle durations
        end_value=lr_conf.end,
        peak_decay=lr_conf.peak_decay,
    )

    # === Initialization ===
    key = jr.PRNGKey(jax_random_seed)
    key_enc, key_inf_predictor, key_dec, key_predictor = jr.split(key, 4)
    metrics = jnp.sort(lookup_table.relevant_metrics)
    # TODO dont do this for the windowed
    _sofas, counts = np.unique(test_y[..., 0], return_counts=True, sorted=True)
    counts = np.pad(counts, 24 - counts.size, constant_values=0)
    sofa_dist = counts / counts.sum()
    deltas = metrics[jnp.round(jnp.cumsum(sofa_dist) * len(metrics)).astype(dtype=jnp.int32)]
    deltas = jnp.ones_like(deltas)  # NOTE
    deltas = deltas / jnp.sum(deltas)
    deltas = jnp.asarray(deltas)

    encoder = CoordinateEncoder(
        key_enc,
        model_conf.input_dim,
        model_conf.enc_hidden,
        model_conf.predictor_z_hidden,
        dropout_rate=model_conf.dropout_rate,
        dtype=dtype,
    )
    inf_predictor = InfectionPredictor(
        key_inf_predictor,
        model_conf.input_dim,
        model_conf.inf_pred_hidden,
        dropout_rate=model_conf.dropout_rate,
        dtype=dtype,
    )
    stop_inf = EarlyStopping(patience=0, direction=-1)
    gru = GRUPredictor(
        key=key_predictor,
        z_dim=model_conf.z_latent_dim,
        z_hidden_dim=model_conf.predictor_z_hidden,
        dtype=dtype,
    )
    decoder = Decoder(
        key_dec,
        model_conf.input_dim,
        model_conf.z_latent_dim,
        model_conf.dec_hidden,
        dtype=dtype,
    )
    key_enc_weights, key_dec_weights, key_gru_weights = jr.split(key, 3)
    encoder = init_encoder_weights(encoder, key_enc_weights)
    decoder = init_decoder_weights(decoder, key_dec_weights)
    gru = init_gru_weights(gru, key_gru_weights)
    model = LatentDynamicsModel(
        encoder=encoder,
        inf_predictor=inf_predictor,
        predictor=gru,
        decoder=decoder,
        alpha=ALPHA,
        ordinal_deltas=deltas,
        sofa_dist=sofa_dist,
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
    model = eqx.nn.inference_mode(model, value=False)

    filter_spec = jtu.tree_map(lambda _: eqx.is_inexact_array, model)

    logger.info(
        f"Instatiated model with {model.n_params} parameters. "
        f"Encoder {model.encoder.n_params}, "
        f"Inf-Pred {model.inf_predictor.n_params}, "
        f"GRU {model.predictor.n_params}, "
        f"Decoder {model.decoder.n_params}"
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
        # kernel_size = int(1 + 2 * (1 + (np.round((1 - cosine_annealing(0.0, 500, 50, epoch))*10))))
        # if int(kernel_size) != int(model.kernel_size):
        #     model = eqx.tree_at(lambda x: x.kernel_size, model, kernel_size)
        #     print(model.kernel_size, kernel_size != model.kernel_size, kernel_size, type(model.kernel_size), type(kernel_size))
        params_model, opt_state, train_metrics, key = process_train_epoch(
            model,
            opt_state=opt_state,
            x_data=x_shuffled,
            y_data=y_shuffled,
            update=optimizer.update,
            key=key,
            calibrate_probs_func=calibration_model.get_sepsis_probs,
            # lookup_func=lookup_table.soft_get_global,
            lookup_func=lookup_table.soft_get_local,
            # lookup_func=lookup_table.soft_get_global if epoch < 200 else lookup_table.soft_get_local,
            loss_params=loss_conf,
            param_filter=filter_spec,
        )
        model = eqx.combine(params_model, static_model)

        log_msg = log_train_metrics(train_metrics, model.params_to_dict(), epoch, writer)
        writer.add_scalar(
            "lr/learning_rate", np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))), epoch
        )
        logger.info(log_msg)

        if epoch % train_conf.validate_every == 0:
            # calibration_model = calibration_model.calibrate(
            #     p_sofa=train_metrics.sofa_d2_p.flatten(),
            #     p_inf=train_metrics.susp_inf_p.flatten(),
            #     p_sepsis=jnp.asarray((y_shuffled[..., -1] == 1.0).any(axis=-1).flatten(), dtype=jnp.float32),
            # )
            # logger.error(f"betas: {calibration_model.betas}, a: {calibration_model.a} b: {calibration_model.b}")
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

            # trigger early stopping
            if stop_inf.step(val_metrics.infection_p_loss_t.mean()):
                logger.info("Freezing Infection Encoder Parameter")
                filter_spec = eqx.tree_at(
                    lambda m: m.inf_predictor,
                    filter_spec,
                    replace=False,
                )

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
