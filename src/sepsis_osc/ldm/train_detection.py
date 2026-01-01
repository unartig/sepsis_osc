import gc
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import (
    Decoder,
    LatentEncoder,
    init_latent_encoder_weights,
)
from sepsis_osc.ldm.calibration_model import CalibrationModel

# from sepsis_osc.ldm.checkpoint_utils import load_checkpoint, save_checkpoint
from sepsis_osc.ldm.commons import (
    binary_logits,
    causal_probs,
    causal_smoothing,
    custom_warmup_cosine,
    prob_increase_steps,
    smooth_labels,
)
from sepsis_osc.ldm.data_loading import get_data_sets_detection, prepare_batches_mask
from sepsis_osc.ldm.gru_detector import GRUDetector, init_gru_weights
from sepsis_osc.ldm.helper_structs import LoadingConfig, LRConfig, SaveConfig, TrainingConfig
from sepsis_osc.ldm.logging_utils import flatten_dict, log_train_metrics, log_val_metrics
from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed
from sepsis_osc.utils.jax_config import EPS, setup_jax, typechecker
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.utils import timing

INT_INF = jnp.astype(jnp.inf, jnp.int32)
_1 = jnp.array(1)


@register_dataclass
@dataclass
class AuxLosses:
    beta: Array
    sigma: Array

    total_loss: Array
    recon_loss: Array

    hists_sofa_score: Array
    hists_sofa_metric: Array
    hists_inf_prob: Array

    total_loss: Array
    recon_loss: Array
    spreading_loss: Array
    sofa_loss_t: Array
    sofa_loss_sum: Array
    sofa_d2_p_loss: Array
    infection_p_loss_t: Array
    sep3_p_loss: Array
    sofa_directional_loss: Array

    sofa_d2_p: Array
    susp_inf_p: Array
    sep3_p: Array

    @staticmethod
    def empty() -> "AuxLosses":
        initialized_fields = {f.name: jnp.zeros(()) for f in fields(AuxLosses)}
        return AuxLosses(**initialized_fields)

    def to_dict(self) -> dict[str, dict[str, jnp.ndarray]]:
        return {
            "latents": {
                "alpha": jnp.ones_like(self.beta) * ALPHA,
                "beta": self.beta,
                "sigma": self.sigma,
            },
            "losses": {
                "total_loss": self.total_loss,
                "spreading_loss": self.spreading_loss,
                "recon_loss": self.recon_loss,
                "sofa_directional_loss": self.sofa_directional_loss,
                "sofa": self.sofa_loss_t,
                "sofa_sum": self.sofa_loss_sum,
                "sofa_d2": self.sofa_d2_p_loss,
                "infection": self.infection_p_loss_t,
                "sepsis-3": self.sep3_p_loss,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
            },
            "mult": {
                "sofa_t": self.sofa_loss_t,
            },
            "sepsis_metrics": {
                "sofa_d2_p": self.sofa_d2_p,
                "susp_inf_p": self.susp_inf_p,
                "sep3_p": self.sep3_p,
            },
        }


@register_dataclass
@dataclass
class LossesConfig:
    w_spreading: float
    w_recon: float
    w_sofa_direction: float
    w_sofa_classification: float
    w_sofa_d2: float
    w_inf: float
    w_sep3: float
    steps_per_epoch: int = 0


@jaxtyped(typechecker=typechecker)
def calc_full_directional_loss(
    pred_sofa: Float[Array, " time"],
    true_sofa: Float[Array, " time"],
    mask: Bool[Array, " time"],
) -> Float[Array, " time"]:
    T = pred_sofa.shape[0]
    dp = pred_sofa[None, :] - pred_sofa[:, None]
    dt = true_sofa[None, :] - true_sofa[:, None]
    triu_mask = jnp.triu(jnp.ones((T, T)), k=1)
    mask2d = mask[None, :] & mask[:, None]
    valid = jnp.abs(dt) > EPS
    weight = jnp.abs(dt) + 1.0

    alignment = dp * dt
    penalty = jax.nn.relu(-alignment)
    return (penalty * mask2d * triu_mask * valid * weight).sum(axis=0)


@jaxtyped(typechecker=typechecker)
def masked_cov(
    x: Float[Array, "batch time latent_dim"], mask: Bool[Array, "batch time"]
) -> Float[Array, "latent_dim latent_dim"]:
    mask_f = mask.astype(x.dtype)
    w = jnp.sum(mask_f)
    mean = jnp.sum(x * mask_f[..., None], axis=(0, 1)) / w
    xc = ((x - mean) * mask_f[..., None]).reshape(-1, x.shape[-1])
    cov = xc.T @ xc / (w - 1)
    return cov


def constrain_z(z: Float[Array, "batch time zlatent_dim"]) -> Float[Array, "batch time zlatent_dim"]:
    # z = jnp.clip(z, 0, 1)
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


def mask_padding(loss: Float[Array, "*"], mask: Float[Array, "*"] | None = None) -> Float[Array, "*"]:
    if mask is None:
        mask = jnp.ones_like(loss)
    print(mask.shape, loss.shape)
    return (loss * mask).sum() / mask.sum()


# === Loss Function ===
@jaxtyped(typechecker=typechecker)
def loss(
    model: GRUDetector,
    x: Float[Array, "batch time input_dim"],
    true_concepts: Float[Array, "batch time 3"],
    mask: Bool[Array, "batch time"] = _1,
    step: Int[Array, ""] = INT_INF,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    B, T, D = x.shape
    aux = AuxLosses.empty()
    sofa_true, infection_true, sepsis_true = true_concepts[..., 0], true_concepts[..., 1], true_concepts[..., 2]
    batch_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(B, dtype=jnp.int32))
    z_seq, inf_seq = jax.vmap(model.online_sequence)(x, batch_keys)
    z_seq = constrain_z(z_seq)

    ann_s3 = cosine_annealing(
        0.0, num_steps=400 * params.steps_per_epoch, num_dead_steps=800 * params.steps_per_epoch, current_step=step
    )
    ann_recon = cosine_annealing(
        0.0, num_steps=200 * params.steps_per_epoch, num_dead_steps=200 * params.steps_per_epoch, current_step=step
    )
    alpha, aux.beta, aux.sigma = jnp.ones_like(z_seq[..., 0]) * model.alpha, z_seq[..., 0], z_seq[..., 1]

    sofa_pred = jax.vmap(lookup_func, in_axes=(0, None, None))(
        jnp.concat([alpha[..., None], z_seq], axis=-1), model.lookup_temperature, model.kernel_size
    )
    # --------- Recon Loss
    x_recon = jax.vmap(jax.vmap(model.decoder))(z_seq)
    x_vals = x[..., : D // 2]
    aux.recon_loss = jnp.mean((x_vals - x_recon) ** 2, axis=-1)

    # --------- Concept losses
    class_weights = jnp.log(1.0 + 1.0 / (model.sofa_dist + EPS))
    weights_per_sample = class_weights[sofa_true.astype(jnp.int32)]
    aux.sofa_loss_t = (sofa_pred - (sofa_true / 23.0)) ** 2 * (weights_per_sample)

    aux.infection_p_loss_t = optax.sigmoid_binary_cross_entropy(inf_seq.squeeze(), infection_true)

    aux.susp_inf_p = jax.nn.sigmoid(inf_seq).squeeze()
    # sofa_pred = sofa_true.astype(jnp.float32) / 23.0
    # aux.susp_inf_p = infection_true

    aux.sofa_d2_p = jax.vmap(prob_increase_steps, in_axes=(0, None, None))(sofa_pred, model.d_thresh, model.d_scale)

    sofa_d2_true = jnp.concat(
        [jnp.expand_dims(jnp.zeros(B, dtype=jnp.bool), axis=1), (jnp.diff(sofa_true, axis=-1) > 0)], axis=-1
    )
    aux.sofa_d2_p_loss = (
        optax.sigmoid_focal_loss(binary_logits(aux.sofa_d2_p), sofa_d2_true, alpha=0.7, gamma=1.5)  # * ann_d2
    )

    # aux.sofa_d2_p = jax.vmap(causal_probs)(p_increase)
    aux.sep3_p = causal_smoothing(aux.sofa_d2_p, decay=model.sofa_d2_label_smooth, radius=12) * aux.susp_inf_p
    aux.sep3_p_loss = optax.sigmoid_binary_cross_entropy(binary_logits(aux.sep3_p), sepsis_true)  #  * ann_s3

    # --------- Directional Loss
    aux.sofa_directional_loss = jax.vmap(calc_full_directional_loss)(sofa_pred, sofa_true / 23.0, mask)

    # --------- Spreading Loss
    zcov = masked_cov(z_seq, mask)
    det_loss  = -jnp.log(jnp.linalg.det(zcov + EPS))
    trace_loss = -jnp.log(jnp.trace(zcov) + EPS)

    aux.spreading_loss =  det_loss# trace_loss #

    # --------- Total Loss
    thresholds = jnp.cumsum(jnp.ones(25) / 25)
    sofa_score_pred = jnp.sum(sofa_pred[..., None] > thresholds, axis=-1)
    aux.hists_sofa_score = jax.lax.stop_gradient(sofa_score_pred)
    aux.total_loss = (
        mask_padding(aux.recon_loss * params.w_recon, mask)
        + mask_padding(aux.sofa_d2_p_loss * params.w_sofa_d2, mask)
        + mask_padding(aux.infection_p_loss_t * params.w_inf, mask)
        + mask_padding(aux.sofa_loss_t * params.w_sofa_classification, mask)
        + mask_padding(aux.sep3_p_loss * params.w_sep3, mask)
        + mask_padding(aux.sofa_directional_loss * params.w_sofa_direction, mask)
        + mask_padding(aux.sofa_loss_sum * params.w_sofa_classification)
        + mask_padding(aux.spreading_loss * params.w_spreading)
    )
    return aux.total_loss, aux


@timing
@eqx.filter_jit
def process_train_epoch(
    model: GRUDetector,
    opt_state: optax.OptState,
    x_data: Float[Array, "ntbatches batch time input_dim"],
    y_data: Float[Array, "ntbatches batch time 3"],
    mask_data: Bool[Array, "nvbatches batch time 1"],
    update: Callable,
    param_filter: PyTree,
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
        batch: tuple[Float[Array, "batch time input_dim"], Float[Array, "bath time 3"], Bool[Array, "bath time 1"]],
    ) -> tuple[tuple[PyTree, optax.OptState, jnp.ndarray], AuxLosses]:
        _model_params, opt_state, key = carry
        batch_x, batch_true_c, batch_mask = batch
        _model = eqx.combine(_model_params, model_static)

        (_, aux_losses), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            _model,
            x=batch_x,
            true_concepts=batch_true_c,
            mask=batch_mask.astype(jnp.bool),
            step=opt_state[1][0].count,
            key=key,
            lookup_func=lookup_func,
            params=loss_params,
        )

        updates, opt_state = update(
            grads,
            opt_state,
            _model_params,
        )
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
    batches = (x_data, y_data, mask_data)
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
    model: GRUDetector,
    x_data: Float[Array, "nvbatches batch t input_dim"],
    y_data: Float[Array, "nvbatches batch t 3"],
    mask_data: Bool[Array, "nvbatches batch t 1"],
    step: Int[Array, ""],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> AuxLosses:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    def scan_step(
        carry: tuple[PyTree, jnp.ndarray, int],
        batch: tuple[Float[Array, "batch time input_dim"], Float[Array, "bath time 3"], Bool[Array, "bath time 1"]],
    ) -> tuple[tuple[PyTree, jnp.ndarray, int], AuxLosses]:
        model_flat, key, i = carry
        batch_x, batch_true_c, batch_mask = batch
        batch_key = jr.fold_in(key, i)

        model_full = eqx.combine(model_flat, model_static)

        _, aux_losses = loss(
            model=model_full,
            x=batch_x,
            true_concepts=batch_true_c,
            mask=batch_mask.astype(jnp.bool),
            key=batch_key,
            lookup_func=lookup_func,
            params=loss_params,
            step=step,
        )

        return (model_flat, key, i + 1), aux_losses

    carry = (model_params, key, int(0.0))
    batches = (x_data, y_data, mask_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    model_params, key, _ = carry

    return step_losses


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info")
    dtype = jnp.float32
    logger = logging.getLogger(__name__)

    train_conf = TrainingConfig(
        batch_size=1024, window_len=1 + 6, epochs=int(3e3), perc_train_set=0.333, validate_every=5
    )
    lr_conf = LRConfig(init=0.0, peak=1e-3, peak_decay=0.5, end=1e-3, warmup_epochs=1, enc_wd=1e-4, grad_norm=100.0)
    loss_conf = LossesConfig(
        w_recon=5.0,
        w_sofa_direction=10.0,
        w_sofa_classification=80.0,
        w_sofa_d2=30.0,
        w_inf=1.0,
        w_sep3=10,
        w_spreading=3e-3,
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=100, perform=True)

    # === Data ===
    (
        train_x,
        train_y,
        train_m,
        val_x,
        val_y,
        val_m,
        _x,
        _y,
        _m,
    ) = get_data_sets_detection(swapaxes_y=(1, 2, 0), dtype=jnp.float32)
    train_m, val_m = train_m.astype(np.bool), val_m.astype(np.bool)
    logger.error(f"{jnp.unique(train_y[..., 0])}")
    logger.error(f"{jnp.unique(train_y[..., 1])}")
    logger.error(f"{jnp.unique(train_y[..., 2])}")
    logger.info(f"Train shape      - Y {train_y.shape}, X {train_x.shape}, m {train_m.shape}")
    logger.info(f"Validation shape - Y {val_y.shape},  X {val_x.shape},  m {val_m.shape}")

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
        dtype=jnp.float32,
    )
    steps_per_epoch = (train_x.shape[0] // train_conf.batch_size) * train_conf.batch_size
    schedule = custom_warmup_cosine(
        init_value=lr_conf.init,
        peak_value=lr_conf.peak,
        warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
        steps_per_cycle=[steps_per_epoch * n for n in [100]],  # cycle durations
        end_value=lr_conf.end,
        peak_decay=lr_conf.peak_decay,
    )

    # === Initialization ===
    key = jr.PRNGKey(jax_random_seed)
    metrics = jnp.sort(lookup_table.relevant_metrics)
    # TODO dont do this for the windowed
    _sofas, counts = np.unique(train_y[train_m][..., 0], return_counts=True, sorted=True)
    counts = np.pad(counts, 24 - counts.size, constant_values=0)
    sofa_dist = counts / counts.sum()

    print(jnp.log(1.0 + 1.0 / (sofa_dist + EPS)))
    print(train_y[train_m, 2].sum(), train_m.sum(), train_y[train_m, 2].sum() / train_m.sum())
    print(train_y[1, train_m[1], 2])

    key_enc, key_inf, key_dec, key_gru = jr.split(key, 4)
    key_enc_weights, key_gru_weights = jr.split(key, 2)
    latent_encoder = LatentEncoder(
        key_enc,
        input_dim=104,
        latent_enc_hidden=32,
        latent_pred_hidden=16,
        dropout_rate=0.3,
        dtype=dtype,
    )
    latent_encoder = init_latent_encoder_weights(latent_encoder, key_enc_weights)
    decoder = Decoder(
        key_dec,
        input_dim=52,
        z_latent_dim=2,
        dec_hidden=32,
        dtype=dtype,
    )
    model = GRUDetector(
        key=key,
        input_dim=104,
        latent_hidden_dim=latent_encoder.latent_pred_hidden,
        latent_dim=2,
        inf_dim=1,
        inf_hidden_dim=8,
        latent_pre_encoder=latent_encoder,
        decoder=decoder,
        alpha=ALPHA,
        sofa_dist=jnp.asarray(sofa_dist),
    )
    model = init_gru_weights(model, key_gru_weights, scale=1e-4)

    hyper_enc, hyper_dec, hyper_gru = model.hypers_dict()

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

    logger.info(f"Instatiated model with {model.n_params} parameters. Decoder {model.decoder.n_params}")
    # === Training Loop ===
    writer = SummaryWriter()
    hyper_params = flatten_dict(
        {
            # "model": asdict(model_conf),
            "losses": asdict(loss_conf),
            "train": asdict(train_conf),
            "lr": asdict(lr_conf),
        }
    )
    writer.add_hparams(hyper_params, metric_dict={}, run_name=".")

    shuffle_val_key = jr.fold_in(key, jax_random_seed)
    val_x, val_y, val_m, nval_batches = prepare_batches_mask(
        val_x, val_y, val_m, train_conf.batch_size, key=shuffle_val_key
    )
    for epoch in range(load_conf.epoch if load_conf.from_dir else 0, train_conf.epochs):
        if epoch > 0:
            shuffle_key = jr.fold_in(key, epoch)
            train_key = jr.fold_in(key, epoch + 1)
            x_shuffled, y_shuffled, m_shuffled, ntrain_batches = prepare_batches_mask(
                train_x,
                train_y,
                train_m,
                key=shuffle_key,
                batch_size=train_conf.batch_size,
                perc=train_conf.perc_train_set,
                pos_fraction=0.1,
            )

            loss_conf.steps_per_epoch = ntrain_batches
            params_model, opt_state, train_metrics, key = process_train_epoch(
                model,
                opt_state=opt_state,
                x_data=x_shuffled,
                y_data=y_shuffled,
                mask_data=m_shuffled,
                update=optimizer.update,
                key=key,
                lookup_func=lookup_table.soft_get_local,
                loss_params=loss_conf,
                param_filter=filter_spec,
            )
            model = eqx.combine(params_model, static_model)

            log_msg = log_train_metrics(train_metrics, model.params_to_dict(), epoch, writer)
            writer.add_scalar(
                "lr/learning_rate", np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))), epoch
            )
            logger.info(log_msg)
            del x_shuffled, y_shuffled
            del train_metrics

        if epoch % train_conf.validate_every == 0:
            val_model = eqx.nn.inference_mode(model, value=True)
            val_key = jr.fold_in(key, epoch + 2)
            val_metrics = process_val_epoch(
                val_model,
                x_data=val_x,
                y_data=val_y,
                mask_data=val_m,
                step=opt_state[1][0].count,
                key=val_key,
                lookup_func=lookup_table.hard_get_fsq,
                loss_params=loss_conf,
            )
            # cal = CalibrationModel().calibrate(
            #     val_metrics.sofa_d2_p[val_m], val_metrics.susp_inf_p[val_m], val_metrics.sep3_p[val_m]
            # )
            # val_metrics.sep3_p = cal.get_sepsis_probs(val_metrics.sofa_d2_p, val_metrics.susp_inf_p)
            # logger.error(cal.coeffs)
            log_msg = log_val_metrics(val_metrics, val_y, lookup_table, epoch, writer, mask=val_m)
            pred_sep = np.asarray(val_metrics.sep3_p)[val_m]
            true_sep = val_y[..., 2][val_m]
            logger.warning(log_msg)

            model = eqx.nn.inference_mode(val_model, value=False)
            del val_metrics, val_model

        gc.collect()

        # --- Save checkpoint ---
    #     if (epoch + 1) % save_conf.save_every == 0 and save_conf.perform:
    #         save_dir = writer.get_logdir() if not load_conf.from_dir else load_conf.from_dir
    #         save_checkpoint(
    #             save_dir + "/checkpoints",
    #             epoch,
    #             model,
    #             opt_state,
    #             hyper_dec,
    #             hyper_enc,
    #         )
    writer.close()
