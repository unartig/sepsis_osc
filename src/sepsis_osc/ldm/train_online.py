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
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorboardX import SummaryWriter

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.calibration_model import PlattScaling, TemperatureScaling
from sepsis_osc.ldm.checkpoint_utils import load_checkpoint, save_checkpoint
from sepsis_osc.ldm.commons import binary_logits, custom_warmup_cosine
from sepsis_osc.ldm.data_loading import get_data_sets_online, prepare_batches_mask
from sepsis_osc.ldm.early_stopping import EarlyStopping
from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel, init_ldm_weights
from sepsis_osc.ldm.logging_utils import flatten_dict, log_train_metrics, log_val_metrics
from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
from sepsis_osc.ldm.model_structs import AuxLosses, LoadingConfig, LossesConfig, LRConfig, SaveConfig, TrainingConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed
from sepsis_osc.utils.jax_config import EPS, setup_jax, typechecker
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.utils import timing

INT_INF = jnp.astype(jnp.inf, jnp.int32)
_1 = jnp.array(1)


@jaxtyped(typechecker=typechecker)
def prob_increase_steps(
    preds: Float[Array, " time"], threshold: Float[Array, "1"], scale: Float[Array, "1"]
) -> Float[Array, " time"]:
    """
    Computes the probability of a value increase between consecutive time steps.

    This function calculates the difference between adjacent elements and passes
    the result through a sigmoid function. It essentially acts as a soft
    thresholding mechanism to detect 'jumps'.
    """
    # return jnp.concat([jnp.array([0.0]), jnp.tanh((jax.nn.relu(jnp.diff(preds, axis=-1) - threshold)) * scale)])
    return jnp.concat([jnp.array([0.0]), jax.nn.sigmoid((jnp.diff(preds, axis=-1) - threshold) * scale)])


@jaxtyped(typechecker=typechecker)
def causal_smoothing(
    labels: Float[Array, "batch time"], radius: int = 3, decay: float | Float[Array, "1"] = 0.5
) -> Float[Array, "batch time"]:
    """
    Applies a causal exponential smoothing filter across the time dimension.

    The filter is 'causal' because it only considers current and past values
    to calculate the smoothed average at any given time step, preventing
    information leakage from the future. It uses a normalized exponential
    decay kernel.
    """
    offsets = jnp.arange(0, radius + 1)

    kernel = jnp.exp(-decay * offsets)
    kernel = kernel / kernel.sum()  # normalize

    def convolve_1d(x: Float[Array, " time"]) -> Float[Array, " time"]:
        return jnp.convolve(x, kernel, mode="full")

    smooth = jax.vmap(convolve_1d)(labels)

    smooth = smooth[:, : labels.shape[1]]

    return jnp.clip(smooth, 0.0, 1.0)


@jaxtyped(typechecker=typechecker)
def masked_cov(
    x: Float[Array, "batch time latent_dim"], mask: Bool[Array, "batch time"]
) -> Float[Array, "latent_dim latent_dim"]:
    """
    Calculates the empirical covariance matrix across a batch and time, considering only valid timesteps.
    """
    mask_f = mask.astype(x.dtype)
    w = jnp.sum(mask_f)
    mean = jnp.sum(x * mask_f[..., None], axis=(0, 1)) / w
    xc = ((x - mean) * mask_f[..., None]).reshape(-1, x.shape[-1])
    cov = xc.T @ xc / (w - 1)
    return cov


def boundary_loss(
    z_sigmoid: Float[Array, "batch time zlatent_dim"],
    margin_fraction: float = 0.1,
) -> Float[Array, ""]:
    """
    Computes a hinge loss to penalize latent values that approach the edges of the sigmoid activation range.
    Ensuring they stay within a specified margin.
    """
    lower_violation = jax.nn.relu(margin_fraction - z_sigmoid)
    upper_violation = jax.nn.relu(z_sigmoid - (1.0 - margin_fraction))

    return (lower_violation + upper_violation).mean(axis=-1)


def constrain_z(z: Float[Array, "batch time zlatent_dim"]) -> Float[Array, "batch time zlatent_dim"]:
    """
    Linearly scales raw latent outputs from the unit interval to the predefined physical ranges for the parameters.
    """
    beta, sigma = jnp.split(z, 2, axis=-1)
    beta = beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma = sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]
    return jnp.concatenate([beta, sigma], axis=-1)


def constrain_z_margin(
    z: Float[Array, "batch time zlatent_dim"], margin: int = 1
) -> Float[Array, "batch time zlatent_dim"]:
    """
    Maps latent outputs to a narrowed parameter space to ensure the lookup kernel
    remains within valid bounds latent space.
    """
    beta, sigma = jnp.split(z, 2, axis=-1)

    safe_beta_min = BETA_SPACE[0] + margin * BETA_SPACE[2]
    safe_beta_max = BETA_SPACE[1] - margin * BETA_SPACE[2]
    safe_sigma_min = SIGMA_SPACE[0] + margin * SIGMA_SPACE[2]
    safe_sigma_max = SIGMA_SPACE[1] - margin * SIGMA_SPACE[2]

    beta = beta * (safe_beta_max - safe_beta_min) + safe_beta_min
    sigma = sigma * (safe_sigma_max - safe_sigma_min) + safe_sigma_min
    return jnp.concatenate([beta, sigma], axis=-1)


def bound_z(z: Float[Array, "batch time zlatent_dim"]) -> Float[Array, "batch time zlatent_dim"]:
    """
    Applies a hard clip to latent values to ensure they do not exceed the defined boundaries of the parameter space.
    """
    beta, sigma = jnp.split(z, 2, axis=-1)
    beta = beta.clip(BETA_SPACE[0], BETA_SPACE[1])
    sigma = sigma.clip(SIGMA_SPACE[0], SIGMA_SPACE[1])
    return jnp.concatenate([beta, sigma], axis=-1)


def cosine_annealing(base_val: float, num_steps: int, num_dead_steps: int, current_step: jnp.int32) -> Float[Array, ""]:
    """
    Computes a cosine-based decay factor for a value over a set number of steps (for temperature or loss scheduling).
    """
    step = jnp.clip(current_step - num_dead_steps, 0, num_steps)
    cosine = 0.5 * (1 + jnp.cos(jnp.pi * step / num_steps))
    return base_val + (1.0 - base_val) * (1.0 - cosine)


def mask_padding(loss: Float[Array, "*"], mask: Float[Array, "*"] | None = None) -> Float[Array, "*"]:
    """
    Computes the mean of a loss tensor while ignoring padded indices as defined by a boolean mask.
    """
    if mask is None:
        mask = jnp.ones_like(loss)
    print(mask.shape, loss.shape)
    return (loss * mask).sum() / mask.sum()


# === Loss Function ===
@jaxtyped(typechecker=typechecker)
def loss(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    true_concepts: Float[Array, "batch time 3"],
    mask: Bool[Array, "batch time"] = _1,
    step: Int[Array, ""] = INT_INF,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    """
    The primary objective function.
    It performs 1. the forward pass,
                2. computes reconstruction loss,
                3. conceptual losses for clinical markers,
                4. and regularization terms (spreading and boundary losses).
    """
    B, T, D = x.shape
    aux = AuxLosses.empty()
    sofa_true, infection_true, sepsis_true = true_concepts[..., 0], true_concepts[..., 1], true_concepts[..., 2]
    batch_keys = jr.split(key, B)
    z_seq_sigm, inf_seq_raw = jax.vmap(model.online_sequence)(x, batch_keys)
    z_seq = constrain_z_margin(z_seq_sigm, model.lookup_kernel_size // 2)

    fine = cosine_annealing(0.01, 200 * params.steps_per_epoch, 100 * params.steps_per_epoch, step)

    aux.boundary_loss = boundary_loss(z_seq_sigm, margin_fraction=0.1)

    aux.beta, aux.sigma = z_seq[..., 0], z_seq[..., 1]

    sofa_pred = jax.vmap(lookup_func, in_axes=(0, None, None))(
        z_seq, model.lookup_temperature, model.lookup_kernel_size
    )
    # --------- Recon Loss
    x_recon = jax.vmap(jax.vmap(model.decoder))(z_seq)
    x_vals = x[..., : D // 2]
    aux.recon_loss = jnp.mean((x_vals - x_recon) ** 2, axis=-1)

    # --------- Concept losses
    class_weights = jnp.log(1.0 + 1.0 / (model.sofa_dist + EPS))
    weights_per_sample = class_weights[sofa_true.astype(jnp.int32)]
    aux.sofa_loss_t = (sofa_pred - (sofa_true / 23.0)) ** 2 * (weights_per_sample)

    aux.infection_p_loss_t = optax.sigmoid_binary_cross_entropy(inf_seq_raw, infection_true)

    aux.susp_inf_p = jax.nn.sigmoid(inf_seq_raw).squeeze()

    aux.sofa_d2_risk = jax.vmap(prob_increase_steps, in_axes=(0, None, None))(sofa_pred, model.d_thresh, model.d_scale)

    aux.sep3_risk = causal_smoothing(aux.sofa_d2_risk, decay=model.sofa_d2_pred_smooth, radius=12) * aux.susp_inf_p
    aux.sep3_loss_t = optax.sigmoid_binary_cross_entropy(binary_logits(aux.sep3_risk), sepsis_true)  # * fine

    # --------- Spreading Loss
    zcov = masked_cov(z_seq, mask)
    det_loss = -jnp.log(jnp.linalg.det(zcov + EPS))
    aux.spreading_loss = det_loss

    # --------- Total Loss
    thresholds = jnp.cumsum(jnp.ones(25) / 25)
    sofa_score_pred = jnp.sum(sofa_pred[..., None] > thresholds, axis=-1)
    aux.hists_sofa_score = jax.lax.stop_gradient(sofa_score_pred)
    aux.total_loss = (
        mask_padding(aux.recon_loss * params.lambda_recon, mask)
        + mask_padding(aux.infection_p_loss_t * params.lambda_inf, mask)
        + mask_padding(aux.sofa_loss_t * params.lambda_sofa_classification, mask)
        + mask_padding(aux.sep3_loss_t * params.lambda_sep3, mask)
        + mask_padding(aux.spreading_loss * params.lambda_spreading)
        + mask_padding(aux.boundary_loss * params.lambda_boundary, mask)
    )
    return aux.total_loss, aux


@timing
@eqx.filter_jit
def process_train_epoch(
    model: LatentDynamicsModel,
    opt_state: optax.OptState,
    x_data: Float[Array, "ntbatches batch time input_dim"],
    y_data: Float[Array, "ntbatches batch time 3"],
    mask_data: Bool[Array, "ntbatches batch time 1"],
    update: Callable,
    param_filter: PyTree,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[PyTree, PyTree, AuxLosses, jnp.ndarray]:
    """
    Executes a single training epoch.
    It uses `jax.lax.scan` to iterate through batches, computing gradients and updating model parameters.
    """
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
            mask=batch_mask,
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
    model: LatentDynamicsModel,
    x_data: Float[Array, "nvbatches batch t input_dim"],
    y_data: Float[Array, "nvbatches batch t 3"],
    mask_data: Bool[Array, "nvbatches batch t 1"],
    step: Int[Array, ""],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> AuxLosses:
    """
    Performs an inference-only pass over the validation dataset to compute metrics and losses without updating model.
    Expects the full data to fit into device memory.
    """
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
            mask=batch_mask,
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
        batch_size=128,
        epochs=int(3e3),
        mini_epochs=4,
        validate_every=1,
        calibrate=False,
        early_stop=True,
    )
    lr_conf = LRConfig(
        init=0.0,
        warmup_epochs=1,
        peak=5e-5,
        enc_wd=2e-1,
    )
    loss_conf = LossesConfig(
        lambda_sep3=600.0,
        lambda_inf=1.0,
        lambda_sofa_classification=2000.0,
        lambda_spreading=6e-3,
        lambda_boundary=30.0,
        lambda_recon=2.5,
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=1, perform=True)

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
    ) = get_data_sets_online(swapaxes_y=(1, 2, 0), dtype=jnp.float32)
    train_m, val_m = train_m.astype(np.bool), val_m.astype(np.bool)
    logger.error(f"{jnp.unique(train_y[..., 0])}")
    logger.error(f"{jnp.unique(train_y[..., 1])}")
    logger.error(f"{jnp.unique(train_y[..., 2])}")
    logger.info(f"Train shape      - Y {train_y.shape}, X {train_x.shape}, m {train_m.shape}")
    logger.info(f"Validation shape - Y {val_y.shape},  X {val_x.shape},  m {val_m.shape}")
    logger.info(f"Frac Pos labels in Test {train_y[train_m, 2].sum() / train_m.sum() * 100:.2f}%")
    for y, m, s in ((train_y, train_m, "Train"), (val_y, val_m, "Val"), (_y, _m, "Test")):
        logger.info(
           f"Patient Prevalence {s:5}   {((y * m[..., None]).max(axis=1) == 1.0).mean(axis=0).round(4)[[1, 2]] * 100}%"
        )
    logger.info("")
    for y, m, s in ((train_y, train_m, "Train"), (val_y, val_m, "Val"), (_y, _m, "Test")):
        logger.info(f"Time Prevalence {s:5}      {((y * m[..., None]) > 0.0).mean(axis=(0, 1)).round(4)[[1, 2]] * 100}%")
    for y, m, s in ((train_y, train_m, "Train"), (val_y, val_m, "Val"), (_y, _m, "Test")):
        logger.info(f"Mean sequence length {s:5} {m.sum(axis=-1).mean().round(2)}h")
    for y, m, s in ((train_y, train_m, "Train"), (val_y, val_m, "Val"), (_y, _m, "Test")):
        logger.info(f"Max values {s:5}           {(y * m[..., None]).max(axis=((0,1)))}")

    logger.info(f"First septic patient labels {train_y[1, train_m[1], 2]}")
    del _x, _y, _m

    db_str = "DaisyFinal"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    sim_storage.close()

    b, s = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
    a = np.ones_like(b) * ALPHA
    indices_2d = jnp.concatenate([b[..., np.newaxis], s[..., np.newaxis]], axis=-1)
    spacing_2d = jnp.array([BETA_SPACE[2], SIGMA_SPACE[2]])
    params = DNMConfig.batch_as_index(a, b, s, 0.2)
    metrics_2d, _ = sim_storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
    metrics_2d = metrics_2d.to_jax()
    lookup_table = LatentLookup(
        metrics=metrics_2d.reshape((-1, 1)),
        indices=indices_2d.reshape((-1, 2)),  # since param space only beta sigma
        metrics_2d=metrics_2d,
        indices_2d=indices_2d,
        grid_spacing=spacing_2d,
        dtype=jnp.float32,
    )

    steps_per_epoch = (train_x.shape[0] // train_conf.batch_size) * train_conf.batch_size
    schedule = custom_warmup_cosine(
        init_value=lr_conf.init,
        peak_value=lr_conf.peak,
        warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
        cycles=[
            (steps_per_epoch * n, scale) for (n, scale) in [(200, 1.0)]
        ],  # (cycle durations, end_value relative to peak)
    )

    # === Initialization ===
    key = jr.PRNGKey(jax_random_seed)
    metrics = jnp.sort(lookup_table.relevant_metrics)
    _sofas, counts = np.unique(train_y[train_m][..., 0], return_counts=True, sorted=True)
    counts = np.pad(counts, 24 - counts.size, constant_values=0)
    sofa_dist = counts / counts.sum()

    logger.info(f"SOFA class weights {jnp.log(1.0 + 1.0 / (sofa_dist + EPS))}")

    key_model, key_weights = jr.split(key, 2)
    model = LatentDynamicsModel(
        key=key_model,
        input_dim=52,
        latent_enc_hidden_dim=64,
        latent_hidden_dim=4,
        latent_dim=2,
        inf_hidden_dim=32,
        inf_dim=1,
        dec_hidden_dim=32,
        sofa_dist=jnp.asarray(sofa_dist),
        lookup_kernel_size=3,
    )
    model = init_ldm_weights(model, key_weights, scale=1e-4)

    hyper_ldm = model.hypers_dict()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
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
        f"Instatiated model with total {model.n_params} parameters. "
        f"Encoder {model.latent_pre_encoder.n_params}, "
        f"Latent {sum(x.size for x in jtu.tree_leaves(model.latent_encoder))} + {sum(x.size for x in jtu.tree_leaves(model.latent_proj_out))}, "
        f"Inf {sum(x.size for x in jtu.tree_leaves(model.inf_encoder))} + {sum(x.size for x in jtu.tree_leaves(model.inf_proj_out))}, "
        f"Decoder {model.decoder.n_params}"
    )
    # === Training Loop ===
    writer = SummaryWriter()
    hyper_params = flatten_dict(
        {
            "model": hyper_ldm,
            "losses": asdict(loss_conf),
            "train": asdict(train_conf),
            "lr": asdict(lr_conf),
        }
    )
    writer.add_hparams(hyper_params, metric_dict={}, name="")

    stop = False
    early_stop_auprc = EarlyStopping(direction=1, patience=30, min_steps=20)
    early_stop_auroc = EarlyStopping(direction=1, patience=30, min_steps=20)

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
                mini_epochs=train_conf.mini_epochs,
                pos_fraction=-1,
            )
            loss_conf.steps_per_epoch = ntrain_batches
            train_metrics = []
            for mini_epoch in range(train_conf.mini_epochs):
                params_model, opt_state, mini_train_metrics, key = process_train_epoch(
                    model,
                    opt_state=opt_state,
                    x_data=x_shuffled[mini_epoch],
                    y_data=y_shuffled[mini_epoch],
                    mask_data=m_shuffled[mini_epoch],
                    update=optimizer.update,
                    key=key,
                    lookup_func=lookup_table.soft_get_local,
                    loss_params=loss_conf,
                    param_filter=filter_spec,
                )
                train_metrics.append(mini_train_metrics)
                model = eqx.combine(params_model, static_model)

            train_metrics = jtu.tree_map(lambda *xs: jnp.stack(xs), *train_metrics)
            log_msg = log_train_metrics(train_metrics, model.params_to_dict(), epoch, writer, mask=m_shuffled)
            writer.add_scalar(
                "lr/learning_rate",
                np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))),
                epoch,
            )
            logger.info(log_msg)

        if epoch % train_conf.validate_every == 0:
            val_model = eqx.nn.inference_mode(model, value=True)
            val_key = jr.fold_in(key, epoch + 2)
            val_metrics = process_val_epoch(
                val_model,
                x_data=val_x[None],
                y_data=val_y[None],
                mask_data=val_m[None],
                step=opt_state[1][0].count,
                key=val_key,
                lookup_func=lookup_table.hard_get_fsq,
                loss_params=loss_conf,
            )

            if train_conf.calibrate and epoch > 0:
                cal = TemperatureScaling().calibrate(
                    train_metrics.sofa_d2_p[m_shuffled],
                    train_metrics.susp_inf_p[m_shuffled],
                    y_shuffled[m_shuffled, 2],
                )
                val_metrics.sep3_risk = np.zeros_like(val_metrics.sep3_risk)
                val_metrics.sep3_risk[val_m[None]] = cal.get_sepsis_probs(
                    val_metrics.sofa_d2_risk[val_m[None]], val_metrics.susp_inf_p[val_m[None]]
                )
                logger.error(cal.coeffs)

            if train_conf.early_stop:
                auprc = average_precision_score((val_y[val_m, 2] == 1.0), val_metrics.sep3_risk[val_m[None]])
                auroc = roc_auc_score((val_y[val_m, 2] == 1.0), val_metrics.sep3_risk[val_m[None]])
                auprc_stop = early_stop_auprc.step(auprc)
                auroc_stop = early_stop_auroc.step(auroc)
                stop = auprc_stop & auroc_stop
                logger.info(
                    f"Bad Steps|Stopped: AUROC {early_stop_auroc.bad_steps}|{early_stop_auroc.stopped} and "
                    f"AUPRC {early_stop_auprc.bad_steps}|{early_stop_auprc.stopped}"
                )

            log_msg = log_val_metrics(val_metrics, val_y[None], lookup_table, epoch, writer, mask=val_m[None])
            pred_sep = np.asarray(val_metrics.sep3_risk)[val_m[None]]
            true_sep = val_y[..., 2][val_m]
            logger.warning(log_msg)

            model = eqx.nn.inference_mode(val_model, value=False)

        if epoch > 0:
            del train_metrics, x_shuffled, y_shuffled, m_shuffled, val_metrics
            gc.collect()

        # --- Save checkpoint ---
        if ((epoch + 1) % save_conf.save_every == 0 or stop) and save_conf.perform:
            save_dir = writer.logdir if not load_conf.from_dir else load_conf.from_dir
            save_checkpoint(
                save_dir + "/checkpoints",
                epoch,
                model,
                opt_state,
                hyper_ldm,
            )

        if stop:
            break
    writer.close()
