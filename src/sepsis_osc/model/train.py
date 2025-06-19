import logging
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.model.data_loading import data
from sepsis_osc.model.model_utils import infer_grid_params, load_checkpoint, prepare_batches, save_checkpoint, timing
from sepsis_osc.model.vae import (
    LATENT_DIM,
    Decoder,
    Encoder,
    init_decoder_weights,
    init_encoder_weights,
)
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging

ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE = (-1.0, 1.0), (0.2, 1.5), (0.0, 1.5)


def ordinal_logits(floats: Float[Array, "batch 1"], n_classes: int = 24) -> Float[Array, "batch n_classes_minus_1"]:
    thresholds = jnp.linspace(-2.5, 2.5, n_classes - 1)
    # [0, 1] -> [-2.5, 2.5]
    floats = 5.0 * (floats - 0.5)
    logits = floats.squeeze()[:, None] - thresholds[None, :]
    return logits


# TODO
# try
# https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/
def ordinal_loss(
    predicted_sofa: Float[Array, "batch 1"], true_sofa: Float[Array, "batch 1"], n_classes: int = 24
) -> Float[Array, "batch 1"]:
    # https://arxiv.org/abs/1901.07884
    # CORN (Cumulative Ordinal Regression)
    logits = ordinal_logits(predicted_sofa, n_classes)  # [B, 23]

    targets = jnp.arange(n_classes - 1)  # [0, 1, ..., 22]
    true_sofa = true_sofa.squeeze()
    true_ord_targets = true_sofa[:, None] > targets[None, :]  # [B, 23]
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_ord_targets))


def binary_logits(probs: Float[Array, "batch 1"], eps: float = 1e-6) -> Float[Array, "batch 1"]:
    probs = jnp.clip(probs, eps, 1 - eps)
    return jnp.log(probs) - jnp.log1p(-probs)


def binary_loss(
    predicted_infection: Float[Array, "batch 1"], true_infection: Float[Array, "batch 1"]
) -> Float[Array, "batch 1"]:
    logits = binary_logits(predicted_infection)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_infection))


def calc_concept_loss(
    prediction: Float[Array, "batch 2"],
    true_sofa: Float[Array, "batch 1"],
    true_infection: Float[Array, "batch 1"],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[Float[Array, "batch 1"], Float[Array, "batch 1"]]:
    pred_sofa = prediction[:, 0]  # SOFA prediction
    pred_infection = prediction[:, 1]  # Infection prediction

    loss_sofa = ordinal_loss(pred_sofa, true_sofa)
    loss_infection = binary_loss(pred_infection, true_infection)

    return alpha * loss_sofa, beta * loss_infection


def calc_locality_loss(input: Float[Array, "batch 52"], latent: Float[Array, "batch 3"], sigma=1.0):
    # x: (B, D_in), z: (B, D_latent)
    x_dists = jnp.sum((input[:, None, :] - input[None, :, :]) ** 2, axis=-1)  # (B, B)
    z_dists = jnp.sum((latent[:, None, :] - latent[None, :, :]) ** 2, axis=-1)

    sim_weights = jnp.exp(-x_dists / (2 * sigma**2))
    loss = jnp.mean(sim_weights * z_dists)  # weighted MSE in latent space
    return loss


# === Loss Function ===
@eqx.filter_jit  # (donate="all")
def loss(
    models: tuple[Encoder, Decoder],
    x: jnp.ndarray,
    true_concepts: jnp.ndarray,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    lambda1: float = 1e3,
    lambda2: float = 1e1,
    lambda3: float = 1e1,
) -> tuple[Array, dict[str, dict[str, Array]]]:
    aux_losses = {"latents": {}, "concepts": {}, "losses": {}}
    encoder, decoder = models

    key, *drop_keys = jr.split(key, x.shape[0] + 1)
    drop_keys = jnp.array(drop_keys)

    alpha, beta, sigma, sofa, infection, lookup_temp = jax.vmap(encoder)(x, key=drop_keys)
    alpha = alpha * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0]
    beta = beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma = sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]
    aux_losses["latents"]["alpha"], aux_losses["latents"]["beta"], aux_losses["latents"]["sigma"] = (
        alpha.mean(),
        beta.mean(),
        sigma.mean(),
    )

    concepts_direct = jnp.concatenate([sofa, infection], axis=-1)
    z_lookup = jnp.concatenate([alpha, beta, sigma], axis=-1)
    # lookup_indices = SystemConfig.batch_as_index(alpha=alpha, beta=beta, sigma=sigma, C=0.2)
    concepts_lookup = lookup_func(
        z_lookup,
        temperatures=lookup_temp,
    )

    # TODO change, we actually want to predict both :^)
    aux_losses["concepts"]["sofa_lookup"], aux_losses["concepts"]["infection_lookup"] = calc_concept_loss(
        concepts_lookup, true_concepts[:, 0], true_concepts[:, 1], alpha=1.0, beta=0.0
    )
    aux_losses["concepts"]["sofa_direct"], aux_losses["concepts"]["infection_direct"] = calc_concept_loss(
        concepts_direct, true_concepts[:, 0], true_concepts[:, 1], alpha=1.0, beta=0.0
    )
    lookup_loss = aux_losses["concepts"]["sofa_lookup"] + aux_losses["concepts"]["infection_lookup"]
    direct_loss = aux_losses["concepts"]["sofa_direct"] + aux_losses["concepts"]["infection_direct"]
    aux_losses["losses"]["concept_loss"] = lookup_loss * 0.7 + direct_loss * 0.3

    z = jnp.concatenate([alpha, beta, sigma], axis=-1)
    cov = z.T @ z / z.shape[0]
    aux_losses["losses"]["tc_loss"]= jnp.sum(jnp.abs(cov - jnp.diag(jnp.diag(cov))))

    aux_losses["losses"]["locality_loss"] = calc_locality_loss(x, z_lookup)

    x_recon = jax.vmap(decoder)(z_lookup)
    aux_losses["losses"]["recon_loss"] = jnp.mean((x - x_recon) ** 2, axis=-1)

    aux_losses["losses"]["total_loss"] = (
        aux_losses["losses"]["recon_loss"]
        + lambda1 * aux_losses["losses"]["concept_loss"]
        + lambda2 * aux_losses["losses"]["locality_loss"]
        + lambda3 * aux_losses["losses"]["tc_loss"]  # https://github.com/YannDubs/disentangling-vae
    )

    aux_losses["concepts"]["lookup_temperature"] = lookup_temp.mean(axis=-1)
    aux_losses = jax.tree_util.tree_map(jnp.mean, aux_losses)
    return aux_losses["losses"]["total_loss"], aux_losses


@eqx.filter_jit
def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    models: tuple[Encoder, Decoder],
    opt_state_enc: optax.OptState,
    opt_state_dec: optax.OptState,
    update_enc: Callable,
    update_dec: Callable,
    *,
    key,
    lookup_func: Callable,
) -> tuple[tuple[Encoder, Decoder], tuple[optax.OptState, optax.OptState], tuple[Array, dict[str, dict[str, Array]]]]:
    encoder, decoder = models
    (total_loss, aux_losses), (grads_enc, grads_dec) = eqx.filter_value_and_grad(loss, has_aux=True)(
        models, x_batch, true_c_batch, key=key, lookup_func=lookup_func
    )

    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    updates_enc, opt_state_enc = update_enc(grads_enc, opt_state_enc, encoder_params)
    encoder_params = eqx.apply_updates(encoder_params, updates_enc)
    encoder = eqx.combine(encoder_params, encoder_static)

    updates_dec, opt_state_dec = update_dec(grads_dec, opt_state_dec, decoder_params)
    decoder_params = eqx.apply_updates(decoder_params, updates_dec)
    decoder = eqx.combine(decoder_params, decoder_static)

    return (encoder, decoder), (opt_state_enc, opt_state_dec), (total_loss, aux_losses)


@timing
@eqx.filter_jit
def process_train_epoch(
    encoder: Encoder,
    decoder: Decoder,
    opt_state_enc,
    opt_state_dec,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    update_enc: Callable,
    update_dec: Callable,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
) -> tuple[PyTree, PyTree, optax.OptState, optax.OptState, dict[str, dict[str, Float[Array, "nbatches"]]], jnp.ndarray]:
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    key, _ = jr.split(key)
    step_losses = []

    def scan_step(carry, batch):
        (encoder_f, decoder_f), (opt_state_enc, opt_state_dec), key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        encoder_full = eqx.combine(encoder_f, encoder_static)
        decoder_full = eqx.combine(decoder_f, decoder_static)

        (encoder_new, decoder_new), (opt_state_enc, opt_state_dec), (_, aux_losses) = step_model(
            batch_x,
            batch_true_c,
            (encoder_full, decoder_full),
            opt_state_enc,
            opt_state_dec,
            update_enc,
            update_dec,
            key=batch_key,
            lookup_func=lookup_func,
        )

        encoder_f = eqx.filter(encoder_new, eqx.is_inexact_array)
        decoder_f = eqx.filter(decoder_new, eqx.is_inexact_array)

        return (
            (encoder_f, decoder_f),
            (opt_state_enc, opt_state_dec),
            key,
        ), aux_losses

    carry = (
        (encoder_params, decoder_params),
        (opt_state_enc, opt_state_dec),
        key,
    )
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    (encoder_params, decoder_params), (opt_state_enc, opt_state_dec), key = carry

    return (
        encoder_params,
        decoder_params,
        opt_state_enc,
        opt_state_dec,
        step_losses,
        key,
    )


@timing
@eqx.filter_jit
def process_val_epoch(
    encoder: Encoder,
    decoder: Decoder,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
) -> tuple[jnp.ndarray, dict[str, dict[str, Array]]]:
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    key, _ = jr.split(key)
    step_losses = []

    def scan_step(carry, batch):
        (encoder_f, decoder_f), key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        encoder_full = eqx.combine(encoder_f, encoder_static)
        decoder_full = eqx.combine(decoder_f, decoder_static)

        _, aux_losses = loss(
            (encoder_full, decoder_full), batch_x, batch_true_c, key=batch_key, lookup_func=lookup_func
        )

        return ((encoder_f, decoder_f), key), aux_losses

    carry = ((encoder_params, decoder_params), key)
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    (encoder_params, decoder_params), key = carry

    return (key, step_losses)


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info")
    logger = logging.getLogger(__name__)

    BATCH_SIZE = 128
    EPOCHS = 100
    LOAD_FROM_CHECKPOINT = ""  # set load dir, e.g. "runs/..."
    LOAD_EPOCH = 0
    SAVE_CHECKPOINTS = True
    SAVE_EVERY_EPOCH = 5

    # cosine decay
    LR_INIT = 1e-3
    LR_END = 1e-5
    LR_EPOCH_END = 200
    ENC_WD = 1e-3

    # === Data ===
    train_y, train_x, val_y, val_x, test_y, test_x = [
        jnp.array(
            v.drop([
                col
                for col in v.columns
                if col.startswith("Missing") or col in {"stay_id", "time", "sep3_alt", "__index_level_0__", "los_icu"}
            ]),
            dtype=jnp.float32,
        )
        for inner in data.values()
        for v in inner.values()
    ]

    logger.info(f"Train shape      - X {train_y.shape}, Y {train_x.shape}")
    logger.info(f"Validation shape - X {val_y.shape},  Y {val_x.shape}")
    logger.info(f"Test shape       - X {test_y.shape},  Y {test_x.shape}")

    # train_x = train_x[: 1000 * BATCH_SIZE]
    # train_y = train_y[: 1000 * BATCH_SIZE]

    db_str = "Daisy"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    indices, np_metrics = sim_storage.get_np_lookup()
    lookup_table = JAXLookup(
        metrics=np_metrics.to_jax(),
        indices=jnp.asarray(indices[:, 5:8]),  # since param space only lapha beta sigma
    )

    # === Initialization ===
    key = jr.PRNGKey(jax_random_seed)
    key_enc, key_dec = jr.split(key)
    encoder = Encoder(key_enc)
    decoder = Decoder(key_dec)

    schedule = optax.cosine_decay_schedule(
        init_value=LR_INIT,
        decay_steps=int(LR_EPOCH_END * (train_x.shape[0] // BATCH_SIZE) * BATCH_SIZE),
        alpha=LR_END / LR_INIT,
    )
    opt_enc = optax.adamw(schedule, weight_decay=ENC_WD)
    opt_dec = optax.adamw(LR_INIT)

    key_enc_weights, key_dec_weights = jr.split(key, 2)
    encoder = init_encoder_weights(encoder, key_enc_weights)
    decoder = init_decoder_weights(decoder, key_dec_weights)

    hyper_enc = {
        "input_dim": encoder.input_dim,
        "latent_dim": encoder.latent_dim,
        "enc_hidden": encoder.enc_hidden,
        "dropout_rate": encoder.dropout_rate,
    }
    hyper_dec = {
        "input_dim": decoder.input_dim,
        "latent_dim": decoder.latent_dim,
        "dec_hidden": decoder.dec_hidden,
    }
    params_enc, static_enc = eqx.partition(encoder, eqx.is_inexact_array)
    params_dec, static_dec = eqx.partition(decoder, eqx.is_inexact_array)
    opt_state_enc = opt_enc.init(params_enc)
    opt_state_dec = opt_dec.init(params_dec)
    if LOAD_FROM_CHECKPOINT:
        try:
            params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec = load_checkpoint(
                LOAD_FROM_CHECKPOINT + "/checkpoints", LOAD_EPOCH, opt_enc, opt_dec
            )
            encoder = eqx.combine(params_enc, static_enc)
            decoder = eqx.combine(params_dec, static_dec)
            logger.info(f"Resuming training from epoch {LOAD_EPOCH + 1}")
        except FileNotFoundError as e:
            logger.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
            LOAD_FROM_CHECKPOINT = ""

    # === Training Loop ===
    writer = SummaryWriter()
    shuffle_val_key, key = jr.split(key, 2)
    val_x, val_y, nval_batches = prepare_batches(val_x, val_y, BATCH_SIZE, shuffle_val_key)
    for epoch in range(LOAD_EPOCH, EPOCHS):
        key, shuffle_key = jr.split(key)
        x_shuffled, y_shuffled, ntrain_batches = prepare_batches(train_x, train_y, BATCH_SIZE, shuffle_key)

        params_enc, params_dec, opt_state_enc, opt_state_dec, train_metrics, key = process_train_epoch(
            encoder,
            decoder,
            opt_state_enc=opt_state_enc,
            opt_state_dec=opt_state_dec,
            x_data=x_shuffled,
            y_data=y_shuffled,
            update_enc=opt_enc.update,
            update_dec=opt_dec.update,
            key=key,
            lookup_func=lookup_table.soft_get_full,
        )
        encoder = eqx.combine(params_enc, static_enc)
        decoder = eqx.combine(params_dec, static_dec)

        # ntrain_batches = train_metrics["losses"]["total_loss"].shape[0]
        log_msg = f"Epoch {epoch} Training  Metrics "
        for k in train_metrics.keys():
            for metric_name, metric_values in train_metrics[k].items():
                if k == "losses":
                    log_msg += f"{metric_name} = {metric_values.mean():.4f} ({metric_values.std():.4f}), "
                if epoch == 0:
                    for step in range(ntrain_batches):
                        writer.add_scalar(
                            f"train_{k}/{metric_name}_step", np.asarray(metric_values[step]), epoch * ntrain_batches + step
                        )
                else:
                    step_interval = max(1, int(ntrain_batches * 0.05))
                    for i in range(0, ntrain_batches, step_interval):
                        start_index = i
                        end_index = min(i + step_interval, ntrain_batches)
                        interval_values = metric_values[start_index:end_index]
                        if len(interval_values) > 0:
                            interval_mean = interval_values.mean()
                            writer.add_scalar(
                                f"train_{k}/{metric_name}_step", np.asarray(interval_mean), epoch * ntrain_batches + start_index
                            )
        logger.info(log_msg)
        del x_shuffled, y_shuffled

        key, _ = jr.split(key)
        key, val_metrics = process_val_epoch(
            encoder,
            decoder,
            x_data=val_x,
            y_data=val_y,
            key=key,
            lookup_func=lookup_table.hard_get,
        )
        log_msg = f"Epoch {epoch} Valdation Metrics "
        for k in val_metrics.keys():
            for metric_name, metric_values in val_metrics[k].items():
                log_msg += (
                    f"{metric_name} = {metric_values.mean():.4f} ({metric_values.std():.4f}), " if k == "losses" else ""
                )
                writer.add_scalar(f"val_{k}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
                writer.add_scalar(f"val_{k}/{metric_name}_std", np.asarray(metric_values.std()), epoch)
        logger.warning(log_msg)
        writer.add_scalar("lr/learning_rate", np.asarray(schedule(epoch * train_x.shape[0])), epoch)

        # --- Save checkpoint ---
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0 and SAVE_CHECKPOINTS:
            save_dir = writer.get_logdir() if not LOAD_FROM_CHECKPOINT else LOAD_FROM_CHECKPOINT
            save_checkpoint(
                save_dir + "/checkpoints",
                epoch,
                params_enc,
                static_enc,
                params_dec,
                static_dec,
                opt_state_enc,
                opt_state_dec,
                hyper_enc,
                hyper_dec,
            )
    writer.close()
