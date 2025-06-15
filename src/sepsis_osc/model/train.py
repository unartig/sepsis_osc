import logging
from functools import wraps
from time import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.model.data_loading import data
from sepsis_osc.model.vae import (
    Decoder,
    Encoder,
    get_pred_concepts,
    init_decoder_weights,
    init_encoder_weights,
    load_checkpoint,
    save_checkpoint,
    LATENT_DIM,
)
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging

setup_jax(simulation=False)
setup_logging("info")
logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("func:%r took: %2.6f sec" % (f.__name__, te - ts))
        return result

    return wrap


BATCH_SIZE = 256
EPOCHS = 2000
LOAD_FROM_CHECKPOINT = ""  # set load dir, e.g. "runs/..."
LOAD_EPOCH = 0
SAVE_CHECKPOINTS = True
SAVE_EVERY_EPOCH = 10


# cosine decay
LR_INIT = 5e-3
LR_END = 1e-8
LR_EPOCH_END = 50
ENC_WD = 1e-3

ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE = (-1.0, 1.0), (0.2, 1.5), (0.0, 1.5)


def ordinal_logits(floats: Float[Array, "batch"], n_classes: int = 24) -> Float[Array, "batch n_classes_minus_1"]:
    thresholds = jnp.linspace(-2.5, 2.5, n_classes - 1)
    # [0, 1] -> [-2.5, 2.5]
    floats = 5.0 * (floats - 0.5)
    logits = floats.squeeze()[:, None] - thresholds[None, :]
    return logits


# TODO
# try
# https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/
def ordinal_loss(
    predicted_sofa: Float[Array, "batch"], true_sofa: Float[Array, "batch"], n_classes: int = 24
) -> Float[Array, ""]:
    # https://arxiv.org/abs/1901.07884
    # CORN (Cumulative Ordinal Regression)
    logits = ordinal_logits(predicted_sofa, n_classes)  # [B, 23]

    targets = jnp.arange(n_classes - 1)  # [0, 1, ..., 22]
    true_sofa = true_sofa.squeeze()
    true_ord_targets = (true_sofa[:, None] > targets[None, :]).astype(jnp.bool)  # [B, 23]
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_ord_targets))


def binary_logits(probs: Float[Array, "batch"], eps: float = 1e-6) -> Float[Array, "batch"]:
    probs = jnp.clip(probs, eps, 1 - eps)
    return jnp.log(probs) - jnp.log1p(-probs)


def binary_loss(predicted_infection: Float[Array, "batch"], true_infection: Float[Array, "batch"]) -> Float[Array, ""]:
    logits = binary_logits(predicted_infection)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_infection))


def calc_concept_loss(
    prediction: Float[Array, "batch 2"],
    true_sofa: Float[Array, "batch"],
    true_infection: Float[Array, "batch"],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Float[Array, ""]:
    pred_sofa = prediction[:, 0]  # SOFA prediction
    pred_infection = prediction[:, 1]  # Infection prediction

    loss_sofa = ordinal_loss(pred_sofa, true_sofa)
    loss_infection = binary_loss(pred_infection, true_infection)

    return alpha * loss_sofa + beta * loss_infection


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
    lookup_table: JAXLookup,
    lambda1=1e2,
    lambda2=1e1,
) -> tuple[Array, dict[str, Array]]:
    aux_losses = {}
    encoder, decoder = models

    key, *drop_keys = jax.random.split(key, BATCH_SIZE + 1)
    drop_keys = jnp.array(drop_keys)

    alpha, beta, sigma = jax.vmap(encoder)(x, key=drop_keys)
    alpha = alpha * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0]
    beta = beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma = sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]

    z = jnp.concatenate([alpha, beta, sigma], axis=-1)

    aux_losses["locality_loss"] = calc_locality_loss(x, z)

    x_recon = jax.vmap(decoder)(z)
    aux_losses["recon_loss"] = jnp.mean((x - x_recon) ** 2, axis=-1)

    nearest_metrics = SystemConfig.batch_as_index(alpha=alpha, beta=beta, sigma=sigma, C=0.2)
    pred_concepts = get_pred_concepts(nearest_metrics, lookup_table)
    # TODO change, we actually want to predict both :^)
    aux_losses["concept_loss"] = calc_concept_loss(
        pred_concepts, true_concepts[:, 0], true_concepts[:, 1], alpha=1.0, beta=0.0
    )

    aux_losses["total_loss"] = (
        aux_losses["recon_loss"] + lambda1 * aux_losses["concept_loss"] + lambda2 + aux_losses["locality_loss"]
    )

    aux_losses = jax.tree_util.tree_map(jnp.mean, aux_losses)
    return aux_losses["total_loss"], aux_losses


@eqx.filter_jit
def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    models: tuple[Encoder, Decoder],
    opt_state_enc: optax.OptState,
    opt_state_dec: optax.OptState,
    update_enc,
    update_dec,
    *,
    key,
    lookup_table: JAXLookup,
):
    encoder, decoder = models
    (total_loss, aux_losses), (grads_enc, grads_dec) = eqx.filter_value_and_grad(loss, has_aux=True)(
        models, x_batch, true_c_batch, key=key, lookup_table=lookup_table
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


def update_mean(m, M2, batch_size, total_count, values):
    batch_mean = jnp.mean(values, axis=0) if values.ndim > 0 else values
    delta = batch_mean - m
    new_mean = m + delta * batch_size / total_count
    return new_mean


def update_M2(m, M2, batch_size, total_count, values):
    batch_mean = jnp.mean(values, axis=0) if values.ndim > 0 else values
    delta = batch_mean - m
    batch_M2 = jnp.sum((values - batch_mean) ** 2, axis=0) if values.ndim > 0 else 0.0
    mean_diff_contrib = delta**2 * (total_count - batch_size) * batch_size / total_count
    new_M2 = M2 + batch_M2 + mean_diff_contrib
    return new_M2


def update_tree_stats(mean_tree, M2_tree, values_tree, count):
    example_leaf = jax.tree_util.tree_leaves(values_tree)[0]
    batch_size = example_leaf.shape[0] if example_leaf.ndim > 0 else 1
    total_count = count + batch_size

    new_means = jax.tree_util.tree_map(
        lambda m, M2, v: update_mean(m, M2, batch_size, total_count, v), mean_tree, M2_tree, values_tree
    )
    new_M2s = jax.tree_util.tree_map(
        lambda m, M2, v: update_M2(m, M2, batch_size, total_count, v), mean_tree, M2_tree, values_tree
    )

    return new_means, new_M2s, total_count


@timing
@eqx.filter_jit
def process_train_epoch(
    encoder,
    decoder,
    opt_state_enc,
    opt_state_dec,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    update_enc,
    update_dec,
    *,
    key,
    lookup_table: JAXLookup,
):
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    key, init_key = jax.random.split(key)
    _, initial_aux_losses = loss((encoder, decoder), x_data[0], y_data[0], key=init_key, lookup_table=lookup_table)

    running_means = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), initial_aux_losses)
    running_M2s = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), initial_aux_losses)
    running_count = jnp.array(0)

    def scan_step(carry, batch):
        (encoder_f, decoder_f), (opt_state_enc, opt_state_dec), running_means, running_M2s, running_count, key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jax.random.split(key)

        encoder_full = eqx.combine(encoder_f, encoder_static)
        decoder_full = eqx.combine(decoder_f, decoder_static)

        (encoder_new, decoder_new), (opt_state_enc, opt_state_dec), losses = step_model(
            batch_x,
            batch_true_c,
            (encoder_full, decoder_full),
            opt_state_enc,
            opt_state_dec,
            update_enc,
            update_dec,
            key=batch_key,
            lookup_table=lookup_table,
        )

        running_means, running_M2s, running_count = update_tree_stats(
            running_means, running_M2s, losses[1], running_count
        )
        encoder_f = eqx.filter(encoder_new, eqx.is_inexact_array)
        decoder_f = eqx.filter(decoder_new, eqx.is_inexact_array)

        return (
            (encoder_f, decoder_f),
            (opt_state_enc, opt_state_dec),
            running_means,
            running_M2s,
            running_count,
            key,
        ), None

    carry = (
        (encoder_params, decoder_params),
        (opt_state_enc, opt_state_dec),
        running_means,
        running_M2s,
        running_count,
        key,
    )
    batches = (x_data, y_data)
    carry, _ = jax.lax.scan(scan_step, carry, batches)

    (encoder_params, decoder_params), (opt_state_enc, opt_state_dec), running_means, running_M2s, running_count, key = (
        carry
    )
    overall_mean_losses = jax.tree.map(lambda stat_mean: stat_mean, running_means)
    overall_std_losses = jax.tree.map(lambda stat_M2: jnp.sqrt(jnp.maximum(stat_M2 / running_count, 0.0)), running_M2s)

    return (
        encoder_params,
        decoder_params,
        opt_state_enc,
        opt_state_dec,
        key,
        overall_mean_losses,
        overall_std_losses,
    )


@timing
@eqx.filter_jit
def process_val_epoch(
    encoder,
    decoder,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    *,
    key,
    lookup_table: JAXLookup,
):
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    key, init_key = jax.random.split(key)
    _, initial_aux_losses = loss((encoder, decoder), x_data[0], y_data[0], key=init_key, lookup_table=lookup_table)

    running_means = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), initial_aux_losses)
    running_M2s = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), initial_aux_losses)
    running_count = jnp.array(0)

    def scan_step(carry, batch):
        (encoder_f, decoder_f), running_means, running_M2s, running_count, key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jax.random.split(key)

        encoder_full = eqx.combine(encoder_f, encoder_static)
        decoder_full = eqx.combine(decoder_f, decoder_static)

        _, aux_losses = loss(
            (encoder_full, decoder_full), batch_x, batch_true_c, key=batch_key, lookup_table=lookup_table
        )

        running_means, running_M2s, running_count = update_tree_stats(
            running_means, running_M2s, aux_losses, running_count
        )

        return ((encoder_f, decoder_f), running_means, running_M2s, running_count, key), None

    carry = ((encoder_params, decoder_params), running_means, running_M2s, running_count, key)
    batches = (x_data, y_data)
    carry, _ = jax.lax.scan(scan_step, carry, batches)

    (encoder_params, decoder_params), running_means, running_M2s, running_count, key = carry
    overall_mean_losses = jax.tree.map(lambda stat_mean: stat_mean, running_means)
    overall_std_losses = jax.tree.map(lambda stat_M2: jnp.sqrt(jnp.maximum(stat_M2 / running_count, 0.0)), running_M2s)

    return (
        key,
        overall_mean_losses,
        overall_std_losses,
    )


# === Data ===
def prepare_batches(
    x_data: Float[Array, "num_samples dim"],
    y_data: Float[Array, "num_samples dim"],
    batch_size: int,
    key,
) -> tuple[Float[Array, "num_batches batch_size dim"], Float[Array, "num_batches batch_size dim"], int]:
    num_samples = x_data.shape[0]
    num_features = x_data.shape[1]

    # Shuffle data
    perm = jax.random.permutation(key, num_samples)
    x_shuffled = x_data[perm]
    y_shuffled = y_data[perm]

    # Ensure full batches only
    num_full_batches = num_samples // batch_size
    x_truncated = x_shuffled[: num_full_batches * batch_size]
    y_truncated = y_shuffled[: num_full_batches * batch_size]

    # Reshape into batches
    x_batched = x_truncated.reshape(num_full_batches, batch_size, num_features)
    y_batched = y_truncated.reshape(num_full_batches, batch_size, 2)

    return x_batched, y_batched, num_full_batches


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
lookup_table = JAXLookup(metrics=np_metrics.to_jax(), indices=jnp.asarray(indices))

# === Initialization ===
key = jax.random.PRNGKey(0)
key_enc, key_dec = jax.random.split(key)

encoder = Encoder(key_enc)
decoder = Decoder(key_dec)

key_enc_init, key_dec_init = jax.random.split(key, 2)
encoder = init_encoder_weights(encoder, key_enc_init)
decoder = init_decoder_weights(decoder, key_dec_init)

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


schedule = optax.cosine_decay_schedule(
    init_value=LR_INIT,
    decay_steps=int(LR_EPOCH_END * (train_x.shape[0] // BATCH_SIZE) * BATCH_SIZE),
    alpha=LR_END / LR_INIT,
)
opt_enc = optax.flatten(optax.adamw(schedule, weight_decay=ENC_WD))
opt_dec = optax.flatten(optax.adamw(LR_INIT))

params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec = None, None, None, None, None, None
if LOAD_FROM_CHECKPOINT:
    try:
        params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec = load_checkpoint(
            LOAD_FROM_CHECKPOINT + "/checkpoints", LOAD_EPOCH, opt_enc, opt_dec
        )
        initial_epoch = LOAD_EPOCH + 1
        logger.info(f"Resuming training from epoch {initial_epoch}")
    except FileNotFoundError as e:
        logger.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
        LOAD_FROM_CHECKPOINT = ""
if not all((params_enc, static_enc, params_dec, params_dec, opt_state_enc, opt_state_dec)):
    logger.info("Instantiating new models")
    params_enc, static_enc = eqx.partition(encoder, eqx.is_inexact_array)
    params_dec, static_dec = eqx.partition(decoder, eqx.is_inexact_array)
    opt_state_enc = opt_enc.init(params_enc)
    opt_state_dec = opt_dec.init(params_dec)

# === Training Loop ===

writer = SummaryWriter()
current_losses = {
    "total_loss": jnp.asarray(jnp.inf),
    "recon_loss": jnp.asarray(jnp.inf),
    "concept_loss": jnp.asarray(jnp.inf),
    "locality_loss": jnp.asarray(jnp.inf),
}
shuffle_val, key = jax.random.split(key, 2)
val_x, val_y, nval_batches = prepare_batches(val_x, val_y, BATCH_SIZE, shuffle_val)
for epoch in range(LOAD_EPOCH, EPOCHS):
    shuffle_key, key = jax.random.split(key)
    x_shuffled, y_shuffled, ntrain_batches = prepare_batches(train_x, train_y, BATCH_SIZE, shuffle_key)

    params_enc, params_dec, opt_state_enc, opt_state_dec, key, epoch_mean_losses, epoch_std_losses = (
        process_train_epoch(
            encoder,
            decoder,
            opt_state_enc=opt_state_enc,
            opt_state_dec=opt_state_dec,
            x_data=x_shuffled,
            y_data=y_shuffled,
            update_enc=opt_enc.update,
            update_dec=opt_dec.update,
            key=key,
            lookup_table=lookup_table,
        )
    )
    encoder = eqx.combine(params_enc, static_enc)
    decoder = eqx.combine(params_dec, static_dec)

    logger.info(
        f"Epoch {epoch}: Training Metrics: "
        f"Total Loss = {epoch_mean_losses['total_loss']:.4f}, "
        f"Recon Loss = {epoch_mean_losses['recon_loss']:.4f}, "
        f"Concept Loss = {epoch_mean_losses['concept_loss']:.4f}, "
        f"Locality Loss = {epoch_mean_losses['locality_loss']:.4f}"
    )
    for loss_name, loss_value in epoch_mean_losses.items():
        writer.add_scalar(f"train/{loss_name}_mean", np.asarray(loss_value), epoch)
    del x_shuffled, y_shuffled

    key, _ = jax.random.split(key)
    key, val_mean_losses, val_std_losses = process_val_epoch(
        encoder,
        decoder,
        x_data=val_x,
        y_data=val_y,
        key=key,
        lookup_table=lookup_table,
    )
    logger.warning(
        f"Epoch {epoch}: Validation Metrics: "
        f"Total Loss = {val_mean_losses['total_loss']:.4f} ({val_std_losses['total_loss']:.4f}), "
        f"Recon Loss = {val_mean_losses['recon_loss']:.4f} ({val_std_losses['recon_loss']:.4f}), "
        f"Concept Loss = {val_mean_losses['concept_loss']:.4f} ({val_std_losses['concept_loss']:.4f}), "
        f"Locality Loss = {val_mean_losses['locality_loss']:.4f} ({val_std_losses['locality_loss']:.4f})"
    )

    for loss_name, loss_value in val_mean_losses.items():
        writer.add_scalar(f"val/{loss_name}_mean", np.asarray(loss_value), epoch)
    for loss_name, loss_value in val_std_losses.items():
        writer.add_scalar(f"val/{loss_name}_std", np.asarray(loss_value), epoch)
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
