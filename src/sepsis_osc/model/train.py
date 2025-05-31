import logging
from functools import wraps
from time import time
from typing import Any

import distrax
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
    init_decoder_weights,
    init_encoder_weights,
    load_checkpoint,
    save_checkpoint,
)
from sepsis_osc.simulation.data_classes import JAXLookup, SystemMetrics
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging

from dataclasses import dataclass

setup_jax(simulation=False)
setup_logging("info")
logger = logging.getLogger(__name__)
logging.getLogger("sepsis_osc.storage.storage_interface").setLevel(logging.ERROR)
logging.getLogger("sepsis_osc.model.vae").setLevel(logging.INFO)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("func:%r took: %2.6f sec" % (f.__name__, te - ts))
        return result

    return wrap


BATCH_SIZE = 512
EPOCHS = 1000
LOAD_FROM_CHECKPOINT = ""  # set load dir
LOAD_EPOCH = 0
SAVE_CHECKPOINTS = True
SAVE_EVERY_EPOCH = 10


# cosine decay
LEARNING_RATE_START = 1e-4
LEARNING_RATE_END = 1e-7


def ordinal_logits(floats: Float[Array, "batch"], n_classes: int = 24) -> Float[Array, "batch n_classes_minus_1"]:
    thresholds = jnp.linspace(-2.5, 2.5, n_classes - 1)
    floats = 5.0 * (floats - 0.5)
    logits = floats.squeeze()[:, None] - thresholds[None, :]
    return logits


def binary_logits(probs: Float[Array, "batch"], eps: float = 1e-6) -> Float[Array, "batch"]:
    probs = jnp.clip(probs, eps, 1 - eps)
    return jnp.log(probs) - jnp.log1p(-probs)


def ordinal_loss(
    predicted_sofa: Float[Array, "batch"], true_sofa: Float[Array, "batch"], n_classes: int = 24
) -> Float[Array, ""]:
    # https://arxiv.org/abs/1901.07884
    # CORN (Cumulative Ordinal Regression)
    logits = ordinal_logits(predicted_sofa, n_classes)  # [B, 23]
    targets = jnp.arange(n_classes - 1)  # [0, 1, ..., 22]
    true_sofa = true_sofa.squeeze()
    true_ord_targets = (true_sofa[:, None] > targets[None, :]).astype(jnp.float32)  # [B, 23]
    loss = optax.sigmoid_binary_cross_entropy(logits, true_ord_targets)
    return jnp.mean(loss)


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


@eqx.filter_jit
def get_pred_concepts(
    z: Float[Array, "batch_size latent_dim"], lookup_table: JAXLookup
) -> Float[Array, "batch_size 2"]:
    z = jax.lax.stop_gradient(z)
    z = z * jnp.array([1 / jnp.pi, 1 / jnp.pi, 1 / 2])[None, :]
    sim_results: SystemMetrics = lookup_table.get(z, threshold=150.0)

    # Extract f_1 and sr_2 + f_2 from the JAXSystemMetrics
    # SOFA and infection prob
    pred_c = jnp.array([sim_results.f_1, jnp.clip(sim_results.sr_2 + sim_results.f_2, 0, 1)])

    return pred_c.squeeze().T


# === Loss Function ===
@eqx.filter_jit
def per_sample_loss(
    encoder, decoder, x, true_concepts, key, lookup_table, lambda1=1.0, lambda2=1e-4
) -> dict[str, Array]:
    multi_keys = jax.random.split(key, BATCH_SIZE + 1)
    key, drop_key = multi_keys[-1], multi_keys[:-1]
    alpha_conc1, alpha_conc0, beta_conc1, beta_conc0, sigma_conc1, sigma_conc0 = jax.vmap(encoder)(x, drop_key)

    posterior_alpha = distrax.Beta(alpha=alpha_conc1, beta=alpha_conc0)
    posterior_beta = distrax.Beta(alpha=beta_conc1, beta=beta_conc0)
    posterior_sigma = distrax.Beta(alpha=sigma_conc1, beta=sigma_conc0)

    # Sample from each posterior using the reparameterization trick
    key_alpha, key_beta, key_sigma, _ = jax.random.split(key, 4)
    z_alpha = posterior_alpha.sample(seed=key_alpha)
    z_beta = posterior_beta.sample(seed=key_beta)
    z_sigma = posterior_sigma.sample(seed=key_sigma)

    z = jnp.concatenate([z_alpha * 2 - 1, z_beta * 1.3 + 0.2, z_sigma * 1.5], axis=-1)

    x_recon = jax.vmap(decoder)(z)
    recon_loss = jnp.mean((x - x_recon) ** 2, axis=-1)

    pred_concepts = get_pred_concepts(z, lookup_table)
    concept_loss = calc_concept_loss(pred_concepts, true_concepts[:, 0], true_concepts[:, 1])

    # Beta(1,1) for uniform prior
    prior = distrax.Beta(alpha=jnp.ones_like(alpha_conc1), beta=jnp.ones_like(alpha_conc0))

    # KL divergence for each latent variable
    kl_loss_alpha = posterior_alpha.kl_divergence(prior)
    kl_loss_beta = posterior_beta.kl_divergence(prior)
    kl_loss_sigma = posterior_sigma.kl_divergence(prior)

    kl_loss = jnp.sum(kl_loss_alpha, axis=-1) + jnp.sum(kl_loss_beta, axis=-1) + jnp.sum(kl_loss_sigma, axis=-1)

    total_loss = recon_loss + lambda1 * concept_loss + lambda2 * kl_loss
    return {
        "recon_loss": recon_loss,
        "concept_loss": concept_loss,
        "kl_loss": kl_loss,
        "total_loss": total_loss,
    }


@eqx.filter_jit
def total_loss_fn(encoder, decoder, x, true_c, key, lookup_table, lambda1=1.0, lambda2=1e-4):
    per_sample_losses = per_sample_loss(encoder, decoder, x, true_c, key, lookup_table, lambda1, lambda2)
    mean_losses = jax.tree.map(jnp.mean, per_sample_losses)
    return mean_losses["total_loss"], mean_losses


@eqx.filter_jit
def loss_fn_for_grad(models, x, true_c, current_key):
    encoder_inner, decoder_inner = models
    return total_loss_fn(encoder_inner, decoder_inner, x, true_c, current_key, lookup_table, 1e2, 1e-4)


def pad_and_batch_data(
    data: Float[Array, "num_samples dim"], batch_size: int
) -> tuple[Float[Array, "num_batches batch_size dim"], int, int]:
    num_original_samples = data.shape[0]
    remainder = num_original_samples % batch_size
    padding_needed = 0 if remainder == 0 else batch_size - remainder

    padded_data = jnp.pad(data, ((0, padding_needed), (0, 0)), mode="edge")

    num_batches = padded_data.shape[0] // batch_size
    batched_data = padded_data.reshape(num_batches, batch_size, data.shape[1])
    return batched_data, num_original_samples, num_batches


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
def process_epoch(
    params_enc,
    static_enc,
    params_dec,
    static_dec,
    opt_state_enc,
    opt_state_dec,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    key,
    opt_enc,
    opt_dec,
    lookup_table: JAXLookup,
    with_grad: bool = True,
):
    # --- pre-batch the data ---
    x_batched, num_original_samples, num_batches = pad_and_batch_data(x_data, BATCH_SIZE)
    y_batched, _, _ = pad_and_batch_data(y_data, BATCH_SIZE)

    # --- initialize accumulators ---
    sub_key, _ = jax.random.split(key)
    encoder_init = eqx.combine(params_enc, static_enc)
    decoder_init = eqx.combine(params_dec, static_dec)
    _, initial_aux_losses = total_loss_fn(encoder_init, decoder_init, x_batched[0], y_batched[0], sub_key, lookup_table)

    running_means = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), initial_aux_losses)
    running_M2s = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), initial_aux_losses)
    running_count = 0

    initial_carry = (
        params_enc,
        params_dec,
        opt_state_enc,
        opt_state_dec,
        key,
        running_means,
        running_M2s,
        running_count,
    )

    def fori_loop_body(batch_idx: int, carry_tup: tuple[Any, ...]):
        params_enc, params_dec, opt_state_enc, opt_state_dec, key_carry, running_means, running_M2s, running_count = (
            carry_tup
        )

        x_batch = x_batched[batch_idx]
        true_c_batch = y_batched[batch_idx]

        key_carry, subkey = jax.random.split(key_carry)

        encoder_batch = eqx.combine(params_enc, static_enc)
        decoder_batch = eqx.combine(params_dec, static_dec)
        models = (encoder_batch, decoder_batch)

        if with_grad:
            (total_loss, aux_losses), grads = eqx.filter_value_and_grad(loss_fn_for_grad, has_aux=True)(
                models, x_batch, true_c_batch, subkey
            )
            grads_enc, grads_dec = grads

            updates_enc, opt_state_enc = opt_enc.update(grads_enc, opt_state_enc, params_enc, value=total_loss)
            params_enc = eqx.apply_updates(params_enc, updates_enc)

            updates_dec, opt_state_dec = opt_dec.update(grads_dec, opt_state_dec, params_dec)
            params_dec = eqx.apply_updates(params_dec, updates_dec)
        else:
            _, aux_losses = loss_fn_for_grad(models, x_batch, true_c_batch, subkey)

        running_means, running_M2s, running_count = update_tree_stats(
            running_means, running_M2s, aux_losses, running_count
        )

        return (
            params_enc,
            params_dec,
            opt_state_enc,
            opt_state_dec,
            key_carry,
            running_means,
            running_M2s,
            running_count,
        )

    final_carry = jax.lax.fori_loop(0, num_batches, fori_loop_body, initial_carry)

    (
        final_params_enc,
        final_params_dec,
        final_opt_state_enc,
        final_opt_state_dec,
        final_key,
        running_means,
        running_M2s,
        running_count,
    ) = final_carry

    overall_mean_losses = jax.tree.map(lambda stat_mean: stat_mean, running_means)
    overall_std_losses = jax.tree.map(lambda stat_M2: jnp.sqrt(jnp.maximum(stat_M2 / running_count, 0.0)), running_M2s)

    return (
        final_params_enc,
        final_params_dec,
        final_opt_state_enc,
        final_opt_state_dec,
        final_key,
        overall_mean_losses,
        overall_std_losses,
    )


# === Data ===
train_y, train_x, val_y, val_x, test_y, test_x = [
    np.array(
        v.drop([
            col
            for col in v.columns
            if col.startswith("Missing") or col in {"stay_id", "time", "sep3_alt", "__index_level_0__", "los_icu"}
        ]),
        dtype=np.float32,
    )
    for inner in data.values()
    for v in inner.values()
]

num_samples = train_x.shape[0]
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

params_enc, static_enc = eqx.partition(encoder, eqx.is_array)
params_dec, static_dec = eqx.partition(decoder, eqx.is_array)

schedule = optax.cosine_decay_schedule(
    init_value=LEARNING_RATE_START, decay_steps=int(EPOCHS * 0.7 * train_x.shape[0]), alpha=LEARNING_RATE_END
)
opt_enc = optax.adam(schedule)
opt_dec = optax.adam(schedule)


opt_state_enc = opt_enc.init(params_enc)
opt_state_dec = opt_dec.init(params_dec)

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


# === Training Loop ===
writer = SummaryWriter()
current_losses = {"total_loss": jnp.inf, "recon_loss": jnp.inf, "concept_loss": jnp.inf, "kl_loss": jnp.inf}
for epoch in range(EPOCHS):
    perm = jax.random.permutation(key, num_samples)
    key, _ = jax.random.split(key)
    x_shuffled = train_x[perm]
    y_shuffled = train_y[perm]

    params_enc, params_dec, opt_state_enc, opt_state_dec, key, epoch_mean_losses, epoch_std_losses = process_epoch(
        params_enc=params_enc,
        static_enc=static_enc,
        params_dec=params_dec,
        static_dec=static_dec,
        opt_state_enc=opt_state_enc,
        opt_state_dec=opt_state_dec,
        x_data=jnp.asarray(x_shuffled),
        y_data=jnp.asarray(y_shuffled),
        key=key,
        opt_enc=opt_enc,
        opt_dec=opt_dec,
        lookup_table=lookup_table,
        with_grad=True,
    )

    logger.info(
        f"Epoch {epoch}: Training Metrics: "
        f"Total Loss = {epoch_mean_losses['total_loss']:.4f}, "
        f"Recon Loss = {epoch_mean_losses['recon_loss']:.4f}, "
        f"Concept Loss = {epoch_mean_losses['concept_loss']:.4f}, "
        f"KL Loss = {epoch_mean_losses['kl_loss']:.4f}"
    )
    for loss_name, loss_value in epoch_mean_losses.items():
        writer.add_scalar(f"train/{loss_name}_mean", np.asarray(loss_value), epoch)

    key, _ = jax.random.split(key)
    _, _, _, _, key, val_mean_losses, val_std_losses = process_epoch(
        params_enc=params_enc,
        static_enc=static_enc,
        params_dec=params_dec,
        static_dec=static_dec,
        opt_state_enc=opt_state_enc,  # unused
        opt_state_dec=opt_state_dec,  # unused
        x_data=jnp.asarray(val_x),
        y_data=jnp.asarray(val_y),
        key=key,
        opt_enc=opt_enc,  # unused
        opt_dec=opt_dec,  # unused
        lookup_table=lookup_table,
        with_grad=False,
    )
    logger.warning(
        f"Epoch {epoch}: Validation Metrics: "
        f"Total Loss = {val_mean_losses['total_loss']:.4f} ({val_std_losses['total_loss']:.4f}), "
        f"Recon Loss = {val_mean_losses['recon_loss']:.4f} ({val_std_losses['recon_loss']:.4f}), "
        f"Concept Loss = {val_mean_losses['concept_loss']:.4f} ({val_std_losses['concept_loss']:.4f}), "
        f"KL Loss = {val_mean_losses['kl_loss']:.4f} ({val_std_losses['kl_loss']:.4f})"
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

# TODO
# different latent prior
# result visualization
