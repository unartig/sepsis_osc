import logging
from functools import wraps
from time import time
from typing import Optional, Callable

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree  # https://github.com/google/jaxtyping

from sepsis_osc.model.data_loading import data
from sepsis_osc.simulation.data_classes import SystemConfig, SystemMetrics
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging

setup_logging("info")
logger = logging.getLogger(__name__)
logging.getLogger("sepsis_osc.storage.storage_interface").setLevel(logging.ERROR)
logging.getLogger("sepsis_osc.model.vae").setLevel(logging.INFO)

setup_jax(simulation=False)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("func:%r took: %2.6f sec" % (f.__name__, te - ts))
        return result

    return wrap


# === Model Definitions ===
LATENT_DIM = 3
INPUT_DIM = 52
# NUM_CATEGORIES = 2
BATCH_SIZE = 512
EPOCHS = 100
ENC_HIDDEN = 512
DEC_HIDDEN = 512


class Encoder(eqx.Module):
    layers: list

    alpha_layer: eqx.nn.Linear
    beta_layer: eqx.nn.Linear
    sigma_layer: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4, key_alpha, key_beta, key_sigma, key = jax.random.split(key, 8)
        self.layers = [
            eqx.nn.Linear(in_features=INPUT_DIM, out_features=ENC_HIDDEN, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(in_features=ENC_HIDDEN, out_features=ENC_HIDDEN, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(in_features=ENC_HIDDEN, out_features=ENC_HIDDEN, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(in_features=ENC_HIDDEN, out_features=LATENT_DIM, key=key4),
        ]
        # 2 features for beta distribution
        self.alpha_layer = eqx.nn.Linear(in_features=LATENT_DIM, out_features=2, key=key_alpha)
        self.beta_layer = eqx.nn.Linear(in_features=LATENT_DIM, out_features=2, key=key_beta)
        self.sigma_layer = eqx.nn.Linear(in_features=LATENT_DIM, out_features=2, key=key_sigma)

    def __call__(self, x: Float[Array, "batch input_dim"]):
        for layer in self.layers:
            x = layer(x)

        alpha_raw = self.alpha_layer(x)
        beta_raw = self.beta_layer(x)
        sigma_raw = self.sigma_layer(x)

        alpha_conc1 = jax.nn.softplus(alpha_raw[0:1]) + 1e-6
        alpha_conc0 = jax.nn.softplus(alpha_raw[1:2]) + 1e-6

        beta_conc1 = jax.nn.softplus(beta_raw[0:1]) + 1e-6
        beta_conc0 = jax.nn.softplus(beta_raw[1:2]) + 1e-6

        sigma_conc1 = jax.nn.softplus(sigma_raw[0:1]) + 1e-6
        sigma_conc0 = jax.nn.softplus(sigma_raw[1:2]) + 1e-6

        return (alpha_conc1, alpha_conc0, beta_conc1, beta_conc0, sigma_conc1, sigma_conc0)


class Decoder(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(in_features=LATENT_DIM, out_features=DEC_HIDDEN, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(in_features=DEC_HIDDEN, out_features=INPUT_DIM, key=key2),
            jax.nn.sigmoid,
        ]

    def __call__(self, z: Float[Array, "batch latent_dim"]) -> Float[Array, "batch input_dim"]:
        for layer in self.layers:
            z = layer(z)
        return z


# def get_pred_cats(z):
#     return jnp.zeros((z.shape[0], 2))


@jax.custom_jvp
def get_pred_cats(z) -> Float[Array, "batch_size 2"]:
    result_shape = jax.ShapeDtypeStruct(shape=(z.shape[0], 2), dtype=z.dtype)
    return eqx.filter_pure_callback(get_pred_cats_host, z, result_shape_dtypes=result_shape)


# @timing
def get_pred_cats_host(z) -> Float[Array, "batch_size 2"]:
    z_alpha_mapped = (z[:, 0] * 2) - 1.0  # Maps (0,1) to (-1,1)
    z_beta_mapped = z[:, 1] * 2.0  # Maps (0,1) to (0,2)
    z_sigma_mapped = z[:, 2]  # (0,1) remains (0,1)

    z_confs: np.ndarray = SystemConfig.batch_as_index(
        z=jnp.concatenate([z_alpha_mapped[:, None], z_beta_mapped[:, None], z_sigma_mapped[:, None]], axis=-1),
        C=0.2,
    )
    sim_results: Optional[SystemMetrics] = sim_storage.read_multiple_results(params=z_confs, threshold=150)

    pred_c: Float[Array, "batch_size 2"] = np.zeros((z.shape[0], 2), dtype=z.dtype)  # we return sofa + susp_inf
    if not sim_results:
        return pred_c

    assert sim_results.sr_2 is not None
    pred_c[:, 0] = sim_results.f_1.T.squeeze()
    pred_c[:, 1] = np.clip(sim_results.sr_2.T.squeeze() + sim_results.f_2.T.squeeze(), 0, 1)
    return jnp.asarray(pred_c)


@get_pred_cats.defjvp
def get_pred_cats_jvp(primals, tangents) -> tuple[Float[Array, "batch_size 2"], Float[Array, "batch_size 2"]]:
    # primals: (z,) - the original input to get_pred_cats
    # tangents: (dz,) - the tangent vector corresponding to z

    # The JVP of a pure callback is zero because the
    # computation happens outside the JAX tracer.
    return get_pred_cats(*primals), jax.numpy.zeros_like(get_pred_cats(*primals))


# === Loss Function ===
@eqx.filter_jit
def per_sample_loss(encoder, decoder, x, true_c, key, lambda1=1.0, lambda2=1e-4) -> dict[str, Array]:
    alpha_conc1, alpha_conc0, beta_conc1, beta_conc0, sigma_conc1, sigma_conc0 = jax.vmap(encoder)(x)

    posterior_alpha = distrax.Beta(alpha=alpha_conc1, beta=alpha_conc0)
    posterior_beta = distrax.Beta(alpha=beta_conc1, beta=beta_conc0)
    posterior_sigma = distrax.Beta(alpha=sigma_conc1, beta=sigma_conc0)

    # Sample from each posterior using the reparameterization trick
    key_alpha, key_beta, key_sigma, _ = jax.random.split(key, 4)
    z_alpha = posterior_alpha.sample(seed=key_alpha)
    z_beta = posterior_beta.sample(seed=key_beta)
    z_sigma = posterior_sigma.sample(seed=key_sigma)

    # Concatenate samples to form the full latent vector z for the decoder and concept prediction
    z = jnp.concatenate([z_alpha, z_beta, z_sigma], axis=-1)

    x_recon = jax.vmap(decoder)(z)
    recon_loss = jnp.mean((x - x_recon) ** 2, axis=-1)

    pred_c = get_pred_cats(z)
    true_c.at[:, 0].set(true_c[:, 0] / 24)
    concept_loss = jnp.mean((true_c - pred_c) ** 2, axis=-1)

    # Define prior Beta distributions for each latent variable (e.g., Beta(1,1) for uniform prior)
    prior_alpha = distrax.Beta(alpha=jnp.ones_like(alpha_conc1), beta=jnp.ones_like(alpha_conc0))
    prior_beta = distrax.Beta(alpha=jnp.ones_like(beta_conc1), beta=jnp.ones_like(beta_conc0))
    prior_sigma = distrax.Beta(alpha=jnp.ones_like(sigma_conc1), beta=jnp.ones_like(sigma_conc0))

    # Calculate KL divergence for each latent variable and sum them
    kl_loss_alpha = posterior_alpha.kl_divergence(prior_alpha)
    kl_loss_beta = posterior_beta.kl_divergence(prior_beta)
    kl_loss_sigma = posterior_sigma.kl_divergence(prior_sigma)

    kl_loss = jnp.sum(kl_loss_alpha, axis=-1) + jnp.sum(kl_loss_beta, axis=-1) + jnp.sum(kl_loss_sigma, axis=-1)

    total_loss = recon_loss + lambda1 * concept_loss + lambda2 * kl_loss
    return {
        "recon_loss": recon_loss,
        "concept_loss": concept_loss,
        "kl_loss": kl_loss,
        "total_loss": total_loss,
    }


@eqx.filter_jit
def total_loss_fn(encoder, decoder, x, true_c, key, lambda1=1.0, lambda2=1e-4):
    per_sample_losses = per_sample_loss(encoder, decoder, x, true_c, key, lambda1, lambda2)
    mean_losses = jax.tree.map(jnp.mean, per_sample_losses)
    return mean_losses["total_loss"], mean_losses


# === Training Step ===
@eqx.filter_jit
def training_step(
    params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec, x, true_c, key, opt_enc, opt_dec
):
    encoder = eqx.combine(params_enc, static_enc)
    decoder = eqx.combine(params_dec, static_dec)

    def loss_fn(models, x, true_c, key):
        encoder, decoder = models
        return total_loss_fn(encoder, decoder, x, true_c, key)

    models = (encoder, decoder)
    # loss and gradients. `aux_losses` contains  total_loss_fn.
    (total_loss, aux_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(models, x, true_c, key)

    grads_enc, grads_dec = grads

    updates_enc, opt_state_enc = opt_enc.update(grads_enc, opt_state_enc, params_enc)
    params_enc = eqx.apply_updates(params_enc, updates_enc)

    updates_dec, opt_state_dec = opt_dec.update(grads_dec, opt_state_dec, params_dec)
    params_dec = eqx.apply_updates(params_dec, updates_dec)

    return params_enc, params_dec, opt_state_enc, opt_state_dec, aux_losses


# === Validation ===
@eqx.filter_jit
def validation(
    params_enc,
    static_enc,
    params_dec,
    static_dec,
    x,
    true_c,
    key,
    batch_size,
):
    encoder = eqx.combine(params_enc, static_enc)
    decoder = eqx.combine(params_dec, static_dec)

    num_samples = x.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    padding_needed = num_batches * batch_size - num_samples
    if padding_needed > 0:
        x_padded = jnp.concatenate([x, jnp.zeros_like(x[:padding_needed])], axis=0)
        true_c_padded = jnp.concatenate([true_c, jnp.zeros_like(true_c[:padding_needed])], axis=0)
    else:
        x_padded = x
        true_c_padded = true_c

    x_batched = x_padded.reshape(num_batches, batch_size, *x.shape[1:])
    true_c_batched = true_c_padded.reshape(num_batches, batch_size, *true_c.shape[1:])

    dummy_key = jax.random.PRNGKey(0)

    initial_per_sample_losses = per_sample_loss(encoder, decoder, x_batched[0], true_c_batched[0], dummy_key)
    accumulated_sums = jax.tree.map(lambda _: jnp.array(0.0, dtype=jnp.float32), initial_per_sample_losses)
    accumulated_sum_of_squares = jax.tree.map(lambda _: jnp.array(0.0, dtype=jnp.float32), initial_per_sample_losses)

    def body_fn(i, carry):
        key_carry, current_sums, current_sum_of_squares = carry
        key_carry, subkey = jax.random.split(key_carry)

        x_batch = x_batched[i]
        true_c_batch = true_c_batched[i]

        batch_per_sample_losses = per_sample_loss(encoder, decoder, x_batch, true_c_batch, subkey)

        updated_sums = jax.tree.map(lambda s, l: s + jnp.sum(l), current_sums, batch_per_sample_losses)
        updated_sum_of_squares = jax.tree.map(
            lambda ss, l: ss + jnp.sum(l**2), current_sum_of_squares, batch_per_sample_losses
        )

        return (key_carry, updated_sums, updated_sum_of_squares)

    final_key, total_sums, total_sum_of_squares = jax.lax.fori_loop(
        0, num_batches, body_fn, (key, accumulated_sums, accumulated_sum_of_squares)
    )

    overall_mean_losses = jax.tree.map(lambda s: s / num_samples, total_sums)

    # Calculate variance: E[X^2] - (E[X])^2
    # E[X^2] = total_sum_of_squares / num_samples
    # E[X] = overall_mean_losses
    overall_variances = jax.tree.map(
        lambda ss, m: (ss / num_samples) - (m**2), total_sum_of_squares, overall_mean_losses
    )
    overall_std_losses = jax.tree.map(lambda v: jnp.sqrt(jnp.maximum(0.0, v)), overall_variances)

    return overall_mean_losses, overall_std_losses


# === Initialization ===
key = jax.random.PRNGKey(0)
key_enc, key_dec = jax.random.split(key)

encoder = Encoder(key_enc)
decoder = Decoder(key_dec)

params_enc, static_enc = eqx.partition(encoder, eqx.is_array)
params_dec, static_dec = eqx.partition(decoder, eqx.is_array)

opt_enc = optax.adam(1e-3)
opt_dec = optax.adam(1e-3)

opt_state_enc = opt_enc.init(params_enc)
opt_state_dec = opt_dec.init(params_dec)

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

num_samples = train_x.shape[0]
print(train_y.shape, train_x.shape)
print(val_y.shape, val_x.shape)
print(test_y.shape, test_x.shape)

db_str = "Daisy"
sim_storage = Storage(
    key_dim=9,
    metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
    parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
    use_mem_cache=True,
)

# === Training Loop ===
current_losses = {"total_loss": jnp.inf, "recon_loss": jnp.inf, "concept_loss": jnp.inf, "kl_loss": jnp.inf}
for epoch in range(EPOCHS):
    perm = jax.random.permutation(key, num_samples)
    key, _ = jax.random.split(key)
    x_shuffled = train_x[perm]
    y_shuffled = train_y[perm]

    for i in range(0, num_samples, BATCH_SIZE):
        key, subkey = jax.random.split(key)
        params_enc, params_dec, opt_state_enc, opt_state_dec, current_losses = training_step(
            params_enc=params_enc,
            static_enc=static_enc,
            params_dec=params_dec,
            static_dec=static_dec,
            opt_state_enc=opt_state_enc,
            opt_state_dec=opt_state_dec,
            x=x_shuffled[i : i + BATCH_SIZE],
            true_c=y_shuffled[i : i + BATCH_SIZE],
            key=subkey,
            opt_enc=opt_enc,
            opt_dec=opt_dec,
        )
        if i % 100 == 0 and current_losses["total_loss"] is not None:
            logger.info(
                f"Epoch {epoch} | Step {i}: "
                f"Total Loss = {current_losses['total_loss']:.4f}, "
                f"Recon Loss = {current_losses['recon_loss']:.4f}, "
                f"Concept Loss = {current_losses['concept_loss']:.4f}, "
                f"KL Loss = {current_losses['kl_loss']:.4f}"
            )

    key, _ = jax.random.split(key)  # Update key for validation
    val_mean_losses, val_std_losses = validation(
        params_enc, static_enc, params_dec, static_dec, val_x, val_y, key, BATCH_SIZE
    )
    # Log validation statistics
    logger.warning(
        f"Epoch {epoch}: Validation Metrics: "
        f"Total Loss = {val_mean_losses['total_loss']:.4f} (Std Dev: {val_std_losses['total_loss']:.4f}), "
        f"Recon Loss = {val_mean_losses['recon_loss']:.4f} (Std Dev: {val_std_losses['recon_loss']:.4f}), "
        f"Concept Loss = {val_mean_losses['concept_loss']:.4f} (Std Dev: {val_std_losses['concept_loss']:.4f}), "
        f"KL Loss = {val_mean_losses['kl_loss']:.4f} (Std Dev: {val_std_losses['kl_loss']:.4f})"
    )


# TODO
# better validation
# profiling (especially storage)
# different latent prior
# logger
