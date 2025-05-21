import logging
from functools import wraps
from time import time

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree  # https://github.com/google/jaxtyping

from sepsis_osc.model.data_loading import data
from sepsis_osc.simulation.data_classes import SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging

setup_logging("info")
logger = logging.getLogger(__name__)
# logging.getLogger("sepsis_osc.storage.storage_interface").setLevel(logging.ERROR)
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


# TODO
# structured vae?
# ranges for parameter

# === Model Definitions ===
LATENT_DIM = 3
INPUT_DIM = 52
# NUM_CATEGORIES = 2
BATCH_SIZE = 1024
EPOCHS = 100


class Encoder(eqx.Module):
    layers: list

    alpha_layer: eqx.nn.Linear
    beta_layer: eqx.nn.Linear
    sigma_layer: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4, key_alpha, key_beta, key_sigma, key = jax.random.split(key, 8)
        self.layers = [
            eqx.nn.Linear(INPUT_DIM, 2048, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(2048, 2048, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(2048, 512, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(512, LATENT_DIM, key=key4),
        ]
        # Separate linear layers for each latent parameter before applying activation functions
        self.alpha_layer = eqx.nn.Linear(LATENT_DIM, 1, key=key_alpha)
        self.beta_layer = eqx.nn.Linear(LATENT_DIM, 1, key=key_beta)
        self.sigma_layer = eqx.nn.Linear(LATENT_DIM, 1, key=key_sigma)

    def __call__(self, x: Float[Array, "batch input_dim"]) -> Float[Array, "batch latent_dim"]:
        for layer in self.layers:
            x = layer(x)
        alpha_raw = self.alpha_layer(x)
        beta_raw = self.beta_layer(x)
        sigma_raw = self.sigma_layer(x)

        # alpha in [-1, 1]: tanh
        alpha = jax.nn.tanh(alpha_raw)

        # beta in [0, 2.0]: sigmoid and scale by 2.0
        beta = jax.nn.sigmoid(beta_raw) * 2.0

        # sigma in [0.0, 1.0]: sigmoid
        sigma = jax.nn.sigmoid(sigma_raw)

        z = jnp.concatenate([alpha, beta, sigma], axis=-1)
        return z


class Decoder(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(LATENT_DIM, 512, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(512, INPUT_DIM, key=key2),
            jax.nn.sigmoid,
        ]

    def __call__(self, z: Float[Array, "batch latent_dim"]) -> Float[Array, "batch input_dim"]:
        for layer in self.layers:
            z = layer(z)
        return z


# def get_pred_cats(z):
#     return jnp.zeros((z.shape[0], 2))


@jax.custom_jvp
def get_pred_cats(z):
    result_shape = jax.ShapeDtypeStruct((z.shape[0], 2), z.dtype)
    return eqx.filter_pure_callback(get_pred_cats_host, z, result_shape_dtypes=result_shape)


@timing
def get_pred_cats_host(z):
    z_confs = np.array([
        SystemConfig(
            N=100,
            C=int(100 * 0.2),
            omega_1=0.0,
            omega_2=0.0,
            a_1=1.0,
            epsilon_1=0.03,
            epsilon_2=0.3,
            alpha=zi[0].astype(float),
            beta=zi[1].astype(float),
            sigma=zi[2].astype(float),
        ).as_index
        for zi in z
    ])
    sim_results = sim_storage.read_multiple_results(z_confs, 100)

    pred_c = jnp.zeros((z.shape[0], 2), device=jax.devices("gpu")[0])  # we return sofa + susp_inf
    if not sim_results:
        return pred_c

    pred_c.at[:, 0].set(sim_results.f_1.T.squeeze() * 24)
    pred_c.at[:, 1].set(np.clip(sim_results.sr_2.T.squeeze() + sim_results.f_2.T.squeeze(), 0, 1))
    return pred_c


@get_pred_cats.defjvp
def get_pred_cats_jvp(primals, tangents):
    # primals: (z,) - the original input to get_pred_cats
    # tangents: (dz,) - the tangent vector corresponding to z

    # The JVP of a pure callback is typically zero because the
    # computation happens outside the JAX tracer.
    return get_pred_cats(*primals), jax.numpy.zeros_like(get_pred_cats(*primals))


# === Loss Function ===
@eqx.filter_jit
def total_loss_fn(encoder, decoder, x, true_c, key, lambda1=1.0, lambda2=1e-4):
    enc_out = jax.vmap(encoder)(x)
    # Split the encoder output into loc and log_scale.
    loc, log_scale = enc_out, enc_out.copy()
    scale = jnp.exp(log_scale)

    eps_key, sample_key = jax.random.split(key)
    eps = jax.random.normal(eps_key, loc.shape)
    z = loc + scale * eps

    x_recon = jax.vmap(decoder)(z)
    recon_loss = jnp.mean((x - x_recon) ** 2)

    pred_c = get_pred_cats(z)
    category_loss = -jnp.mean(jnp.sum(true_c * jnp.log(pred_c + 1e-8), axis=-1))

    prior = distrax.Normal(0, 1)
    posterior = distrax.Normal(loc, scale)
    kl_loss = jnp.mean(jnp.sum(posterior.kl_divergence(prior), axis=-1))
    return recon_loss + lambda1 * category_loss + lambda2 * kl_loss


# === Training Step ===
@eqx.filter_jit
def training_step(
    params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec, x, true_c, key, opt_enc, opt_dec
):
    encoder = eqx.combine(params_enc, static_enc)
    decoder = eqx.combine(params_dec, static_dec)

    # Note: We pass the models as a tuple to compute the loss.
    def loss_fn(models, x, true_c, key):
        encoder, decoder = models
        return total_loss_fn(encoder, decoder, x, true_c, key)

    models = (encoder, decoder)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(models, x, true_c, key)

    grads_enc, grads_dec = grads  # unpack the tuple

    updates_enc, opt_state_enc = opt_enc.update(grads_enc, opt_state_enc, params_enc)
    params_enc = eqx.apply_updates(params_enc, updates_enc)

    updates_dec, opt_state_dec = opt_dec.update(grads_dec, opt_state_dec, params_dec)
    params_dec = eqx.apply_updates(params_dec, updates_dec)

    return params_enc, params_dec, opt_state_enc, opt_state_dec, loss


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
    models = (encoder, decoder)

    num_samples = x.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    padding_needed = num_batches * batch_size - num_samples
    if padding_needed > 0:
        x_pad_shape = list(x.shape)
        x_pad_shape[0] = padding_needed
        x_padded = jnp.concatenate([x, jnp.zeros(x_pad_shape, dtype=x.dtype)], axis=0)

        true_c_pad_shape = list(true_c.shape)
        true_c_pad_shape[0] = padding_needed
        true_c_padded = jnp.concatenate([true_c, jnp.zeros(true_c_pad_shape, dtype=true_c.dtype)], axis=0)
    else:
        x_padded = x
        true_c_padded = true_c

    x_batched = x_padded.reshape(num_batches, batch_size, *x.shape[1:])
    true_c_batched = true_c_padded.reshape(num_batches, batch_size, *true_c.shape[1:])

    def loss_fn(models, x_batch, true_c_batch, key_batch):
        encoder, decoder = models
        return total_loss_fn(encoder, decoder, x_batch, true_c_batch, key_batch)

    @eqx.filter_jit
    def single_batch_loss(models, x_batch, true_c_batch, key_batch):
        loss, _ = eqx.filter_value_and_grad(loss_fn)(models, x_batch, true_c_batch, key_batch)
        return loss

    def body_fn(i, carry):
        key_carry, losses_array = carry
        key_carry, subkey = jax.random.split(key_carry)

        x_batch = x_batched[i]
        true_c_batch = true_c_batched[i]
        loss = single_batch_loss(models, x_batch, true_c_batch, subkey)

        losses_array = losses_array.at[i].set(loss)

        return (key_carry, losses_array)

    init_losses = jnp.zeros(num_batches)

    final_key, batch_losses_sums = jax.lax.fori_loop(0, num_batches, body_fn, (key, init_losses))

    total_sum_of_losses = jnp.sum(batch_losses_sums)
    mean_loss = total_sum_of_losses / num_samples
    std_loss = jnp.std(batch_losses_sums)

    return mean_loss, std_loss


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
loss = jnp.inf
for epoch in range(EPOCHS):
    perm = jax.random.permutation(key, num_samples)
    key, _ = jax.random.split(key)
    x_shuffled = train_x[perm]
    y_shuffled = train_y[perm]

    for i in range(0, num_samples, BATCH_SIZE):
        key, subkey = jax.random.split(key)
        params_enc, params_dec, opt_state_enc, opt_state_dec, loss = training_step(
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
        if i % 10 == 0 and loss is not None:
            logger.info(f"Epoch {epoch} | Step {i}: loss = {loss:.4f}")

    key, _ = jax.random.split(key)
    # Call validation with the BATCH_SIZE
    val_mean_loss, val_std_loss = validation(
        params_enc, static_enc, params_dec, static_dec, val_x, val_y, key, BATCH_SIZE
    )
    logger.warning(f"Epoch {epoch}: Validation Loss = {val_mean_loss:.4f} (Std Dev: {val_std_loss:.4f})")


# TODO
# better validation
# profiling (especially storage)
# different latent prior
# logger
