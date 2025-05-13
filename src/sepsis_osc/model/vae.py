import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import distrax
from jaxtyping import Array, Float, PyTree  # https://github.com/google/jaxtyping

from utils.jax_config import setup_jax

setup_jax()

# === Model Definitions ===
latent_dim = 5
input_dim = 100
num_categories = 4
batch_size = 128


class Encoder(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(input_dim, 512, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(512, 1024, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(1024, 512, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(512, latent_dim, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "batch input_dim"]) -> Float[Array, "batch latent_dim"]:
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(latent_dim, 512, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(512, input_dim, key=key2),
            jax.nn.sigmoid,
        ]

    def __call__(self, z: Float[Array, "batch latent_dim"]) -> Float[Array, "batch input_dim"]:
        for layer in self.layers:
            z = layer(z)
        return z


def black_box_predict(z):
    # z: (batch, latent_dim)
    # Sum over the latent dimension yielding shape (batch, 1)
    logits = jnp.sin(jnp.sum(z, axis=-1, keepdims=True) * jnp.linspace(1, 2, num_categories))
    return jax.nn.softmax(logits, axis=-1)


# === Loss Function ===
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

    pred_c = black_box_predict(z)
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


# === Dummy Data ===

x_data = jax.random.normal(jax.random.PRNGKey(1), (batch_size, input_dim))
c_indices = jax.random.randint(jax.random.PRNGKey(2), (batch_size,), 0, num_categories)
true_c = jax.nn.one_hot(c_indices, num_classes=num_categories)

print("x_data:", x_data.shape)
print("c_indices:", c_indices.shape)
print("true_c:", true_c.shape)


# === Training Loop ===
for step in range(int(5e4)):
    key, subkey = jax.random.split(key)
    params_enc, params_dec, opt_state_enc, opt_state_dec, loss = training_step(
        params_enc,
        static_enc,
        params_dec,
        static_dec,
        opt_state_enc,
        opt_state_dec,
        x_data,
        true_c,
        subkey,
        opt_enc,
        opt_dec,
    )
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss:.4f}")
