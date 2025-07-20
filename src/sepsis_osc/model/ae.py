import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

logger = logging.getLogger(__name__)


# === Model Definitions ===
class Encoder(eqx.Module):
    # Layers
    initial_linear1: eqx.nn.Linear
    initial_norm1: eqx.nn.LayerNorm
    dropout1: eqx.nn.Dropout

    initial_linear2: eqx.nn.Linear
    initial_norm2: eqx.nn.LayerNorm
    dropout2: eqx.nn.Dropout

    attention_layer: eqx.nn.Linear
    post_attention_dropout: eqx.nn.Dropout

    final_linear1: eqx.nn.Linear
    final_norm1: eqx.nn.LayerNorm
    dropout3: eqx.nn.Dropout

    final_linear2: eqx.nn.Linear

    # Output heads
    alpha_layer: eqx.nn.Linear
    beta_layer: eqx.nn.Linear
    sigma_layer: eqx.nn.Linear

    # Params
    alpha_var: Array
    beta_var: Array
    sigma_var: Array

    # Hyperparams
    input_dim: int
    latent_dim: int
    enc_hidden: int
    dropout_rate: float

    def __init__(
        self,
        key: Array,
        input_dim: int,
        latent_dim: int,
        enc_hidden: int,
        dropout_rate: float = 0.5,
        dtype=jnp.float32,
    ):
        (
            key_lin1,
            key_lin2,
            key_lin3,
            key_lin4,
            key_lin5,
            key_alpha,
            key_beta,
            key_sigma,
            key_sofa,
            key_inf,
        ) = jax.random.split(key, 10)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_hidden = enc_hidden
        self.dropout_rate = dropout_rate

        # Initial layers
        self.initial_linear1 = eqx.nn.Linear(input_dim, enc_hidden, key=key_lin1, dtype=dtype)
        self.initial_norm1 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)

        self.initial_linear2 = eqx.nn.Linear(enc_hidden, enc_hidden, key=key_lin2, dtype=dtype)
        self.initial_norm2 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

        # Attention
        self.attention_layer = eqx.nn.Linear(enc_hidden, input_dim, key=key_lin3, dtype=dtype)
        self.post_attention_dropout = eqx.nn.Dropout(dropout_rate)

        # Final encoder layers
        self.final_linear1 = eqx.nn.Linear(enc_hidden + input_dim, enc_hidden, key=key_lin4, dtype=dtype)
        self.final_norm1 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)
        self.dropout3 = eqx.nn.Dropout(dropout_rate)
        self.final_linear2 = eqx.nn.Linear(enc_hidden, latent_dim, key=key_lin5, dtype=dtype)

        # Output heads
        self.alpha_layer = eqx.nn.Linear(latent_dim, 2, key=key_alpha, dtype=dtype)
        self.beta_layer = eqx.nn.Linear(latent_dim, 2, key=key_beta, dtype=dtype)
        self.sigma_layer = eqx.nn.Linear(latent_dim, 2, key=key_sigma, dtype=dtype)

        self.alpha_var = jnp.ones((1,))
        self.beta_var = jnp.ones((1,))
        self.sigma_var = jnp.ones((1,))

    def __call__(
        self, x: Float[Array, "input_dim"], *, dropout_keys: jnp.ndarray, sampling_keys: jnp.ndarray
    ) -> tuple[
        tuple[Float[Array, "1"], Float[Array, "1"], Float[Array, "1"]],
        tuple[Float[Array, "1"], Float[Array, "1"], Float[Array, "1"]],
        tuple[Float[Array, "1"], Float[Array, "1"], Float[Array, "1"]],
    ]:
        k1, k2, k3, k4 = dropout_keys
        ka, kb, ks = sampling_keys

        # === Initial Layers ===
        x_hidden = self.initial_linear1(x)
        x_hidden = self.initial_norm1(x_hidden)
        x_hidden = jax.nn.swish(x_hidden)
        x_hidden = self.dropout1(x_hidden, key=k1)

        x_hidden = self.initial_linear2(x_hidden)
        x_hidden = self.initial_norm2(x_hidden)
        x_hidden = jax.nn.swish(x_hidden)
        x_hidden = self.dropout2(x_hidden, key=k2)

        # === Attention Block ===
        attention_scores = self.attention_layer(x_hidden)
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        x_attended = x * attention_weights

        x_combined = jnp.concatenate([x_hidden, x_attended], axis=-1)
        x_combined = self.post_attention_dropout(x_combined, key=k3)

        # === Final Layers ===
        x_final_enc = self.final_linear1(x_combined)
        x_final_enc = self.final_norm1(x_final_enc)
        x_final_enc = jax.nn.swish(x_final_enc)
        x_final_enc = self.dropout3(x_final_enc, key=k4)

        x_final_enc = self.final_linear2(x_final_enc)
        x_final_enc = jax.nn.swish(x_final_enc)

        return (
            self.get_sample(ka, self.alpha_layer(x_final_enc)),
            self.get_sample(kb, self.beta_layer(x_final_enc)),
            self.get_sample(ks, self.sigma_layer(x_final_enc)),
        )

    def get_sample(self, key, v):
        mu, logvar = jnp.split(v, 2, axis=-1)
        std = jnp.exp(0.5 * logvar)
        sample = jax.nn.sigmoid(mu + jax.random.normal(key, shape=mu.shape) * std)
        return mu, std, sample

    def get_prior_vars(self):
        return jnp.exp(jnp.concatenate([self.alpha_var, self.beta_var, self.sigma_var], axis=-1))


class Decoder(eqx.Module):
    layers: list

    input_dim: int
    latent_dim: int
    dec_hidden: int

    def __init__(self, key, input_dim: int, latent_dim: int, dec_hidden: int, dtype: jnp.dtype = jnp.float32):
        key1, key2 = jax.random.split(key, 2)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dec_hidden = dec_hidden
        self.layers = [
            eqx.nn.Linear(in_features=latent_dim, out_features=dec_hidden, key=key1, dtype=dtype),
            jax.nn.relu,
            eqx.nn.Linear(in_features=dec_hidden, out_features=input_dim, key=key2, dtype=dtype),
            jax.nn.softplus,
        ]

    def __call__(self, z: Float[Array, "batch latent_dim"]) -> Float[Array, "batch input_dim"]:
        z = z.reshape(
            self.latent_dim,
        )
        for layer in self.layers:
            z = layer(z)
        return z


# He Uniform Initialization for ReLU activation
def he_uniform_init(weight: Array, key: Array) -> Array:
    out, in_ = weight.shape
    stddev = jnp.sqrt(2 / in_)  # He init scale
    return jax.random.uniform(key, shape=(out, in_), minval=-stddev, maxval=stddev, dtype=weight.dtype)


# Bias initialization to target softplus output around 1
def softplus_bias_init(bias: Array, key: Array, dtype=jnp.float32) -> Array:
    return jnp.full(bias.shape, jnp.log(jnp.exp(1.0) - 1.0), dtype=bias.dtype)


def zero_bias_init(bias: Array, key: Array) -> Array:
    return jnp.zeros_like(bias, dtype=bias.dtype)


def apply_initialization(model, init_fn_weight, init_fn_bias, key):
    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)

    linear_weights = [x.weight for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear) if is_linear(x)]
    linear_biases = [
        x.bias for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear) if is_linear(x) and x.bias is not None
    ]

    num_weights = len(linear_weights)
    num_biases = len(linear_biases)

    key_weights, key_biases = jax.random.split(key, 2)

    new_weights = [
        init_fn_weight(w, subkey) for w, subkey in zip(linear_weights, jax.random.split(key_weights, num_weights))
    ]
    model = eqx.tree_at(
        lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)],
        model,
        new_weights,
    )

    new_biases = [init_fn_bias(b, subkey) for b, subkey in zip(linear_biases, jax.random.split(key_biases, num_biases))]
    model = eqx.tree_at(
        lambda m: [
            x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x) and x.bias is not None
        ],
        model,
        new_biases,
    )
    return model


def init_encoder_weights(encoder: Encoder, key: jnp.ndarray, dtype=jnp.float32):
    encoder = apply_initialization(encoder, he_uniform_init, zero_bias_init, key)

    key_alpha_b, key_beta_b, key_sigma_b, _ = jax.random.split(key, 4)
    encoder = eqx.tree_at(
        lambda e: e.alpha_layer.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.alpha_layer.bias), key_alpha_b, dtype=dtype),
    )
    encoder = eqx.tree_at(
        lambda e: e.beta_layer.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.beta_layer.bias), key_beta_b, dtype=dtype),
    )
    encoder = eqx.tree_at(
        lambda e: e.sigma_layer.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.sigma_layer.bias), key_sigma_b, dtype=dtype),
    )

    return encoder


def init_decoder_weights(decoder: Decoder, key: jnp.ndarray):
    decoder = apply_initialization(decoder, he_uniform_init, zero_bias_init, key)
    return decoder


def make_encoder(key, input_dim: int, latent_dim: int, enc_hidden: int, dropout_rate: float):
    return Encoder(key, input_dim, latent_dim, enc_hidden, dropout_rate=dropout_rate)


def make_decoder(key, input_dim: int, latent_dim: int, dec_hidden: int):
    return Decoder(key, input_dim, latent_dim, dec_hidden)
