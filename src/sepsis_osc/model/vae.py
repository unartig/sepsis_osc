import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.simulation.data_classes import JAXLookup, SystemMetrics

setup_jax(simulation=False)
setup_logging("info")
logger = logging.getLogger(__name__)


# === Model Definitions ===
LATENT_DIM = 3
INPUT_DIM = 52
ENC_HIDDEN = 32
DEC_HIDDEN = 32


class Encoder(eqx.Module):
    initial_layers: list
    attention_layer: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    final_layers: list

    alpha_layer: eqx.nn.Linear
    beta_layer: eqx.nn.Linear
    sigma_layer: eqx.nn.Linear

    sofa_layer: eqx.nn.Linear
    infection_layer: eqx.nn.Linear

    temperature_layer: eqx.nn.Linear

    input_dim: int
    latent_dim: int
    enc_hidden: int
    dropout_rate: float

    def __init__(
        self,
        key,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        enc_hidden: int = ENC_HIDDEN,
        dropout_rate: float = 0.5,
    ):
        key1, key2, key_attn_score, key3, key4, key_alpha, key_beta, key_sigma, key_temp, key_sofa, key_inf = (
            jax.random.split(key, 11)
        )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_hidden = enc_hidden
        self.dropout_rate = dropout_rate
        self.initial_layers = [
            eqx.nn.Linear(in_features=input_dim, out_features=enc_hidden, key=key1),
            eqx.nn.LayerNorm(enc_hidden),
            jax.nn.relu,
            eqx.nn.Linear(in_features=enc_hidden, out_features=enc_hidden, key=key2),
            eqx.nn.LayerNorm(enc_hidden),
            jax.nn.relu,
        ]

        self.attention_layer = eqx.nn.Linear(in_features=enc_hidden, out_features=input_dim, key=key_attn_score)
        self.dropout = eqx.nn.Dropout(self.dropout_rate)

        self.final_layers = [
            eqx.nn.Linear(in_features=enc_hidden + input_dim, out_features=enc_hidden, key=key3),
            eqx.nn.LayerNorm(enc_hidden),
            jax.nn.relu,
            eqx.nn.Linear(in_features=enc_hidden, out_features=latent_dim, key=key4),
        ]

        self.alpha_layer = eqx.nn.Linear(in_features=latent_dim, out_features=1, key=key_alpha)
        self.beta_layer = eqx.nn.Linear(in_features=latent_dim, out_features=1, key=key_beta)
        self.sigma_layer = eqx.nn.Linear(in_features=latent_dim, out_features=1, key=key_sigma)

        self.sofa_layer = eqx.nn.Linear(latent_dim, out_features=1, key=key_sofa)
        self.infection_layer = eqx.nn.Linear(latent_dim, out_features=1, key=key_inf)

        self.temperature_layer = eqx.nn.Linear(latent_dim, out_features=1, key=key_temp)

    def __call__(self, x: Float[Array, "input_dim"], *, key):
        x_original = x

        x_hidden = x
        for layer in self.initial_layers:
            x_hidden = layer(x_hidden)

        attention_scores = self.attention_layer(x_hidden)
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        x_attended = x_original * attention_weights

        x_combined = jnp.concatenate([x_hidden, x_attended], axis=-1)

        x_final_enc = self.dropout(x_combined, key=key)

        for layer in self.final_layers:
            x_final_enc = layer(x_final_enc)

        alpha_raw = jax.nn.sigmoid(self.alpha_layer(x_final_enc))
        beta_raw = jax.nn.sigmoid(self.beta_layer(x_final_enc))
        sigma_raw = jax.nn.sigmoid(self.sigma_layer(x_final_enc))

        sofa_raw = jax.nn.sigmoid(self.sofa_layer(x_final_enc))
        infection_raw = jax.nn.sigmoid(self.infection_layer(x_final_enc))

        temperature = (jax.nn.sigmoid(self.temperature_layer(x_final_enc)) + 1e-6) * 3

        return alpha_raw, beta_raw, sigma_raw, sofa_raw, infection_raw, temperature


class Decoder(eqx.Module):
    layers: list

    input_dim: int
    latent_dim: int
    dec_hidden: int

    def __init__(self, key, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM, dec_hidden: int = DEC_HIDDEN):
        key1, key2 = jax.random.split(key, 2)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dec_hidden = dec_hidden
        self.layers = [
            eqx.nn.Linear(in_features=latent_dim, out_features=dec_hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(in_features=dec_hidden, out_features=input_dim, key=key2),
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
def he_uniform_init(weight: jax.Array, key: jnp.ndarray) -> Array:
    out, in_ = weight.shape
    stddev = jnp.sqrt(2 / in_)  # He init scale
    return jax.random.uniform(key, shape=(out, in_), minval=-stddev, maxval=stddev)


# Bias initialization to target softplus output around 1
def softplus_bias_init(bias: jax.Array, key: jnp.ndarray) -> Array:
    return jnp.full(bias.shape, jnp.log(jnp.exp(1.0) - 1.0), dtype=bias.dtype)


def zero_bias_init(bias: jax.Array, key: jnp.ndarray) -> jax.Array:
    return jnp.zeros_like(bias)


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


def init_encoder_weights(encoder: Encoder, key: jnp.ndarray):
    encoder = apply_initialization(encoder, he_uniform_init, zero_bias_init, key)

    key_alpha_b, key_beta_b, key_sigma_b, _ = jax.random.split(key, 4)
    encoder = eqx.tree_at(
        lambda e: e.alpha_layer.bias, encoder, softplus_bias_init(jnp.asarray(encoder.alpha_layer.bias), key_alpha_b)
    )
    encoder = eqx.tree_at(
        lambda e: e.beta_layer.bias, encoder, softplus_bias_init(jnp.asarray(encoder.beta_layer.bias), key_beta_b)
    )
    encoder = eqx.tree_at(
        lambda e: e.sigma_layer.bias, encoder, softplus_bias_init(jnp.asarray(encoder.sigma_layer.bias), key_sigma_b)
    )

    return encoder


def init_decoder_weights(decoder: Decoder, key: jnp.ndarray):
    decoder = apply_initialization(decoder, he_uniform_init, zero_bias_init, key)
    return decoder


def make_encoder(key, input_dim: int, latent_dim: int, enc_hidden: int, dropout_rate: float):
    return Encoder(key, input_dim, latent_dim, enc_hidden, dropout_rate=dropout_rate)


def make_decoder(key, input_dim: int, latent_dim: int, dec_hidden: int):
    return Decoder(key, input_dim, latent_dim, dec_hidden)
