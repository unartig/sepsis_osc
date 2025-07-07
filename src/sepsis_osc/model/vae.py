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
    sofa_layer: eqx.nn.Linear
    infection_layer: eqx.nn.Linear

    # Parameter
    label_temperature: Array
    lookup_temperature: Array
    ordinal_deltas: Array
    z_scaling: Array

    # Loss Unvertainties
    sigma_recon: Array
    sigma_locality: Array
    sigma_concept: Array
    sigma_tc: Array

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
        sofa_dist: Array,
        dropout_rate: float = 0.3,
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
        self.alpha_layer = eqx.nn.Linear(latent_dim, 1, key=key_alpha, dtype=dtype)
        self.beta_layer = eqx.nn.Linear(latent_dim, 1, key=key_beta, dtype=dtype)
        self.sigma_layer = eqx.nn.Linear(latent_dim, 1, key=key_sigma, dtype=dtype)
        self.sofa_layer = eqx.nn.Linear(latent_dim, 1, key=key_sofa, dtype=dtype)
        self.infection_layer = eqx.nn.Linear(latent_dim, 1, key=key_inf, dtype=dtype)

        # Parameter
        self.label_temperature = jnp.log(jnp.ones((1,)) * 0.1)
        self.lookup_temperature = jnp.log(jnp.ones((1,)) * 0.5)

        self.sigma_recon = jnp.ones((1,)) * 0.1
        self.sigma_locality = jnp.ones((1,)) * 0.1
        self.sigma_concept = jnp.ones((1,)) * 0.1
        self.sigma_tc = jnp.ones((1,)) * 0.1

        self.ordinal_deltas = sofa_dist
        self.z_scaling = jnp.ones((3,))

    def __call__(self, x: Float[Array, "input_dim"], *, key):
        # Split keys for dropout layers
        k1, k2, k3, k4 = jax.random.split(key, 4)

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

        # === Outputs ===
        alpha_raw = jax.nn.sigmoid(self.alpha_layer(x_final_enc))
        beta_raw = jax.nn.sigmoid(self.beta_layer(x_final_enc))
        sigma_raw = jax.nn.sigmoid(self.sigma_layer(x_final_enc))

        sofa_raw = jax.nn.sigmoid(self.sofa_layer(x_final_enc))
        infection_raw = jax.nn.sigmoid(self.infection_layer(x_final_enc))

        return (
            alpha_raw,
            beta_raw,
            sigma_raw,
            sofa_raw,
            infection_raw,
        )

    def get_parameters(self) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        ordinal_thresholds = jnp.cumsum(jax.nn.softmax(self.ordinal_deltas))  # monotonicity
        z_scaling = jax.nn.softplus(self.z_scaling) + 0.1
        return (
            jnp.exp(self.lookup_temperature),
            jnp.exp(self.label_temperature),
            self.sigma_recon,
            self.sigma_locality,
            self.sigma_concept,
            self.sigma_tc,
            ordinal_thresholds,
            z_scaling,
        )


class Decoder(eqx.Module):
    layers: list

    input_dim: int
    latent_dim: int
    dec_hidden: int

    def __init__(self, key, input_dim: int, latent_dim: int, dec_hidden: int):
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
