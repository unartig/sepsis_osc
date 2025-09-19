import logging
from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped, PyTree
from numpy.typing import DTypeLike

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

    key_layer: eqx.nn.Linear
    query_layer: eqx.nn.Linear
    value_layer: eqx.nn.Linear
    post_attention_dropout: eqx.nn.Dropout

    final_linear1: eqx.nn.Linear
    final_norm1: eqx.nn.LayerNorm
    dropout3: eqx.nn.Dropout

    final_linear2: eqx.nn.Linear

    # Output heads
    alpha_layer: eqx.nn.Linear
    beta_layer: eqx.nn.Linear
    sigma_layer: eqx.nn.Linear

    h_layer: eqx.nn.Linear

    # Hyperparams
    input_dim: int
    latent_dim: int
    enc_hidden: int
    pred_hidden: int
    dropout_rate: float

    def __init__(
        self,
        key: Array,
        input_dim: int,
        latent_dim: int,
        enc_hidden: int,
        pred_hidden: int,
        dropout_rate: float = 0.2,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        (
            key_lin1,
            key_lin2,
            key_key,
            key_value,
            key_query,
            key_lin4,
            key_lin5,
            key_alpha,
            key_beta,
            key_sigma,
        ) = jax.random.split(key, 10)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_hidden = enc_hidden
        self.pred_hidden = pred_hidden
        self.dropout_rate = dropout_rate

        # Initial layers
        self.initial_linear1 = eqx.nn.Linear(input_dim, enc_hidden, key=key_lin1, dtype=dtype)
        self.initial_norm1 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)

        self.initial_linear2 = eqx.nn.Linear(enc_hidden, enc_hidden, key=key_lin2, dtype=dtype)
        self.initial_norm2 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

        # Attention
        self.key_layer = eqx.nn.Linear(enc_hidden, input_dim, key=key_key, dtype=dtype)
        self.query_layer = eqx.nn.Linear(input_dim, input_dim, key=key_query, dtype=dtype)
        self.value_layer = eqx.nn.Linear(enc_hidden, input_dim, key=key_value, dtype=dtype)
        self.post_attention_dropout = eqx.nn.Dropout(dropout_rate)

        # Final encoder layers
        self.final_linear1 = eqx.nn.Linear(enc_hidden + input_dim, 64, key=key_lin4, dtype=dtype)
        self.final_norm1 = eqx.nn.LayerNorm(64, dtype=dtype)
        self.dropout3 = eqx.nn.Dropout(dropout_rate)
        self.final_linear2 = eqx.nn.Linear(64, 32, key=key_lin5, dtype=dtype)

        # Output heads
        self.alpha_layer = eqx.nn.Linear(32, 1, key=key_alpha, dtype=dtype)
        self.beta_layer = eqx.nn.Linear(32, 1, key=key_beta, dtype=dtype)
        self.sigma_layer = eqx.nn.Linear(32, 1, key=key_sigma, dtype=dtype)
        self.h_layer = eqx.nn.Linear(32, pred_hidden, key=key_sigma, dtype=dtype)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, " input_dim"], *, dropout_keys: jnp.ndarray
    ) -> tuple[Float[Array, "1"], Float[Array, "1"], Float[Array, "1"], Float[Array, " pred_hidden"]]:
        k1, k2, k3, k4 = dropout_keys

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
        query = self.query_layer(x)
        key = self.key_layer(x_hidden)
        value = self.value_layer(x_hidden)

        attn_weights = jax.nn.softmax(query * key, axis=-1)
        x_attended = attn_weights * value

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
            self.alpha_layer(x_final_enc),
            self.beta_layer(x_final_enc),
            self.sigma_layer(x_final_enc),
            jax.nn.tanh(self.h_layer(x_final_enc)),
        )


class Decoder(eqx.Module):
    layers: list

    input_dim: int
    latent_dim: int
    dec_hidden: int

    def __init__(
        self, key: jnp.ndarray, input_dim: int, latent_dim: int, dec_hidden: int, dtype: jnp.dtype = jnp.float32
    ) -> None:
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dec_hidden = dec_hidden
        self.layers = [
            eqx.nn.Linear(in_features=latent_dim, out_features=16, key=key1, dtype=dtype),
            jax.nn.swish,
            eqx.nn.Linear(in_features=16, out_features=32, key=key2, dtype=dtype),
            jax.nn.swish,
            eqx.nn.Linear(in_features=32, out_features=dec_hidden, key=key3, dtype=dtype),
            jax.nn.swish,
            eqx.nn.Linear(in_features=dec_hidden, out_features=input_dim, key=key4, dtype=dtype),
        ]

    @jaxtyped(typechecker=typechecker)
    def __call__(self, z: Float[Array, " latent_dim"]) -> Float[Array, " input_dim"]:
        z = z.reshape(
            self.latent_dim,
        )
        for layer in self.layers:
            z = layer(z)
        return z * 5


EncDec = TypeVar("EncDec", Encoder, Decoder)
InitFn = Callable[[Array, Array], Array]


# He Uniform Initialization for ReLU activation
def he_uniform_init(weight: Array, key: Array) -> Array:
    out, in_ = weight.shape
    stddev = jnp.sqrt(2 / in_)  # He init scale
    return jax.random.uniform(key, shape=(out, in_), minval=-stddev, maxval=stddev, dtype=weight.dtype)


# Bias initialization to target softplus output around 1
def softplus_bias_init(bias: Array, _key: jnp.ndarray) -> Array:
    return jnp.full(bias.shape, jnp.log(jnp.exp(1.0) - 1.0), dtype=bias.dtype)


def zero_bias_init(bias: Array, _key: jnp.ndarray) -> Array:
    return jnp.zeros_like(bias, dtype=bias.dtype)


def apply_initialization(model: EncDec, init_fn_weight: InitFn, init_fn_bias: InitFn, key: jnp.ndarray) -> EncDec:
    def is_linear(x: PyTree) -> bool:
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
    return eqx.tree_at(
        lambda m: [
            x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x) and x.bias is not None
        ],
        model,
        new_biases,
    )


def init_encoder_weights(encoder: Encoder, key: jnp.ndarray) -> Encoder:
    encoder = apply_initialization(encoder, he_uniform_init, zero_bias_init, key)

    key_alpha_b, key_beta_b, key_sigma_b, _ = jax.random.split(key, 4)
    encoder = eqx.tree_at(
        lambda e: e.alpha_layer.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.alpha_layer.bias), key_alpha_b),
    )
    encoder = eqx.tree_at(
        lambda e: e.beta_layer.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.beta_layer.bias), key_beta_b),
    )
    return eqx.tree_at(
        lambda e: e.sigma_layer.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.sigma_layer.bias), key_sigma_b),
    )


def init_decoder_weights(decoder: Decoder, key: jnp.ndarray) -> Decoder:
    return apply_initialization(decoder, he_uniform_init, zero_bias_init, key)


def make_encoder(
    key: jnp.ndarray, input_dim: int, latent_dim: int, enc_hidden: int, pred_hidden: int, dropout_rate: float
) -> Encoder:
    return Encoder(key, input_dim, latent_dim, enc_hidden, pred_hidden=pred_hidden, dropout_rate=dropout_rate)


def make_decoder(key: jnp.ndarray, input_dim: int, latent_dim: int, dec_hidden: int) -> Decoder:
    return Decoder(key, input_dim, latent_dim, dec_hidden)
