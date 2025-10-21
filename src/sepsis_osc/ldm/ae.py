import logging
from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, jaxtyped, PyTree
from numpy.typing import DTypeLike

from sepsis_osc.utils.jax_config import typechecker

logger = logging.getLogger(__name__)


# === Model Definitions ===
class Encoder(eqx.Module):
    # Layers
    dropout: eqx.nn.Dropout
    linear1: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm

    linear2: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm

    gate_g: eqx.nn.Linear
    gate_v: eqx.nn.Linear
    gate_q: eqx.nn.Linear
    norm3: eqx.nn.LayerNorm

    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    head_proj: eqx.nn.Linear

    # Output heads
    beta: eqx.nn.Linear
    sigma: eqx.nn.Linear

    vsofa: eqx.nn.Linear
    vinf: eqx.nn.Linear

    h: eqx.nn.Linear

    # Hyperparams
    input_dim: int
    enc_hidden: int
    pred_hidden: int
    dropout_rate: float

    def __init__(
        self,
        key: Array,
        input_dim: int,
        enc_hidden: int,
        pred_hidden: int,
        dropout_rate: float = 0.2,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        (
            key_lin1,
            key_lin2,
            key_gate,
            key_value,
            key_query,
            key_lin3,
            key_lin4,
            key_head,
            key_beta,
            key_sigma,
            key_vsofa,
            key_vinf,
            key_h,
        ) = jax.random.split(key, 13)

        self.input_dim = input_dim
        self.enc_hidden = enc_hidden
        self.pred_hidden = pred_hidden
        self.dropout_rate = dropout_rate

        # Initial layers
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.norm1 = eqx.nn.LayerNorm(input_dim, dtype=dtype)
        self.linear1 = eqx.nn.Linear(input_dim, enc_hidden, key=key_lin1, dtype=dtype)

        self.norm2 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)
        self.linear2 = eqx.nn.Linear(enc_hidden, enc_hidden, key=key_lin2, dtype=dtype)

        # Gating
        self.gate_g= eqx.nn.Linear(enc_hidden, enc_hidden, key=key_gate, dtype=dtype)
        self.gate_q= eqx.nn.Linear(input_dim, enc_hidden, key=key_query, dtype=dtype)
        self.gate_v= eqx.nn.Linear(enc_hidden, enc_hidden, key=key_value, dtype=dtype)
        self.norm3 = eqx.nn.LayerNorm(enc_hidden, dtype=dtype)

        # Final encoder layers
        self.linear3 = eqx.nn.Linear(enc_hidden, 128, key=key_lin3, dtype=dtype)
        self.linear4 = eqx.nn.Linear(128, 32, key=key_lin4, dtype=dtype)
        self.head_proj = eqx.nn.Linear(32, 32, key=key_head)

        # Output heads
        self.beta= eqx.nn.Linear(32, 1, key=key_beta, dtype=dtype)
        self.sigma= eqx.nn.Linear(32, 1, key=key_sigma, dtype=dtype)

        self.vsofa= eqx.nn.Linear(32, 1, key=key_vsofa, dtype=dtype)
        self.vinf= eqx.nn.Linear(32, 1, key=key_vinf, dtype=dtype)
        self.h= eqx.nn.Linear(32, pred_hidden, key=key_h, dtype=dtype)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    def make_keys(self, base_key: jnp.ndarray) -> jnp.ndarray:
        return jr.split(base_key, 2)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, " input_dim"], *, dropout_keys: jnp.ndarray
    ) -> tuple[
        Float[Array, "1"], Float[Array, "1"], Float[Array, "1"], Float[Array, "1"], Float[Array, " pred_hidden"]
    ]:
        k1, k2 = dropout_keys

        # === Initial Layers ===
        h = jax.nn.gelu(self.linear1(self.norm1(x)))
        h = jax.nn.gelu(self.linear2(self.norm2(h)))
        h = self.dropout(h, key=k1)

        # === Gating Block ===
        qx = self.gate_q(x)
        gate = jax.nn.sigmoid(self.gate_g(h) + qx)
        gated = gate * jax.nn.swish(self.gate_v(h)) + (1 - gate) * jax.nn.swish(qx)
        combined = self.norm3(h + gated)  # residual connection

        # === Final Layers ===
        enc = jax.nn.gelu(self.linear3(combined))
        enc = self.dropout(enc, key=k2)
        enc = jax.nn.gelu(self.linear4(enc))

        head= jax.nn.tanh(self.head_proj(enc))

        return (
            self.beta(head),
            self.sigma(head),
            self.vsofa(head),
            self.vinf(head),
            jax.nn.tanh(self.h(head)),
        )


class Decoder(eqx.Module):
    layers: list

    input_dim: int
    z_latent_dim: int
    v_latent_dim: int
    dec_hidden: int

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int,
        z_latent_dim: int,
        v_latent_dim: int,
        dec_hidden: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.input_dim = input_dim
        self.z_latent_dim = z_latent_dim
        self.v_latent_dim = v_latent_dim
        self.dec_hidden = dec_hidden
        self.layers = [
            eqx.nn.Linear(in_features=z_latent_dim + v_latent_dim, out_features=16, key=key1, dtype=dtype),
            jax.nn.swish,
            eqx.nn.Linear(in_features=16, out_features=32, key=key2, dtype=dtype),
            jax.nn.swish,
            eqx.nn.Linear(in_features=32, out_features=dec_hidden, key=key3, dtype=dtype),
            jax.nn.swish,
            eqx.nn.Linear(in_features=dec_hidden, out_features=input_dim, key=key4, dtype=dtype),
        ]

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @jaxtyped(typechecker=typechecker)
    def __call__(self, z: Float[Array, " latent_dim"]) -> Float[Array, " input_dim"]:
        z = z.reshape(
            self.z_latent_dim + self.v_latent_dim,
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

    key_beta_b, key_sigma_b, _ = jax.random.split(key, 3)
    encoder = eqx.tree_at(
        lambda e: e.beta.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.beta.bias), key_beta_b),
    )
    return eqx.tree_at(
        lambda e: e.sigma.bias,
        encoder,
        softplus_bias_init(jnp.asarray(encoder.sigma.bias), key_sigma_b),
    )


def init_decoder_weights(decoder: Decoder, key: jnp.ndarray) -> Decoder:
    return apply_initialization(decoder, he_uniform_init, zero_bias_init, key)


def make_encoder(key: jnp.ndarray, input_dim: int, enc_hidden: int, pred_hidden: int, dropout_rate: float) -> Encoder:
    return Encoder(key, input_dim, enc_hidden, pred_hidden, dropout_rate=dropout_rate)


def make_decoder(key: jnp.ndarray, input_dim: int, z_latent_dim: int, v_latent_dim: int, dec_hidden: int) -> Decoder:
    return Decoder(key, input_dim, z_latent_dim, v_latent_dim, dec_hidden)
