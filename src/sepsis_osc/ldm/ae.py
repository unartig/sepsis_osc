import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.utils.jax_config import typechecker

logger = logging.getLogger(__name__)


class LatentEncoder(eqx.Module):
    # Layers
    gating: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm
    linear2: eqx.nn.Linear
    norm3: eqx.nn.LayerNorm
    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    z: eqx.nn.Linear
    h: eqx.nn.Linear

    # Hyperparams
    input_dim: int
    latent_enc_hidden: int
    latent_pred_hidden: int

    def __init__(
        self,
        key: Array,
        input_dim: int,
        latent_enc_hidden: int,
        latent_pred_hidden: int,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        (key_linz, key_attnz, key_z, key_linh, key_h) = jax.random.split(key, 5)

        self.input_dim = input_dim
        self.latent_enc_hidden = latent_enc_hidden
        self.latent_pred_hidden = latent_pred_hidden

        # Gating
        self.gating = eqx.nn.Linear(input_dim // 2, input_dim // 2, key=key_attnz, dtype=dtype, use_bias=False)
        self.norm1 = eqx.nn.LayerNorm(input_dim, dtype=dtype)
        self.linear1 = eqx.nn.Linear(input_dim, latent_enc_hidden, key=key_linz, dtype=dtype)
        self.norm2 = eqx.nn.LayerNorm(latent_enc_hidden, dtype=dtype)
        self.linear2 = eqx.nn.Linear(latent_enc_hidden, latent_enc_hidden, key=key_linz, dtype=dtype)
        self.norm3 = eqx.nn.LayerNorm(latent_enc_hidden, dtype=dtype)
        self.linear3 = eqx.nn.Linear(latent_enc_hidden, latent_enc_hidden, key=key_linh, dtype=dtype)
        self.linear4 = eqx.nn.Linear(latent_enc_hidden, 16, key=key_linh, dtype=dtype)
        self.z = eqx.nn.Linear(16, 2, key=key_z, dtype=dtype)
        self.h = eqx.nn.Linear(16, latent_pred_hidden, key=key_h, dtype=dtype)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, " input_dim"]
    ) -> tuple[Float[Array, " pred_hidden_dim"], Float[Array, " output_dim"]]:
        assert x.shape[0] == self.input_dim
        assert self.input_dim % 2 == 0

        half = self.input_dim // 2
        x_val = x[:half]
        x_mask = x[half:]

        # gating only on real-valued features
        weights = jax.nn.sigmoid(self.gating(x_val))
        x_gate = x_val * weights

        x_gated = jnp.concat([x_gate, x_mask])

        hidden = x_gated

        hidden = self.linear1(jax.nn.gelu(self.norm1(hidden)))
        hidden = hidden + self.linear2(jax.nn.gelu(self.norm2(hidden)))
        hidden = hidden + self.linear3(jax.nn.gelu(self.norm3(hidden)))

        head = jax.nn.gelu(self.linear4(hidden))

        return self.h(head), self.z(head)


class Decoder(eqx.Module):
    layers: list

    input_dim: int
    z_latent_dim: int
    dec_hidden: int

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int,
        z_latent_dim: int,
        dec_hidden: int,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.input_dim = input_dim
        self.z_latent_dim = z_latent_dim
        self.dec_hidden = dec_hidden
        self.layers = [
            eqx.nn.Linear(in_features=z_latent_dim, out_features=16, key=key1, dtype=dtype),
            eqx.nn.LayerNorm(16),
            jax.nn.gelu,
            eqx.nn.Linear(in_features=16, out_features=32, key=key2, dtype=dtype),
            eqx.nn.LayerNorm(32),
            jax.nn.gelu,
            eqx.nn.Linear(in_features=32, out_features=dec_hidden, key=key3, dtype=dtype),
            eqx.nn.LayerNorm(dec_hidden),
            jax.nn.gelu,
            eqx.nn.Linear(in_features=dec_hidden, out_features=input_dim, key=key4, dtype=dtype),
        ]

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @jaxtyped(typechecker=typechecker)
    def __call__(self, z: Float[Array, " latent_dim"]) -> Float[Array, " input_dim"]:
        for layer in self.layers:
            z = layer(z)
        return jax.nn.tanh(z) * 5

