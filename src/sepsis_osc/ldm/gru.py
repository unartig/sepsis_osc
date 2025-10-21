import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.utils.jax_config import typechecker


class GRUPredictor(eqx.Module):
    z_gru_cell: eqx.nn.GRUCell
    z_proj_out: eqx.nn.Linear
    v_gru_cell: eqx.nn.GRUCell
    v_proj_out: eqx.nn.Linear

    z_hidden_dim: int = eqx.field(static=True)
    v_hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        z_dim: int,
        v_dim: int,
        z_hidden_dim: int,
        v_hidden_dim: int,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        keyz, keyv = jr.split(key, 2)

        self.z_hidden_dim = z_hidden_dim
        self.v_hidden_dim = v_hidden_dim

        self.z_gru_cell = eqx.nn.GRUCell(z_dim, z_hidden_dim, key=keyz, dtype=dtype)
        self.v_gru_cell = eqx.nn.GRUCell(v_dim, v_hidden_dim, key=keyv, dtype=dtype)
        self.z_proj_out = eqx.nn.Linear(z_hidden_dim, z_dim, key=keyz, dtype=dtype)
        self.v_proj_out = eqx.nn.Linear(v_hidden_dim, v_dim, key=keyv, dtype=dtype)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        zv_t: Float[Array, " latent_dim"],
        h_prev: Float[Array, " pred_hidden"],
    ) -> tuple[Float[Array, " latent_dim"], Float[Array, " pred_hidden"]]:
        hz_prev, hv_prev = h_prev[: self.z_hidden_dim], h_prev[self.z_hidden_dim :]
        z_t, v_t = zv_t[: self.z_proj_out.out_features], zv_t[self.z_proj_out.out_features :]

        hz_next = self.z_gru_cell(z_t, hz_prev)
        z_pred = self.z_proj_out(hz_next)

        hv_next = self.v_gru_cell(v_t, hv_prev)
        v_pred = self.v_proj_out(hv_next)
        return jnp.concat([z_pred, v_pred]), jnp.concat([hz_next, hv_next])


def make_predictor(key: jnp.ndarray, z_dim: int, v_dim: int, z_hidden_dim: int, v_hidden_dim: int) -> GRUPredictor:
    return GRUPredictor(key, z_dim, v_dim, z_hidden_dim, v_hidden_dim)
