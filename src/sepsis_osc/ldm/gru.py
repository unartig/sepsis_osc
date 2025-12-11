import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.utils.jax_config import typechecker


class GRUPredictor(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    z_proj_out: eqx.nn.Linear

    z_hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jnp.ndarray,
        z_dim: int,
        z_hidden_dim: int,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        keyz, _ = jr.split(key, 2)

        self.z_hidden_dim = z_hidden_dim

        self.gru_cell = eqx.nn.GRUCell(z_dim, z_hidden_dim, key=keyz, dtype=dtype)
        self.z_proj_out = eqx.nn.Linear(z_hidden_dim, z_dim, key=keyz, dtype=dtype, use_bias=False)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        z_t: Float[Array, " latent_dim"],
        h_prev: Float[Array, " pred_hidden"],
    ) -> tuple[Float[Array, " latent_dim"], Float[Array, " pred_hidden"]]:
        h_next = self.gru_cell(z_t, h_prev)
        z_pred = self.z_proj_out(h_next)
        return z_pred, h_next


def init_gru_weights(gru: GRUPredictor, key: jnp.ndarray) -> GRUPredictor:
    A = jax.random.normal(key, gru.gru_cell.weight_ih.shape, dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(A)
    gru = eqx.tree_at(lambda e: e.gru_cell.weight_ih, gru, Q)
    gru = eqx.tree_at(lambda e: e.gru_cell.bias, gru, gru.gru_cell.bias)
    return gru


def make_predictor(key: jnp.ndarray, z_dim: int, z_hidden_dim: int) -> GRUPredictor:
    return GRUPredictor(key, z_dim, z_hidden_dim)

