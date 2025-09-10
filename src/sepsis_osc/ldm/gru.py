import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


class GRUPredictor(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    proj_out: eqx.nn.Linear

    hidden_dim: int = eqx.field(static=True)

    def __init__(self, key, dim: int, hidden_dim: int, dtype=jnp.float32):
        key1, key2, key3 = jr.split(key, 3)
        self.hidden_dim = hidden_dim

        self.gru_cell = eqx.nn.GRUCell(dim, hidden_dim, key=key1, dtype=dtype)
        self.proj_out = eqx.nn.Linear(hidden_dim, dim, key=key2, dtype=dtype)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        z_t: Float[Array, " latent_dim"],
        h_prev: Float[Array, " pred_hidden"],
    ) -> tuple[Float[Array, " latent_dim"], Float[Array, " pred_hidden"]]:
        h_next = self.gru_cell(z_t, h_prev)
        z_pred = self.proj_out(h_next)
        return z_pred, h_next


def make_predictor(key, dim, hidden_dim):
    return GRUPredictor(key, dim, hidden_dim)
