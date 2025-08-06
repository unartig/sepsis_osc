import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


class GRUPredictor(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    proj_out: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    hidden_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, key, dim: int, hidden_dim: int, dropout_rate: float = 0.4, dtype=jnp.float32):
        key1, key2, key3 = jr.split(key, 3)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.gru_cell = eqx.nn.GRUCell(dim, hidden_dim, key=key1, dtype=dtype)
        self.proj_out = eqx.nn.Linear(hidden_dim, dim, key=key2, dtype=dtype)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        z_t: Float[Array, " latent_dim"],
        h_prev: Float[Array, " pred_hidden"],
        *,
        key,
    ) -> tuple[Float[Array, " latent_dim"], Float[Array, " pred_hidden"]]:
        h_next = self.gru_cell(z_t, h_prev)
        h_next_dropped = self.dropout(h_next, key=key)
        z_pred = jax.nn.tanh(self.proj_out(h_next_dropped)) * 0.01
        return z_pred, h_next_dropped


def make_predictor(key, dim, hidden_dim, dropout_rate):
    return GRUPredictor(key, dim, hidden_dim, dropout_rate)
