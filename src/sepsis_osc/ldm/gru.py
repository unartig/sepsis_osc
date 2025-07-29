import equinox as eqx
from jaxtyping import Float, Array
import jax.random as jr


class GRUPredictor(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    proj_out: eqx.nn.Linear

    hidden_dim: int = eqx.static_field()

    def __init__(self, key, dim: int, hidden_dim: int):
        key1, key2 = jr.split(key)
        self.hidden_dim = hidden_dim

        self.gru_cell = eqx.nn.GRUCell(dim, hidden_dim, key=key1)
        self.proj_out = eqx.nn.Linear(hidden_dim, dim, key=key2)

    def __call__(
        self,
        z_t: Float[Array, "latent_dim"],
        h_prev: Float[Array, "hidden_dim"],
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "hidden"]]:
        h_next = self.gru_cell(z_t, h_prev)
        z_pred = self.proj_out(h_next)
        return z_pred, h_next

def make_predictor(key, dim, hidden_dim):
    return GRUPredictor(key, dim, hidden_dim)
