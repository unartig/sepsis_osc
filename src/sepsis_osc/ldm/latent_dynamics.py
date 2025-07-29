import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float

from sepsis_osc.ldm.ae import Decoder, Encoder
from sepsis_osc.ldm.gru import GRUPredictor


class LatentDynamicsModel(eqx.Module):
    encoder: Encoder
    predictor: GRUPredictor
    decoder: Decoder

    # Parameter
    label_temperature: Array
    lookup_temperature: Array
    ordinal_deltas: Array = eqx.static_field()

    sofa_dist: Array = eqx.static_field()

    def __init__(
        self,
        encoder: Encoder,
        predictor: GRUPredictor,
        decoder: Decoder,
        ordinal_deltas: Float[Array, "25"] = jnp.ones((25)),
        sofa_dist: Float[Array, "24"] = jnp.ones((24)),
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder

        # Parameter
        self.label_temperature = jnp.log(jnp.ones((1,)) * 0.05)
        self.lookup_temperature = jnp.log(jnp.ones((1,)) * 0.5)
        self.ordinal_deltas = ordinal_deltas

        self.sofa_dist = sofa_dist


    def get_parameters(self) -> tuple[Array, Array, Array]:
        ordinal_thresholds = jnp.cumsum(jax.nn.softmax(self.ordinal_deltas))[:-1]  # monotonicity
        return (
            jnp.exp(self.lookup_temperature),
            jnp.exp(self.label_temperature),
            ordinal_thresholds,
        )
