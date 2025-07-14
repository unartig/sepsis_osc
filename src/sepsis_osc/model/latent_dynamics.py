import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float

from sepsis_osc.model.ae import Decoder, Encoder
from sepsis_osc.model.transformer import TransformerForecaster


class LatentDynamicsModel(eqx.Module):
    encoder: Encoder
    forecaster: TransformerForecaster
    decoder: Decoder

    # Parameter
    label_temperature: Array
    lookup_temperature: Array
    ordinal_deltas: Array

    def __init__(self, encoder: Encoder, forecaster: TransformerForecaster, decoder: Decoder, sofa_dist: Float[Array, "24 1"]):
        self.encoder = encoder
        self.forecaster = forecaster
        self.decoder = decoder

        # Parameter
        self.label_temperature = jnp.log(jnp.ones((1,)) * 0.01)
        self.lookup_temperature = jnp.log(jnp.ones((1,)) * 0.5)
        self.ordinal_deltas = sofa_dist

    def get_parameters(self) -> tuple[Array, Array, Array]:
        ordinal_thresholds = jnp.cumsum(jax.nn.softmax(self.ordinal_deltas))  # monotonicity
        return (
            jnp.exp(self.lookup_temperature),
            jnp.exp(self.label_temperature),
            ordinal_thresholds,
        )
