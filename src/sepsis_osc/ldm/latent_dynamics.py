import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from sepsis_osc.ldm.ae import Decoder, Encoder
from sepsis_osc.ldm.gru import GRUPredictor

ones_24 = jnp.ones((24))
ones_25 = jnp.ones((25))


class LatentDynamicsModel(eqx.Module):
    encoder: Encoder
    predictor: GRUPredictor
    decoder: Decoder

    # Parameter
    _sofa_dist: Array = eqx.field(static=True)

    _label_temperature: Array
    _lookup_temperature: Array
    _delta_temperature: Array
    _prior_deltas: Array = eqx.field(static=True)
    _learned_deltas: Array

    _sofa_exp: Array
    _inf_exp: Array

    _sofa_lsigma: Array
    _inf_lsigma: Array
    _sep3_lsigma: Array

    def __init__(
        self,
        encoder: Encoder,
        predictor: GRUPredictor,
        decoder: Decoder,
        ordinal_deltas: Float[Array, "25"] = ones_25,
        sofa_dist: Float[Array, "24"] = ones_24,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder

        # Parameter
        self._sofa_dist = sofa_dist

        self._label_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.05)
        self._lookup_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.5)
        self._delta_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.5)
        self._prior_deltas = ordinal_deltas
        self._learned_deltas = ordinal_deltas

        self._sofa_exp = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_exp = jnp.zeros((1,), dtype=jnp.float32)

        self._sofa_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sep3_lsigma = jnp.zeros((1,), dtype=jnp.float32)

    @property
    def learned_deltas(self) -> Float[Array, "25"]:
        return self._learned_deltas

    @property
    def sofa_dist(self) -> Float[Array, "24"]:
        return self._sofa_dist

    @property
    def label_temperature(self) -> Float[Array, "1"]:
        return jnp.exp(self._label_temperature)

    @property
    def lookup_temperature(self) -> Float[Array, "1"]:
        return jnp.exp(self._lookup_temperature)

    @property
    def delta_temperature(self) -> Float[Array, "1"]:
        return jnp.exp(self._delta_temperature)

    def ordinal_thresholds(self, lam: Float[Array, "1"]) -> Float[Array, "24"]:
        combined_deltas = lam * self._learned_deltas + (1.0 - lam) * self._prior_deltas

        return jnp.cumsum(jax.nn.softmax(combined_deltas))[:-1]

    @property
    def sofa_exp(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_exp)

    @property
    def inf_exp(self) -> Float[Array, "1"]:
        return jnp.exp(self._inf_exp)

    @property
    def sofa_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_lsigma)

    @property
    def inf_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._inf_lsigma)

    @property
    def sep3_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sep3_lsigma)

    def params_to_dict(self) -> dict["str", jnp.ndarray]:
        return {
            "label_temperature": self.label_temperature,
            "lookup_temperature": self.lookup_temperature,
            "delta_temperature": self.delta_temperature,
            "sofa_exp": self.sofa_exp,
            "inf_exp": self.inf_exp,
            "sofa_lsigma": self.sofa_lsigma,
            "inf_lsigma": self.inf_lsigma,
            "sep3_lsigma": self.sep3_lsigma,
        }
