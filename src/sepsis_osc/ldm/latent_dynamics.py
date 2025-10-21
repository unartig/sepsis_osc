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

    _alpha: float = eqx.field(static=True)

    _d_diff: Array
    _d_scale: Array

    _inf_w: Array
    _inf_b: Array
    _inf_p: Array

    _sofa_w: Array
    _sofa_b: Array
    _sofa_p: Array

    _sofa_lsigma: Array
    _inf_lsigma: Array
    _sep3_lsigma: Array

    def __init__(
        self,
        encoder: Encoder,
        predictor: GRUPredictor,
        decoder: Decoder,
        alpha: float,
        ordinal_deltas: Float[Array, "25"] = ones_25,
        sofa_dist: Float[Array, "24"] = ones_24,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder

        # Parameter
        self._alpha = alpha
        self._sofa_dist = sofa_dist

        self._label_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.05)
        self._lookup_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.5)
        self._delta_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.5)
        self._prior_deltas = ordinal_deltas
        self._learned_deltas = ordinal_deltas

        self._d_diff = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 1 / 25)
        self._d_scale = jnp.log(jnp.ones((1,), dtype=jnp.float32))

        self._inf_w = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_b = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_p = jnp.zeros((1,), dtype=jnp.float32)
        self._sofa_w = jnp.ones((1,), dtype=jnp.float32)
        self._sofa_b = jnp.zeros((1,), dtype=jnp.float32)
        self._sofa_p = jnp.zeros((1,), dtype=jnp.float32)

        self._sofa_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sep3_lsigma = jnp.zeros((1,), dtype=jnp.float32)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @property
    def learned_deltas(self) -> Float[Array, "25"]:
        return self._learned_deltas

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def sofa_dist(self) -> Float[Array, "24"]:
        return self._sofa_dist

    @property
    def label_temperature(self) -> Float[Array, "1"]:
        return jnp.exp(self._label_temperature)

    @property
    def lookup_temperature(self) -> Float[Array, "3"]:
        return jnp.exp(self._lookup_temperatures)

    @property
    def delta_temperature(self) -> Float[Array, "1"]:
        return jnp.exp(self._delta_temperature)

    def ordinal_thresholds(self, lam: Float[Array, "1"]) -> Float[Array, "24"]:
        combined_deltas = lam * self._learned_deltas + (1.0 - lam) * self._prior_deltas

        return jnp.cumsum(jax.nn.softmax(combined_deltas))[:-1]

    def transform_infs(self, pred_infs: Float[Array, "*"]) -> Float[Array, "*"]:
        return jax.nn.sigmoid(((pred_infs + self._inf_b) ** jnp.exp(self._inf_p)) * self._inf_w)

    def transform_sofas(self, pred_sofas: Float[Array, "*"]) -> Float[Array, "*"]:
        return jax.nn.sigmoid((pred_sofas + self._sofa_b) * self._sofa_w)

    @property
    def d_diff(self) -> Float[Array, "1"]:
        return jnp.exp(self._d_diff)

    @property
    def d_scale(self) -> Float[Array, "1"]:
        return jnp.exp(self._d_scale)

    @property
    def sofa_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_lsigma)

    @property
    def inf_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._inf_lsigma)

    @property
    def sep3_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sep3_lsigma)

    def params_to_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "label_temperature": self.label_temperature,
            "lookup_temperature": self.lookup_temperature,
            "delta_temperature": self.delta_temperature,
            "d_diff": self.d_diff,
            "d_scale": self.d_scale,
            "sofa_lsigma": self.sofa_lsigma,
            "inf_lsigma": self.inf_lsigma,
            "sep3_lsigma": self.sep3_lsigma,
        }

    def hypers_dict(self) -> tuple[dict[str, int | float], dict[str, int], dict[str, int]]:
        hyper_enc = {
            "input_dim": self.encoder.input_dim,
            "enc_hidden": self.encoder.enc_hidden,
            "pred_hidden": self.encoder.pred_hidden,
            "dropout_rate": self.encoder.dropout_rate,
        }
        hyper_dec = {
            "input_dim": self.decoder.input_dim,
            "z_latent_dim": self.decoder.z_latent_dim,
            "v_latent_dim": self.decoder.v_latent_dim,
            "dec_hidden": self.decoder.dec_hidden,
        }

        hyper_pred = {
            "z_dim": self.decoder.z_latent_dim,
            "z_hidden_dim": self.predictor.z_hidden_dim,
            "v_dim": self.decoder.v_latent_dim,
            "v_hidden_dim": self.predictor.v_hidden_dim,
        }
        return hyper_enc, hyper_dec, hyper_pred
