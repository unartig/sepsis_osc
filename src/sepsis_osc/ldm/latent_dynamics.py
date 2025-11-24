import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from sepsis_osc.ldm.ae import CoordinateEncoder, Decoder, InfectionPredictor
from sepsis_osc.ldm.gru import GRUPredictor

ones_24 = jnp.ones(24)
ones_25 = jnp.ones(25)


class LatentDynamicsModel(eqx.Module):
    encoder: CoordinateEncoder
    inf_predictor: InfectionPredictor
    predictor: GRUPredictor
    decoder: Decoder

    kernel_size: int

    # Parameter
    _sofa_dist: Array = eqx.field(static=True)

    _lookup_temperature: Array
    _prior_deltas: Array = eqx.field(static=True)

    _alpha: float = eqx.field(static=True)

    _d_diff: Array
    _d_scale: Array
    _input_sim_scale: Array

    _sofa_dir_lsigma: Array
    _sofa_d2_lsigma: Array
    _sofa_class_lsigma: Array
    _inf_lsigma: Array
    _sep3_lsigma: Array
    _velo_lsigma: Array
    _recon_lsigma: Array
    _seq_lsigma: Array
    _spread_lsigma: Array
    _tc_lsigma: Array
    _distr_lsigma: Array

    def __init__(
        self,
        encoder: CoordinateEncoder,
        inf_predictor: InfectionPredictor,
        predictor: GRUPredictor,
        decoder: Decoder,
        alpha: float,
        ordinal_deltas: Float[Array, "25"] = ones_25,
        sofa_dist: Float[Array, "24"] = ones_24,
        kernel_size: int = 5,
    ) -> None:
        self.encoder = encoder
        self.inf_predictor = inf_predictor
        self.predictor = predictor
        self.decoder = decoder

        self.kernel_size = kernel_size

        # Parameter
        self._alpha = alpha
        self._sofa_dist = sofa_dist

        self._lookup_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.05)
        self._prior_deltas = ordinal_deltas

        self._d_diff = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 1 / 25)
        self._d_scale = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 10)
        self._input_sim_scale = jnp.ones((encoder.input_dim,), dtype=jnp.float32)

        self._sofa_dir_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sofa_d2_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sofa_class_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sep3_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._velo_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._recon_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._seq_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._spread_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._tc_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._distr_lsigma = jnp.zeros((1,), dtype=jnp.float32)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def sofa_dist(self) -> Float[Array, "24"]:
        return self._sofa_dist

    @property
    def lookup_temperature(self) -> Float[Array, "3"]:
        return jnp.exp(self._lookup_temperature)

    @property
    def ordinal_thresholds(self) -> Float[Array, "24"]:
        combined_deltas = self._prior_deltas
        return jnp.cumsum(jax.nn.softmax(combined_deltas))[:-1]

    @property
    def input_sim_scale(self) -> Float[Array, " input_dim"]:
        return jax.nn.soft_sign(self._input_sim_scale)

    @property
    def d_diff(self) -> Float[Array, "1"]:
        return jnp.exp(self._d_diff)

    @property
    def d_scale(self) -> Float[Array, "1"]:
        return jnp.exp(self._d_scale)

    @property
    def sofa_dir_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_dir_lsigma)

    @property
    def sofa_class_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_class_lsigma)

    @property
    def sofa_d2_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_d2_lsigma)

    @property
    def velo_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._velo_lsigma)

    @property
    def seq_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._seq_lsigma)

    @property
    def spread_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._spread_lsigma)

    @property
    def tc_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._tc_lsigma)
    @property
    def distr_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._distr_lsigma)

    @property
    def recon_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._recon_lsigma)

    @property
    def inf_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._inf_lsigma)

    @property
    def sep3_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sep3_lsigma)

    def params_to_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "lookup_temperature": self.lookup_temperature,
            "d_diff": self.d_diff,
            "d_scale": self.d_scale,
            "sofa_lsigma": self.sofa_class_lsigma,
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
            "dec_hidden": self.decoder.dec_hidden,
        }

        hyper_pred = {
            "z_dim": self.decoder.z_latent_dim,
            "z_hidden_dim": self.predictor.z_hidden_dim,
        }
        return hyper_enc, hyper_dec, hyper_pred
