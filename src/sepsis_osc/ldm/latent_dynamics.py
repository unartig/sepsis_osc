import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, ScalarLike, jaxtyped
from optimistix import BFGS, minimise

from sepsis_osc.ldm.ae import Decoder, Encoder
from sepsis_osc.ldm.commons import binary_logits
from sepsis_osc.ldm.gru import GRUPredictor
from sepsis_osc.utils.jax_config import typechecker

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

    _sofa_exp: Array
    _inf_exp: Array

    _sofa_lsigma: Array
    _inf_lsigma: Array
    _sep3_lsigma: Array

    _beta0: float
    _beta1: float
    _beta2: float
    _beta12: float

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

        self._sofa_exp = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_exp = jnp.zeros((1,), dtype=jnp.float32)

        self._sofa_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sep3_lsigma = jnp.zeros((1,), dtype=jnp.float32)

        self._beta0 = 1.0
        self._beta1 = 1.0
        self._beta2 = 1.0
        self._beta12 = 1.0

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

    @property
    def betas(self) -> Float[Array, "4"]:
        return jnp.array([self._beta0, self._beta1, self._beta2, self._beta12])

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_logits(
        self,
        p_sofa: Float[Array, " samples"] | Float[Array, " batch time"],
        p_inf: Float[Array, " samples"] | Float[Array, " batch time"],
        betas: Float[Array, "4"] | None = None,
    ) -> Float[Array, " samples"] | Float[Array, " batch time"]:
        if betas is None:
            betas = self.betas

        l_sofa = binary_logits(p_sofa)
        l_inf = binary_logits(p_inf)
        X = jax.lax.stop_gradient(
            jnp.stack(
                [
                    jnp.ones_like(l_sofa),  # intercept
                    l_sofa,  # sofa
                    l_inf,  # infection
                    l_sofa * l_inf,  # interaction
                ],
                axis=-1,
            )
        )
        return jnp.tensordot(X, betas, axes=[-1, 0])

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_probs(
        self,
        p_sofa: Float[Array, " samples"] | Float[Array, " batch time"],
        p_inf: Float[Array, " samples"] | Float[Array, " batch time"],
        betas: Float[Array, "4"] | None = None,
    ) -> Float[Array, " samples"] | Float[Array, " batch time"]:
        return jax.nn.sigmoid(self.get_sepsis_logits(p_sofa, p_inf, betas))

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


@jaxtyped(typechecker=typechecker)
def set_betas(model: LatentDynamicsModel, betas: Float[Array, "4"]) -> LatentDynamicsModel:
    return eqx.tree_at(
        lambda m: (m._beta0, m._beta1, m._beta2, m._beta12),  # noqa: SLF001
        model,
        replace=tuple(betas),
    )


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def update_betas(
    model: LatentDynamicsModel,
    p_sofa: Float[Array, " samples"],
    p_inf: Float[Array, " samples"],
    p_sepsis: Float[Array, " samples"],
) -> LatentDynamicsModel:
    def loss_fn(betas: Float[Array, "4"], args: tuple[Float[Array, " samples"], ...]) -> ScalarLike:
        p_sofa, p_inf, y = args
        logits = model.get_sepsis_logits(p_sofa, p_inf, betas)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32)))

    solver = BFGS(rtol=1e-8, atol=1e-8)

    # Run optimization
    res = minimise(
        fn=loss_fn,
        solver=solver,
        y0=jnp.zeros_like(model.betas),  # initial betas
        args=(p_sofa, p_inf, p_sepsis),
        throw=False,
    )
    return set_betas(model, res.value)
