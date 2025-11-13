import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, ScalarLike, jaxtyped
from optimistix import BFGS, minimise

from sepsis_osc.ldm.commons import binary_logits
from sepsis_osc.utils.jax_config import typechecker

arr_1 = jnp.array(1)
arr_0 = jnp.array(0)


class CalibrationModel(eqx.Module):
    # Interaction Model
    beta0: Float[Array, ""]
    beta1: Float[Array, ""]
    beta2: Float[Array, ""]
    beta12: Float[Array, ""]

    # Calibration
    a: Float[Array, ""]
    b: Float[Array, ""]

    def __init__(
        self,
        beta0: Float[Array, ""] = arr_0,
        beta1: Float[Array, ""] = arr_0,
        beta2: Float[Array, ""] = arr_0,
        beta12: Float[Array, ""] = arr_0,
        a: Float[Array, ""] = arr_1,
        b: Float[Array, ""] = arr_0,
    ) -> None:
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta12 = beta12
        self.a = a
        self.b = b

    @property
    def betas(self) -> Float[Array, ""]:
        return jnp.array([self.beta0, self.beta1, self.beta2, self.beta12])

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_logits(
        self,
        p_sofa: Float[Array, " samples"] | Float[Array, " batch time"],
        p_inf: Float[Array, " samples"] | Float[Array, " batch time"],
        betas: Float[Array, "4"] | None = None,
    ) -> Float[Array, " samples"] | Float[Array, " batch time"]:
        betas = betas if betas is not None else self.betas
        # l_sofa = binary_logits(p_sofa)
        # l_inf = binary_logits(p_inf)
        X = jnp.stack(
            [
                jnp.ones_like(p_sofa),  # intercept
                p_sofa,  # sofa
                p_inf,  # infection
                p_sofa * p_inf,  # interaction
            ],
            axis=-1,
        )
        return jnp.tensordot(X, betas, axes=[-1, 0])

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_probs(
        self,
        p_sofa: Float[Array, " samples"] | Float[Array, " batch time"],
        p_inf: Float[Array, " samples"] | Float[Array, " batch time"],
        betas: Float[Array, "4"] | None = None,
        a: Float[Array, ""] | None = None,
        b: Float[Array, ""] | None = None,
    ) -> Float[Array, " samples"] | Float[Array, " batch time"]:
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        return jax.nn.sigmoid(a * self.get_sepsis_logits(p_sofa, p_inf, betas) + b)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def calibrate(
        self,
        p_sofa: Float[Array, " samples"],
        p_inf: Float[Array, " samples"],
        p_sepsis: Float[Array, " samples"],
    ) -> "CalibrationModel":
        solver = BFGS(rtol=1e-8, atol=1e-8)

        def loss_fn_interaction(parameters: CalibrationModel, args: tuple[Float[Array, " samples"], ...]) -> ScalarLike:
            p_sofa, p_inf, y = args
            logits = self.get_sepsis_logits(p_sofa, p_inf, parameters.betas)
            return jnp.mean(optax.sigmoid_focal_loss(logits, y.astype(jnp.float32), alpha=0.99, gamma=1.0))

        res = minimise(
            fn=loss_fn_interaction,
            solver=solver,
            y0=CalibrationModel(),  # initial parameter guess
            args=(p_sofa, p_inf, p_sepsis),
            throw=False,
            max_steps=int(5e3),
        )

        def loss_fn_calibration(parameters: CalibrationModel, args: tuple[Float[Array, " samples"], ...]) -> ScalarLike:
            betas, p_sofa, p_inf, y = args
            probs = self.get_sepsis_probs(p_sofa, p_inf, betas, parameters.a, parameters.b)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(binary_logits(probs), y.astype(jnp.float32)))

        res = minimise(
            fn=loss_fn_calibration,
            solver=solver,
            y0=res.value,
            args=(res.value.betas, p_sofa, p_inf, p_sepsis),
            throw=False,
            max_steps=int(1e3),
        )

        return res.value
