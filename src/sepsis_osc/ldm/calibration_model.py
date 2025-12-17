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

    # Calibration
    a: Float[Array, ""]
    b: Float[Array, ""]

    def __init__(
        self,
        a: Float[Array, ""] = arr_1,
        b: Float[Array, ""] = arr_0,
    ) -> None:
        self.a = a
        self.b = b

    @property
    def coeffs(self) -> Float[Array, ""]:
        return jnp.array([self.a, self.b])

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_logits(
        self,
        sofa: Float[Array, " samples"] | Float[Array, " batch time"] | Float[Array, "*"],
        inf: Float[Array, " samples"] | Float[Array, " batch time"] | Float[Array, "*"],
    ) -> Float[Array, " samples"] | Float[Array, " batch time"] | Float[Array, "*"]:
        return sofa * inf  # interaction

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_probs(
        self,
        p_sofa: Float[Array, " samples"] | Float[Array, " batch time"] | Float[Array, "*"],
        p_inf: Float[Array, " samples"] | Float[Array, " batch time"] | Float[Array, "*"],
        a: Float[Array, ""] | None = None,
        b: Float[Array, ""] | None = None,
    ) -> Float[Array, " samples"] | Float[Array, " batch time"] | Float[Array, "*"]:
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        return jax.nn.sigmoid(a * self.get_sepsis_logits(p_sofa, p_inf) + b)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def calibrate(
        self,
        p_sofa: Float[Array, " samples"],
        p_inf: Float[Array, " samples"],
        p_sepsis: Float[Array, " samples"],
    ) -> "CalibrationModel":
        solver = BFGS(rtol=1e-8, atol=1e-8)

        def loss_fn_calibration(parameters: CalibrationModel, args: tuple[Float[Array, " samples"], ...]) -> ScalarLike:
            p_sofa, p_inf, y = args
            probs = self.get_sepsis_probs(p_sofa, p_inf, parameters.a, parameters.b)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(binary_logits(probs), y.astype(jnp.float32)))

        res = minimise(
            fn=loss_fn_calibration,
            solver=solver,
            y0=self,
            args=(p_sofa, p_inf, p_sepsis),
            throw=False,
            max_steps=int(1e3),
        )

        return res.value
