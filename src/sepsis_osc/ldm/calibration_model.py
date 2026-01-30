from abc import abstractmethod

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
    """
    Base class for probability calibration methods.
    Subclasses implement different calibration strategies (Platt, Temperature, etc.)
    """

    @abstractmethod
    def transform_logits(
        self,
        logits: Float[Array, "*"],
    ) -> Float[Array, "*"]:
        """
        Apply calibration transformation to logits.
        Must be implemented by subclasses.
        """

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_logits(
        self,
        sofa: Float[Array, "*"],
        inf: Float[Array, "*"],
    ) -> Float[Array, "*"]:
        """
        Calculates the raw interaction term between SOFA and infection.
        """
        return binary_logits(sofa * inf)

    @jaxtyped(typechecker=typechecker)
    def get_sepsis_probs(
        self,
        p_sofa: Float[Array, "*"],
        p_inf: Float[Array, "*"],
    ) -> Float[Array, "*"]:
        """
        Computes calibrated sepsis probabilities.
        """
        logits = self.get_sepsis_logits(p_sofa, p_inf)
        calibrated_logits = self.transform_logits(logits)
        return jax.nn.sigmoid(calibrated_logits)

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def calibrate(
        self,
        p_sofa: Float[Array, " samples"],
        p_inf: Float[Array, " samples"],
        p_sepsis: Float[Array, " samples"],
    ) -> "CalibrationModel":
        """
        Optimizes calibration parameters using BFGS minimization.
        """
        solver = BFGS(rtol=1e-8, atol=1e-8)

        def loss_fn_calibration(parameters: CalibrationModel, args: tuple[Float[Array, " samples"], ...]) -> ScalarLike:
            p_sofa, p_inf, y = args
            # Use the parameter's transform, not self's
            logits = self.get_sepsis_logits(p_sofa, p_inf)
            calibrated_logits = parameters.transform_logits(logits)
            probs = jax.nn.sigmoid(calibrated_logits)

            return jnp.mean(optax.sigmoid_binary_cross_entropy(binary_logits(probs), y.astype(jnp.float32)))

        res = minimise(
            fn=loss_fn_calibration,
            solver=solver,
            y0=type(self)(),  # Initialize same type as self
            args=(p_sofa, p_inf, p_sepsis),
            throw=False,
            max_steps=int(1e3),
        )
        return res.value


class PlattScaling(CalibrationModel):
    """
    Platt scaling: sigmoid(a * logits + b)
    Learns both slope (a) and bias (b) parameters.
    """

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
    def coeffs(self) -> Float[Array, "2"]:
        """Returns [a, b] as a JAX array."""
        return jnp.array([self.a, self.b])

    def transform_logits(
        self,
        logits: Float[Array, "*"],
    ) -> Float[Array, "*"]:
        """Apply Platt scaling: a * logits + b"""
        return self.a * logits + self.b


class TemperatureScaling(CalibrationModel):
    """
    Temperature scaling: sigmoid(logits / T)
    Learns temperature parameter T (no bias term).
    """

    temperature: Float[Array, ""]

    def __init__(
        self,
        temperature: Float[Array, ""] = arr_1,
    ) -> None:
        self.temperature = temperature

    @property
    def coeffs(self) -> Float[Array, ""]:
        """Alias for temperature."""
        return self.temperature

    def transform_logits(
        self,
        logits: Float[Array, "*"],
    ) -> Float[Array, "*"]:
        """Apply temperature scaling: logits / T"""
        return logits / self.temperature

