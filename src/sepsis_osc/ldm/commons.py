import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, jaxtyped

from sepsis_osc.utils.jax_config import EPS, typechecker


@jaxtyped(typechecker=typechecker)
def ordinal_logits(
    x: Float[Array, "*"], thresholds: Float[Array, " n_classes_minus_1"], temperature: Float[Array, "1"]
) -> Float[Array, "* n_classes_minus_1"]:
    return (x.squeeze()[:, None] - thresholds) / (temperature + EPS)


@jaxtyped(typechecker=typechecker)
def binary_logits(probs: Float[Array, "*"]) -> Float[Array, "*"]:
    probs = jnp.clip(probs, EPS, 1 - EPS)
    return jnp.log(probs) - jnp.log1p(-probs)


@jaxtyped(typechecker=typechecker)
def smooth_labels(
    sofa_true: Float[Array, "batch time"], threshold: float = 1.0, radius: int = 3, decay: float = 0.5
) -> Float[Array, "batch time-1"]:
    events = (jnp.diff(sofa_true, axis=-1) > threshold).astype(jnp.float32)

    offsets = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-decay * jnp.abs(offsets))
    kernel = kernel / kernel.max()

    def convolve_1d(x: Float[Array, " time-1"]) -> Float[Array, " time-1"]:
        return jnp.convolve(x, kernel, mode="same")[:x.shape[0]]

    smooth = vmap(convolve_1d)(events)

    return jnp.clip(smooth, 0.0, 1.0)
