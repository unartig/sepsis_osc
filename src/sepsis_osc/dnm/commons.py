import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

from sepsis_osc.utils.jax_config import typechecker


@jaxtyped(typechecker=typechecker)
def mean_angle(angles: Float[Array, "* N"], axis: int=-1) -> Float[Array, "*"]:
    angles = jnp.asarray(angles)
    sin_vals = jnp.sin(angles)
    cos_vals = jnp.cos(angles)
    return jnp.arctan2(jnp.mean(sin_vals, axis=axis), jnp.mean(cos_vals, axis=axis))


@jaxtyped(typechecker=typechecker)
def diff_angle(a1: Float[Array, "* N"], a2: Float[Array, "* 1"]) -> Float[Array, "* N"]:
    return jnp.angle(jnp.exp(1j * (a1 - a2)))  # Wrap differences to [-pi, pi]


@jaxtyped(typechecker=typechecker)
def std_angle(angles: Float[Array, "* N"], axis: int=-1) -> Float[Array, "*"]:
    angles = jnp.asarray(angles)
    mean_ang = mean_angle(angles, axis=axis)
    angular_diff = diff_angle(angles, jnp.expand_dims(mean_ang, axis))
    return jnp.sqrt(jnp.mean(angular_diff**2, axis=axis))

@jaxtyped(typechecker=typechecker)
def phase_entropy(phis: Float[Array, "* N"], num_bins: int = 36) -> Float[Array, ""]:
    hist, bin_edges = jnp.histogram(phis, bins=num_bins, range=(0, 2 * jnp.pi), density=True)

    hist = jnp.clip(hist, 1e-10, 1)

    return -jnp.sum(hist * jnp.log(hist) * (bin_edges[1] - bin_edges[0]))  # bin width is constant


@jaxtyped(typechecker=typechecker)
def entropy(x: Float[Array, "* N"], num_bins: int = 100, lims: tuple[float, float] = (0, 2)) -> Float[Array, ""]:
    hist, bin_edges = jnp.histogram(x, bins=num_bins, range=lims, density=True)

    hist = jnp.clip(hist, 1e-10, 1)

    # Shannon entropy
    return -jnp.sum(hist * jnp.log(hist) * (bin_edges[1] - bin_edges[0]))  # bin width is constant
