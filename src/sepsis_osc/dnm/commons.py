import jax.numpy as jnp


def mean_angle(angles, axis=-1) -> jnp.ndarray:
    angles = jnp.asarray(angles)
    sin_vals = jnp.sin(angles)
    cos_vals = jnp.cos(angles)
    return jnp.arctan2(jnp.mean(sin_vals, axis=axis), jnp.mean(cos_vals, axis=axis))


def diff_angle(a1, a2) -> jnp.ndarray:
    return jnp.angle(jnp.exp(1j * (a1 - a2)))  # Wrap differences to [-pi, pi]


def std_angle(angles, axis=-1) -> jnp.ndarray:
    angles = jnp.asarray(angles)
    mean_ang = mean_angle(angles, axis=axis)
    angular_diff = diff_angle(angles, jnp.expand_dims(mean_ang, axis))
    return jnp.sqrt(jnp.mean(angular_diff**2, axis=axis))


def phase_entropy(phis: jnp.ndarray, num_bins: int = 36) -> jnp.ndarray:
    hist, bin_edges = jnp.histogram(phis, bins=num_bins, range=(0, 2 * jnp.pi), density=True)

    hist = jnp.clip(hist, 1e-10, 1)

    return -jnp.sum(hist * jnp.log(hist) * (bin_edges[1] - bin_edges[0]))  # bin width is constant


def entropy(x: jnp.ndarray, num_bins: int = 100, lims: tuple[float, float] = (0, 2)) -> jnp.ndarray:
    hist, bin_edges = jnp.histogram(x, bins=num_bins, range=lims, density=True)

    hist = jnp.clip(hist, 1e-10, 1)

    # Shannon entropy
    return -jnp.sum(hist * jnp.log(hist) * (bin_edges[1] - bin_edges[0]))  # bin width is constant
