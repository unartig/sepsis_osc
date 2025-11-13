from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from sepsis_osc.utils.jax_config import EPS, typechecker


@jaxtyped(typechecker=typechecker)
def binary_logits(probs: Float[Array, "*"]) -> Float[Array, "*"]:
    probs = jnp.clip(probs, EPS, 1 - EPS)
    return jnp.log(probs) - jnp.log1p(-probs)


def custom_warmup_cosine(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    steps_per_cycle: list[int],
    end_value: float,
    peak_decay: float = 0.5,
) -> Callable:
    cycle_boundaries = jnp.array([warmup_steps + sum(steps_per_cycle[:i]) for i in range(len(steps_per_cycle))])
    cycle_lengths = jnp.array(steps_per_cycle)
    num_cycles = len(steps_per_cycle)

    def schedule(step: Int[Array, ""]) -> Float[Array, ""]:
        step = jnp.asarray(step)

        # --- Warmup ---
        def in_warmup_fn(step: Int[Array, ""]) -> Float[Array, ""]:
            frac = step / jnp.maximum(warmup_steps, 1)
            return (init_value + frac * (peak_value - init_value)).astype(jnp.float32)

        # --- Cosine Decay Cycles ---
        def in_decay_fn(step: Int[Array, ""]) -> Float[Array, ""]:
            # Which cycle are we in?
            rel_step = step - warmup_steps
            cycle_idx = jnp.sum(rel_step >= cycle_boundaries - warmup_steps) - 1
            cycle_idx = jnp.clip(cycle_idx, 0, num_cycles - 1)

            cycle_start = cycle_boundaries[cycle_idx]
            cycle_len = cycle_lengths[cycle_idx]
            step_in_cycle = step - cycle_start

            cycle_frac = jnp.clip(step_in_cycle / jnp.maximum(cycle_len, 1), 0.0, 1.0)
            peak = peak_value * (peak_decay**cycle_idx)
            return end_value + 0.5 * (peak - end_value) * (1 + jnp.cos(jnp.pi * cycle_frac))

        # Select warmup vs decay
        return jax.lax.cond(step < warmup_steps, in_warmup_fn, in_decay_fn, step)

    return schedule


def sq_dist(a: Float[Array, "batch input"]) -> Float[Array, "batch batch"]:
    diff = a[:, None, :] - a[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def median_sigma_sq(distances: Float[Array, "batch batch"]) -> Float[Array, ""]:
    B = distances.shape[0]
    iu = jnp.triu_indices(B, k=1)
    return jnp.median(distances[iu])


def local_sigma_sq(distances: Float[Array, "batch batch"], k: int = 10) -> Float[Array, " batch batch"]:
    sorted_dists = jnp.sort(distances, axis=1)
    local_sigmas = sorted_dists[:, k]  # k-th nearest distance
    return (local_sigmas[:, None] + local_sigmas[None, :]) / 2.0


def approx_similarity_loss(
    inputs: Float[Array, "batch input_dim"],
    latents: Float[Array, "batch latent_dim"],
    weights: Float[Array, " input_dim"],
) -> Float[Array, ""]:
    B = inputs.shape[0]

    z_dists = sq_dist(latents)
    z_sim = jnp.exp(-z_dists / (local_sigma_sq(z_dists) + EPS))

    x_dists = sq_dist(inputs * weights[None, :])
    x_sim = jnp.exp(-x_dists / (local_sigma_sq(x_dists) + EPS))

    mask = ~jnp.eye(B, dtype=bool)
    return jnp.where(mask, (z_sim - x_sim) ** 2, 0).mean()


@jaxtyped(typechecker=typechecker)
def mahalanobis_similarity_loss(
    inputs: Float[Array, "batch input_dim"],
    latents: Float[Array, "batch latent_dim"],
    weights: Float[Array, " input_dim"],  # for compatibility with approx version
) -> Float[Array, ""]:
    B = inputs.shape[0]

    z_dists = sq_dist(latents)
    z_sim = jnp.exp(-z_dists / (local_sigma_sq(z_dists) + EPS))

    cov = jnp.cov(inputs, rowvar=False)

    L = jnp.linalg.inv(jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(cov.shape[0])))
    x_dists = sq_dist(jnp.linalg.solve(L, inputs.T).T)
    x_sim = jnp.exp(-x_dists / (local_sigma_sq(x_dists) + EPS))

    mask = ~jnp.eye(B, dtype=bool)
    return jnp.where(mask, (z_sim - x_sim) ** 2, 0).mean()


@jaxtyped(typechecker=typechecker)
def prob_increase(preds: Float[Array, " time"], diff: Float[Array, "1"], scale: Float[Array, "1"]) -> Float[Array, ""]:
    return 1.0 - jnp.prod(1.0 - jax.nn.sigmoid((jnp.diff(preds, axis=-1) - diff) * scale), axis=-1)
