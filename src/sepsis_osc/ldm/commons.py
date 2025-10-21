from typing import Callable
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, jaxtyped

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
        return jnp.convolve(x, kernel, mode="same")[: x.shape[0]]

    smooth = vmap(convolve_1d)(events)

    return jnp.clip(smooth, 0.0, 1.0)


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


# EVENT PROBS


@jaxtyped(typechecker=typechecker)
def sofa_ps_from_logits(logits: Float[Array, "time n_classes_minus_1"]) -> Float[Array, "time n_classes"]:
    p_gt = jax.nn.sigmoid(logits)  # (T, K)
    p0 = 1.0 - p_gt[..., 0:1]  # P(S=0) shape (T,1)
    p_middle = p_gt[..., :-1] - p_gt[..., 1:]  # P(S=k) for 1..K-1  -> (T,K-1)
    plast = p_gt[..., -1:]  # P(S=K) shape (T,1)
    probs = jnp.concatenate([p0, p_middle, plast], axis=-1)  # (T,S)
    # numeric safety
    probs = jnp.clip(probs, EPS, 1.0 - EPS)
    return probs / probs.sum(axis=-1, keepdims=True)


@jaxtyped(typechecker=typechecker)
def sofa_event_prob_all_times(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    mask: Bool[Array, "time time"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, ""]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]  # (S, S)
    soft_I = jax.nn.sigmoid((diff - delta) / tau)  # (S, S)

    # All pairwise combinations of times
    p1 = sofa_probs[:, None, :, None]  # (T, 1, S, 1)
    p2 = sofa_probs[None, :, None, :]  # (1, T, 1, S)
    joint = p1 * p2  # (T, T, S, S)

    # Probability for each pair
    p_events = jnp.sum(joint * soft_I, axis=(2, 3)) * mask  # (T, T)

    # Soft OR across all pairs
    return 1.0 - jnp.prod(1.0 - p_events)


@jaxtyped(typechecker=typechecker)
def sofa_event_prob_consecutive_times(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, ""]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]  # (S, S)
    soft_I = jax.nn.sigmoid((diff - delta) / tau)  # (S, S)

    p1 = sofa_probs[:-1, :, None]
    p2 = sofa_probs[1:, None, :]
    joint = p1 * p2  # (T-1, S, S)

    # Probability for each pair
    p_events = jnp.sum(joint * soft_I, axis=(1, 2))

    # Soft OR across
    return 1.0 - jnp.prod(1.0 - p_events)


@jaxtyped(typechecker=typechecker)
def sofa_event_prob_any_future(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, " time"]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]
    soft_I = jax.nn.sigmoid((diff - delta) / tau)

    # all pairs
    p1 = sofa_probs[:, None, :, None]  # (T, 1, S, 1)
    p2 = sofa_probs[None, :, None, :]  # (1, T, 1, S)
    joint = p1 * p2  # (T, T, S, S)
    p_events = jnp.sum(joint * soft_I, axis=(2, 3))  # (T, T)

    return jnp.sum(jax.nn.softmax(jnp.tril(p_events, k=-1), axis=1), axis=-1)  # (T,)


@jaxtyped(typechecker=typechecker)
def sofa_increase_probs(
    predicted_sofa: Float[Array, " time"],
    thresholds: Float[Array, " n_classes_minus_1"],
    label_temperature: Float[Array, "1"],
    tau: Float[Array, "1"],
    delta: float = 2.0,
) -> Float[Array, " time-1"]:
    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)
    sofa_probs = sofa_ps_from_logits(logits)  # (T, S)
    _T, S = sofa_probs.shape
    support = jnp.arange(S, dtype=jnp.float32)
    diff = support[None, :] - support[:, None]  # (S, S)
    soft_I = jax.nn.sigmoid((diff - delta) / tau)  # (S, S)

    # joint (T-1, S, S)
    p1 = sofa_probs[:-1, :, None]
    p2 = sofa_probs[1:, None, :]
    joint = p1 * p2

    # probability of increase at each step
    return jnp.sum(joint * soft_I, axis=(1, 2))  # (T-1,)


@jaxtyped(typechecker=typechecker)
def organ_failure_increase_probs(
    sofa_frac: Float[Array, " time"],
    tau: Float[Array, "1"],
    delta: float = 0.1,
) -> Float[Array, " time-1"]:
    diffs = sofa_frac[1:] - sofa_frac[:-1]  # (T-1,)
    return jax.nn.sigmoid((diffs - delta) / tau)  # (T-1,)


def pairwise_probs(preds: Float[Array, " time"], scale: Float[Array, ""], tau: float = 0.04) -> Float[Array, " time-1"]:
    i_idx, j_idx = jnp.triu_indices(preds.shape[0], k=1)  # each has shape (M,)
    diffs = preds[j_idx] - preds[i_idx]  # (N, M)
    return jax.nn.relu((diffs - tau) / scale)

def soft_positive(diff):
    return jnp.clip(diff / (jnp.abs(diff) + EPS) * 0.5 + 0.5, 0.0, 1.0)

def soft_any(x):
    return 1.0 - jnp.exp(jnp.sum(jnp.log(1.0 - x + EPS), axis=-1))

@jaxtyped(typechecker=typechecker)
def prob_increase(
    preds: Float[Array, " time"], diff: Float[Array, "1"], scale: Float[Array, "1"]
) -> Float[Array, ""]:
    return 1.0 - jnp.prod(1.0 - jax.nn.sigmoid((jnp.diff(preds, axis=-1) - diff) * scale), axis=-1)

# def prob_increase(
#     preds: Float[Array, " time"], diff: Float[Array, "1"], scale: Float[Array, "1"]
#           ):
#     dpreds= jnp.diff(preds, axis=-1)
#     soft_increase = soft_positive(dpreds)
#     return soft_any(soft_increase)
# def prob_increase(
#     preds: Float[Array, " time"], diff: Float[Array, "1"], scale: Float[Array, "1"]
#           ):
#     return (jnp.diff(preds, axis=-1) > 0.0).any(axis=-1).astype(jnp.float32)
