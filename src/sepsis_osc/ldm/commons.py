from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped

from sepsis_osc.utils.jax_config import EPS, typechecker


def he_uniform_init(weight: Array, key: Array) -> Array:
    # He Uniform Initialization for ReLU activation
    out, in_ = weight.shape
    stddev = jnp.sqrt(2 / in_)  # He init scale
    return jax.random.uniform(key, shape=(out, in_), minval=-stddev, maxval=stddev, dtype=weight.dtype)

def gru_bias_init(b: Array) -> Array:
    hidden_size = b.shape[0] // 3
    b = b.at[:].set(0.0)
    return b.at[hidden_size:2*hidden_size].set(1.0)  # update gate

def xavier_uniform(w: Array, key: Array) -> Array:
    fan_in, fan_out = w.shape[1], w.shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, w.shape, minval=-limit, maxval=limit)


def softplus_bias_init(bias: Array, _key: jnp.ndarray) -> Array:
    return jnp.full(bias.shape, jnp.log(jnp.exp(1.0) - 1.0), dtype=bias.dtype)


def zero_bias_init(bias: Array, _key: jnp.ndarray) -> Array:
    return jnp.zeros_like(bias, dtype=bias.dtype)


def qr_init(weight: Array, key: Array) -> Array:
    rows, cols = weight.shape
    A = jax.random.normal(key, (rows, cols), dtype=jnp.float32)

    if rows < cols:
        Q, _ = jnp.linalg.qr(A.T)
        Q = Q.T
    else:
        Q, _ = jnp.linalg.qr(A)
    Q = Q[:rows, :cols].astype(jnp.float32)
    return Q


def apply_initialization(model: PyTree, init_fn_weight: Callable, init_fn_bias: Callable, key: jnp.ndarray) -> PyTree:
    def is_linear(x: PyTree) -> bool:
        return isinstance(x, eqx.nn.Linear)

    linear_weights = [x.weight for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear) if is_linear(x)]
    linear_biases = [
        x.bias for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear) if is_linear(x) and x.bias is not None
    ]

    num_weights = len(linear_weights)
    num_biases = len(linear_biases)

    key_weights, key_biases = jax.random.split(key, 2)

    new_weights = [
        init_fn_weight(w, subkey) for w, subkey in zip(linear_weights, jax.random.split(key_weights, num_weights))
    ]
    model = eqx.tree_at(
        lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)],
        model,
        new_weights,
    )

    new_biases = [init_fn_bias(b, subkey) for b, subkey in zip(linear_biases, jax.random.split(key_biases, num_biases))]
    return eqx.tree_at(
        lambda m: [
            x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x) and x.bias is not None
        ],
        model,
        new_biases,
    )


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


@jaxtyped(typechecker=typechecker)
def prob_increase(
    preds: Float[Array, " time"], threshold: Float[Array, "1"], scale: Float[Array, "1"]
) -> Float[Array, ""]:
    return 1.0 - jnp.prod(1.0 - jax.nn.sigmoid((jnp.diff(preds, axis=-1) - threshold) * scale), axis=-1)


@jaxtyped(typechecker=typechecker)
def prob_increase_steps(
    preds: Float[Array, " time"], threshold: Float[Array, "1"], scale: Float[Array, "1"]
) -> Float[Array, " time"]:
    return jnp.concat([jnp.array([0.0]), jax.nn.sigmoid((jnp.diff(preds, axis=-1) - threshold) * scale)])


@jaxtyped(typechecker=typechecker)
def smooth_labels(
    labels: Float[Array, "batch time"] | Bool[Array, "batch time"],
    radius: int = 3,
    decay: float | Float[Array, "1"] = 0.5,
) -> Float[Array, "batch time"]:
    offsets = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-decay * jnp.abs(offsets))
    kernel = kernel / kernel.sum()

    def convolve_1d(x: Float[Array, " time"]) -> Float[Array, " time"]:
        return jnp.convolve(x, kernel, mode="same")

    smooth = vmap(convolve_1d)(labels)

    return jnp.clip(smooth, 0.0, 1.0)


@jaxtyped(typechecker=typechecker)
def causal_smoothing(
    labels: Float[Array, "batch time"], radius: int = 3, decay: float | Float[Array, "1"] = 0.5
) -> Float[Array, "batch time"]:
    offsets = jnp.arange(0, radius + 1)

    kernel = jnp.exp(-decay * offsets)
    kernel = kernel / kernel.sum()  # normalize

    def convolve_1d(x: Float[Array, " time"]) -> Float[Array, " time"]:
        return jnp.convolve(x, kernel, mode="full")

    smooth = vmap(convolve_1d)(labels)

    smooth = smooth[:, : labels.shape[1]]

    return jnp.clip(smooth, 0.0, 1.0)


def causal_probs(probs, window=6, eps=1e-6):
    padded = jnp.pad(probs, (window - 1, 0), constant_values=0.0)
    # log-space cumulative sum trick for rolling product
    log1m = jnp.log1p(-padded + eps)
    cumsum = jnp.cumsum(log1m)
    cumsum_shifted = jnp.pad(cumsum[:-window], (window, 0), constant_values=0.0)
    rolling_log = cumsum - cumsum_shifted
    return 1.0 - jnp.exp(rolling_log)[window - 1 :]
