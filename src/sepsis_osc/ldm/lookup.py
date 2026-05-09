# ruff:  noqa: ARG002
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray, jaxtyped

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics, MetricBase
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.jax_config import EPS, typechecker

_1 = jnp.ones((1,))


@runtime_checkable
class LookupProtocol(Protocol):
    """Common interface shared by all lookup implementations."""

    def hard_get(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"],
        kernel_size: int,
    ) -> Float[Array, " batch"]: ...

    def hard_get_fsq(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"],
        kernel_size: int,
    ) -> Float[Array, " batch"]: ...

    def soft_get_local(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"],
        kernel_size: int,
    ) -> Float[Array, " batch"]: ...

    def soft_get_global(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"],
        kernel_size: int,
    ) -> Float[Array, " batch"]: ...


@jaxtyped(typechecker=typechecker)
class LatentLookup(eqx.Module):
    """
    An Equinox module that performs differentiable and non-differentiable lookups into a precomputed metric database.
    """

    metrics: MetricBase
    indices_t: Float[Array, "db_size 2"]
    relevant_metrics: Float[Array, " db_size"]

    # Precomputations
    # norm for faster distance calculation
    i_norm: Float[Array, "db_size 2"]

    grid_shape: Int[Array, "2"]
    grid_origin: Float[Array, "2"]
    grid_spacing: Float[Array, "2"]
    indices_2d: Float[Array, "nb ns 2"]
    relevant_metrics_2d: Float[Array, "nb ns"]
    metrics_2d: MetricBase

    dtype: DTypeLike = jnp.float32

    def __init__(
        self,
        metrics: MetricBase,
        indices: Float[Array, "db_size 2"],
        metrics_2d: MetricBase,
        indices_2d: Float[Array, "nb ns 2"],
        grid_spacing: Float[Array, "2"],
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        object.__setattr__(self, "metrics", metrics.astype(dtype))
        object.__setattr__(self, "indices_t", indices.T.astype(dtype))
        relevant_metrics = self._extract_relevant(metrics)
        object.__setattr__(self, "relevant_metrics", relevant_metrics)

        i_norm = jnp.sum(indices**2, axis=-1, keepdims=True).T
        object.__setattr__(self, "i_norm", i_norm)

        nx, ny, _ = indices_2d.shape
        object.__setattr__(self, "grid_shape", jnp.array([nx, ny]))
        object.__setattr__(self, "grid_origin", indices_2d[0, 0].astype(dtype))
        object.__setattr__(self, "grid_spacing", grid_spacing.astype(dtype))
        object.__setattr__(self, "indices_2d", indices_2d.astype(dtype))
        object.__setattr__(self, "metrics_2d", metrics_2d.astype(dtype))
        relevant_metrics_3d = self._extract_relevant(metrics_2d)
        object.__setattr__(self, "relevant_metrics_2d", relevant_metrics_3d)

    @staticmethod
    def build(storage: Storage, alpha: float, beta_space: tuple[float, float, float], sigma_space: tuple[float, float, float]) -> "LatentLookup":
        b, s = as_2d_indices(beta_space, sigma_space)
        a = np.ones_like(b) * alpha

        indices_2d = jnp.concatenate([b[..., None], s[..., None]], axis=-1)
        spacing_2d = jnp.array([beta_space[2], sigma_space[2]])

        params = DNMConfig.batch_as_index(a, b, s, 0.2)
        metrics_2d, _ = storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)

        metrics_2d = metrics_2d.to_jax()

        return LatentLookup(
            metrics=metrics_2d.reshape((-1, 1)),
            indices=indices_2d.reshape((-1, 2)),
            metrics_2d=metrics_2d,
            indices_2d=indices_2d,
            grid_spacing=spacing_2d,
            dtype=jnp.float32,
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def _extract_relevant(_metrics: MetricBase) -> Float[Array, "nb ns"] | Float[Array, " nbxns"]:
        sofa_metric = _metrics.s_1

        return ((sofa_metric - sofa_metric.min()) / (sofa_metric.max() - sofa_metric.min())).squeeze(axis=-1)

    @jaxtyped(typechecker=typechecker)
    def hard_get(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"] = _1,  # placeholder to make compatible with soft get
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        """
        A non-differentiable lookup that retrieves the metric value of the absolute nearest neighbor in the database.
        Uses a global `jnp.argmin` distance calculation.
        """
        orig_dtype = query_vectors.dtype
        query_vectors = jax.lax.stop_gradient(query_vectors)
        query_vectors = query_vectors.astype(self.dtype)
        q_norm = jnp.sum(query_vectors**2, axis=-1, keepdims=True)
        dot_prod = query_vectors @ self.indices_t
        squared_distances = q_norm + self.i_norm - 2 * dot_prod

        min_indices = jnp.argmin(squared_distances, axis=-1)

        pred_c = self.relevant_metrics[min_indices]

        return pred_c.astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    def hard_get_fsq(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"] = _1,  # placeholder to make compatible with soft get
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        """
        An efficient grid-based lookup that retrieves metric values by rounding latent coordinates to the nearest discrete grid index.
        It uses a Straight-Through Estimator (STE) pattern for gradients.
        """
        rel_pos = (query_vectors - self.grid_origin) / self.grid_spacing
        center_idx = self.round_ste(rel_pos).astype(jnp.int32)

        @jaxtyped(typechecker=typechecker)
        def index_single(center_idx_row: Int[Array, " latent"]) -> Float[Array, ""]:
            return self.relevant_metrics_2d[
                center_idx_row[0],
                center_idx_row[1],
            ]

        return jax.vmap(index_single)(center_idx)

    @jaxtyped(typechecker=typechecker)
    def soft_get_local(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"],
        kernel_size: int = 15,
    ) -> Float[Array, " batch"]:
        """
        A differentiable lookup that computes a weighted average of metrics within a local (NxN) kernel around the query point.
        Weights are determined by a Softmax over negative squared Euclidean distances.
        """
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)
        temp = temperature.astype(self.dtype)

        # Convert query points into fractional voxel coordinates
        rel_pos = (q - self.grid_origin) / self.grid_spacing
        voxel_idx = jnp.round(rel_pos).astype(jnp.int32)

        radius = kernel_size // 2
        offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
        neighbor_offsets = jnp.stack(jnp.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)

        @jaxtyped(typechecker=typechecker)
        def gather_neighbors(vi: Int[Array, "2"], q_point: Float[Array, "2"]) -> Float[Array, ""]:
            coords = vi + neighbor_offsets

            x, y = coords[:, 0], coords[:, 1]

            neighbor_xy = self.indices_2d[x, y]
            neighbor_metrics = self.relevant_metrics_2d[x, y]

            dists = jnp.sum((q_point - neighbor_xy) ** 2, axis=-1)

            weights = jax.nn.softmax(-dists / (temp + EPS), axis=-1)
            return jnp.sum(weights * neighbor_metrics)

        return jax.vmap(gather_neighbors)(voxel_idx, q).astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    def soft_get_global(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: Float[Array, "1"],
        kernel_size: int | Int[Array, ""] = 3,  # placeholder to make compatible with local get
    ) -> Float[Array, " batch"]:
        """
        A differentiable lookup that computes a Softmax-weighted average across the entire metric database.
        It includes a thresholding operation to zero out negligible weights for numerical stability.
        """
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)

        q_norm = jnp.sum(q**2, axis=-1, keepdims=True)

        dot_prod = q @ self.indices_t
        dists = q_norm + self.i_norm - 2 * dot_prod

        weights = jax.nn.softmax(-dists / (temperature + EPS), axis=-1).squeeze()
        weights = weights * (1 - (weights < 0.001))

        return jnp.sum(weights * self.relevant_metrics, axis=-1).astype(orig_dtype)

    def round_ste(self, x: Float[Array, "batch 2"]) -> Float[Array, "batch 2"]:
        """
        Implements a Straight-Through Estimator for the rounding operation,
        allowing gradients to bypass the non-differentiable round function during backpropagation.
        """
        return x + jax.lax.stop_gradient(jnp.round(x) - x)


@jaxtyped(typechecker=typechecker)
class AnalyticalLookup(eqx.Module):
    _f: Callable

    def __init__(self, f: Callable) -> None:
        self._f = f

    @jaxtyped(typechecker=typechecker)
    def to_latent_lookup(
        self,
        beta_range: tuple[float, float, float],  # (start, stop, step)
        sigma_range: tuple[float, float, float],
        dtype: DTypeLike = jnp.float32,
    ) -> LatentLookup:
        """
        Discretizes the analytical function into a grid and returns a LatentLookup.
        """
        beta_grid, sigma_grid = as_2d_indices(beta_range, sigma_range)

        vals = self._f(jnp.asarray(beta_grid).ravel(), jnp.asarray(sigma_grid).ravel())

        _empty = jnp.empty_like(vals[..., None])
        metrics_flat = DNMMetrics(
            r_1=_empty,
            r_2=_empty,
            m_1=_empty,
            m_2=_empty,
            s_2=_empty,
            q_1=_empty,
            q_2=_empty,
            f_1=_empty,
            f_2=_empty,
            cq_1=_empty,
            cq_2=_empty,
            cs_1=_empty,
            cs_2=_empty,
            s_1=vals[..., None],
        )
        metrics_2d = metrics_flat.reshape((*beta_grid.shape, 1))

        indices_2d = jnp.stack([beta_grid, sigma_grid], axis=-1)
        indices_flat = indices_2d.reshape(-1, 2)
        grid_spacing = jnp.array([beta_range[2], sigma_range[2]])

        return LatentLookup(
            metrics=metrics_flat,
            indices=indices_flat,
            metrics_2d=metrics_2d,
            indices_2d=indices_2d,
            grid_spacing=grid_spacing,
            dtype=dtype,
        )

    @jaxtyped(typechecker=typechecker)
    def _query(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: float | Float[Array, "1"] = 0.0,
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        return self._f(query_vectors[:, 0], query_vectors[:, 1])

    # all methods share identical behaviour.
    hard_get = hard_get_fsq = soft_get_local = soft_get_global = _query


@jaxtyped(typechecker=typechecker)
class LearnedLookup(eqx.Module):
    _layers: list

    def __init__(self, key: PRNGKeyArray, latent_size: int = 2, hidden_dim: int = 32) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self._layers = [
            eqx.nn.Linear(latent_size, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=k2),
            eqx.nn.Linear(hidden_dim, 1, key=k3),
        ]

    def _query(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: float | Float[Array, "1"] = 0.0,
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        def forward(z: Float[Array, ""]) -> Float[Array, ""]:
            z = jax.nn.tanh(self._layers[0](z))
            z = jax.nn.tanh(self._layers[1](z))
            return jax.nn.sigmoid(self._layers[2](z))

        return jax.vmap(forward)(query_vectors).squeeze(-1)

    # all methods share identical behaviour.
    hard_get = hard_get_fsq = soft_get_local = soft_get_global = _query


class ODESurrogate(eqx.Module):
    layers: list

    def __init__(self, key: PRNGKeyArray, hidden: int = 64) -> None:
        keys = jr.split(key, 4)
        self.layers = [
            eqx.nn.Linear(2, hidden, key=keys[0]),
            eqx.nn.LayerNorm(hidden, hidden),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.LayerNorm(hidden, hidden),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.LayerNorm(hidden, hidden),
            eqx.nn.Linear(hidden, 1, key=keys[2]),
        ]

    def __call__(self, x: Float[Array, "2"]) -> Float[Array, "1"]:
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))

        return jax.nn.sigmoid(self.layers[-1](x))


@jaxtyped(typechecker=typechecker)
class SurrogateLookup(eqx.Module):
    _surrogate: ODESurrogate
    _beta_min: float
    _beta_max: float
    _sigma_min: float
    _sigma_max: float

    def __init__(
        self,
        betas: Float[Array, " b"],
        sigmas: Float[Array, " s"],
        target_space: Float[Array, "b s"],
        key: PRNGKeyArray,
        *,
        pretrained: ODESurrogate | None = None,
        hidden: int = 64,
        n_epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        self._beta_min = float(betas.min())
        self._beta_max = float(betas.max())
        self._sigma_min = float(sigmas.min())
        self._sigma_max = float(sigmas.max())

        if pretrained is not None:
            self._surrogate = pretrained
        else:
            _, subkey = jr.split(key)
            b2d, s2d = jnp.meshgrid(betas, sigmas, indexing="ij")
            params = jnp.stack([b2d.ravel(), s2d.ravel()], axis=-1)  # (b*s, 2)
            target_space = (target_space - target_space.min()) / (target_space.max() - target_space.min())

            targets = target_space.ravel()

            self._surrogate = self._train(subkey, params, targets, hidden, n_epochs, batch_size, lr)

    def _normalize(self, query_vectors: Float[Array, "batch 2"]) -> Float[Array, "batch 2"]:
        beta = (query_vectors[:, 0] - self._beta_min) / (self._beta_max - self._beta_min)
        sigma = (query_vectors[:, 1] - self._sigma_min) / (self._sigma_max - self._sigma_min)
        return jnp.stack([beta, sigma], axis=-1)

    def _train(
        self,
        key: PRNGKeyArray,
        params: Float[Array, "b*s 2"],
        targets: Float[Array, " b*s"],
        hidden: int,
        n_epochs: int,
        batch_size: int,
        lr: float,
    ) -> ODESurrogate:
        params_norm = self._normalize(params)

        key, subkey = jr.split(key)
        surrogate = ODESurrogate(subkey, hidden)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=100,
            decay_steps=int(n_epochs * 0.75),
            end_value=lr * 0.01,
        )
        optim = optax.radam(schedule)
        opt_state = optim.init(eqx.filter(surrogate, eqx.is_array))

        @eqx.filter_jit
        def step(
            model: ODESurrogate,
            opt_state: optax.OptState,
            p_batch: Float[Array, "sbatch 2"],
            t_batch: Float[Array, " sbatch"],
        ) -> tuple[ODESurrogate, optax.OptState, Float[Array, ""]]:
            def loss_fn(model: ODESurrogate) -> Float[Array, ""]:
                preds = jax.vmap(model)(p_batch).squeeze(-1)  # (batch,)
                return jnp.mean((preds - t_batch) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            jax.lax.cond(
                opt_state[0].count % 10000 == 0,  # ty:ignore[not-subscriptable, unresolved-attribute]
                lambda: jax.debug.print("Training ODESurrogate: Loss {x} Step {y}", x=loss, y=opt_state[0].count),  # ty:ignore[not-subscriptable, unresolved-attribute]
                lambda: None,
            )
            updates, new_opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            return eqx.apply_updates(model, updates), new_opt_state, loss

        _, subkey = jr.split(key)
        all_indices = jax.random.randint(subkey, (n_epochs, batch_size), 0, params_norm.shape[0])

        def scan_step(
            carry: tuple[ODESurrogate, optax.OptState], indices: Float[Array, ""]
        ) -> tuple[tuple[ODESurrogate, optax.OptState], Float[Array, ""]]:
            model, opt_state = carry
            p_batch = params_norm[indices]
            t_batch = targets[indices]
            model, opt_state, loss = step(model, opt_state, p_batch, t_batch)
            return (model, opt_state), loss

        (surrogate, _), losses = jax.lax.scan(scan_step, (surrogate, opt_state), all_indices)

        jax.debug.print("Final loss: {}", losses[-1])
        jax.debug.print("Min loss: {}", losses.min())
        return surrogate

    @eqx.filter_jit
    def _query(
        self,
        query_vectors: Float[Array, "batch 2"],
        key: PRNGKeyArray,
        temperature: float | Float[Array, "1"] = 0.0,
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        # freeze surrogate - no gradients through weights, only through inputs
        frozen = jax.lax.stop_gradient(eqx.filter(self._surrogate, eqx.is_array))
        surrogate = eqx.combine(frozen, eqx.filter(self._surrogate, eqx.is_inexact_array))

        normed = self._normalize(query_vectors)
        return jax.vmap(surrogate)(normed).squeeze(-1)  # (batch,)

    hard_get = hard_get_fsq = soft_get_local = soft_get_global = _query


def as_2d_indices(
    x_space: tuple[float, float, float],
    y_space: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D coordinate meshgrid from two space specifications,
    typically used to define the latent manifold for Beta and Sigma parameters.
    """
    xs = np.arange(*x_space)
    ys = np.arange(*y_space)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    return x_grid, y_grid


def get_aligned_subgrid(
    betas: np.ndarray | Array,
    sigmas: np.ndarray | Array,
    beta_space: tuple[float, float, float],
    sigma_space: tuple[float, float, float],
    window_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract subgrid around given parameter values, aligned with original grid.
    """
    betas_space = np.arange(*beta_space)
    sigmas_space = np.arange(*sigma_space)

    # Find indices of nearest grid points
    beta_idx = np.argmin(np.abs(betas_space[:, None] - betas[None, :]), axis=0)
    sigma_idx = np.argmin(np.abs(sigmas_space[:, None] - sigmas[None, :]), axis=0)

    # Get unique indices and extend window around them
    beta_idx_unique = np.unique(beta_idx)
    sigma_idx_unique = np.unique(sigma_idx)

    # Expand to window
    beta_min = max(0, beta_idx_unique.min() - window_size)
    beta_max = min(len(betas_space), beta_idx_unique.max() + window_size)
    sigma_min = max(0, sigma_idx_unique.min() - window_size)
    sigma_max = min(len(sigmas_space), sigma_idx_unique.max() + window_size)

    betas_subspace = betas_space[beta_min:beta_max]
    sigmas_subspace = sigmas_space[sigma_min:sigma_max]

    beta_grid, sigma_grid = np.meshgrid(betas_subspace, sigmas_subspace, indexing="ij")
    param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    return param_grid, betas_subspace, sigmas_subspace


def compute_window_bounds(
    betas: np.ndarray | Array,
    sigmas: np.ndarray | Array,
    betas_space: np.ndarray | Array,
    sigmas_space: np.ndarray | Array,
    window_size: int = 5,
) -> tuple[int, int, int, int]:
    betas_space_np = np.asarray(betas_space)
    sigmas_space_np = np.asarray(sigmas_space)

    betas, sigmas = betas.flatten(), sigmas.flatten()
    beta_idx = np.argmin(np.abs(betas_space_np[:, None] - betas[None, :]), axis=0)
    sigma_idx = np.argmin(np.abs(sigmas_space_np[:, None] - sigmas[None, :]), axis=0)

    beta_min = max(0, beta_idx.min() - window_size)
    beta_max = min(len(betas_space_np), beta_idx.max() + window_size)
    sigma_min = max(0, sigma_idx.min() - window_size)
    sigma_max = min(len(sigmas_space_np), sigma_idx.max() + window_size)

    return beta_min, beta_max, sigma_min, sigma_max


class RadialFn(eqx.Module):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    k: float

    def __call__(
        self,
        xs: Float[Array, " batch"],
        ys: Float[Array, " batch"],
    ) -> Float[Array, " batch"]:
        x_symm = (xs - self.xmin) / (self.xmax - self.xmin)
        y_symm = (ys - self.ymin) / (self.ymax - self.ymin)
        r = jnp.sqrt((x_symm - 0.5) ** 2 + (y_symm - 0.5) ** 2)
        return jax.nn.sigmoid(self.k * (0.2 - r))


@jaxtyped(typechecker=typechecker)
def get_radial(
    x: Float[Array, "nb ns"],
    y: Float[Array, "nb ns"],
    k: int | float,
) -> RadialFn:
    return RadialFn(
        xmin=float(x.min()),
        xmax=float(x.max()),
        ymin=float(y.min()),
        ymax=float(y.max()),
        k=float(k),
    )


def bump(x: Float[Array, " batch"]) -> Float[Array, " batch"]:
    return x * jnp.exp(-x)

class LinearFn(eqx.Module):
    xmid: float
    k: float

    def __call__(
        self,
        xs: Float[Array, " batch"],
        ys: Float[Array, " batch"],
    ) -> Float[Array, " batch"]:
        return jax.nn.sigmoid(self.k * (xs - self.xmid))


def get_linear(
    x: Float[Array, "nb ns"],
    y: Float[Array, "nb ns"],
    k: int | float,
) -> LinearFn:
    return LinearFn(
        xmid=float(x.min() + (x.max() - x.min()) / 2),
        k=float(k),
    )


class ApproxFn(eqx.Module):
    def __call__(
        self,
        xs: Float[Array, " batch"],
        ys: Float[Array, " batch"],
    ) -> Float[Array, " batch"]:
        return (jnp.sin((jnp.sin((jnp.sin(xs / (0.50825256 - bump(jnp.sin(jnp.sin((ys / 0.60103625) - (xs * 1.1470096))) + 0.4785785))) - 0.10480727)**4))**2 + (ys * 0.11145829)))  # fmt: off

def get_approx() -> ApproxFn:
    return ApproxFn()
