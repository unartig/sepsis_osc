# ruff:  noqa: ARG002
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import Tsit5
from equinox import field, filter_jit
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray, jaxtyped

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DynamicNetworkModel, MetricBase
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

    metrics: MetricBase = field(static=True)
    indices_t: Float[Array, "db_size 2"] = field(static=True)
    relevant_metrics: Float[Array, " db_size"] = field(static=True)

    # Precomputations
    # norm for faster distance calculation
    i_norm: Float[Array, "db_size 2"] = field(static=True)

    grid_shape: Int[Array, "2"] = field(static=True)
    grid_origin: Float[Array, "2"] = field(static=True)
    grid_spacing: Float[Array, "2"] = field(static=True)
    indices_2d: Float[Array, "nb ns 2"] = field(static=True)
    relevant_metrics_2d: Float[Array, "nb ns"] = field(static=True)
    metrics_2d: MetricBase = field(static=True)

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
        voxel_idx = self.round_ste(rel_pos).astype(jnp.int32)

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

    # TODO to latent_lookup

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


@jaxtyped(typechecker=typechecker)
class IntegrationLookup(eqx.Module):
    _N: int = 200
    _T_max_base: int = 2000
    _T_step_base: int = 100

    _solver: Tsit5 = Tsit5()
    _dnm: DynamicNetworkModel = DynamicNetworkModel(full_save=False, steady_state_check=False, progress_bar=False)

    def _query(
        self,
        query_vectors: Float[Array, "batch latent"],
        key: PRNGKeyArray,
        temperature: float | Float[Array, "1"] = 0.0,
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        _, rand_key = jr.split(key)

        def integrate_single(carry, inputs):
            params, key = inputs
            run_conf = DNMConfig(
                N=self._N,
                C=0.2,
                alpha=-0.28,
                beta=params[0],
                sigma=params[1],
            )
            sol = self._dnm.integrate(
                config=run_conf,
                M=1,
                solver=self._solver,
                key=key,
                T_init=0.0,
                T_max=self._T_max_base,
                T_step=self._T_step_base,
                ts=jnp.arange(0.0, self._T_max_base, self._T_step_base),
            )
            # sol.ys.s_1 shape is (*t, ensemble=1, N) — take final timestep, squeeze ensemble
            return None, sol.ys.s_1[-1].mean()  # shape: (N,) or scalar depending on your reduction

        batch_size = query_vectors.shape[0]
        keys = jr.split(rand_key, batch_size)

        integrate_single_remat = jax.checkpoint(integrate_single)
        _, results = jax.lax.scan(integrate_single_remat, None, (query_vectors, keys))
        return results

    # all methods share identical behaviour.
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


@jaxtyped(typechecker=typechecker)
def get_linear(x: Float[Array, "nb ns"], y: Float[Array, "nb ns"], k: int | float) -> Callable:
    xmid = x.min() + (x.max() - x.min()) / 2

    @jaxtyped(typechecker=typechecker)
    def linear(xs: Float[Array, " batch"], ys: Float[Array, " batch"]) -> Float[Array, " batch"]:
        return jax.nn.sigmoid(k * (xs - xmid))

    return linear


@jaxtyped(typechecker=typechecker)
def get_radial(x: Float[Array, "nb ns"], y: Float[Array, "nb ns"], k: int | float) -> Callable:
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    @jaxtyped(typechecker=typechecker)
    def radial(xs: Float[Array, " batch"], ys: Float[Array, " batch"]) -> Float[Array, " batch"]:
        x_symm = (xs - xmin) / (xmax - xmin)
        y_symm = (ys - ymin) / (ymax - ymin)
        r = jnp.sqrt((x_symm - 0.5) ** 2 + (y_symm - 0.5) ** 2)
        return jax.nn.sigmoid(k * (0.2 - r))

    return radial


def bump(x):
    return x * jnp.exp(-x)


def get_approx() -> Callable:
    def approx(xs: Float[Array, " batch"], ys: Float[Array, " batch"]) -> Float[Array, " batch"]:
        return (jnp.sin((jnp.sin((jnp.sin(xs / (0.50825256 - bump(jnp.sin(jnp.sin((ys / 0.60103625) - (xs * 1.1470096))) + 0.4785785))) - 0.10480727)**4))**2 + (ys * 0.11145829)))  # fmt: off

    return approx
