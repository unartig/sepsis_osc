import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import field, filter_jit
from jaxtyping import Array, DTypeLike, Float, Int, jaxtyped

from sepsis_osc.dnm.dynamic_network_model import MetricBase
from sepsis_osc.utils.jax_config import EPS, typechecker

_1 = jnp.ones((1,))


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

    @filter_jit
    @jaxtyped(typechecker=typechecker)
    def hard_get(
        self,
        query_vectors: Float[Array, "batch latent"],
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

    @filter_jit
    @jaxtyped(typechecker=typechecker)
    def hard_get_fsq(
        self,
        query_vectors: Float[Array, "batch latent"],
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

    @filter_jit
    @jaxtyped(typechecker=typechecker)
    def soft_get_local(
        self,
        query_vectors: Float[Array, "batch latent"],
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

    @filter_jit
    @jaxtyped(typechecker=typechecker)
    def soft_get_global(
        self,
        query_vectors: Float[Array, "batch latent"],
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
