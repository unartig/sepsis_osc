from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype import beartype as typechecker
from equinox import field, filter_jit
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.dnm.abstract_ode import MetricBase

@jaxtyped(typechecker=typechecker)
@dataclass(frozen=True)
class LatentLookup:
    metrics: MetricBase = field(static=True)
    indices_T: Float[Array, "db_size 3"] = field(static=True)
    relevant_metrics: Float[Array, "db_size 2"] = field(static=True)

    # Precomputations
    # norm for faster distance calculation
    i_norm: Float[Array, "db_size 3"] = field(static=True)

    grid_shape: tuple[int, int, int] = field(static=True)
    grid_origin: Float[Array, "3"] = field(static=True)
    grid_spacing: Float[Array, "3"] = field(static=True)
    indices_3d: Float[Array, "na nb ns 3"] = field(static=True)
    relevant_metrics_3d: Float[Array, "na nb ns 2"] = field(static=True)
    metrics_3d: MetricBase = field(static=True)

    dtype: DTypeLike = jnp.float32

    def __init__(
        self,
        metrics: MetricBase,
        indices: Float[Array, "db_size 3"],
        metrics_3d,
        indices_3d,
        grid_spacing,
        dtype: DTypeLike = jnp.float32,
    ):
        object.__setattr__(self, "metrics", metrics.astype(dtype))
        object.__setattr__(self, "indices_T", indices.T.astype(dtype))
        relevant_metrics = self.extract_relevant(metrics)
        object.__setattr__(self, "relevant_metrics", relevant_metrics)

        i_norm = jnp.sum(indices**2, axis=-1, keepdims=True).T
        object.__setattr__(self, "i_norm", i_norm)

        nx, ny, nz = indices_3d.shape[:-1]
        object.__setattr__(self, "grid_shape", jnp.array([nx, ny, nz]))
        object.__setattr__(self, "grid_origin", indices_3d[0, 0, 0].astype(dtype))
        object.__setattr__(self, "grid_spacing", grid_spacing.astype(dtype))
        object.__setattr__(self, "indices_3d", indices_3d.astype(dtype))
        object.__setattr__(self, "metrics_3d", metrics_3d.astype(dtype))
        relevant_metrics_3d = self.extract_relevant(metrics_3d)
        object.__setattr__(self, "relevant_metrics_3d", relevant_metrics_3d)

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def extract_relevant(metrics_: MetricBase) -> Float[Array, "na nb ns 2"] | Float[Array, "naxnbxns 2"]:
        sofa_metric = metrics_.s_1
        stacked = jnp.concatenate(
            [
                (sofa_metric - sofa_metric.min()) / (sofa_metric.max() - sofa_metric.min()) + 1e-12,
                # jnp.clip(metrics_.sr_2 + metrics_.f_2, 0, 1),
                jnp.zeros_like(sofa_metric, dtype=sofa_metric.dtype),
            ],
            axis=-1,
        )
        return stacked

    def tree_flatten(self):
        return (self.metrics, self.indices_T.T, self.metrics_3d, self.indices_3d, self.grid_spacing), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def hard_get(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperatures,  # placeholder to make compatible with soft get
    ) -> Float[Array, "batch 2"]:
        orig_dtype = query_vectors.dtype
        query_vectors = jax.lax.stop_gradient(query_vectors)
        query_vectors = query_vectors.astype(self.dtype)
        q_norm = jnp.sum(query_vectors**2, axis=-1, keepdims=True)
        dot_prod = query_vectors @ self.indices_T
        squared_distances = q_norm + self.i_norm - 2 * dot_prod

        min_indices = jnp.argmin(squared_distances, axis=-1)

        pred_c = self.relevant_metrics[min_indices]

        return pred_c.astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def hard_get_local(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperatures,  # placeholder to make compatible with soft get
    ) -> Float[Array, "batch 2"]:
        orig_dtype = query_vectors.dtype
        query_vectors = query_vectors.astype(self.dtype)
        rel_pos = (query_vectors - self.grid_origin) / self.grid_spacing
        center_idx = jnp.round(rel_pos).astype(jnp.int32)

        offsets = jnp.array([-1, 0, 1])
        neighbor_offsets = jnp.stack(jnp.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)
        neighbor_coords = center_idx[:, None, :] + neighbor_offsets[None, :, :]
        neighbor_coords = jnp.clip(neighbor_coords, 0, jnp.array(self.grid_shape) - 1)
        x, y, z = neighbor_coords[..., 0], neighbor_coords[..., 1], neighbor_coords[..., 2]
        neighbor_xyz = self.indices_3d[x, y, z]
        neighbor_metrics = self.relevant_metrics_3d[x, y, z]
        dists = jnp.sum((query_vectors[:, None, :] - neighbor_xyz) ** 2, axis=-1)
        best_idx = jnp.argmin(dists, axis=-1)
        pred_c = jnp.take_along_axis(neighbor_metrics, best_idx[:, None, None], axis=1).squeeze(1)
        return pred_c.astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def soft_get_local(
        self,
        query_vectors: Float[Array, "batch 3"],
        temperatures: Float[Array, "1"],
    ) -> Float[Array, "batch 2"]:
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)
        temp = temperatures.astype(self.dtype)

        # Convert query points into fractional voxel coordinates
        rel_pos = (q - self.grid_origin) / self.grid_spacing
        voxel_idx = self.round_ste(rel_pos).astype(jnp.int32)
        voxel_idx = jnp.clip(voxel_idx, 1, self.grid_shape - 2)  # orig -2

        offsets = jnp.array([-1, 0, 1])  # orig 1
        neighbor_offsets = jnp.stack(jnp.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)

        def gather_neighbors(vi, q_point):
            coords = vi[None, :] + neighbor_offsets
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            neighbor_xyz = self.indices_3d[x, y, z]
            neighbor_metrics = self.relevant_metrics_3d[x, y, z]

            # Compute distances to neighbors
            dists = jnp.sum((q_point - neighbor_xyz) ** 2, axis=-1)

            weights = jax.nn.softmax(-dists / temp, axis=-1)
            weighted = jnp.sum(weights[:, None] * neighbor_metrics, axis=0)

            return weighted

        pred_c = jax.vmap(gather_neighbors)(voxel_idx, q)
        return pred_c.astype(orig_dtype)

    def round_ste(self, x):
        return x + jax.lax.stop_gradient(jnp.round(x) - x)


jtu.register_pytree_node(LatentLookup, LatentLookup.tree_flatten, LatentLookup.tree_unflatten)
