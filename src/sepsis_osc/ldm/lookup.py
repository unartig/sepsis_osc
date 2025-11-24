import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import field, filter_jit
from jaxtyping import Array, Float, Int, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.dnm.dynamic_network_model import MetricBase
from sepsis_osc.utils.jax_config import EPS, typechecker


@jaxtyped(typechecker=typechecker)
class LatentLookup(eqx.Module):
    metrics: MetricBase = field(static=True)
    indices_T: Float[Array, "db_size 3"] = field(static=True)
    relevant_metrics: Float[Array, " db_size"] = field(static=True)

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
        metrics_3d: MetricBase,
        indices_3d: Float[Array, "na nb ns 3"],
        grid_spacing: Float[Array, "3"],
        dtype: DTypeLike = jnp.float32,
    ) -> None:
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
    def extract_relevant(_metrics: MetricBase) -> Float[Array, "na nb ns"] | Float[Array, " naxnbxns"]:
        sofa_metric = _metrics.s_1

        return ((sofa_metric - sofa_metric.min()) / (sofa_metric.max() - sofa_metric.min())).squeeze(axis=-1)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def hard_get(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperature: Float[Array, "1"],  # placeholder to make compatible with soft get
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
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
    def hard_get_fsq(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperature: Float[Array, "1"] | Float[Array, "batch latent"],  # placeholder to make compatible with soft get
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        rel_pos = (query_vectors - self.grid_origin) / self.grid_spacing
        center_idx = self.round_ste(rel_pos).astype(jnp.int32)

        @jaxtyped(typechecker=typechecker)
        def index_single(center_idx_row: Int[Array, " latent"]) -> Float[Array, ""]:
            return self.relevant_metrics_3d[
                center_idx_row[0],
                center_idx_row[1],
                center_idx_row[2],
            ]

        return jax.vmap(index_single)(center_idx)


    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def soft_get_local(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperature: Float[Array, "1"],
        kernel_size: int = 15,
    ) -> Float[Array, " batch"]:
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)
        temp = temperature.astype(self.dtype)

        # Convert query points into fractional voxel coordinates
        rel_pos = (q - self.grid_origin) / self.grid_spacing
        voxel_idx = self.round_ste(rel_pos).astype(jnp.int32)
        voxel_idx = jnp.clip(voxel_idx, 1, self.grid_shape - 2)

        radius = kernel_size // 2
        offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
        neighbor_offsets = jnp.stack(jnp.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)

        @jaxtyped(typechecker=typechecker)
        def gather_neighbors(vi: Int[Array, "3"], q_point: Float[Array, "3"]) -> Float[Array, ""]:
            coords = vi[None, :] + neighbor_offsets

            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            neighbor_xyz = self.indices_3d[x, y, z]
            neighbor_metrics = self.relevant_metrics_3d[x, y, z]

            dists = jnp.sum((q_point - neighbor_xyz) ** 2, axis=-1)

            weights = jax.nn.softmax(-dists / (temp + EPS), axis=-1)
            return jnp.sum(weights * neighbor_metrics)

        return jax.vmap(gather_neighbors)(voxel_idx, q).astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def soft_get_local_slice(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperature: Float[Array, "1"],
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)
        temp = temperature.astype(self.dtype)

        rel_pos = (q - self.grid_origin) / self.grid_spacing
        voxel_idx = self.round_ste(rel_pos).astype(jnp.int32)

        alpha_idx = jnp.clip(voxel_idx[:, 0], 0, self.grid_shape[0] - 1)

        radius = kernel_size // 2
        offsets = jnp.arange(-radius, radius + 1)
        beta_sigma_idx = jnp.clip(voxel_idx[:, 1:], radius, self.grid_shape[1:] - radius - 1)

        offsets = jnp.arange(-radius, radius)
        neighbor_offsets = jnp.stack(jnp.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)

        @jaxtyped(typechecker=typechecker)
        def gather_neighbors(
            bs_idx: Int[Array, "2"], q_point: Float[Array, "3"], a_idx: Int[Array, ""]
        ) -> Float[Array, ""]:
            coords = bs_idx[None, :] + neighbor_offsets
            x, y = coords[:, 0], coords[:, 1]

            neighbor_xyz = self.indices_3d[a_idx, x, y, :]
            neighbor_metrics = self.relevant_metrics_3d[a_idx, x, y, :]

            dists = jnp.sum((q_point - neighbor_xyz) ** 2, axis=-1)

            weights = jax.nn.softmax(-dists / (temp + EPS), axis=-1)
            return jnp.sum(weights * neighbor_metrics)

        return jax.vmap(gather_neighbors)(beta_sigma_idx, q, alpha_idx).astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def soft_get_global(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperature: Float[Array, "1"],
        kernel_size: int | Int[Array, ""] = 3,
    ) -> Float[Array, " batch"]:
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)

        q_norm = jnp.sum(q**2, axis=-1, keepdims=True)

        dot_prod = q @ self.indices_T  # (num_indices,)
        dists = q_norm + self.i_norm - 2 * dot_prod

        weights = jax.nn.softmax(-dists / (temperature + EPS), axis=-1).squeeze()
        weights = weights * (1 - (weights < 0.001))

        return jnp.sum(weights * self.relevant_metrics, axis=-1).astype(orig_dtype)

    @jaxtyped(typechecker=typechecker)
    @filter_jit
    def soft_get_global_slice(
        self,
        query_vectors: Float[Array, "batch latent"],
        temperature: Float[Array, "1"],
        kernel_size: int | Int[Array, ""] = 11,
    ) -> Float[Array, " batch"]:
        orig_dtype = query_vectors.dtype
        q = query_vectors.astype(self.dtype)

        rel_pos = (q - self.grid_origin) / self.grid_spacing
        voxel_idx = self.round_ste(rel_pos).astype(jnp.int32)

        alpha_idx = voxel_idx[:, 0]
        q_beta_sigma = q[:, 1:]

        @jaxtyped(typechecker=typechecker)
        def per_query(q_bs: Float[Array, "2"], a_idx: Int[Array, ""]) -> Float[Array, ""]:
            slice_vals = self.relevant_metrics_3d[a_idx, :, :]

            diffs = self.indices_3d[a_idx, :, :, 1:] - q_bs
            squared_distances = jnp.sum(diffs**2, axis=-1)

            flat_weights = jax.nn.softmax(-squared_distances.reshape(-1) / (temperature + EPS))  # flattened
            weights = flat_weights.reshape(squared_distances.shape)

            return jnp.sum(weights * slice_vals)  # (2,)

        return jax.vmap(per_query)(q_beta_sigma, alpha_idx).astype(orig_dtype)

    def round_ste(self, x: Float[Array, "batch 3"]) -> Float[Array, "batch 3"]:
        return x + jax.lax.stop_gradient(jnp.round(x) - x)


def as_3d_indices(
    alpha_space: tuple[float, float, float],
    beta_space: tuple[float, float, float],
    sigma_space: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.arange(*alpha_space)
    betas = np.arange(*beta_space)
    sigmas = np.arange(*sigma_space)
    alpha_grid, beta_grid, sigma_grid = np.meshgrid(alphas, betas, sigmas, indexing="ij")
    permutations = np.stack([alpha_grid, beta_grid, sigma_grid], axis=-1)
    a, b, s = permutations[:, :, :, 0:1], permutations[:, :, :, 1:2], permutations[:, :, :, 2:3]
    return a, b, s


def as_2d_indices(
    x_space: tuple[float, float, float],
    y_space: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(*x_space)
    ys = np.arange(*y_space)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    return x_grid, y_grid
