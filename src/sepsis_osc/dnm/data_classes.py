from dataclasses import dataclass, fields
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from beartype import beartype as typechecker
from equinox import filter_jit, field
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import DTypeLike
from scipy.ndimage import uniform_filter1d


@jaxtyped(typechecker=typechecker)
@dataclass
class SystemConfig:
    N: int  # number of oscillators per layer
    C: int  # number of infected cells
    omega_1: float  # natural frequency parenchymal layer
    omega_2: float  # natural frequency immune layer
    a_1: float  # intralayer connectivity weights
    epsilon_1: float  # adaption rate
    epsilon_2: float  # adaption rate
    alpha: float  # phase lag
    beta: float  # plasticity (age parameter)
    sigma: float  # interlayer coupling
    T_init: Optional[int] = None
    T_trans: Optional[int] = None
    T_max: Optional[int] = None
    T_step: Optional[int] = None

    @property
    def as_args(self) -> tuple[jnp.ndarray, ...]:
        ja_1_ij = jnp.ones((self.N, self.N), jnp.float64) * self.a_1
        ja_1_ij = ja_1_ij.at[jnp.diag_indices(self.N)].set(0)  # NOTE no self coupling
        jsin_alpha = jnp.sin(self.alpha * jnp.pi)
        jcos_alpha = jnp.cos(self.alpha * jnp.pi)
        jsin_beta = jnp.sin(self.beta * jnp.pi)
        jcos_beta = jnp.cos(self.beta * jnp.pi)
        jadj = jnp.array(1 / (self.N - 1), jnp.float64)
        diag = (jnp.ones((self.N, self.N), jnp.float64) - jnp.eye(self.N))[None, :]  # again no self coupling
        jepsilon_1 = jnp.array(self.epsilon_1, jnp.float64) * diag
        jepsilon_2 = jnp.array(self.epsilon_2, jnp.float64) * diag
        jomega_1_i = jnp.ones((self.N,), jnp.float64) * self.omega_1
        jomega_2_i = jnp.ones((self.N,), jnp.float64) * self.omega_2
        jsigma = jnp.array(self.sigma, jnp.float64)
        return (
            ja_1_ij,
            jsin_alpha,
            jcos_alpha,
            jsin_beta,
            jcos_beta,
            jadj,
            jepsilon_1,
            jepsilon_2,
            jsigma,
            jomega_1_i,
            jomega_2_i,
        )

    @property
    def as_index(self) -> tuple[float, ...]:
        return (
            float(self.omega_1),
            float(self.omega_2),
            float(self.a_1),
            float(self.epsilon_1),
            float(self.epsilon_2),
            float(self.alpha / jnp.pi),
            float(self.beta / jnp.pi),
            float(self.sigma / 2),
            float(self.C / self.N),
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def batch_as_index(
        alpha: Float[Array, "... 1"] | np.ndarray,
        beta: Float[Array, "... 1"] | np.ndarray,
        sigma: Float[Array, "... 1"] | np.ndarray,
        C: float,
    ) -> jnp.ndarray:
        batch_shape = alpha.shape

        # variable
        alpha = alpha / jnp.pi
        beta = beta / jnp.pi
        sigma = sigma / 2
        _C = jnp.full(batch_shape, C)

        # constant
        omega_1_batch = jnp.full(batch_shape, 0.0)
        omega_2_batch = jnp.full(batch_shape, 0.0)
        a_1_batch = jnp.full(batch_shape, 1.0)
        epsilon_1_batch = jnp.full(batch_shape, 0.03)
        epsilon_2_batch = jnp.full(batch_shape, 0.3)

        batch_indices = jnp.stack(
            [
                omega_1_batch,
                omega_2_batch,
                a_1_batch,
                epsilon_1_batch,
                epsilon_2_batch,
                alpha,
                beta,
                sigma,
                _C,
            ],
            axis=-1,
            dtype=jnp.float32,
        )

        return batch_indices.squeeze()


@jaxtyped(typechecker=typechecker)
@dataclass
class SystemState:
    # NOTE shapes: Ensemble Simulation | Single Simulation | Visualisations
    phi_1: Float[Array, "*t ensemble N"] | Float[Array, " N"] | np.ndarray | object
    phi_2: Float[Array, "*t ensemble N"] | Float[Array, " N"] | np.ndarray | object
    kappa_1: Float[Array, "*t ensemble N N"] | Float[Array, "N N"] | np.ndarray | object
    kappa_2: Float[Array, "*t ensemble N N"] | Float[Array, "N N"] | np.ndarray | object

    # Required for JAX to recognize it as a PyTree
    def tree_flatten(self):
        return (self.phi_1, self.phi_2, self.kappa_1, self.kappa_2), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def shape(self):
        return jax.tree.map(lambda x: x.shape if x is not None else None, self.__dict__)

    def enforce_bounds(self) -> "SystemState":
        self.phi_1 = self.phi_1 % (2 * jnp.pi)
        self.phi_2 = self.phi_2 % (2 * jnp.pi)
        # its written but not actually done
        # self.kappa_1 = jnp.clip(self.kappa_1, -1, 1)
        # self.kappa_2 = jnp.clip(self.kappa_2, -1, 1)
        return self

    def last(self) -> "SystemState":
        return jax.tree.map(lambda x: x[-1], self)

    def squeeze(self) -> "SystemState":
        # when running with ensemble size 1, we want to get rid of the ensemble dimension
        self.phi_1 = self.phi_1.squeeze()
        self.phi_2 = self.phi_2.squeeze()
        self.kappa_1 = self.kappa_1.squeeze(axis=1)
        self.kappa_2 = self.kappa_2.squeeze(axis=1)
        return self

    def astype(self, dtype: jnp.dtype = jnp.float64) -> "SystemState":
        return jax.tree.map(lambda x: x.astype(dtype), self)

    def remove_infs(self) -> "SystemState":
        phi_1_has_inf = jnp.isinf(self.phi_1).any(axis=(1, 2))
        phi_2_has_inf = jnp.isinf(self.phi_2).any(axis=(1, 2))
        kappa_1_has_inf = jnp.isinf(self.kappa_1).any(axis=(1, 2, 3))
        kappa_2_has_inf = jnp.isinf(self.kappa_2).any(axis=(1, 2, 3))
        combined_mask = ~(phi_1_has_inf | phi_2_has_inf | kappa_1_has_inf | kappa_2_has_inf)
        return jax.tree.map(lambda x: x[combined_mask], self)

    def copy(self) -> "SystemState":
        return SystemState(self.phi_1, self.phi_2, self.kappa_1, self.kappa_2)


# Register SystemState as a JAX PyTree
jtu.register_pytree_node(SystemState, SystemState.tree_flatten, SystemState.tree_unflatten)


@dataclass
class SystemMetrics:
    # NOTE shapes: Simulation | DB/Lookup Query | Visualisations

    # Kuramoto Order Parameter
    r_1: Float[Array, "*t ensemble"] | Float[Array, "... 1"] | np.ndarray
    r_2: Float[Array, "*t ensemble"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble average velocity
    m_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    m_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble average std
    s_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    s_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble phase entropy
    q_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    q_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Frequency cluster ratio
    f_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    f_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Splay State Ratio
    sr_1: Optional[Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray] = None
    sr_2: Optional[Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray] = None
    # Measured mean transient time
    tt: Optional[Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray] = None

    @property
    def shape(self):
        return jax.tree.map(lambda x: x.shape if x is not None else None, self.__dict__)

    def tree_flatten(self):
        flat_children = []
        for field_name in [f.name for f in fields(self)]:
            value = getattr(self, field_name)
            flat_children.append(value)
        return flat_children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def copy(self) -> "SystemMetrics":
        return SystemMetrics(**{f.name: getattr(self, f.name) for f in fields(self)})

    def remove_infs(self):
        def is_valid(arr):
            if arr is None:
                return None
            arr = jnp.asarray(arr)
            if arr.ndim == 1:
                return ~jnp.isinf(arr)
            else:
                return ~jnp.isinf(arr).any(axis=1)

        masks = [
            is_valid(self.r_1),
            is_valid(self.r_2),
            is_valid(self.m_1),
            is_valid(self.m_2),
            is_valid(self.s_1),
            is_valid(self.s_2),
            is_valid(self.q_1),
            is_valid(self.q_2),
            is_valid(self.f_1),
            is_valid(self.f_2),
            is_valid(self.sr_1) if hasattr(self, "sr_1") else None,
            is_valid(self.sr_2) if hasattr(self, "sr_2") else None,
        ]

        masks = jnp.asarray([m for m in masks if m is not None])
        combined_mask = jnp.logical_and.reduce(masks)

        return jax.tree.map(
            lambda x: x[combined_mask] if x is not None and x.shape[0] == combined_mask.shape[0] else x,
            self,
        )

    def add_follow_ups(self) -> "SystemMetrics":
        if self.sr_1 is None and self.r_1.size > 1:
            self.sr_1 = jnp.sum(self.r_1 < 0.2, axis=-1) / self.r_1.shape[-1]
            self.sr_2 = jnp.sum(self.r_2 < 0.2, axis=-1) / self.r_2.shape[-1]
            std_over_time = jnp.std(self.r_1, axis=1)
            smoothed_std = uniform_filter1d(std_over_time, size=10)
            threshold = jnp.percentile(smoothed_std, jnp.clip((self.r_1[-1].std() * 100) ** 2, 1, 10))
            transient_end_candidates = np.where(smoothed_std < threshold)[0]
            if len(transient_end_candidates) > 0:
                self.tt = transient_end_candidates.max()
            else:
                self.tt = jnp.array([self.r_1.shape[0]])
        return self

    def as_single(self) -> "SystemMetrics":
        if self.r_1.size <= 1:  # already single
            return self

        self.add_follow_ups()
        print(self.shape)
        return SystemMetrics(
            r_1=jnp.mean(jnp.asarray(self.r_1)[..., -1, :], axis=(-1,)),
            r_2=jnp.mean(jnp.asarray(self.r_2)[..., -1, :], axis=(-1,)),
            s_1=jnp.mean(jnp.asarray(self.s_1)),
            s_2=jnp.mean(jnp.asarray(self.s_2)),
            m_1=jnp.mean(jnp.asarray(self.m_1), axis=-1),
            m_2=jnp.mean(jnp.asarray(self.m_2), axis=-1),
            q_1=jnp.mean(jnp.asarray(self.q_1), axis=-1),
            q_2=jnp.mean(jnp.asarray(self.q_2), axis=-1),
            f_1=jnp.mean(jnp.asarray(self.f_1), axis=-1),
            f_2=jnp.mean(jnp.asarray(self.f_2), axis=-1),
            sr_1=self.sr_1[..., -1] if self.sr_1 is not None else None,
            sr_2=self.sr_2[..., -1] if self.sr_2 is not None else None,
            tt=self.tt.max() if self.tt is not None else None,
        )

    @staticmethod
    def np_empty(shape: tuple[int, ...], dtype: DTypeLike = np.float32) -> "SystemMetrics":
        initialized_fields = {}
        for f in fields(SystemMetrics):
            initialized_fields[f.name] = np.zeros(shape, dtype=dtype)
        return SystemMetrics(**initialized_fields)

    def to_jax(self) -> "SystemMetrics":
        return jax.tree.map(lambda x: jnp.asarray(x) if x is not None else None, self)

    def reshape(self, shape: tuple[int, ...]) -> "SystemMetrics":
        return jax.tree.map(lambda x: jnp.reshape(x, shape) if x is not None else None, self)

    def astype(self, dtype: jnp.dtype = jnp.float32) -> "SystemMetrics":
        return jax.tree.map(lambda x: x.astype(dtype) if x is not None else None, self)


jtu.register_pytree_node(SystemMetrics, SystemMetrics.tree_flatten, SystemMetrics.tree_unflatten)


@jaxtyped(typechecker=typechecker)
# @register_dataclass
@dataclass(frozen=True)
class LatentLookup:
    metrics: SystemMetrics = field(static=True)
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
    metrics_3d: SystemMetrics = field(static=True)

    dtype: DTypeLike = jnp.float32

    def __init__(
        self,
        metrics: SystemMetrics,
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
    def extract_relevant(metrics_: SystemMetrics) -> Float[Array, "na nb ns 2"] | Float[Array, "naxnbxns 2"]:
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
