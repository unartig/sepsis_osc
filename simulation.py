from collections.abc import Callable
from equinox.debug import assert_max_traces
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ScalarLike

import os
from dataclasses import dataclass


@dataclass
class SystemConfig:
    N: int
    C: int
    omega_1: float
    omega_2: float
    a_1: float
    epsilon_1: float
    epsilon_2: float
    alpha: float
    beta: float
    sigma: float
    T_init: int | None = None
    T_trans: int | None = None
    T_max: int | None = None
    T_step: int | None = None

    @property
    def as_args(self) -> tuple[jnp.ndarray, ...]:
        ja_1_ij = jnp.ones((self.N, self.N)) * self.a_1
        ja_1_ij = ja_1_ij.at[jnp.diag_indices(self.N)].set(0)  # NOTE no self coupling
        jsin_alpha = jnp.sin(self.alpha * jnp.pi)
        jcos_alpha = jnp.cos(self.alpha * jnp.pi)
        jsin_beta = jnp.sin(self.beta * jnp.pi)
        jcos_beta = jnp.cos(self.beta * jnp.pi)
        jpi2 = jnp.array(jnp.pi * 2)
        jadj = jnp.array(1 / (self.N - 1))
        diag = (jnp.ones((self.N, self.N)) - jnp.eye(self.N))[None, :]  # again no self coupling
        jepsilon_1 = self.epsilon_1 * diag
        jepsilon_2 = self.epsilon_2 * diag
        jomega_1_i = jnp.ones((self.N,)) * self.omega_1
        jomega_2_i = jnp.ones((self.N,)) * self.omega_2
        jsigma = jnp.array(self.sigma)
        return (
            ja_1_ij,
            jsin_alpha,
            jcos_alpha,
            jsin_beta,
            jcos_beta,
            jpi2,
            jadj,
            jepsilon_1,
            jepsilon_2,
            jsigma,
            jomega_1_i,
            jomega_2_i,
        )

    @property
    def as_index(self) -> tuple[float | int, ...]:
        return (
            self.omega_1,
            self.omega_2,
            self.a_1,
            self.epsilon_1,
            self.epsilon_2,
            self.alpha / jnp.pi,
            self.beta / jnp.pi,
            self.sigma / 2,
            self.C / self.N,
        )


@dataclass
class SystemState:
    phi_1: jnp.ndarray  # Shape (N,)
    phi_2: jnp.ndarray  # Shape (N,)
    kappa_1: jnp.ndarray  # Shape (N, N)
    kappa_2: jnp.ndarray  # Shape (N, N)

    # Required for JAX to recognize it as a PyTree
    def tree_flatten(self):
        return (self.phi_1, self.phi_2, self.kappa_1, self.kappa_2), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def enforce_bounds(self):
        self.phi_1 = self.phi_1 % (2 * jnp.pi)
        self.phi_2 = self.phi_2 % (2 * jnp.pi)
        self.kappa_1 = jnp.clip(self.kappa_1, -1, 1)
        self.kappa_2 = jnp.clip(self.kappa_2, -1, 1)
        return self

    def last(self):
        self.phi_1 = self.phi_1[-1]
        self.phi_2 = self.phi_2[-1]
        self.kappa_1 = self.kappa_1[-1]
        self.kappa_2 = self.kappa_2[-1]
        return self

    def astype(self, dtype: jnp.dtype = jnp.float64):
        return SystemState(
            phi_1=self.phi_1.astype(dtype),
            phi_2=self.phi_2.astype(dtype),
            kappa_1=self.kappa_1.astype(dtype),
            kappa_2=self.kappa_2.astype(dtype)
        )
        


# Register SystemState as a JAX PyTree
jtu.register_pytree_node(SystemState, SystemState.tree_flatten, SystemState.tree_unflatten)


@dataclass
class SystemMetrics:
    # Kuramoto Order Parameter
    r_1: jnp.ndarray  # shape (batch_size,)
    r_2: jnp.ndarray  # shape (batch_size,)
    # Ensemble average std
    s_1: ScalarLike
    s_2: ScalarLike
    # Ensemble average normed std
    ns_1: ScalarLike
    ns_2: ScalarLike
    # Frequency cluster ratio
    f_1: ScalarLike
    f_2: ScalarLike

    def tree_flatten(self):
        return (
            self.r_1,
            self.r_2,
            self.s_1,
            self.s_2,
            self.ns_1,
            self.ns_2,
            self.f_1,
            self.f_2,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


jtu.register_pytree_node(SystemMetrics, SystemMetrics.tree_flatten, SystemMetrics.tree_unflatten)

#### Configurations
# jax flags
jax.config.update("jax_enable_x64", True)  #  MATLAB defaults to double precision
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)
# jax.config.update("jax_disable_jit", True)

# cpu/gpu flags
os.environ["XLA_FLAGS"] = (
    # "--xla_gpu_enable_latency_hiding_scheduler=true "
    # "--xla_gpu_enable_triton_gemm=false "
    # "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_autotune_level=4 "  # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
    # "--xla_gpu_exhaustive_tiling_search=true "
    # "--xla_cpu_multi_thread_eigen=false "
    # "--xla_cpu_use_thunk_runtime=false "
    # "--xla_cpu_multi_thread_eigen=true "
    # "intra_op_parallelism_threads=1 "
    "--xla_force_host_platform_device_count=10 "
)
devices = jax.devices()
print("jax.devices()", devices)


def generate_init_conditions_fixed(N: int, beta: float, C: int) -> Callable:
    def inner(key: jnp.ndarray) -> SystemState:
        phi_1_init = jr.uniform(key, (N,)) * (2 * jnp.pi)
        phi_2_init = jr.uniform(key, (N,)) * (2 * jnp.pi)

        # kappaIni1 = sin(parStruct.beta(1))*ones(N*N,1)+0.01*(2*rand(N*N,1)-1);
        kappa_1_init = jnp.sin(beta) * jnp.ones((N, N)) + 0.01 * (2 * jr.uniform(key, (N, N)) - 1)
        # kappa_1_init = jr.uniform(key, (N, N)) * 2 - 1
        kappa_1_init = kappa_1_init.at[jnp.diag_indices(N)].set(0)

        kappa_2_init = jnp.ones((N, N))
        kappa_2_init = kappa_2_init.at[jnp.diag_indices(N)].set(0)
        kappa_2_init = kappa_2_init.at[C:, :C].set(0)
        kappa_2_init = kappa_2_init.at[:C, C:].set(0)

        return SystemState(
            phi_1=phi_1_init,
            phi_2=phi_2_init,
            kappa_1=kappa_1_init,
            kappa_2=kappa_2_init,
        )

    return inner


# @assert_max_traces(max_traces=20)  # TODO: why is it traced that often?
def system_deriv(
    t: ScalarLike,
    y: SystemState,
    args: tuple[jnp.ndarray, ...],
) -> SystemState:
    (
        ja_1_ij,
        sin_alpha,
        cos_alpha,
        sin_beta,
        cos_beta,
        pi2,
        adj,
        jepsilon_1,
        jepsilon_2,
        jsigma,
        jomega_1_i,
        jomega_2_i,
    ) = args
    # recover states from the py_tree
    phi_1_i, phi_2_i = y.phi_1 % pi2, y.phi_2 % pi2
    kappa_1_ij, kappa_2_ij = y.kappa_1.clip(-1, 1), y.kappa_2.clip(-1, 1)

    # sin/cos in radians
    sin_phi_1, cos_phi_1 = jnp.sin(phi_1_i), jnp.cos(phi_1_i)
    sin_phi_2, cos_phi_2 = jnp.sin(phi_2_i), jnp.cos(phi_2_i)

    sin_diff_phi_1 = jnp.einsum("bi,bj->bij", sin_phi_1, cos_phi_1) - jnp.einsum("bi,bj->bij", cos_phi_1, sin_phi_1)
    cos_diff_phi_1 = jnp.einsum("bi,bj->bij", cos_phi_1, cos_phi_1) + jnp.einsum("bi,bj->bij", sin_phi_1, sin_phi_1)

    sin_diff_phi_2 = jnp.einsum("bi,bj->bij", sin_phi_2, cos_phi_2) - jnp.einsum("bi,bj->bij", cos_phi_2, sin_phi_2)
    cos_diff_phi_2 = jnp.einsum("bi,bj->bij", cos_phi_2, cos_phi_2) + jnp.einsum("bi,bj->bij", sin_phi_2, sin_phi_2)

    sin_phi_1_diff_alpha = sin_diff_phi_1 * cos_alpha + cos_diff_phi_1 * sin_alpha
    sin_phi_2_diff_alpha = sin_diff_phi_2 * cos_alpha + cos_diff_phi_2 * sin_alpha

    # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
    # reuse y as dy
    y.phi_1 = (
        jomega_1_i
        - adj * jnp.einsum("bij,bij->bi", (ja_1_ij + kappa_1_ij), sin_phi_1_diff_alpha)
        - jsigma * (sin_phi_1 * cos_phi_2 - cos_phi_1 * sin_phi_2)
    )
    y.phi_2 = (
        jomega_2_i
        - adj * jnp.einsum("bij,bij->bi", kappa_2_ij, sin_phi_2_diff_alpha)
        - jsigma * (sin_phi_2 * cos_phi_1 - cos_phi_2 * sin_phi_1)
    )

    y.kappa_1 = -jepsilon_1 * (kappa_1_ij + sin_diff_phi_1 * cos_beta - cos_diff_phi_1 * sin_beta)
    y.kappa_2 = -jepsilon_2 * (kappa_2_ij + sin_diff_phi_2 * cos_beta - cos_diff_phi_2 * sin_beta)

    return y


def make_full_compressed_save(dtype: jnp.dtype = jnp.float16) -> Callable:
    def full_compressed_save(t: ScalarLike, y: SystemState, args: tuple[jnp.ndarray, ...] | None) -> jnp.ndarray:
        y.enforce_bounds()
        return y.astype(dtype)

    return full_compressed_save


def make_metric_save(deriv) -> Callable:
    def mean_angle(angles, axis=None):
        angles = jnp.asarray(angles)
        sin_vals = jnp.sin(angles)
        cos_vals = jnp.cos(angles)
        return jnp.arctan2(jnp.mean(sin_vals, axis=axis), jnp.mean(cos_vals, axis=axis))

    def std_angle(angles, axis=None):
        angles = jnp.asarray(angles)
        mean_ang = mean_angle(angles, axis=axis)
        angular_diff = jnp.angle(jnp.exp(1j * (angles - jnp.expand_dims(mean_ang, axis))))  # Wrap differences to [-pi, pi]
        return jnp.sqrt(jnp.mean(angular_diff ** 2, axis=axis))
    
    def metric_save(t: ScalarLike, y: SystemState, args: tuple[jnp.ndarray, ...]):
        y.enforce_bounds

        ###### Kuramoto Order Parameter
        r_1 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_1), axis=-1))
        r_2 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_2), axis=-1))

        ###### Ensemble average of the standard deviations
        # For the derivatives we need to evaluate again ...
        dy = deriv(0, y, args)
        mean_1 = mean_angle(dy.phi_1, axis=-1)
        mean_2 = mean_angle(dy.phi_2, axis=-1)
        std_1 = std_angle(dy.phi_1, axis=-1)
        std_2 = std_angle(dy.phi_2, axis=-1)
        s_1 = jnp.mean(std_1)
        s_2 = jnp.mean(std_2)
        nstd_1 = std_1 / mean_1
        nstd_2 = std_2 / mean_1
        ns_1 = jnp.mean(nstd_1)
        ns_2 = jnp.mean(nstd_2)

        ###### Frequency cluster ratio
        # check for desynchronized nodes
        eps = 1e-1  # TODO what is the value here?
        desync_1 = jnp.any(jnp.abs(y.phi_1 - mean_1[:, None]) > eps, axis=-1)
        desync_2 = jnp.any(jnp.abs(y.phi_2 - mean_2[:, None]) > eps, axis=-1)

        # Count number of frequency clusters
        # Number of ensembles where at least one node deviates
        n_f_1 = jnp.sum(desync_1)
        n_f_2 = jnp.sum(desync_2)

        N_E = y.phi_1.shape[0]
        f_1 = n_f_1 / N_E  # N_f / N_E
        f_2 = n_f_2 / N_E

        return SystemMetrics(r_1=r_1, r_2=r_2, s_1=s_1, s_2=s_2, ns_1=ns_1, ns_2=ns_2, f_1=f_1, f_2=f_2)

    return metric_save
