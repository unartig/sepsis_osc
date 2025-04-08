from collections.abc import Callable
from equinox.debug import assert_max_traces
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import ScalarLike
import os

from simulation.data_classes import SystemState, SystemMetrics

#### Configurations
# jax flags
jax.config.update("jax_enable_x64", True)  #  MATLAB defaults to double precision
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)
# jax.config.update("jax_disable_jit", True)

# cpu/gpu flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_autotune_level=4 "  # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
    "--xla_gpu_exhaustive_tiling_search=true "
)
devices = jax.devices()
print("jax.devices()", devices)


def generate_init_conditions_fixed(N: int, beta: float, C: int) -> Callable:
    def inner(key: jnp.ndarray) -> SystemState:

        keys = jr.split(key, 3)
        phi_1_init = jr.uniform(keys[0], (N,)) * (2 * jnp.pi)
        phi_2_init = jr.uniform(keys[1], (N,)) * (2 * jnp.pi)

        # kappaIni1 = sin(parStruct.beta(1))*ones(N*N,1)+0.01*(2*rand(N*N,1)-1);
        kappa_1_init = jnp.ones((N, N)) + 0.01 * (2 * jr.uniform(keys[2], (N, N)) - 1)
        # kappa_1_init = jr.uniform(keys[3], (N, N)) * 2 - 1
        kappa_1_init = kappa_1_init.at[jnp.diag_indices(N)].set(0)

        kappa_2_init = jnp.ones((N, N))
        kappa_2_init = kappa_2_init.at[C:, :C].set(0)
        kappa_2_init = kappa_2_init.at[:C, C:].set(0)
        kappa_2_init = kappa_2_init.at[jnp.diag_indices(N)].set(0)

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
    # https://mediatum.ub.tum.de/doc/1638503/1638503.pdf
    sin_phi_1, cos_phi_1 = jnp.sin(phi_1_i), jnp.cos(phi_1_i)
    sin_phi_2, cos_phi_2 = jnp.sin(phi_2_i), jnp.cos(phi_2_i)

    # expand dims to broadcast outer product [i:]*[:j]->[ij]
    sin_diff_phi_1 = jnp.einsum("bi,bj->bij", sin_phi_1, cos_phi_1) - jnp.einsum("bi,bj->bij", cos_phi_1, sin_phi_1)
    cos_diff_phi_1 = jnp.einsum("bi,bj->bij", cos_phi_1, cos_phi_1) + jnp.einsum("bi,bj->bij", sin_phi_1, sin_phi_1)

    sin_diff_phi_2 = jnp.einsum("bi,bj->bij", sin_phi_2, cos_phi_2) - jnp.einsum("bi,bj->bij", cos_phi_2, sin_phi_2)
    cos_diff_phi_2 = jnp.einsum("bi,bj->bij", cos_phi_2, cos_phi_2) + jnp.einsum("bi,bj->bij", sin_phi_2, sin_phi_2)
    sin_phi_1_diff_alpha = sin_diff_phi_1 * cos_alpha + cos_diff_phi_1 * sin_alpha
    sin_phi_2_diff_alpha = sin_diff_phi_2 * cos_alpha + cos_diff_phi_2 * sin_alpha

    # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
    phi_1 = (
        jomega_1_i
        - adj * jnp.einsum("bij,bij->bi", (ja_1_ij + kappa_1_ij), sin_phi_1_diff_alpha)
        - jsigma * (sin_phi_1 * cos_phi_2 - cos_phi_1 * sin_phi_2)
    )
    phi_2 = (
        jomega_2_i
        - adj * jnp.einsum("bij,bij->bi", kappa_2_ij, sin_phi_2_diff_alpha)
        - jsigma * (sin_phi_2 * cos_phi_1 - cos_phi_2 * sin_phi_1)
    )
    kappa_1 = -jepsilon_1 * (kappa_1_ij + (sin_diff_phi_1 * cos_beta - cos_diff_phi_1 * sin_beta))
    kappa_2 = -jepsilon_2 * (kappa_2_ij + (sin_diff_phi_2 * cos_beta - cos_diff_phi_2 * sin_beta))

    return SystemState(phi_1=phi_1, phi_2=phi_2, kappa_1=kappa_1, kappa_2=kappa_2)


def matlab_deriv(
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

    # expand dims to broadcast outer product [i:]*[:j]->[ij]
    sin_diff_phi_1 = jnp.einsum("bi,bj->bij", sin_phi_1, cos_phi_1) - jnp.einsum("bi,bj->bij", cos_phi_1, sin_phi_1)
    cos_diff_phi_1 = jnp.einsum("bi,bj->bij", cos_phi_1, cos_phi_1) + jnp.einsum("bi,bj->bij", sin_phi_1, sin_phi_1)

    sin_diff_phi_2 = jnp.einsum("bi,bj->bij", sin_phi_2, cos_phi_2) - jnp.einsum("bi,bj->bij", cos_phi_2, sin_phi_2)
    cos_diff_phi_2 = jnp.einsum("bi,bj->bij", cos_phi_2, cos_phi_2) + jnp.einsum("bi,bj->bij", sin_phi_2, sin_phi_2)

    sin_phi_1_diff_alpha = sin_diff_phi_1 * cos_alpha + cos_diff_phi_1 * sin_alpha
    sin_phi_2_diff_alpha = sin_diff_phi_2 * cos_alpha + cos_diff_phi_2 * sin_alpha

    # (phi1 (bxN), phi2 (bxN), k1 (bxNxN), k2 (bxNxN)))
    phi_1 = jomega_1_i - jsigma * jnp.einsum("bij,bij->bi", sin_phi_1_diff_alpha, ja_1_ij * (1 + kappa_1_ij))
    phi_2 = jomega_2_i - jsigma * jnp.einsum("bij,bij->bi", sin_phi_2_diff_alpha, ja_1_ij * kappa_2_ij)

    kappa_1 = -jepsilon_1 * (kappa_1_ij + (sin_diff_phi_1 * cos_beta - cos_diff_phi_1 * sin_beta))
    kappa_2 = -jepsilon_2 * (kappa_2_ij + (sin_diff_phi_2 * cos_beta - cos_diff_phi_2 * sin_beta))
    return SystemState(phi_1=phi_1, phi_2=phi_2, kappa_1=kappa_1, kappa_2=kappa_2)


def make_full_compressed_save(dtype: jnp.dtype = jnp.float16) -> Callable:
    def full_compressed_save(t: ScalarLike, y: SystemState, args: tuple[jnp.ndarray, ...] | None) -> SystemState:
        y.enforce_bounds()
        return y.astype(dtype)

    return full_compressed_save


def make_metric_save(deriv) -> Callable:
    def mean_angle(angles, axis=-1):
        angles = jnp.asarray(angles)
        sin_vals = jnp.sin(angles)
        cos_vals = jnp.cos(angles)
        return jnp.arctan2(jnp.mean(sin_vals, axis=axis), jnp.mean(cos_vals, axis=axis))

    def std_angle(angles, axis=-1):
        angles = jnp.asarray(angles)
        mean_ang = mean_angle(angles, axis=axis)
        angular_diff = jnp.angle(
            jnp.exp(1j * (angles - jnp.expand_dims(mean_ang, axis)))
        )  # Wrap differences to [-pi, pi]
        return jnp.sqrt(jnp.mean(angular_diff**2, axis=axis))

    def phase_entropy(phis: jnp.ndarray, num_bins=30):
        hist, bin_edges = jnp.histogram(phis, bins=num_bins, range=(0, 2 * jnp.pi), density=True)

        hist = jnp.clip(hist, 1e-10, 1)

        # Shannon entropy
        entropy = -jnp.sum(hist * jnp.log(hist) * (bin_edges[1] - bin_edges[0]))  # bin width is constant
        return entropy

    def metric_save(t: ScalarLike, y: SystemState, args: tuple[jnp.ndarray, ...]):
        y.enforce_bounds()

        ###### Kuramoto Order Parameter
        r_1 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_1), axis=-1))
        r_2 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_2), axis=-1))

        ###### Entropy of Phase System
        q_1 = phase_entropy(y.phi_1)
        q_2 = phase_entropy(y.phi_2)

        ###### Ensemble average and avergage of the standard deviations
        # For the derivatives we need to evaluate again ...
        dy = deriv(0, y, args)
        mean_1 = mean_angle(dy.phi_1, axis=-1)
        mean_2 = mean_angle(dy.phi_2, axis=-1)
        m_1 = jnp.mean(mean_1)
        m_2 = jnp.mean(mean_2)

        s_1 = jnp.mean(std_angle(dy.phi_1, axis=-1))
        s_2 = jnp.mean(std_angle(dy.phi_2, axis=-1))

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

        return SystemMetrics(r_1=r_1, r_2=r_2, m_1=m_1, m_2=m_2, s_1=s_1, s_2=s_2, q_1=q_1, q_2=q_2, f_1=f_1, f_2=f_2)

    return metric_save
