from diffrax import (
    diffeqsolve,
    ODETerm,
    ConstantStepSize,
    SaveAt,
    TqdmProgressMeter,
    # solver - from most to least accurate / slow to fast
    Dopri8,
    Dopri5,  # Dopri5 same as MATLAB ode45
    Tsit5,
    Bosh3,
    Ralston,
)
from equinox import error_if, filter_jit
from equinox.debug import assert_max_traces
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import jit, vmap
from jaxtyping import ScalarLike

import json
import os
from numpy import save as np_save
from dataclasses import dataclass


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


# Register SystemState as a JAX PyTree
jtu.register_pytree_node(
    SystemState, SystemState.tree_flatten, SystemState.tree_unflatten
)


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


jtu.register_pytree_node(
    SystemMetrics, SystemMetrics.tree_flatten, SystemMetrics.tree_unflatten
)

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
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_autotune_level=0 "  # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
    "--xla_gpu_exhaustive_tiling_search=true "
    # "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_use_thunk_runtime=false "
    "--xla_cpu_multi_thread_eigen=true "
    "--xla_force_host_platform_device_count=10 "
)
devices = jax.devices()
print("jax.devices()       ", devices)

rand_key = jr.key(123)
num_parallel_runs = 50
rand_keys = jr.split(rand_key, num_parallel_runs)


#### Parameters
N = 200
N_kappa = N**2
alpha = -0.28 * jnp.pi  # phase lag
beta = 0.66 * jnp.pi  # age parameter
a_1 = 1.0
epsilon_1 = 0.03  # adaption rate
epsilon_2 = 0.3  # adaption rate
sigma = 1.0
omega_1 = omega_2 = 0.0
T_init, T_trans, T_max = 0, 1000, 2000
T_step = 0.05

omega_1_i = jnp.ones((N,)) * omega_1
omega_2_i = jnp.ones((N,)) * omega_2

a_1_ij = jnp.ones((N, N)) * a_1
a_1_ij = a_1_ij.at[jnp.diag_indices(N)].set(0)  # NOTE no self coupling

args = (N, N_kappa, alpha, beta, epsilon_1, epsilon_2, sigma)


def make_deriv(
    jomega_1_i: jnp.ndarray,
    jomega_2_i: jnp.ndarray,
    ja_1_ij: jnp.ndarray,
    batch_size: int,
):
    jN, jN_kappa, jalpha, jbeta, jepsilon_1, jepsilon_2, jsigma = (
        N,
        N_kappa,
        jnp.array(alpha),
        jnp.array(beta),
        jnp.array(epsilon_1),
        jnp.array(epsilon_2),
        jnp.array(sigma),
    )
    sin_beta = jnp.sin(beta)
    cos_beta = jnp.cos(beta)
    sin_alpha = jnp.sin(alpha)
    cos_alpha = jnp.cos(alpha)
    pi2 = jnp.array(jnp.pi * 2)
    adj = jnp.array(1 / (jN - 1))
    diag = jnp.diag_indices(N)

    @assert_max_traces(max_traces=8)  # TODO: why is it traced that often?
    def single_system_deriv(
        t: ScalarLike,
        y: SystemState,
        args: tuple[int, ...] | None = None,
    ) -> SystemState:
        # recover states from the py_tree
        phi_1_i, phi_2_i = y.phi_1 % pi2, y.phi_2 % pi2
        kappa_1_ij, kappa_2_ij = y.kappa_1.clip(-1, 1), y.kappa_2.clip(-1, 1)

        # sin/cos in radians
        sin_phi_1, cos_phi_1 = jnp.sin(phi_1_i), jnp.cos(phi_1_i)
        sin_phi_2, cos_phi_2 = jnp.sin(phi_2_i), jnp.cos(phi_2_i)

        sin_diff_phi_1 = jnp.einsum("bi,bj->bij", sin_phi_1, cos_phi_1) - jnp.einsum(
            "bi,bj->bij", cos_phi_1, sin_phi_1
        )
        cos_diff_phi_1 = jnp.einsum("bi,bj->bij", cos_phi_1, cos_phi_1) + jnp.einsum(
            "bi,bj->bij", sin_phi_1, sin_phi_1
        )

        sin_diff_phi_2 = jnp.einsum("bi,bj->bij", sin_phi_2, cos_phi_2) - jnp.einsum(
            "bi,bj->bij", cos_phi_2, sin_phi_2
        )
        cos_diff_phi_2 = jnp.einsum("bi,bj->bij", cos_phi_2, cos_phi_2) + jnp.einsum(
            "bi,bj->bij", sin_phi_2, sin_phi_2
        )

        sin_phi_1_diff_alpha = sin_diff_phi_1 * cos_alpha + cos_diff_phi_1 * sin_alpha
        sin_phi_2_diff_alpha = sin_diff_phi_2 * cos_alpha + cos_diff_phi_2 * sin_alpha

        dphi_1_i = (
            jomega_1_i
            - adj
            * jnp.einsum("bij,bij->bi", (ja_1_ij + kappa_1_ij), sin_phi_1_diff_alpha)
            - jsigma * (sin_phi_1 * cos_phi_2 - cos_phi_1 * sin_phi_2)
        )
        dphi_2_i = (
            jomega_2_i
            - adj * jnp.einsum("bij,bij->bi", kappa_2_ij, sin_phi_2_diff_alpha)
            - jsigma * (sin_phi_2 * cos_phi_1 - cos_phi_2 * sin_phi_1)
        )

        dkappa_1_ij = -jepsilon_1 * (
            kappa_1_ij + sin_diff_phi_1 * cos_beta - cos_diff_phi_1 * sin_beta
        )
        dkappa_2_ij = -jepsilon_2 * (
            kappa_2_ij + sin_diff_phi_2 * cos_beta - cos_diff_phi_2 * sin_beta
        )

        dkappa_1_ij = dkappa_1_ij.at[diag].set(0)
        dkappa_2_ij = dkappa_2_ij.at[diag].set(0)

        # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
        # reuse y as dy
        y.phi_1 = dphi_1_i
        y.phi_2 = dphi_2_i
        y.kappa_1 = dkappa_1_ij
        y.kappa_2 = dkappa_2_ij
        return y

    batched = single_system_deriv

    return batched


deriv = make_deriv(omega_1_i, omega_2_i, a_1_ij, num_parallel_runs)


def generate_init_conditions(key: jnp.ndarray) -> SystemState:
    phi_1_init = jr.uniform(key, (N,)) * (2 * jnp.pi)
    phi_2_init = jr.uniform(key, (N,)) * (2 * jnp.pi)

    # kappaIni1 = sin(parStruct.beta(1))*ones(N*N,1)+0.01*(2*rand(N*N,1)-1);
    # kappa_1_init = jnp.sin(beta)*jnp.ones((N, N))+0.01*(2*jr.uniform(rand_key, (N, N)) - 1);
    kappa_1_init = jr.uniform(key, (N, N)) * 2 - 1
    kappa_1_init = kappa_1_init.at[jnp.diag_indices(N)].set(0)

    kappa_2_init = jnp.ones((N, N))
    kappa_2_init = kappa_2_init.at[jnp.diag_indices(N)].set(0)
    kappa_2_init = kappa_2_init.at[40:, :40].set(0)
    kappa_2_init = kappa_2_init.at[:40, 40:].set(0)

    return SystemState(
        phi_1=phi_1_init,
        phi_2=phi_2_init,
        kappa_1=kappa_1_init,
        kappa_2=kappa_2_init,
    )


# @jit
def enforce_bounds(y: SystemState) -> SystemState:
    return SystemState(
        phi_1=y.phi_1 % (2 * jnp.pi),
        phi_2=y.phi_2 % (2 * jnp.pi),
        kappa_1=jnp.clip(y.kappa_1, -1, 1),
        kappa_2=jnp.clip(y.kappa_2, -1, 1),
    )


def full_save_compressed(
    t: ScalarLike, y: SystemState, args: tuple[int, ...] | None
) -> jnp.ndarray:
    y = enforce_bounds(y)
    return jnp.concatenate(
        [
            y.phi_1,
            y.phi_2,
            y.kappa_1.reshape(y.phi_1.shape[0], -1),
            y.kappa_2.reshape(y.phi_1.shape[0], -1),
        ],
        axis=-1,
        dtype=jnp.float16,
    )


def metric_save(t: ScalarLike, y: SystemState, args: tuple[int, ...] | None):
    y = enforce_bounds(y)

    ###### Kuramoto Order Parameter
    r_1 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_1), axis=-1))
    r_2 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_2), axis=-1))

    ###### Ensemble average of the standard deviations
    # For the derivatives we need to evaluate again ...
    dy = deriv(0, y, None)
    std_1 = jnp.std(dy.phi_1, axis=-1)
    std_2 = jnp.std(dy.phi_2, axis=-1)
    mean_1 = jnp.mean(dy.phi_1, axis=-1)
    mean_2 = jnp.mean(dy.phi_2, axis=-1)
    s_1 = jnp.mean(std_1)
    s_2 = jnp.mean(std_2)
    nstd_1 = std_1 / mean_1
    nstd_2 = std_2 / mean_1
    ns_1 = jnp.mean(nstd_1)
    ns_2 = jnp.mean(nstd_2)

    ###### Frequency cluster ratio
    # check for desynchronized nodes
    desync_1 = jnp.any(jnp.abs(y.phi_1 - mean_1[:, None]) > 1e-3, axis=-1)
    desync_2 = jnp.any(jnp.abs(y.phi_2 - mean_2[:, None]) > 1e-3, axis=-1)

    # Count number of frequency clusters
    # Number of ensembles where at least one node deviates
    n_f_1 = jnp.sum(desync_1)
    n_f_2 = jnp.sum(desync_2)

    f_1 = n_f_1 / y.phi_1.shape[0]  # N_f / N_E
    f_2 = n_f_2 / y.phi_1.shape[0]

    return SystemMetrics(
        r_1=r_1, r_2=r_2, s_1=s_1, s_2=s_2, ns_1=ns_1, ns_2=ns_2, f_1=f_1, f_2=f_2
    )


@filter_jit
def solve(batched_init_condition):
    t = jnp.arange(T_trans, T_max + T_step, T_step)
    term = ODETerm(deriv)
    solver = Ralston()
    stepsize_controller = ConstantStepSize()
    saveat = SaveAt(t0=True, ts=t, fn=metric_save)
    # saveat = SaveAt(ts=[0, T_max], fn=metric_save)
    res = diffeqsolve(
        term,
        solver,
        y0=batched_init_condition,
        args=args,
        t0=T_init,
        t1=T_max,
        dt0=T_step,
        stepsize_controller=stepsize_controller,
        max_steps=int(T_max / T_step) + 1,
        progress_meter=TqdmProgressMeter(),
        saveat=saveat,
    )
    return res


# sh_gen_init_cond = shard_map(
#     generate_init_conditions,
#     mesh=jax.sharding.Mesh(devices, axis_names=("data",)),  # Define the device mesh
#     in_specs=(P("data"),),  # Specify sharding pattern
#     out_specs=P("data"),  # Ensure output is also sharded
#     check_rep=False,
# )
# init_conditions = sh_gen_init_cond(rand_keys)

init_conditions = vmap(generate_init_conditions)(
    rand_keys
)  # shape: (num_parallel_runs, state)

# sharded_solve = shard_map(
#     solve,
#     mesh=jax.sharding.Mesh(devices, axis_names=("data",)),
#     in_specs=(P("data"),),
#     out_specs=P(),
#     check_rep=False,
# )
# sol = sharded_solve(init_conditions)
sol = solve(init_conditions)

if sol.ts is not None and sol.ys is not None:
    print(sol.ts)
    print(sol.ys)
    print(sol.ys.r_1.shape)  # shape (t, state)
    print(sol.ys[-1].s_1)  # shape (t, state)
    # print(sol.ys[-1][0][:N].sort())  # shape (t, batchnr, state)
    # print(sol.ts.shape, sol.ys.shape)
    # dir_name = "test"
    # np_save(f"{dir_name}/t_vals", sol.ts)
    # np_save(f"{dir_name}/y_vals", sol.ys)
    # T_tot = len(sol.ts)

    # info = {
    #     "N": N,
    #     "N_kappa": N_kappa,
    #     "alpha": alpha,
    #     "beta": beta,
    #     "a_1": a_1,
    #     "epsilon_1": epsilon_1,
    #     "epsilon_2": epsilon_2,
    #     "sigma": sigma,
    #     "omega_1": omega_1,
    #     "omega_2": omega_2,
    #     "T_init": T_init,
    #     "T_trans": T_trans,
    #     "T_max": T_max,
    #     "T_step": T_step,
    #     "T_tot": T_tot,
    # }
    # with open(f"{dir_name}/info.json", "w") as json_file:
    #     json.dump(info, json_file)


# TODO
# save
# no globals? https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
