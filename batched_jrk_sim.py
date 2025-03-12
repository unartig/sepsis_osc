import json

from diffrax import (
    diffeqsolve,
    ODETerm,
    Dopri5,  # Dopri5 same as MATLAB ode45
    Tsit5,
    ConstantStepSize,
    SaveAt,
    TqdmProgressMeter,
)
from equinox import error_if
from equinox.debug import assert_max_traces
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import jit, vmap
from numpy import save as np_save
from ml_dtypes import float8_e3m4

import jax
import os
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

    def astype(self, dtype):
        self.phi_1 = self.phi_1.astype(dtype)
        self.phi_2 = self.phi_2.astype(dtype)
        self.kappa_1 = self.kappa_1.astype(dtype)
        self.kappa_2 = self.kappa_2.astype(dtype)


# Register SystemState as a JAX PyTree
jtu.register_pytree_node(
    SystemState, SystemState.tree_flatten, SystemState.tree_unflatten
)

#### Configurations
# jax flags
# jax.config.update("jax_enable_x64", True)  #  MATLAB defaults to double precision
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

# gpu flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_autotune_level=4 "  # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_use_thunk_runtime=false "
)
devices = jax.devices()
print(devices)

rand_key = jr.key(123)
num_parallel_runs = 10
rand_keys = jr.split(rand_key, num_parallel_runs)


#### Parameters
N = 200
N_kappa = (N) ** 2
alpha = -0.28 * jnp.pi  # phase lag
beta = 0.66 * jnp.pi  # age parameter
a_1 = 1
epsilon_1 = 0.03  # adaption rate
epsilon_2 = 0.3  # adaption rate
sigma = 1
omega_1 = omega_2 = 0
T_init, T_trans, T_max = 0, 1000, 2000
T_step = 0.05

omega_1_i = jnp.ones((N,)) * omega_1
omega_2_i = jnp.ones((N,)) * omega_2

a_1_ij = jnp.ones((N, N)) * a_1
a_1_ij = a_1_ij.at[jnp.diag_indices(N)].set(0)  # NOTE no self coupling

args = (N, N_kappa, alpha, beta, epsilon_1, epsilon_2, sigma)


def make_deriv(jomega_1_i, jomega_2_i, ja_1_ij, batch_size):
    jN, jN_kappa, jalpha, jbeta, jepsilon_1, jepsilon_2, jsigma = (
        N,
        N_kappa,
        alpha,
        beta,
        epsilon_1,
        epsilon_2,
        sigma,
    )
    sin_beta = jnp.sin(beta)
    cos_beta = jnp.cos(beta)
    sin_alpha = jnp.sin(alpha)
    cos_alpha = jnp.cos(alpha)
    pi2 = jnp.pi * 2
    adj = 1 / (jN - 1)

    def single_system_deriv(
        t: float, y: SystemState, args: tuple = None
    ) -> tuple[jnp.ndarray]:
        # recover states from the py_tree
        phi_1_i, phi_2_i = y.phi_1 % pi2, y.phi_2 % pi2
        kappa_1_ij, kappa_2_ij = y.kappa_1.clip(-1, 1), y.kappa_2.clip(-1, 1)

        # sin/cos in radians
        sin_phi_1, cos_phi_1 = jnp.sin(phi_1_i), jnp.cos(phi_1_i)
        sin_phi_2, cos_phi_2 = jnp.sin(phi_2_i), jnp.cos(phi_2_i)
        sin_diff_phi_1 = jnp.einsum("i,j->ij", sin_phi_1, cos_phi_1) - jnp.einsum(
            "i,j->ij", cos_phi_1, sin_phi_1
        )
        cos_diff_phi_1 = jnp.einsum("i,j->ij", cos_phi_1, cos_phi_1) + jnp.einsum(
            "i,j->ij", sin_phi_1, sin_phi_1
        )

        sin_diff_phi_2 = jnp.einsum("i,j->ij", sin_phi_2, cos_phi_2) - jnp.einsum(
            "i,j->ij", cos_phi_2, sin_phi_2
        )
        cos_diff_phi_2 = jnp.einsum("i,j->ij", cos_phi_2, cos_phi_2) + jnp.einsum(
            "i,j->ij", sin_phi_2, sin_phi_2
        )

        sin_phi_1_diff_alpha = sin_diff_phi_1 * cos_alpha + cos_diff_phi_1 * sin_alpha
        dphi_1_i = (
            jomega_1_i
            - adj * jnp.einsum("ij,ij->i", (ja_1_ij + kappa_1_ij), sin_phi_1_diff_alpha)
            - jsigma * (sin_phi_1 * cos_phi_2 - cos_phi_1 * sin_phi_2)
        )

        dkappa_1_ij = -jepsilon_1 * (
            kappa_1_ij + sin_diff_phi_1 * cos_beta - cos_diff_phi_1 * sin_beta
        )
        dkappa_1_ij = dkappa_1_ij.at[jnp.diag_indices(jN)].set(0)

        sin_phi_2_diff_alpha = sin_diff_phi_2 * cos_alpha + cos_diff_phi_2 * sin_alpha
        dphi_2_i = (
            jomega_2_i
            - adj * jnp.einsum("ij,ij->i", kappa_2_ij, sin_phi_2_diff_alpha)
            - jsigma * (sin_phi_2 * cos_phi_1 - cos_phi_2 * sin_phi_1)
        )

        dkappa_2_ij = -jepsilon_2 * (
            kappa_2_ij + sin_diff_phi_2 * cos_beta - cos_diff_phi_2 * sin_beta
        )
        dkappa_2_ij = dkappa_2_ij.at[jnp.diag_indices(jN)].set(0)

        # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
        dy = SystemState(dphi_1_i, dphi_2_i, dkappa_1_ij, dkappa_2_ij)
        return dy

    # Vectorizing over batch
    batched = jit(vmap(single_system_deriv, in_axes=(None, 0, None)))

    return batched


def generate_init_conditions(key: jr.key) -> jnp.ndarray:
    phi_1_init = jr.uniform(key, (N)) * (2 * jnp.pi)
    phi_2_init = jr.uniform(key, (N)) * (2 * jnp.pi)

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


def enforce_bounds(y):
    return SystemState(
        phi_1=y.phi_1 % (2 * jnp.pi),
        phi_2=y.phi_2 % (2 * jnp.pi),
        kappa_1=jnp.clip(y.kappa_1, -1, 1),
        kappa_2=jnp.clip(y.kappa_2, -1, 1),
    )


@jit
def compress_save(t, y, args):
    return y.astype(dtype=jnp.float16)


def solve(batched_init_condition):
    # flat_init_condition = batched_init_condition.flatten()
    # print("BATCH SIZE", flat_init_condition.shape[0] // (2 * N + 2 * N_kappa))
    # print(flat_init_condition.shape)
    t = jnp.arange(T_trans, T_max, T_step)
    deriv = make_deriv(omega_1_i, omega_2_i, a_1_ij, num_parallel_runs)
    term = ODETerm(deriv)
    solver = Tsit5()
    stepsize_controller = ConstantStepSize()
    saveat = SaveAt(t0=True, ts=t, fn=compress_save)
    saveat = SaveAt(ts=[0, 1, T_max])
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


init_conditions = vmap(generate_init_conditions)(
    rand_keys
)  # shape: (num_parallel_runs, state_size)

sol = solve(init_conditions)
ys = enforce_bounds(sol.ys)
print(sol.ts)
print(ys.phi_1[-1][0])
# y_vals = jnp.concatenate(
#     [init_conditions[:, None, :], sol.ys], axis=1
# )  # shape: (num_parallel_runs, T_tot, state_size)

# print(y_vals.shape)  # shape: (num_parallel_runs, T_tot, state_size)

# t = jnp.arange(0, T_max)
# t_vals = jnp.concat([jnp.array([-1]), sol.ts]) + 1
# y_vals = jnp.concat([jnp.array([init_condition]), sol.ys])

# print(y_vals.shape)
# print(t_vals[-10:])
# print(y_vals[:, 0])

# dir_name = "test"
# np_save(f"{dir_name}/t_vals", t_vals)
# np_save(f"{dir_name}/y_vals", y_vals)

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
#     "T_tot": len(t_vals),
# }
# with open(f"{dir_name}/info.json", "w") as json_file:
#     json.dump(info, json_file)


# TODO
# save
# no globals? https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
