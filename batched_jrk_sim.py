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
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
from numpy import save as np_save

import jax
import os


#### Configurations
# jax flags
jax.config.update("jax_enable_x64", True)  #  MATLAB defaults to double precision
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

# gpu flags
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=True '
    '--xla_gpu_autotune_level=2 '
    '--xla_cpu_multi_thread_eigen=False '
)

rand_key = jr.key(123)
num_parallel_runs = 1
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
    jN, jN_kappa, jalpha, jbeta, jepsilon_1, jepsilon_2, jsigma = N, N_kappa, alpha, beta, epsilon_1, epsilon_2, sigma
    pi2 = jnp.pi * 2

    @jit
    def deriv(t: float, y: jnp.ndarray, args=None) -> jnp.ndarray:

        # reshape y back into individual systems
        y = y.reshape(batch_size, -1)  # shape: (batch_size, state_size)

        # recover states from batched array and enforce bounds
        phi_1_i = y[:, 0 * jN : 1 * jN] % (pi2)
        phi_2_i = y[:, 1 * jN : 2 * jN] % (pi2)

        kappa_1_ij = y[:, 2 * jN + 0 * jN_kappa : 2 * jN + 1 * jN_kappa].clip(-1, 1).reshape(batch_size, jN, jN)
        kappa_2_ij = y[:, 2 * jN + 1 * jN_kappa : 2 * jN + 2 * jN_kappa].clip(-1, 1).reshape(batch_size, jN, jN)

        # kappa_1_ij = error_if(kappa_1_ij, jnp.sum(jnp.diagonal(kappa_1_ij, axis1=1, axis2=2)) != 0.0, "kappa1 diag non-zero")
        # kappa_2_ij = error_if(kappa_2_ij, jnp.sum(jnp.diagonal(kappa_2_ij, axis1=1, axis2=2)) != 0.0, "kappa2 diag non-zero")

        # sin/cos in radians
        phi_1_diff = phi_1_i[:, :, None] - phi_1_i[:, None, :]
    
        sin_phi_1_diff_alpha = jnp.sin(phi_1_diff + jalpha)
        dphi_1_i = (
            jomega_1_i[None, :]
            - (1 / (jN - 1)) * jnp.sum((ja_1_ij[None, :, :] + kappa_1_ij) * sin_phi_1_diff_alpha, axis=2)
            - jsigma * jnp.sin(phi_1_i - phi_2_i)
        )

        dkappa_1_ij = -jepsilon_1 * (kappa_1_ij + jnp.sin(phi_1_diff - jbeta))
        dkappa_1_ij = dkappa_1_ij.at[:, jnp.diag_indices(jN)].set(0)  # Zero diagonal

        phi_2_diff = phi_2_i[:, :, None] - phi_2_i[:, None, :]
        sin_phi_2_diff_alpha = jnp.sin(phi_2_diff + jalpha)
        dphi_2_i = (
            jomega_2_i[None, :]
            - (1 / (jN - 1)) * jnp.sum((ja_1_ij[None, :, :] + kappa_1_ij) * sin_phi_2_diff_alpha, axis=2)
            - jsigma * jnp.sin(phi_2_i - phi_1_i)
        )

        dkappa_2_ij = -jepsilon_2 * (kappa_2_ij + jnp.sin(phi_2_diff - jbeta))
        dkappa_2_ij = dkappa_2_ij.at[:, jnp.diag_indices(jN)].set(0)  # Zero diagonal

        # return batched and stacked and flattened array (batch_size, (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
        dy = jnp.concatenate([dphi_1_i, dphi_2_i, dkappa_1_ij.reshape(batch_size, -1), dkappa_2_ij.reshape(batch_size, -1)], axis=1).flatten()
        return dy
    return deriv

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

    return jnp.concatenate(
        [
            phi_1_init,
            phi_2_init,
            kappa_1_init.flatten(),
            kappa_2_init.flatten(),
        ]
    )


def solve(batched_init_condition):
    flat_init_condition = batched_init_condition.flatten()
    print("BATCH SIZE", flat_init_condition.shape[0] // (2 * N + 2 * N_kappa))
    print(flat_init_condition.shape)
    t = jnp.arange(T_trans, T_max)
    deriv = make_deriv(omega_1_i, omega_2_i, a_1_ij, num_parallel_runs)
    term = ODETerm(deriv)
    solver = Dopri5()
    stepsize_controller = ConstantStepSize()
    saveat = SaveAt(ts=t, dense=False)  # potentially use fn=...
    # saveat = SaveAt(ts=[0])
    res = diffeqsolve(
        term,
        solver,
        y0=flat_init_condition,
        args=args,
        t0=T_init,
        t1=T_max,
        dt0=T_step,
        stepsize_controller=stepsize_controller,
        max_steps=int(T_max / T_step)+1,
        progress_meter=TqdmProgressMeter(),
        saveat=saveat,
    )
    return res

init_conditions = vmap(generate_init_conditions)(rand_keys)  # shape: (num_parallel_runs, state_size)

sol = solve(init_conditions)

t_vals = jnp.concatenate([jnp.array([-1]), sol.ts]) + 1
y_vals = jnp.concatenate([init_conditions[:, None, :], sol.ys], axis=1)  # shape: (num_parallel_runs, T_tot, state_size)

print(y_vals.shape)  # shape: (num_parallel_runs, T_tot, state_size)

# t = jnp.arange(0, T_max)
# t_vals = jnp.concat([jnp.array([-1]), sol.ts]) + 1
# y_vals = jnp.concat([jnp.array([init_condition]), sol.ys])

# print(y_vals.shape)
print(t_vals[-10:])
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
