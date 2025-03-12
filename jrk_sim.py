import json

import jax

from diffrax import (
    diffeqsolve,
    ODETerm,
    Dopri5,  # Dopri5 same as MATLAB ode45
    SaveAt,
    TqdmProgressMeter,
)
from equinox import error_if
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from numpy import save as np_save

jax.config.update("jax_enable_x64", True)  #  MATLAB defaults to double precision
# jax.config.update("jax_platform_name", "cpu")

rand_key = jr.key(123)

N = 200
N_kappa = (N) ** 2
alpha = -0.28 * jnp.pi  # phase lag
beta = 0.66 * jnp.pi  # age parameter
a_1 = 1
epsilon_1 = 0.03  # adaption rate
epsilon_2 = 0.3  # adaption rate
sigma = 1
omega_1 = omega_2 = 0
T_init, T_dyn, T_trans, T_max = 0, 0, 1000, 2000
T_step = 0.05

omega_1_i = jnp.ones((N,)) * omega_1
omega_2_i = jnp.ones((N,)) * omega_2

a_1_ij = jnp.ones((N, N)) * a_1
a_1_ij = jnp.fill_diagonal(a_1_ij, 0, inplace=False)  # NOTE no self coupling

@jit
def deriv(t: float, y: jnp.ndarray, args=None) -> jnp.ndarray:
    # expect 1d stacked array (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN))
    # recover states from 1d array and enforce bounds
    phi_1_i = y[0 * N : 1 * N] % (jnp.pi * 2)
    phi_2_i = y[1 * N : 2 * N] % (jnp.pi * 2)

    kappa_1_ij = y[2 * N + 0 * N_kappa : 2 * N + 1 * N_kappa].reshape(N, N).clip(-1, 1)
    kappa_2_ij = y[2 * N + 1 * N_kappa : 2 * N + 2 * N_kappa].reshape(N, N).clip(-1, 1)

    kappa_1_ij = error_if(kappa_1_ij, jnp.sum(jnp.diag(kappa_1_ij)) != 0.0, "kappa1 diag non-zero")
    kappa_2_ij = error_if(kappa_2_ij, jnp.sum(jnp.diag(kappa_2_ij)) != 0.0, "kappa2 diag non-zero")

    # track bounded variables
    y = y.at[0 * N : 1 * N].set(phi_1_i)
    y = y.at[1 * N : 2 * N].set(phi_2_i)
    y = y.at[2 * N + 0 * N_kappa : 2 * N + 1 * N_kappa].set(kappa_1_ij.flatten())
    y = y.at[2 * N + 1 * N_kappa : 2 * N + 2 * N_kappa].set(kappa_2_ij.flatten())

    # sin/cos in radians
    phi_1_diff = phi_1_i[:, jnp.newaxis] - phi_1_i
    dphi_1_i = (
        omega_1_i
        - (1 / (N - 1)) * jnp.sum((a_1_ij + kappa_1_ij) * jnp.sin(phi_1_diff + alpha), axis=1)
        - sigma * jnp.sin(phi_1_i - phi_2_i)
    )
    dkappa_1_ij = -epsilon_1 * (kappa_1_ij + jnp.sin(phi_1_diff - beta))
    dkappa_1_ij = dkappa_1_ij.at[jnp.diag_indices(N)].set(0)  # Zero diagonal

    phi_2_diff = phi_2_i[:, jnp.newaxis] - phi_2_i
    dphi_2_i = (
        omega_2_i
        - (1 / (N - 1)) * jnp.sum((a_1_ij + kappa_1_ij) * jnp.sin(phi_2_diff + alpha), axis=1)
        - sigma * jnp.sin(phi_2_i - phi_1_i)
    )
    dkappa_2_ij = -epsilon_2 * (kappa_2_ij + jnp.sin(phi_2_diff - beta))
    dkappa_2_ij = dkappa_2_ij.at[jnp.diag_indices(N)].set(0)  # Zero diagonal

    # return 1d stacked array (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN))
    dy = jnp.concatenate([dphi_1_i, dphi_2_i, dkappa_1_ij.flatten(), dkappa_2_ij.flatten()])
    return dy


phi_1_init = jr.uniform(rand_key, (N)) * (2 * jnp.pi)
phi_2_init = jr.uniform(rand_key, (N)) * (2 * jnp.pi)

# kappaIni1 = sin(parStruct.beta(1))*ones(N*N,1)+0.01*(2*rand(N*N,1)-1);
# kappa_1_init = jnp.sin(beta)*jnp.ones((N, N))+0.01*(2*jr.uniform(rand_key, (N, N)) - 1);
kappa_1_init = jr.uniform(rand_key, (N, N)) * 2 - 1
kappa_1_init = jnp.fill_diagonal(kappa_1_init, 0, inplace=False)

kappa_2_init = jnp.ones((N, N))
kappa_2_init = jnp.fill_diagonal(kappa_2_init, 0, inplace=False)
kappa_2_init = kappa_2_init.at[40:, :40].set(0)
kappa_2_init = kappa_2_init.at[:40, 40:].set(0)

init_condition = jnp.concatenate(
    [
        phi_1_init,
        phi_2_init,
        kappa_1_init.flatten(),
        kappa_2_init.flatten(),
    ]
)


# @jit
def solve():
    t = jnp.arange(T_trans, T_max)
    term = ODETerm(deriv)
    solver = Dopri5()
    saveat = SaveAt(ts=t)
    res = diffeqsolve(
        term,
        solver,
        t0=T_init,
        t1=T_max,
        dt0=T_step,
        y0=init_condition,
        max_steps=int(T_max / T_step)+1,
        progress_meter=TqdmProgressMeter(),
        saveat=saveat,
    )
    return res


sol = solve()
print(sol)

t = jnp.arange(0, T_max)
t_vals = jnp.concat([jnp.array([-1]), sol.ts]) + 1
y_vals = jnp.concat([jnp.array([init_condition]), sol.ys])

print(y_vals.shape)
print(t_vals)
print(y_vals[:, 0])

dir_name = "test"
np_save(f"{dir_name}/t_vals", t_vals)
np_save(f"{dir_name}/y_vals", y_vals)

info = {
    "N": N,
    "N_kappa": N_kappa,
    "alpha": alpha,
    "beta": beta,
    "a_1": a_1,
    "epsilon_1": epsilon_1,
    "epsilon_2": epsilon_2,
    "sigma": sigma,
    "omega_1": omega_1,
    "omega_2": omega_2,
    "T_init": T_init,
    "T_dyn": T_dyn,
    "T_max": T_max,
    "T_step": T_step,
    "T_tot": len(t_vals),
}
with open(f"{dir_name}/info.json", "w") as json_file:
    json.dump(info, json_file)


# TODO
# save
# no globals? https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
