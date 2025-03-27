import logging

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from diffrax import (
    diffeqsolve,
    ODETerm,
    RecursiveCheckpointAdjoint,
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
from equinox import filter_jit
import numpy as np


from config import jax_random_seed
from simulation import (
    build_args,
    system_deriv,
    make_full_compressed_save,
    make_metric_save,
    generate_init_conditions_fixed,
    SystemState,
    SystemMetrics,
)
from storage import Storage

from logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True)

rand_key = jr.key(jax_random_seed)
num_parallel_runs = 50
rand_keys = jr.split(rand_key, num_parallel_runs)

#### Parameters
N = 20
N_kappa = N**2
alpha = -0.28 * jnp.pi  # phase lag
beta = 0.66 * jnp.pi  # age parameter
a_1 = 1.0
epsilon_1 = 0.03  # adaption rate
epsilon_2 = 0.3  # adaption rate
sigma = 1.0
T_init, T_trans, T_max = 0, 100, 200
T_step = 0.05
omega_1 = omega_2 = 0.0
C = 40  # local infection


deriv = system_deriv
metric_save = make_metric_save(system_deriv)


@filter_jit
def solve(
    batched_init_condition,
    args,
    deriv,
):
    term = ODETerm(deriv)
    solver = Bosh3()
    stepsize_controller = ConstantStepSize()
    transient = diffeqsolve(
        term,
        solver,
        y0=batched_init_condition,
        args=args,
        t0=T_init,
        t1=T_trans,
        dt0=T_step,
        stepsize_controller=stepsize_controller,
        max_steps=int(T_trans / T_step) + 1,
        progress_meter=TqdmProgressMeter(),
        saveat=SaveAt(t1=True, solver_state=True),
    )
    if not transient.ys:
        logger.error("Failed Transient Calculation. Aborting...")
        exit(0)
    result = diffeqsolve(
        term,
        solver,
        y0=transient.ys.squeeze().enforce_bounds(),
        solver_state=transient.solver_state,
        args=args,
        t0=T_trans,
        t1=T_max,
        dt0=T_step,
        stepsize_controller=stepsize_controller,
        max_steps=int((T_max - T_trans) / T_step),
        progress_meter=TqdmProgressMeter(),
        saveat=SaveAt(t0=True, t1=True, steps=True, fn=metric_save),
    )
    return result


storage = Storage(key_dim=9)
params = None
for C in [39, 40 , 41]:
    args = build_args(N, omega_1, omega_2, a_1, epsilon_1, epsilon_2, alpha, beta, sigma)
    generate_init_conditions = generate_init_conditions_fixed(N, beta, C)

    init_conditions = vmap(generate_init_conditions)(rand_keys)
    # shape (num_parallel_runs, state)
    sol = solve(init_conditions, args, deriv)
    if sol.ys:
        params =   (
                omega_1,
                omega_2,
                a_1,
                epsilon_1,
                epsilon_2,
                alpha / jnp.pi,
                beta / jnp.pi,
                sigma,
                C,
            )
        storage.add_result(params, sol.ys)
        print(C, sol.ys.r_1.shape)
        print(sol.ts)

storage.close()

if params:
    storage1 = Storage()
    res = storage1.read_result(params)
    if res:
        print(params[-1], res.r_1[-1])

