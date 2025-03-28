import logging

import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
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
from equinox import filter_jit
import numpy as np


from utils.config import jax_random_seed
from utils.logger import setup_logging
from simulation import (
    system_deriv,
    make_full_compressed_save,
    make_metric_save,
    generate_init_conditions_fixed,
    SystemConfig,
)
from storage.storage_interface import Storage


setup_logging()
logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True)

rand_key = jr.key(jax_random_seed)
num_parallel_runs = 50
rand_keys = jr.split(rand_key, num_parallel_runs)

#### Parameters
N = 200
N_kappa = N**2
alpha = -0.28  # phase lag
beta = 0.66  # age parameter
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

    # from 0 to t_transient
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
    # from t_transient + step_size to t_max
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


storage = Storage()
for beta in np.linspace(0.4, 0.7, 50):
    for sigma in np.linspace(0, 1.5, 50):
        run_conf = SystemConfig(
            N=N,
            C=C,
            omega_1=omega_1,
            omega_2=omega_2,
            a_1=a_1,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
        )
        if not storage.read_result(run_conf.as_index, threshold=0.0):
            generate_init_conditions = generate_init_conditions_fixed(run_conf.N, run_conf.beta, run_conf.C)

            init_conditions = vmap(generate_init_conditions)(rand_keys)
            # shape (num_parallel_runs, state)
            sol = solve(init_conditions, run_conf.as_args, deriv)
            if sol.ys:
                storage.add_result(run_conf.as_index, sol.ys)

            storage.close()
            storage = Storage()
