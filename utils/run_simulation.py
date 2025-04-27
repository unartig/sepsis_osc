import logging

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import filter_jit
from jax import vmap
from diffrax import (
    _step_size_controller,
    diffeqsolve,
    ODETerm,
    ConstantStepSize,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    # solver - from most to least accurate / slow to fast
    Dopri8,
    Dopri5,  # Dopri5 same as MATLAB ode45
    Tsit5,
    Bosh3,
    Ralston,
)

from simulation.data_classes import SystemConfig
from simulation.simulation import (
    generate_init_conditions_fixed,
    make_full_compressed_save,
    make_metric_save,
    system_deriv,
)
from storage.storage_interface import Storage
from utils.config import jax_random_seed
from utils.logger import setup_logging


@filter_jit
def solve(
    T_init,
    T_trans,
    T_max,
    T_step,
    batched_init_condition,
    args,
    solver,
    term,
    stepsize_controller,
    save_method,
):
    result = diffeqsolve(
        term,
        solver,
        y0=batched_init_condition,
        args=args,
        t0=T_init,
        t1=T_max,
        dt0=T_step,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e12),
        saveat=SaveAt(t0=True, ts=jnp.arange(T_trans, T_max, T_step), fn=save_method),
        progress_meter=TqdmProgressMeter(),
    )
    return result


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    np.set_printoptions(suppress=True, linewidth=120)

    rand_key = jr.key(jax_random_seed)
    num_parallel_runs = 20
    rand_keys = jr.split(rand_key, num_parallel_runs)

    metric_save = make_metric_save(system_deriv)

    solver = Dopri5()
    term = ODETerm(system_deriv)

    xs_step = 0.00303030303030305
    xs = np.arange(0.0, 1.5, xs_step)
    ys_step = 0.01515151515151515
    ys = np.arange(0.0, 2.0, ys_step)
    db_str = ""
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"storage/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"storage/{db_str}SepsisParameters_index.bin",
        use_mem_cache=False,
    )
    for c_frac in [0.2]:
        for x, beta in enumerate(xs):
            for y, sigma in enumerate(ys):
                N = 100
                C = int(N * c_frac)
                run_conf = SystemConfig(
                    N=N,
                    C=int(c_frac * N),  # local infection
                    omega_1=0.0,
                    omega_2=0.0,
                    a_1=1.0,
                    epsilon_1=0.03,  # adaption rate
                    epsilon_2=0.3,  # adaption rate
                    alpha=-0.28,  # phase lage
                    beta=float(beta),  # age parameter
                    sigma=float(sigma),
                    T_init=0,
                    T_trans=150,
                    T_max=200,
                    T_step=0.1,
                )
                logger.info(f"New config {run_conf.as_index}")
                if not storage.read_result(run_conf.as_index, threshold=0.0):
                    logger.info("Starting solve")
                    generate_init_conditions = generate_init_conditions_fixed(run_conf.N, run_conf.beta, run_conf.C)

                    stepsize_controller = PIDController(rtol=1e-4, atol=1e-7)
                    init_conditions = vmap(generate_init_conditions)(rand_keys)
                    # shape (num_parallel_runs, state)
                    sol = solve(
                        run_conf.T_init,
                        run_conf.T_trans,
                        run_conf.T_max,
                        run_conf.T_step,
                        init_conditions.copy(),
                        run_conf.as_args,
                        solver,
                        term,
                        stepsize_controller,
                        metric_save,
                    )
                    logger.info(f"Solved in {sol.stats['num_steps']} steps")
                    if sol.ys:
                        logger.info("Saving Result")
                        storage.add_result(run_conf.as_index, sol.ys.copy(), overwrite=False)
            storage.write()
