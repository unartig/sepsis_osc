import logging

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import filter_jit
from jax import vmap
from diffrax import (
    diffeqsolve,
    ODETerm,
    PIDController,
    SaveAt,
    Dopri5,  # Dopri5 same as MATLAB ode45
    Tsit5,
)

from sepsis_osc.dnm.data_classes import SystemConfig
from sepsis_osc.dnm.simulation import (
    generate_init_conditions_fixed,
    make_metric_save,
    system_deriv,
)
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
from sepsis_osc.utils.run_simulation import solve
from sepsis_osc.utils.logger import setup_logging


if __name__ == "__main__":
    setup_logging("info", console_log=True)
    logger = logging.getLogger(__name__)
    np.set_printoptions(suppress=True, linewidth=120)

    rand_key = jr.key(jax_random_seed)
    num_parallel_runs = 50
    rand_keys = jr.split(rand_key, num_parallel_runs)

    metric_save = make_metric_save(system_deriv)

    solver = Tsit5()
    term = ODETerm(system_deriv)
    stepsize_controller = PIDController(rtol=1e-3, atol=1e-6)

    beta_step = 0.02
    betas = np.arange(0.0, 0.5, beta_step)
    sigma_step = 0.04
    sigmas = np.arange(0.0, 2.0, sigma_step)
    alpha_step = 0.04
    alphas = jnp.array([-0.28])  # np.arange(-1.0, 1.0, alpha_step)
    T_max_base = 1000
    T_step_base = 10

    logger.info(
        f"Starting to map parameter space of "
        f"{len(betas)} beta, "
        f"{len(sigmas)} sigma, "
        f"{len(alphas)} alpha, "
        f"total {len(betas) * len(sigmas) * len(alphas)}"
    )

    overwrite = False
    db_str = "Steady"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    for alpha in alphas:
        for beta in betas:
            for sigma in sigmas:
                N = 100
                C = int(N * 0.2)
                run_conf = SystemConfig(
                    N=N,
                    C=C,  # local infection
                    omega_1=0.0,
                    omega_2=0.0,
                    a_1=1.0,
                    epsilon_1=0.03,  # adaption rate
                    epsilon_2=0.3,  # adaption rate
                    alpha=float(alpha),  # phase lage
                    beta=float(beta),  # age parameter
                    sigma=float(sigma),
                    T_init=0,
                    T_trans=0,
                    T_max=T_max_base,
                    T_step=T_step_base,
                )
                logger.info(f"New config {run_conf.as_index}")
                if not storage.read_result(run_conf.as_index, threshold=0.0) or overwrite:
                    logger.info("Starting solve")
                    generate_init_conditions = generate_init_conditions_fixed(run_conf.N, run_conf.beta, run_conf.C)

                    init_conditions = vmap(generate_init_conditions)(rand_keys)
                    # shape (num_parallel_runs, state)
                    sol = solve(
                        run_conf.T_init,
                        run_conf.T_trans,
                        run_conf.T_max,
                        run_conf.T_step,
                        init_conditions,
                        run_conf.as_args,
                        solver,
                        term,
                        stepsize_controller,
                        metric_save,
                        steady_state_check=True
                    )
                    logger.info(f"Solved in {sol.stats['num_steps']} steps")
                    if sol.ys:
                        logger.info("Saving Result")
                        storage.add_result(run_conf.as_index, sol.ys.copy(), overwrite=overwrite)
            storage.write()

