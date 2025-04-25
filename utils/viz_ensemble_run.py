import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

from simulation.data_classes import SystemConfig, SystemMetrics


def plot_metric_t(var_1: np.ndarray, var_2: np.ndarray, ax=None):
    if not ax:
        _, ax = plt.subplots(2, 1)
    t = np.arange(var_1.shape[0])
    ax[0].plot(t, var_1)
    ax[1].plot(t, var_2)
    return ax


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax.random as jr
    from diffrax import Bosh3, Dopri5, Dopri8, ODETerm, PIDController
    from jax import vmap

    from utils.run_simulation import solve
    from simulation.simulation import (
        generate_init_conditions_fixed,
        make_metric_save,
        system_deriv,
    )
    from utils.config import jax_random_seed

    rand_key = jr.key(jax_random_seed + 123)
    num_parallel_runs = 100
    rand_keys = jr.split(rand_key, num_parallel_runs)
    full_save = make_metric_save(system_deriv)
    term = ODETerm(system_deriv)
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-4, atol=1e-7, dtmax=0.1)

    #### Parameters
    N = 200
    run_conf = SystemConfig(
        N=N,
        C=int(0.2 * N),  # local infection
        omega_1=0.0,
        omega_2=0.0,
        a_1=1.0,
        epsilon_1=0.03,  # adaption rate
        epsilon_2=0.3,  # adaption rate
        alpha=-0.28,  # phase lage
        beta=0.83,  # age parameter
        sigma=1.1,
        T_init=0,
        T_trans=0,
        T_max=25,
        T_step=0.1,
    )
    generate_init_conditions = generate_init_conditions_fixed(run_conf.N, run_conf.beta, run_conf.C)

    init_conditions = vmap(generate_init_conditions)(rand_keys)
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
        full_save,
    )
    if not sol.ys:
        exit(0)
    metrics = sol.ys
    metrics = metrics.add_follow_ups()
    ts = np.asarray(sol.ts).squeeze()
    ts = ts[~jnp.isinf(ts)]
    print(metrics.f_1.shape)
    print(metrics.s_1.shape)
    print(metrics.r_1.shape)
    ax = plot_metric_t(metrics.r_1, metrics.r_2)
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[0].set_title("Kuramoto Order Parameter\nParenchymal Layer")
    ax[1].set_title("Immune Layer")
    plt.tight_layout()

    plt.show()
