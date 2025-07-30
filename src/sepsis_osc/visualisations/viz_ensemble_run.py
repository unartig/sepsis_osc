import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

from sepsis_osc.dnm.data_classes import SystemConfig, SystemMetrics


def plot_metric_t(var_1: np.ndarray, var_2: np.ndarray, ax=None):
    if not ax:
        _, ax = plt.subplots(2, 1)
    t = np.arange(var_1.shape[0])
    # t = t / t.max() * run_conf.T_max
    ax[0].plot(t, var_1)
    ax[1].plot(t, var_2)
    return ax


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax.random as jr
    from diffrax import Bosh3, Dopri5, Tsit5, Dopri8, ODETerm, PIDController
    from jax import vmap

    from sepsis_osc.utils.run_simulation import solve
    from sepsis_osc.dnm.simulation import (
        generate_init_conditions_fixed,
        make_metric_save,
        system_deriv,
    )
    from sepsis_osc.utils.config import jax_random_seed

    rand_key = jr.key(jax_random_seed + 123)
    num_parallel_runs = 50
    rand_keys = jr.split(rand_key, num_parallel_runs)
    metric_save = make_metric_save(system_deriv)
    term = ODETerm(system_deriv)
    solver = Tsit5()
    stepsize_controller = PIDController(dcoeff=0, rtol=1e-3, atol=1e-6)

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
        beta=1.0,  # age parameter
        sigma=1.0,
        T_init=0,
        T_trans=0,
        T_max=1000,
        T_step=10,
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
        metric_save,
    )
    if not sol.ys or not run_conf.T_max:
        exit(0)
    metrics = sol.ys
    metrics = metrics.add_follow_ups()
    ts = np.asarray(sol.ts).squeeze()
    ts = ts[~jnp.isinf(ts)]
    print(metrics.shape)
    # exit(0)

    # calculate transient time
    last_x = metrics.r_1[-int(0.3 * run_conf.T_max) :]
    # tt = np.where(np.max(np.abs(metrics.r_1 - last_x.mean(axis=0)), axis=-1) > 0.1)[0].max()
    tt = metrics.tt.max()
    print(tt)
    # tt = int(tt[-1] / metrics.r_1.shape[0] * run_conf.T_max)

    ax = plot_metric_t(metrics.s_1, metrics.s_2)
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[0].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[1].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[0].set_title("Average Standard Deviation\nParenchymal Layer")
    ax[1].set_title("Immune Layer")

    ax = plot_metric_t(metrics.r_1, metrics.r_2)
    ax[0].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[1].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[0].set_title("Kuramoto Order Parameter\nParenchymal Layer")
    ax[1].set_title("Immune Layer")

    plt.tight_layout()
    plt.show()

    # print(self.f_1.mean(axis=-1))
    # if self.f_1.mean(axis=-1) < 0.3:
    #     std_ = self.r_1.std(axis=1)
    #     threshold = std_.mean() + 2 * std_[50:].std()
    #     self.tt = jnp.argmax(std_ > threshold) + 1
    # else:
    #     window = 5
    #     diff = jnp.abs(jnp.diff(self.r_1.mean(axis=1)))
    #     smoothed = jnp.convolve(diff, np.ones(window) / window, mode="valid")
    #     threshold = 0.01
    #     self.tt = jnp.argmax(smoothed < threshold) + 1
