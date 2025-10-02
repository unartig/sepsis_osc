import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics, DynamicNetworkModel
from sepsis_osc.dnm.lie_dnm import LieDynamicNetworkModel


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
    from diffrax import Bosh3, Dopri5, Tsit5, Dopri8, ODETerm, PIDController
    from jax import vmap

    from sepsis_osc.utils.config import jax_random_seed

    rand_key = jr.key(jax_random_seed + 123)
    num_parallel_runs = 100

    #### Parameters
    N = 200
    run_conf = DNMConfig(
        N=N,
        C=0.2,  # local infection
        omega_1=0.0,
        omega_2=0.0,
        a_1=1.0,
        epsilon_1=0.03,  # adaption rate
        epsilon_2=0.3,  # adaption rate
        alpha=-0.28,  # phase lage
        beta=0.666,  # age parameter
        sigma=0.42,
        tau=0.55,
    )
    T_init = 0
    T_max = 500
    T_step = 10

    check = True
    # dnm = LieDynamicNetworkModel(full_save=False, steady_state_check=check)
    dnm = DynamicNetworkModel(full_save=False, steady_state_check=check)

    sol = dnm.integrate(
        solver=Tsit5(),
        config=run_conf,
        M=num_parallel_runs,
        key=rand_key,
        T_init=0,
        T_max=T_max,
        T_step=T_step,
    )

    if not sol.ys:
        exit(0)

    metrics = sol.ys
    metrics = metrics.remove_infs().add_follow_ups()

    ts = np.asarray(sol.ts).squeeze()
    ts = ts[~jnp.isinf(ts)]
    print(metrics.shape)

    tt = metrics.tt.max()
    print(tt)

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

    ax = plot_metric_t(metrics.m_1, metrics.m_2)
    ax[0].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[1].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[0].set_title("Mean Velo\nParenchymal Layer")
    ax[1].set_title("Immune Layer")

    ax = plot_metric_t(metrics.cq_1, metrics.cq_2)
    # ax[0].set_ylim([0, 1])
    # ax[1].set_ylim([0, 1])
    ax[0].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[1].vlines(tt, 0, 1, color="tab:red", ls=":")
    ax[0].set_title("Ensemble Average Coupling Entropy\nParenchymal Layer")
    ax[1].set_title("Immune Layer")
    plt.tight_layout()
    plt.show()
