import logging

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from simulation.data_classes import SystemConfig, SystemMetrics
from storage.storage_interface import Storage
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

size = (20, 20)
mat1 = jnp.zeros(size)
mat2 = jnp.zeros(size)
xs = np.linspace(0.4, 0.7, size[0])
ys = np.linspace(0, 1.5, size[1])
storage = Storage(
    key_dim=9,
    metrics_kv_name="storage/SepsisMetrics.db/",
    parameter_k_name="storage/SepsisParameters_index.bin",
)
params = np.ndarray((*size, 9))
metrix = SystemMetrics(
    r_1=mat1.copy(),
    r_2=mat2.copy(),
    m_1=mat1.copy(),
    m_2=mat2.copy(),
    s_1=mat1.copy(),
    s_2=mat2.copy(),
    q_1=mat1.copy(),
    q_2=mat2.copy(),
    f_1=mat1.copy(),
    f_2=mat2.copy(),
)
for x, beta in enumerate(xs):
    for y, sigma in enumerate(ys):
        N = 100
        run_conf = SystemConfig(
            N=100,
            C=int(0.2 * N),
            omega_1=0.0,
            omega_2=0.0,
            a_1=1.0,
            epsilon_1=0.03,
            epsilon_2=0.3,
            alpha=-0.28,
            beta=beta,
            sigma=sigma,
        )
        params[x, y] = np.array(run_conf.as_index)
        metrics = storage.read_result(run_conf.as_index, threshold=0.0)
        if metrics:

            metrix.r_1 = metrix.r_1.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.r_1)[-1, :]), -np.inf, np.inf))
            metrix.r_2 = metrix.r_2.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.r_2)[-1, :]), -np.inf, np.inf))
            metrix.s_1 = metrix.s_1.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.s_1)), -np.inf, np.inf))
            metrix.s_2 = metrix.s_2.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.s_2)), -np.inf, np.inf))
            metrix.m_1 = metrix.m_1.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.m_1)), -np.inf, np.inf))
            metrix.m_2 = metrix.m_2.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.m_2)), -np.inf, np.inf))
            metrix.q_1 = metrix.q_1.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.q_1)), -np.inf, np.inf))
            metrix.q_2 = metrix.q_2.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.q_2)), -np.inf, np.inf))
            metrix.f_1 = metrix.f_1.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.f_1)), -np.inf, np.inf))
            metrix.f_2 = metrix.f_2.at[-y, x].set(np.clip(np.mean(np.asarray(metrics.f_2)), -np.inf, np.inf))

storage.close()
print(metrix)
num_ticks = 5


def plot_metric(m1, m2, ax):
    ax1, ax2 = ax[0], ax[1]
    vmin = jnp.min(jnp.array([m1, m2]))
    vmax = jnp.max(jnp.array([m1, m2]))
    cax1 = ax1.matshow(m1, vmin=vmin, vmax=vmax, interpolation="none")
    cax2 = ax2.matshow(m2, vmin=vmin, vmax=vmax, interpolation="none")
    fig.colorbar(cax2, ax=ax, location="bottom", shrink=0.7)
    xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
    ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
    for a in ax:
        a.set_xticks(xtick_positions)
        a.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
        a.set_yticks(ytick_positions)
        a.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])


fig = plt.figure()
ax = fig.subplots(2, 5)

plot_metric(metrix.r_1, metrix.r_2, ax[:, 0])
ax[0, 0].set_title("Kuramoto Order Parameter R")
plot_metric(metrix.s_1, metrix.s_2, ax[:, 1])
ax[0, 1].set_title("Mean Phase Velocity Std")
plot_metric(metrix.m_1, metrix.m_2, ax[:, 2])
ax[0, 2].set_title("Mean Phase Velocity")
plot_metric(metrix.q_1, metrix.q_2, ax[:, 3])
ax[0, 3].set_title("Mean Entropy")
plot_metric(metrix.f_1, metrix.f_2, ax[:, 4])
ax[0, 4].set_title("Fraction of Cluster")

plt.show()
