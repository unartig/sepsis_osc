import logging

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from simulation.data_classes import SystemConfig, SystemMetrics
from storage.storage_interface import Storage
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

size = (100, 100)
xs = np.linspace(0.4, 0.7, size[0])
ys = np.linspace(0.0, 1.5, size[1])
storage = Storage(
    key_dim=9,
    metrics_kv_name="storage/SepsisMetrics.db/",
    parameter_k_name="storage/SepsisParameters_index.bin",
    use_mem_cache=False,
)
params = np.ndarray((*size, 9))


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
        params[-y, x] = np.array(run_conf.as_index)

metrix = storage.read_multiple_results(params)
if not metrix:
    exit(0)
metrix.r_1 = np.clip(np.mean(np.asarray(metrix.r_1)[:, :, -1, :], axis=(-1,)), -np.inf, np.inf)
metrix.r_2 = np.clip(np.mean(np.asarray(metrix.r_2)[:, :, -1, :], axis=(-1,)), -np.inf, np.inf)
metrix.s_1 = np.clip(np.mean(np.asarray(metrix.s_1), axis=-1), -np.inf, np.inf)
metrix.s_2 = np.clip(np.mean(np.asarray(metrix.s_2), axis=-1), -np.inf, np.inf)
metrix.m_1 = np.clip(np.mean(np.asarray(metrix.m_1), axis=-1), -np.inf, np.inf)
metrix.m_2 = np.clip(np.mean(np.asarray(metrix.m_2), axis=-1), -np.inf, np.inf)
metrix.q_1 = np.clip(np.mean(np.asarray(metrix.q_1), axis=-1), -np.inf, np.inf)
metrix.q_2 = np.clip(np.mean(np.asarray(metrix.q_2), axis=-1), -np.inf, np.inf)
metrix.f_1 = np.clip(np.mean(np.asarray(metrix.f_1), axis=-1), -np.inf, np.inf)
metrix.f_2 = np.clip(np.mean(np.asarray(metrix.f_2), axis=-1), -np.inf, np.inf)
storage.close()
num_ticks = 5


def plot_metric(m1, m2, ax, log=False):
    ax1, ax2 = ax[0], ax[1]
    fig = ax1.figure

    if log:
        norm1 = LogNorm(vmin=float(np.min(m1)), vmax=float(np.max(m1)))
        norm2 = LogNorm(vmin=float(np.min(m2)), vmax=float(np.max(m2)))
        cax1 = ax1.matshow(m1, interpolation="none", norm=norm1)
        cax2 = ax2.matshow(m2, interpolation="none", norm=norm2)
    else:
        cax1 = ax1.matshow(m1, vmin=float(np.min(m1)), vmax=float(np.max(m1)), interpolation="none")
        cax2 = ax2.matshow(m2, vmin=float(np.min(m2)), vmax=float(np.max(m2)), interpolation="none")

    # Add individual colorbars for each subplot
    fig.colorbar(cax1, ax=ax1, location="bottom", shrink=0.7)
    fig.colorbar(cax2, ax=ax2, location="bottom", shrink=0.7)

    # Set tick labels
    xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
    ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)

    for a in ax:
        a.set_xticks(xtick_positions)
        a.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
        a.set_yticks(ytick_positions)
        a.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])


fig = plt.figure()
ax = fig.subplots(2, 5)

log = False
plot_metric(metrix.r_1, metrix.r_2, ax[:, 0], log=log)
ax[0, 0].set_title("Kuramoto Order Parameter R")
ax[0, 0].set_ylabel("Parenchymal Layer\nsigma")
ax[1, 0].set_ylabel("Immune Layer\nsigma")
plot_metric(metrix.s_1, metrix.s_2, ax[:, 1], log=log)
ax[0, 1].set_title("Mean Phase Velocity Std")
plot_metric(metrix.m_1, metrix.m_2, ax[:, 2], log=log)
ax[0, 2].set_title("Mean Phase Velocity")
plot_metric(metrix.q_1, metrix.q_2, ax[:, 3], log=log)
ax[0, 3].set_title("Mean Entropy")
plot_metric(metrix.f_1, metrix.f_2, ax[:, 4], log=log)
ax[0, 4].set_title("Fraction of Cluster")

plt.show()
