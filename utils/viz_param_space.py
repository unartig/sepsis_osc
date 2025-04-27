import logging

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from simulation.data_classes import SystemConfig, SystemMetrics
from storage.storage_interface import Storage
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

xs_step = 0.00303030303030305
ys_step = 0.01515151515151515
xs = np.arange(0.0, 1.5, xs_step)
ys = np.arange(0.0, 2.0, ys_step)

orig_xs = [np.argmin(np.abs(xs - x)) for x in [0.4, 0.7]]
orig_ys = [len(ys) - np.argmin(np.abs(ys - y)) - 1 for y in [0.0, 1.5]]

size = (len(ys), len(xs))
db_str = ""  # other/Tiny"
storage = Storage(
    key_dim=9,
    metrics_kv_name=f"storage/{db_str}SepsisMetrics.db/",
    parameter_k_name=f"storage/{db_str}SepsisParameters_index.bin",
    use_mem_cache=True,
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
            beta=float(beta),
            sigma=float(sigma),
        )
        params[-y, x] = np.array(run_conf.as_index)

metrix = storage.read_multiple_results(params)
storage.close()
if not metrix:
    exit(0)
num_ticks = 10


def pretty_plot(metric_parenchymal, metric_immune, title, filename, figure_dir, fs=(8, 6), show=False):
    fig, axes = plt.subplots(2, 1, figsize=fs)

    # Enhanced heatmap plotting
    im0 = axes[0].imshow(metric_parenchymal, aspect="auto", cmap="viridis")
    im1 = axes[1].imshow(metric_immune, aspect="auto", cmap="viridis")

    # Clearer titles and labels
    axes[0].set_title(f"{title}\nParenchymal Layer", fontsize=14)
    axes[1].set_title(f"{title}\nImmune Layer", fontsize=14)
    axes[0].set_ylabel(r"$\sigma$", fontsize=12)
    axes[1].set_ylabel(r"$\sigma$", fontsize=12)
    axes[1].set_xlabel(r"$\beta / \pi$", fontsize=12)
    axes[0].set_xlabel(r"$\beta / \pi$", fontsize=12)

    xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
    ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
    for a in axes:
        a.set_xticks(xtick_positions)
        a.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
        a.set_yticks(ytick_positions)
        a.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])
        a.plot(orig_xs, [orig_ys[0], orig_ys[0]], color="white", linewidth=0.5)
        a.plot(orig_xs, [orig_ys[1], orig_ys[1]], color="white", linewidth=0.5)
        a.plot([orig_xs[0], orig_xs[0]], orig_ys, color="white", linewidth=0.5)
        a.plot([orig_xs[1], orig_xs[1]], orig_ys, color="white", linewidth=0.5)

    # Add a colorbar to each subplot
    fig.colorbar(im0, ax=axes[0], location="right", shrink=0.8)
    fig.colorbar(im1, ax=axes[1], location="right", shrink=0.8)

    plt.tight_layout()  # Adjust layout
    plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close(fig)


print(metrix.sr_1.shape)
print(metrix.sr_2.sum() / metrix.sr_2.size)
log = False
show = False
figure_dir = "figures"
fs = (8, 8)
pretty_plot(metrix.r_1, metrix.r_2, "Kuramoto Order Parameter R", "kuramoto_beta_sigma", figure_dir, fs, show)
pretty_plot(metrix.sr_1, metrix.sr_2, "Splay Ratio", "splay_ratio_beta_sigma", figure_dir, fs, show)
pretty_plot(metrix.s_1, metrix.s_2, "Mean Phase Velocity Std", "std_beta_sigma", figure_dir, fs, show)
pretty_plot(metrix.m_1, metrix.m_2, "Mean Phase Velocity", "mean_beta_sigma", figure_dir, fs, show)
pretty_plot(metrix.q_1, metrix.q_2, "Entropy", "entropy_beta_sigma", figure_dir, fs, show)
pretty_plot(metrix.f_1, metrix.f_2, "Cluster Fraction", "cluster_beta_sigma", figure_dir, fs, show)
