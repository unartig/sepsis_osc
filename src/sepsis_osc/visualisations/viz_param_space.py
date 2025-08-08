import logging

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from sepsis_osc.dnm.data_classes import SystemConfig, SystemMetrics
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.ldm.model_utils import as_2d_indices

setup_logging()
logger = logging.getLogger(__name__)

num_ticks = 10


def plot_tt(tt, title, filename, figure_dir, fs=(8, 6), show=False):
    fig, ax = plt.subplots(figsize=fs)
    im = ax.imshow(tt, aspect="auto", cmap="viridis")
    ax.set_title(f"{title}\nParenchymal Layer", fontsize=14)
    ax.set_ylabel(r"$\sigma$", fontsize=12)
    ax.set_xlabel(r"$\beta / \pi$", fontsize=12)
    xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
    ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])
    ax.plot(orig_xs, [orig_ys[0], orig_ys[0]], color="white", linewidth=0.5)
    ax.plot(orig_xs, [orig_ys[1], orig_ys[1]], color="white", linewidth=0.5)
    ax.plot([orig_xs[0], orig_xs[0]], orig_ys, color="white", linewidth=0.5)
    ax.plot([orig_xs[1], orig_xs[1]], orig_ys, color="white", linewidth=0.5)
    fig.colorbar(im, ax=ax, location="right", shrink=0.8)
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close(fig)

def space_plot(metric, xs, ys, title, filename, figure_dir, fs=(6, 4)):
    fig, ax = plt.subplots(1, 1, figsize=fs)

    # Enhanced heatmap plotting
    im = ax.imshow(metric.T[::-1, :], cmap="viridis")

    # Clearer titles and labels
    ax.set_title(f"{title}\nParenchymal Layer", fontsize=14)
    ax.set_ylabel(r"$\sigma$", fontsize=12)
    ax.set_xlabel(r"$\beta / \pi$", fontsize=12)

    xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
    ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])

    cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.8)
    cbar.set_label(r"$s^{mu}$")
    return ax

def pretty_plot(metric_parenchymal, metric_immune, title, filename, figure_dir, fs=(8, 6), show=False):
    fig, axes = plt.subplots(2, 1, figsize=fs)

    # Enhanced heatmap plotting
    im0 = axes[0].imshow(metric_parenchymal.T[::-1, :], aspect="auto", cmap="viridis")
    im1 = axes[1].imshow(metric_immune.T[::-1, :], aspect="auto", cmap="viridis")

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
        a.plot([orig_xs[0], orig_xs[1]], [orig_ys[0], orig_ys[0]], color="white", linewidth=0.5)
        a.plot([orig_xs[0], orig_xs[1]], [orig_ys[1], orig_ys[1]], color="white", linewidth=0.5)
        a.plot([orig_xs[0], orig_xs[0]], [orig_ys[0], orig_ys[1]], color="white", linewidth=0.5)
        a.plot([orig_xs[1], orig_xs[1]], [orig_ys[0], orig_ys[1]], color="white", linewidth=0.5)

    # Add a colorbar to each subplot
    fig.colorbar(im0, ax=axes[0], location="right", shrink=0.8)
    fig.colorbar(im1, ax=axes[1], location="right", shrink=0.8)

    plt.tight_layout()  # Adjust layout
    plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":

    # ALPHA_SPACE = (-0.52, -0.52, 1.0)
    BETA_SPACE = (0.2, 2.0, 0.02)
    SIGMA_SPACE = (0.0, 2.0, 0.04)
    xs = np.arange(*BETA_SPACE)
    ys = np.arange(*SIGMA_SPACE)
    
    orig_xs = [np.argmin(np.abs(xs - x)) for x in [0.4, 0.7]]
    orig_ys = [len(ys) - np.argmin(np.abs(ys - y)) - 1 for y in [0.0, 1.5]]
    
    db_str = "Steady1_3"  # other/Tiny"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=False,
    )
    b, s = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
    a = np.ones_like(b) * -0.28
    indices_3d = np.concatenate([a, b, s], axis=-1)
    spacing_3d = np.array([0, BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = storage.read_multiple_results(params, np.inf)
    print(metrics_3d.shape)
    metrix = metrics_3d.squeeze()

    storage.close()
    if not metrix:
        exit(0)

    print(metrix.shape)
    log = False
    show = False
    figure_dir = f"figures/{db_str}"
    fs = (8, 8)
    pretty_plot(metrix.r_1, metrix.r_2, "Kuramoto Order Parameter R", "kuramoto_beta_sigma", figure_dir, fs, show)
    pretty_plot(metrix.sr_1, metrix.sr_2, "Splay Ratio", "splay_ratio_beta_sigma", figure_dir, fs, show)
    pretty_plot(metrix.s_1, metrix.s_2, "Mean Phase Velocity Std", "std_beta_sigma", figure_dir, fs, show)
    pretty_plot(metrix.m_1, metrix.m_2, "Mean Phase Velocity", "mean_beta_sigma", figure_dir, fs, show)
    pretty_plot(metrix.q_1, metrix.q_2, "Entropy", "entropy_beta_sigma", figure_dir, fs, show)
    pretty_plot(metrix.f_1, metrix.f_2, "Cluster Fraction", "cluster_beta_sigma", figure_dir, fs, show)
    pretty_plot(metrix.f_1, metrix.f_2, "Cluster Fraction", "cluster_beta_sigma", figure_dir, fs, show)
    plot_tt(metrix.tt, "Measured Max Transient Time", "transient_time_beta_sigma", figure_dir, (8, 4), show)
