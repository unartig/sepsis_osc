import logging

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.ldm.lookup import as_2d_indices

setup_logging()
logger = logging.getLogger(__name__)

num_ticks = 10


def plot_tt(tt, title, filename, figure_dir, fs=(8, 6), show=False):
    fig, ax = plt.subplots(figsize=fs)
    im = ax.imshow(tt.T[::-1, :], aspect="auto", cmap="viridis")
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


def space_plot(
    metric, xs, ys, title, cmap=True, filename="space_plot.svg", figure_dir="figures", fs=(6, 4), figax=None
):
    if not figax:
        fig, ax = plt.subplots(1, 1, figsize=fs)
    else:
        fig, ax = figax

    im = ax.imshow(metric.T[::-1, :], cmap="viridis")

    ax.set_title(f"{title}\nParenchymal Layer", fontsize=14)
    ax.set_ylabel(r"$\sigma$", fontsize=12)
    ax.set_xlabel(r"$\beta / \pi$", fontsize=12)

    xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
    ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])
    if cmap:
        cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.8)
        cbar.set_label(r"$s^{mu}$")
    return ax


def pretty_plot(metric_parenchymal: np.ndarray, metric_immune: np.ndarray, title: str, filename: str ="", figure_dir: str="", fs: tuple[int, int]=(8, 6), *, show: bool =False) -> None:
    fig, axes = plt.subplots(2, 1, figsize=fs)

    im0 = axes[0].imshow(metric_parenchymal.T[::-1, :], aspect="auto", cmap="viridis")
    im1 = axes[1].imshow(metric_immune.T[::-1, :], aspect="auto", cmap="viridis")

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
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    if show:
        plt.show()


if __name__ == "__main__":
    # ALPHA_SPACE = (-0.52, -0.52, 1.0)
    BETA_SPACE = (0.0, 1.0, 0.01)
    SIGMA_SPACE = (0.0, 1.5, 0.015)
    xs = np.arange(*BETA_SPACE)
    ys = np.arange(*SIGMA_SPACE)

    orig_xs = [np.argmin(np.abs(xs - x)) for x in [0.4, 0.7]]
    orig_ys = [len(ys) - np.argmin(np.abs(ys - y)) - 1 for y in [0.0, 1.5]]

    db_str = "Daisy2"  # other/Tiny"
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
    params = DNMConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = storage.read_multiple_results(params, DNMMetrics, np.inf)
    print(metrics_3d.shape)
    metrix = metrics_3d.squeeze()

    if not metrix:
        raise ValueError("Failed")

    print(metrix.shape)
    log = False
    show = False
    figure_dir = f"figures/{db_str}"
    fs = (8, 8)
    pretty_plot(metrix.r_1, metrix.r_2, "Kuramoto Order Parameter R", "kuramoto_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.sr_1, metrix.sr_2, "Splay Ratio", "splay_ratio_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.s_1, metrix.s_2, "Mean Phase Velocity Std", "std_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.m_1, metrix.m_2, "Mean Phase Velocity", "mean_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.q_1, metrix.q_2, "Entropy", "entropy_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.f_1, metrix.f_2, "Cluster Fraction", "cluster_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.f_1, metrix.f_2, "Cluster Fraction", "cluster_beta_sigma", figure_dir, fs, show=show)
    pretty_plot(metrix.cq_1, metrix.cq_2, "Coupling Entropy", "coupling_entropy", figure_dir, fs, show=show)
    pretty_plot(metrix.cs_1, metrix.cs_2, "Coupling Std", "coupling_std", figure_dir, fs, show=show)
    plot_tt(metrix.tt, "Measured Max Transient Time", "transient_time_beta_sigma", figure_dir, (8, 4), show=show)

    ALPHA_SPACE = (-0.84, 0.84 + 0.28, 0.28)
    for alpha in np.arange(*ALPHA_SPACE):
        neg = "" if alpha > 0 else "neg"
        a = np.ones_like(b) * alpha
        indices_3d = np.concatenate([a, b, s], axis=-1)
        spacing_3d = np.array([0, BETA_SPACE[2], SIGMA_SPACE[2]])
        params = DNMConfig.batch_as_index(a, b, s, 0.2)
        metrics_3d, _ = storage.read_multiple_results(params, DNMMetrics, np.inf)
        print(metrics_3d.shape)
        metrix = metrics_3d.squeeze()

        if not metrix:
            raise ValueError("Failed")
        pretty_plot(metrix.s_1, metrix.s_2, f"Mean Phase Velocity Std alpha@{neg}{abs(alpha):.2f}", f"std_beta_sigma_ {neg}{abs(alpha):.2f}", figure_dir, fs, show=show)
    storage.close()
