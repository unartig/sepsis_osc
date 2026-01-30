from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sepsis_osc.dnm.dynamic_network_model import DNMState

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar


def get_grid(n: int) -> tuple[int, int]:
    cols = np.ceil(np.sqrt(n))
    rows = np.ceil(n / cols)
    return int(rows), int(cols)


def mean_angle(angles: np.ndarray, ax: int = -1) -> np.ndarray:
    angles = np.asarray(angles)
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)
    return np.arctan2(np.mean(sin_vals, axis=ax), np.mean(cos_vals, axis=ax))


def plot_phase_snapshot(
    phis_1: np.ndarray,
    phis_2: np.ndarray,
    t: int = -1,
    s: int | None = None,
    ax: list[plt.Axes] | None = None,
    *,
    deriv: bool = False,
) -> list[plt.Axes]:
    if not ax:
        _, ax = plt.subplots(1, 2)
    n = phis_1.shape[-1]

    if t == 0:
        ax[0].scatter(np.arange(n), phis_1[t, :] / np.pi, s=s, label="Parenchymal Cells")
        ax[1].scatter(np.arange(n), phis_2[t, :] / np.pi, s=s, label="Immune Cells", c="tab:green")
    else:
        ax[0].scatter(np.arange(n), np.sort(phis_1[t, :] / np.pi), s=5, label="Parenchymal Cells")
        ax[1].scatter(np.arange(n), np.sort(phis_2[t, :] / np.pi), s=5, label="Immune Cells", c="tab:green")

    if deriv:
        ax[0].plot([0, n], [0, 0], color="black", ls=":", lw=0.5)
        ax[0].set_ylim(-1, 1)
        ax[1].plot([0, n], [0, 0], color="black", ls=":", lw=0.5)
        ax[1].set_ylim(-1, 1)
    else:
        ax[0].set_ylim(0, 2)
        ax[1].set_ylim(0, 2)
    ax[0].set_xlim(0, n)
    ax[1].set_xlim(0, n)

    return ax


def plot_phase_progression(
    phis_1: np.ndarray,
    phis_2: np.ndarray,
    ts: list[int] | np.ndarray | range,
) -> list[plt.Axes]:
    grid = get_grid(len(ts))
    _, axes = plt.subplots(*grid)
    axes = axes.flatten()
    for t, ax in zip(ts, axes, strict=True):
        plot_phase_snapshot(phis_1, phis_2, int(t), ax)
    return axes


def plot_snapshot(
    ys: DNMState,
    dys: DNMState,
    t: int = -1,
    figax: tuple[plt.Figure, list[plt.Axes]] | None = None,
) -> None:
    if not figax:
        fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(5, 2))
    axes = axes.flatten()

    plot_phase_snapshot(np.asarray(ys.phi_1), np.asarray(ys.phi_2), t, deriv=False, ax=axes[0])
    plot_phase_snapshot(np.asarray(dys.phi_1), np.asarray(dys.phi_2), t, deriv=True, ax=axes[1])

    axes[0].set_title("Phases")
    axes[0].set_xlabel("Index j")
    axes[0].set_ylabel("Phase / π")
    axes[1].set_title("Phase-Velocities")
    axes[1].set_xlabel("Index j")
    fig.tight_layout()


def plot_phase_space_time(
    phis_1: np.ndarray,
    phis_2: np.ndarray,
    sorting_1: np.ndarray,
    sorting_2: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if not ax:
        _, ax = plt.subplots(1, 1)
    phis_1 = phis_1[:, sorting_1]
    phis_2 = phis_2[:, sorting_2]
    ax.matshow(np.concatenate([phis_1, phis_2], axis=-1))
    return ax


def plot_kappa(
    kappas_1: np.ndarray,
    kappas_2: np.ndarray,
    sorting_1: np.ndarray,
    sorting_2: np.ndarray,
    t: int = -1,
    figax: tuple[plt.Figure, list[plt.Axes]] | None = None,
    cax: list[plt.Axes | None] | None = None,
    *,
    cbar1: bool = True,
    cbar2: bool = False,
    cbar_loc: str | None = None,
) -> list[plt.Axes]:
    if not figax:
        if cbar_loc == "top":
            # Create figure with gridspec to reserve colorbar space
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(3, 2, height_ratios=[0.05, 1, 0.05], hspace=0.05, wspace=0.3)
            ax = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
            cax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        else:
            fig, ax = plt.subplots(1, 2)
            cax = [None, None]
    else:
        fig, ax = figax
        if cax is None:
            cax = [None, None]

    for a in ax:
        a.xaxis.set_ticks_position("bottom")
        a.xaxis.set_label_position("bottom")

    n = kappas_1[0].shape[-1]
    vmin = -1
    vmax = 1
    cmap = "seismic"

    if t == 0:
        im1 = ax[0].imshow(kappas_1[0], origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        im2 = ax[1].imshow(kappas_2[0], origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        ims = []
        for i, (sorting, kappas) in enumerate(zip([sorting_1, sorting_2], [kappas_1, kappas_2], strict=True)):
            kappa = kappas[t].copy()
            sorted_indices = np.ix_(sorting, sorting)
            sorted_kappa = kappa[sorted_indices]
            diag_mask = np.eye(n, dtype=bool)
            off_diag_mask = ~diag_mask
            final_kappa = np.zeros_like(kappa)
            final_kappa[diag_mask] = np.diag(kappa)
            final_kappa[off_diag_mask] = sorted_kappa[off_diag_mask]
            ims.append(ax[i].imshow(final_kappa, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap))
        im1, im2 = tuple(ims)

    for i, (axi, im, cb) in enumerate([(ax[0], im1, cbar1), (ax[1], im2, cbar2)]):
        if cb:
            if cbar_loc is None:
                cbar = fig.colorbar(im, cmap=cmap, ax=axi, fraction=0.046, pad=0.14)
                cbar.set_label(r"$\kappa^1_{ij}$ value")
            elif cbar_loc == "inside":
                axins = inset_axes(
                    axi,
                    width="40%",
                    height="3%",
                    loc="upper left",
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=axi.transAxes,
                    borderpad=1,
                )
                cbar = fig.colorbar(im, cmap=cmap, cax=axins, fraction=0.046, pad=0.04, orientation="horizontal")
                for label in cbar.ax.get_xticklabels():
                    label.set_weight("bold")
            elif cbar_loc == "top":
                if cax[i] is not None:  # Check if colorbar axes exists
                    cbar = fig.colorbar(im, cax=cax[i], orientation="horizontal")
                    cax[i].xaxis.set_ticks_position("top")
                else:
                    continue  # Skip if no colorbar axes provided
            cbar.set_ticks([-1, 0, 1])

    return ax


def plot_kuramoto(
    phis: np.ndarray,
    t: int = -1,
    ax: plt.Axes | None = None,
    color: str = "tab:blue",
) -> plt.Axes:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")

    # Kuramoto oscillator phases
    color = color if color else "tab:blue"
    radii = np.ones_like(phis[t])
    ax.scatter(phis[t], radii, c=color, zorder=2)

    # Order parameter vector
    z = np.exp(1j * phis[t])
    order_param = np.mean(z)
    R = np.abs(order_param)
    psi = np.angle(order_param).astype(np.float32)
    ax.annotate(
        "",
        xy=(psi, R),
        xytext=(0, 0),
        arrowprops={"facecolor": "black", "shrink": 0.0, "width": 2, "headwidth": 8},
        zorder=3,
    )

    # Histogram
    cmap_name = "Blues" if color == "tab:blue" else "Reds"
    cmap = plt.colormaps[cmap_name]

    N_bins = 36
    bottom = 0.1
    phis_t = phis[t]
    counts, bin_edges = np.histogram(phis_t, bins=N_bins, range=(0, 2 * np.pi), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_count = counts.max() if counts.max() > 0 else 1  # avoid division by zero
    bar_width = (2 * np.pi) / N_bins

    bars = ax.bar(bin_centers, counts * (1.1 - R), width=bar_width, bottom=bottom, zorder=0)
    for count, bar in zip(counts, bars, strict=True):
        bar.set_facecolor(cmap(count / max_count))
        bar.set_alpha(0.8)

    return ax


def gif_kuramoto_plt(
    phis_1: np.ndarray,
    phis_2: np.ndarray,
    ts: np.ndarray,
    filename: str = "figures/kuramoto.gif",
) -> None:
    T, _ = phis_1.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(8, 4))

    def update(t: int) -> list:
        ax1.clear()
        plot_kuramoto(phis_1, t=t, ax=ax1, color="tab:blue")
        ax1.set_title(f"Parenchymal Layer — t={float(ts[t]):.4f}")

        ax2.clear()
        plot_kuramoto(phis_2, t=t, ax=ax2, color="tab:orange")
        ax2.set_title(f"Immune Layer — t={float(ts[t]):.4f}")

        return []

    ani = FuncAnimation(fig, update, frames=T, blit=False)
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)


def gif_phase_plt(
    phis_1: np.ndarray,
    phis_2: np.ndarray,
    ts: np.ndarray,
    *,
    deriv: bool,
    filename: str = "figures/phis.gif",
) -> None:
    T, N = phis_1.shape
    fig, ax = plt.subplots(nrows=1, ncols=2)

    def update(t: int) -> list:
        ax.clear()
        plot_phase_snapshot(phis_1, phis_2, t=t, deriv=deriv, ax=ax)
        ax.set_xlim(0, 2 * N)
        ax.set_xlabel(f"Index [0-{int(N) - 1}] Parenchymal (blue), [{int(N)} - {N * 2}] Immune Layer (orange)")
        ax.set_ylabel("Phase / π")
        ax.set_title(f"Time step: {float(ts[t]):.4f}")
        return []

    ani = FuncAnimation(fig, update, frames=T, blit=False)
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)


# if __name__ == "__main__":
#     import jax.numpy as jnp
#     import jax.random as jr
#     from diffrax import Dopri5, Dopri8

#     from sepsis_osc.dnm.dynamic_network_model import DynamicNetworkModel
#     from sepsis_osc.utils.config import jax_random_seed

#     rand_key = jr.key(jax_random_seed + 1)
#     M = 1
#     rand_keys = jr.split(rand_key, M)

#     #### Parameters
#     N = 200
#     # sync -0.28, 0.46, 1
#     # desync -0.28, 0.666, 0.42
#     # splay -0.28, 1, 1
#     run_conf = DNMConfig(
#         N=N,
#         C=0.2,  # local infection
#         omega_1=0.0,
#         omega_2=0.0,
#         a_1=1.0,
#         epsilon_1=0.03,  # adaption rate
#         epsilon_2=0.3,  # adaption rate
#         alpha=-0.28,  # phase lage
#         beta=1.0,  # age parameter
#         sigma=1.0,
#     )
#     T_init, T_max = 0, 500
#     T_step = 1
#     print(run_conf.as_index)

#     dnm = DynamicNetworkModel(full_save=True, steady_state_check=False)
#     sol = dnm.integrate(config=run_conf, M=M, key=rand_key, T_init=0, T_max=T_max, T_step=T_step, solver=Dopri8())

#     if not sol.ys:
#         exit(0)

#     ys, dys = sol.ys
#     print(ys.shape, dys.shape)
#     ys, dys = ys.remove_infs().squeeze().enforce_bounds(), dys.remove_infs().squeeze()
#     ts = np.asarray(sol.ts).squeeze()
#     ts = ts[~jnp.isinf(ts)]
#     print(dys.kappa_1.shape)
#     sorting1 = np.lexsort((dys.phi_1.mean(axis=0), ys.phi_1[-1]))
#     sorting2 = np.lexsort((dys.phi_2.mean(axis=0), ys.phi_2[-1]))
#     use_sorting = False
#     # last_t = 5
#     # romega_1 = (sol.ys.phi_1[-1, :] - sol.ys.phi_1[1, :]) / ((T_max - last_t) * T_step)
#     # romega_2 = (sol.ys.phi_2[-1, :] - sol.ys.phi_2[1, :]) / ((T_max - last_t) * T_step)
#     # plot_mean_phase_velos(romega_1, romega_2)

#     print((ys.phi_1[-1].mean() - ys.phi_2[-1].mean()) / np.pi)

#     # plot_phase_snapshot(ys.phi_1, ys.phi_2, -1)
#     # plot_phase_progression(ys.phi_1, ys.phi_2, range(-21, -1))
#     # plot_phase_snapshot(np.asarray(dys.phi_1), np.asarray(dys.phi_2), -1)
#     t = -5
#     # plot_snapshot(ys, dys, t)

#     # plot_kappa(ys.kappa_1, ys.kappa_2, 0)
#     # plot_kappa(ys.kappa_1, ys.kappa_2, -1)

#     # plot_phase_space_time(ys.phi_1, ys.phi_2)

#     # plot_kuramoto(ys.phi_1, 0)
#     # plot_kuramoto(ys.phi_1, t)

#     # gif_phase_plt(ys.phi_1, ys.phi_2, ts, False)
#     # gif_phase_plt(dys.phi_1, dys.phi_2, ts, True, filename="figures/dphis.gif")
#     # gif_kuramoto_plt(ys.phi_1, ys.phi_2, ts)

#     plot_init_cond(DynamicNetworkModel().init_sampler(run_conf, M, rand_key))
#     plt.show()
