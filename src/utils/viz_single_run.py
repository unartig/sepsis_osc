import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

from simulation.data_classes import SystemConfig, SystemState

# import fastplotlib as fpl
# from PIL import Image
# import time


def get_grid(n: int) -> tuple[int, int]:
    cols = np.ceil(np.sqrt(n))
    rows = np.ceil(n / cols)
    return int(rows), int(cols)


def mean_angle(angles, ax=-1):
    angles = np.asarray(angles)
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)
    return jnp.arctan2(np.mean(sin_vals, axis=ax), np.mean(cos_vals, axis=ax))


def plot_phase_snapshot(phis_1: np.ndarray, phis_2: np.ndarray, t: int = -1, deriv: bool = False, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1)
    n = phis_1.shape[-1]

    ax.scatter(np.arange(n), np.sort(phis_1[t, :] / np.pi), s=5, label="Parenchymal Cells")
    ax.scatter(np.arange(n) + n, np.sort(phis_2[t, :] / np.pi), s=5, label="Immune Cells")

    if deriv:
        ax.plot([0, n * 2], [0, 0], color="black", ls=":", lw=0.5)
        ax.set_ylim(-2, 2)
    else:
        ax.set_ylim(0, 2)

    return ax


def plot_phase_progression(phis_1: np.ndarray, phis_2: np.ndarray, ts: list[int] | np.ndarray | range):
    grid = get_grid(len(ts))
    _, axes = plt.subplots(*grid)
    axes = axes.flatten()
    for t, ax in zip(ts, axes):
        plot_phase_snapshot(phis_1, phis_2, int(t), ax)
    return axes


def plot_snapshot(ys: SystemState, dys: SystemState, t: int = -1):
    fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(5, 2))
    axes = axes.flatten()
    plot_phase_snapshot(np.asarray(ys.phi_1), np.asarray(ys.phi_2), t, False, axes[0])
    plot_phase_snapshot(np.asarray(dys.phi_1), np.asarray(dys.phi_2), t, True, axes[1])

    axes[0].set_title("Snapshot of Phases")
    axes[0].set_xlim(0, 2 * N)
    axes[0].set_xlabel("Index j")
    axes[0].legend()
    axes[0].set_ylabel("Phase / π")
    axes[1].set_title("Snapshot of Phase-Velocities")
    axes[1].set_xlim(0, 2 * N)
    axes[1].set_xlabel("Index j")
    fig.tight_layout()


def plot_phase_space_time(phis_1: np.ndarray, phis_2: np.ndarray, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1)
    phis_1 = phis_1[:, sorting1]
    phis_2 = phis_2[:, sorting2]
    ax.matshow(np.concatenate([phis_1, phis_2], axis=-1))
    return ax


def plot_kappa(kappas_1: np.ndarray, kappas_2: np.ndarray, t=-1, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 2)

    if t == 0:
        ax[0].matshow(kappas_1[0], origin="lower")
        ax[1].matshow(kappas_2[0], origin="lower")
        return ax

    for i, (sorting, kappas) in enumerate(zip([sorting1, sorting2], [kappas_1, kappas_2])):
        kappa = kappas[t].copy()
        n = kappa.shape[0]
        sorted_indices = np.ix_(sorting, sorting)
        sorted_kappa = kappa[sorted_indices]
        diag_mask = np.eye(n, dtype=bool)
        off_diag_mask = ~diag_mask
        final_kappa = np.zeros_like(kappa)
        final_kappa[diag_mask] = np.diag(kappa)
        final_kappa[off_diag_mask] = sorted_kappa[off_diag_mask]
        ax[i].matshow(final_kappa, origin="lower")
    return ax


def plot_kuramoto(phis: np.ndarray, t: int = -1, ax=None, color: str = "tab:blue"):
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
    psi = np.angle(order_param)
    ax.annotate(
        "",
        xy=(psi, R),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", shrink=0.0, width=2, headwidth=8),
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
    for count, bar in zip(counts, bars):
        bar.set_facecolor(cmap(count / max_count))
        bar.set_alpha(0.8)

    return ax


def gif_kuramoto_plt(phis_1: np.ndarray, phis_2: np.ndarray, ts: np.ndarray, filename="figures/kuramoto.gif"):
    T, N = phis_1.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(8, 4))

    def update(t):
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


def gif_phase_plt(phis_1: np.ndarray, phis_2: np.ndarray, ts: np.ndarray, deriv: bool, filename="figures/phis.gif"):
    T, N = phis_1.shape
    fig, ax = plt.subplots()

    def update(t):
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


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax.random as jr
    from diffrax import Tsit5, Dopri5, Dopri8, ODETerm, PIDController
    from jax import vmap

    from utils.run_simulation import solve
    from simulation.simulation import (
        generate_init_conditions_fixed,
        make_full_compressed_save,
        system_deriv,
    )
    from utils.config import jax_random_seed

    rand_key = jr.key(jax_random_seed + 0)
    num_parallel_runs = 1
    rand_keys = jr.split(rand_key, num_parallel_runs)
    full_save = make_full_compressed_save(system_deriv, jnp.float32)
    term = ODETerm(system_deriv)
    solver = Tsit5()
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-11)

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
        beta=0.5,  # age parameter
        sigma=1,
    )  # b 0.83, s 0.25, rseed + 0
    T_init, T_trans, T_max = 0, 0, 1000
    T_step = 1
    generate_init_conditions = generate_init_conditions_fixed(run_conf.N, run_conf.beta, run_conf.C)

    init_conditions = vmap(generate_init_conditions)(rand_keys)
    sol = solve(
        T_init,
        T_trans,
        T_max,
        T_step,
        init_conditions.copy(),
        run_conf.as_args,
        solver,
        term,
        stepsize_controller,
        full_save,
    )
    if not sol.ys:
        exit(0)
    ys, dys = sol.ys
    ys, dys = ys.squeeze().remove_infs().enforce_bounds(), dys.squeeze().remove_infs()
    ts = np.asarray(sol.ts).squeeze()
    ts = ts[~jnp.isinf(ts)]
    print(dys.kappa_1.shape)
    sorting1 = np.lexsort((dys.phi_1.mean(axis=0), ys.phi_1[-1]))
    sorting2 = np.lexsort((dys.phi_2.mean(axis=0), ys.phi_2[-1]))
    # last_t = 5
    # romega_1 = (sol.ys.phi_1[-1, :] - sol.ys.phi_1[1, :]) / ((T_max - last_t) * T_step)
    # romega_2 = (sol.ys.phi_2[-1, :] - sol.ys.phi_2[1, :]) / ((T_max - last_t) * T_step)
    # plot_mean_phase_velos(romega_1, romega_2)

    print((ys.phi_1[-1].mean() - ys.phi_2[-1].mean()) / np.pi)

    # plot_phase_snapshot(ys.phi_1, ys.phi_2, -1)
    # plot_phase_progression(ys.phi_1, ys.phi_2, range(-21, -1))
    # plot_phase_snapshot(np.asarray(dys.phi_1), np.asarray(dys.phi_2), -1)
    plot_snapshot(ys, dys)

    # plot_kappa(ys.kappa_1, ys.kappa_2, 0)
    # plot_kappa(ys.kappa_1, ys.kappa_2, -1)

    # plot_phase_space_time(ys.phi_1, ys.phi_2)

    # plot_kuramoto(ys.phi_1, 0)
    # plot_kuramoto(ys.phi_1)

    # gif_phase_plt(ys.phi_1, ys.phi_2, ts, False)
    # gif_phase_plt(dys.phi_1, dys.phi_2, ts, True, filename="figures/dphis.gif")
    # gif_kuramoto_plt(ys.phi_1, ys.phi_2, ts)
    plt.show()
