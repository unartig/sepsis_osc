import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# import fastplotlib as fpl
# from PIL import Image
# import time


def get_grid(n: int) -> tuple[int, int]:
    cols = np.ceil(np.sqrt(n))
    rows = np.ceil(n / cols)
    return int(rows), int(cols)


def mean_angle(angles, axis=-1):
    angles = np.asarray(angles)
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)
    return jnp.arctan2(np.mean(sin_vals, axis=axis), np.mean(cos_vals, axis=axis))


def plot_phase_snapshot(phis_1: np.ndarray, phis_2: np.ndarray, t: int = -1, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1)
    n = phis_1.shape[-1]

    ax.scatter(np.arange(n), phis_1[t, :] / np.pi, s=5)
    ax.scatter(np.arange(n) + n, phis_2[t, :] / np.pi, s=5)
    ax.set_ylim(0, 2)

    return ax


def plot_phase_progression(phis_1: np.ndarray, phis_2: np.ndarray, ts: list[int] | np.ndarray | range):
    grid = get_grid(len(ts))
    _, axes = plt.subplots(*grid)
    axes = axes.flatten()
    for t, ax in zip(ts, axes):
        plot_phase_snapshot(phis_1, phis_2, int(t), ax)
    return axes


def plot_phase_space_time(phis: np.ndarray, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1)
    ax.matshow(phis)
    return ax


def plot_kappa(kappas: np.ndarray, t=-1, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1)
    ax.matshow(kappas[t])
    return ax


def plot_kuramoto(phis: np.ndarray, t: int = -1, ax=None, color: str = ""):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    color = color if color else "tab:blue"
    radii = np.ones_like(phis[t])
    ax.scatter(phis[t], radii, c=color)

    z = np.exp(1j * phis[t])
    order_param = np.mean(z)
    R = np.abs(order_param)
    psi = np.angle(order_param)
    ax.annotate(
        "",
        xy=(psi, R),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", shrink=0.0, width=2, headwidth=8),
    )


def gif_kuramoto_plt(phis_1: np.ndarray, phis_2: np.ndarray, filename="kuramoto.gif"):
    T, N = phis_1.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(8, 4))

    def update(t):
        ax1.clear()
        plot_kuramoto(phis_1, t=t, ax=ax1, color="blue")
        ax1.set_title(f"Parenchymal Layer — t={t}")

        ax2.clear()
        plot_kuramoto(phis_2, t=t, ax=ax2, color="orange")
        ax2.set_title(f"Immune Layer — t={t}")

        return []

    ani = FuncAnimation(fig, update, frames=T, blit=False)
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)


def gif_phase_plt(phis_1: np.ndarray, phis_2: np.ndarray, filename="phis.gif"):
    T, N = phis_1.shape
    fig, ax = plt.subplots()

    def update(t):
        ax.clear()
        plot_phase_snapshot(phis_1, phis_2, t=t, ax=ax)
        ax.set_xlim(0, 2 * N)
        ax.set_xlabel(f"Index [0-{int(N/2)-1}] Parenchymal, [{int(N/2)-1} - {N}] Immune Layer")
        ax.set_ylabel("Phase / π")
        ax.set_title(f"Time step: {t}")
        return []

    ani = FuncAnimation(fig, update, frames=T, blit=False)
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax.random as jr
    from diffrax import Bosh3, Dopri5, ODETerm, PIDController
    from jax import vmap

    from run_simulation import solve
    from simulation.data_classes import SystemConfig, SystemState
    from simulation.simulation import (
        generate_init_conditions_fixed,
        make_full_compressed_save,
        matlab_deriv,
        system_deriv,
    )
    from utils.config import jax_random_seed

    rand_key = jr.key(jax_random_seed)
    num_parallel_runs = 1
    rand_keys = jr.split(rand_key, num_parallel_runs)
    full_save = make_full_compressed_save(jnp.float32)
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
        beta=0.5,  # age parameter
        sigma=1.0,
    )
    T_init, T_trans, T_max = 0, 0, 100
    T_step = 0.05
    generate_init_conditions = generate_init_conditions_fixed(run_conf.N, run_conf.beta, run_conf.C)

    init_conditions = vmap(generate_init_conditions)(rand_keys)
    # shape (num_parallel_runs, state)
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
    ys = sol.ys.squeeze().remove_infs().enforce_bounds()
    print(ys.kappa_1.shape)
    ts = np.asarray(sol.ts).squeeze()
    dys = SystemState(
        phi_1=np.gradient(ys.phi_1, axis=0) * (1 / T_step),
        phi_2=np.gradient(ys.phi_2, axis=0) * (1 / T_step),
        kappa_1=np.gradient(ys.kappa_1 * (1 / T_step), axis=(-2, -1)),
        kappa_2=np.gradient(ys.kappa_2 * (1 / T_step), axis=(-2, -1)),
    )
    # last_t = 5
    # romega_1 = (sol.ys.phi_1[-1, :] - sol.ys.phi_1[1, :]) / ((T_max - last_t) * T_step)
    # romega_2 = (sol.ys.phi_2[-1, :] - sol.ys.phi_2[1, :]) / ((T_max - last_t) * T_step)
    # plot_mean_phase_velos(romega_1, romega_2)

    print((ys.phi_1[-1].mean() - ys.phi_2[-1].mean()) / np.pi)
    ts = ts[~jnp.isinf(ts)]
    print(ys.phi_1.shape)

    # TODO sorting :^)
    plot_phase_snapshot(ys.phi_1, ys.phi_2, -1)
    plot_phase_progression(ys.phi_1, ys.phi_2, range(-21, -1))
    plot_phase_snapshot(np.asarray(dys.phi_1), np.asarray(dys.phi_2 * (1 / T_step)), -1)

    plot_kappa(ys.kappa_2, -1)
    plot_kappa(ys.kappa_2, 0)

    plot_phase_space_time(ys.phi_1)

    plot_kuramoto(ys.phi_1)
    plot_kuramoto(ys.phi_1, 0)

    # gif_phase_plt(ys.phi_1, ys.phi_2)
    gif_kuramoto_plt(ys.phi_1, ys.phi_2)
    plt.show()
