import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Axes, Figure
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from sepsis_osc.ldm.lookup import LatentLookup, get_aligned_subgrid
from sepsis_osc.utils.config import BETA_SPACE, SIGMA_SPACE, plt_params
from sepsis_osc.visualisations.viz_param_space import space_plot

plt.rcParams.update(plt_params)


def viz_starter(
    betas: jnp.ndarray,
    sigmas: jnp.ndarray,
    lookup: LatentLookup,
    *,
    mask: np.ndarray,
    cmaps: bool = True,
    figax: tuple[Figure, Axes] | None = None,
    figure_dir: str = "figures/model",
    filename: str = "latent_starters",
) -> Axes:
    if figax is not None:
        fig, ax = figax
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    betas_space = jnp.arange(*BETA_SPACE)
    sigmas_space = jnp.arange(*SIGMA_SPACE)
    beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing="ij")
    param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    metrics = lookup.hard_get_fsq(jnp.asarray(param_grid)).reshape(len(betas_space), len(sigmas_space))

    # --- SOFA
    std_sym = r"$s^{1}$"
    ax = space_plot(
        metrics,
        xs=np.asarray(betas_space),
        ys=np.asarray(sigmas_space),
        title=rf"SOFA Progression over {std_sym} space{'\n'}Parenchymal Layer",
        cmap=cmaps,
        filename=filename,
        figure_dir=figure_dir,
        figax=(fig, ax),
    )
    N, T = betas.shape
    betas = betas[mask].ravel()  # row-major flatten
    sigmas = sigmas[mask].ravel()

    t_idx = np.tile(np.arange(T), N)
    t_idx = t_idx[mask.ravel()]  # filter with mask

    beta_scale = len(betas_space) * (betas - BETA_SPACE[0]) / (BETA_SPACE[1] - BETA_SPACE[0])
    sigma_scale = len(sigmas_space) * (1 - (sigmas - SIGMA_SPACE[0]) / (SIGMA_SPACE[1] - SIGMA_SPACE[0]))
    ax.scatter(beta_scale, sigma_scale, c=t_idx, cmap="copper", alpha=0.8, s=3.0)

    ax.set_xlabel(r"$\beta / \pi$")
    ax.set_ylabel(r"$\sigma$")
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    return ax


def viz_plane(
    true_sofa: jnp.ndarray | np.ndarray,
    betas: jnp.ndarray | np.ndarray,
    sigmas: jnp.ndarray | np.ndarray,
    lookup: LatentLookup,
    mask: jnp.ndarray | np.ndarray,
    figax: tuple[Figure, Axes] | None = None,
    *,
    window_size: int = 5,
    zoom: bool = True,
    cmaps: bool = True,
    cs: tuple[str, ...] | list[str] = ("tab:cyan", "tab:purple", "tab:pink", "tab:orange"),
) -> tuple[Figure, Axes]:
    if figax is not None:
        fig, ax = figax
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if zoom:
        param_grid, betas_space, sigmas_space = get_aligned_subgrid(
            betas[mask], sigmas[mask], BETA_SPACE, SIGMA_SPACE, window_size=window_size
        )
    else:
        betas_space = jnp.arange(*BETA_SPACE)
        sigmas_space = jnp.arange(*SIGMA_SPACE)
        beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing="ij")
        param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    metrics = lookup.hard_get_fsq(jnp.asarray(param_grid)).reshape((len(betas_space), len(sigmas_space)))

    std_sym = r"$s^{1}$"
    ax = space_plot(
        metrics,
        xs=np.asarray(betas_space),
        ys=np.asarray(sigmas_space),
        title=rf"SOFA Progression over {std_sym} space{'\n'}Parenchymal Layer",
        cmap=cmaps,
        filename="",
        figax=(fig, ax),
    )

    cm = plt.colormaps.get_cmap("copper")
    norm = colors.Normalize(vmin=0, vmax=24)

    if cmaps:
        sm = ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        sm.set_clim(0, 24)
        cbar2 = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar2.set_label("Ground Truth SOFA-score")

    # Ensure arrays are 2D (N, T)
    true_sofa = np.atleast_2d(true_sofa)
    betas = np.atleast_2d(betas)
    sigmas = np.atleast_2d(sigmas)
    mask = np.atleast_2d(mask)

    for i in range(true_sofa.shape[0]):
        m = mask[i]

        beta_scale = len(betas_space) * (betas[i, m] - betas_space[0]) / (betas_space[-1] - betas_space[0])
        sigma_scale = len(sigmas_space) * (1 - (sigmas[i, m] - sigmas_space[0]) / (sigmas_space[-1] - sigmas_space[0]))

        ax.scatter(
            beta_scale,
            sigma_scale,
            c=true_sofa[i, m],
            cmap=cm,
            norm=norm,
            s=20,
        )
        ax.scatter(
            beta_scale[[0, -1]],
            sigma_scale[[0, -1]],
            c=cs[i],
            marker="x",
        )

        ax.annotate("0", (beta_scale[0] + 1, sigma_scale[0] + 1), color=cs[i], weight="bold")
        ax.annotate(f"{beta_scale.size}", (beta_scale[-1] + 1, sigma_scale[-1] + 1), color=cs[i], weight="bold")

    return fig, ax


def viz_heatmap_concepts(
    true_sofa: jnp.ndarray | np.ndarray,
    true_infs: jnp.ndarray | np.ndarray,
    pred_sofa: jnp.ndarray | np.ndarray,
    pred_infs: jnp.ndarray | np.ndarray,
    *,
    cmap: bool,
    figax: tuple[Figure, tuple[Axes, Axes, Axes]] | None = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(1, 3)

    sofa_bins = [np.arange(-0.5, 24.5, 1), np.arange(-0.5, 24.5, 1)]
    sofa_heatmap = np.histogram2d(true_sofa, pred_sofa, bins=sofa_bins)
    sofa = (sofa_heatmap, "SOFA-score", ax[0])

    dsofa_bins = [np.arange(-24.5, 24.5, 1), np.arange(-24.5, 24.5, 1)]  # 1-point bins
    dsofa_heatmap = np.histogram2d(np.diff(true_sofa), np.diff(pred_sofa), bins=dsofa_bins)
    dsofa = (dsofa_heatmap, r"$\Delta$SOFA-score", ax[1])

    infection_bins = [np.arange(-0.05, 1.05, 0.1), np.arange(-0.05, 1.05, 0.1)]
    inf_heatmap = np.histogram2d(true_infs, pred_infs, bins=infection_bins)
    inf = (inf_heatmap, "Suspected Infection", ax[2])

    heats = np.log(
        np.concat(
            [
                np.asarray(sofa_heatmap[0]).flatten(),
                np.asarray(dsofa_heatmap[0]).flatten(),
                np.asarray(inf_heatmap[0]).flatten(),
            ]
        )
        + 1
    )
    norm = colors.Normalize(vmin=np.min(heats), vmax=np.max(heats))

    images = []
    for (heatmap, xedges, yedges), name, axi in [sofa, dsofa, inf]:
        # Plot the 2D histogram as an image
        images.append(
            axi.imshow(
                np.log(heatmap.T + 1),  # transpose so it matches axes orientation
                origin="lower",
                extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                aspect="equal",
                cmap="magma",
                norm=norm,
            )
        )
        axi.set_title(name)
        axi.set_xlabel("Ground Truth")
    if cmap:
        pos = ax[-1].get_position()

        # Create colorbar axes outside the plot
        cax = fig.add_axes((pos.x1 + 0.02, pos.y0, 0.02, pos.height))
        fig.colorbar(images[-1], cax=cax, label="log(Count)")

    ax[0].set_ylabel("Predicted")
    fig.subplots_adjust(wspace=0.2)
    return fig, ax


def viz_curves(
    true_sofa: jnp.ndarray | np.ndarray,
    true_infs: jnp.ndarray | np.ndarray,
    true_sep3: jnp.ndarray | np.ndarray,
    pred_sofa: jnp.ndarray | np.ndarray,
    pred_infs: jnp.ndarray | np.ndarray,
    pred_sep3: jnp.ndarray | np.ndarray,
    figure_dir: str = "figures/model",
    filename: str = "confusion.jpg",
    figax: tuple[Figure, tuple[Axes, Axes]] | None = None,
) -> tuple[Axes, Axes]:
    if figax is not None:
        _fig, ax = figax
    if ax is None or len(ax) < 2:
        _fig, ax = plt.subplots(1, 2)

    RocCurveDisplay.from_predictions(
        true_sofa, pred_sofa, ax=ax[0], curve_kwargs={"label": "SOFA d2", "color": "tab:blue"}
    )
    PrecisionRecallDisplay.from_predictions(true_sofa, pred_sofa, ax=ax[1], label="SOFA d2", color="tab:blue")
    RocCurveDisplay.from_predictions(
        true_infs, pred_infs, ax=ax[0], curve_kwargs={"label": "Infection", "color": "tab:orange"}
    )
    PrecisionRecallDisplay.from_predictions(true_infs, pred_infs, ax=ax[1], label="Infection", color="tab:orange")
    RocCurveDisplay.from_predictions(
        true_sep3, pred_sep3, ax=ax[0], curve_kwargs={"label": "Sepsis-3", "color": "tab:green"}
    )
    PrecisionRecallDisplay.from_predictions(true_sep3, pred_sep3, ax=ax[1], label="Sepsis3", color="tab:green")

    ax[0].set_title("Receiver Operating Characteristics")
    ax[1].set_title("Precision Recall Curve")
    # ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}", transparent=True, dpi=800)
    return ax
