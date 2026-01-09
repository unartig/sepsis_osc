from typing import Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Axes, Figure
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from sepsis_osc.ldm.lookup import LatentLookup
from sepsis_osc.utils.config import BETA_SPACE, SIGMA_SPACE
from sepsis_osc.visualisations.viz_param_space import space_plot

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def viz_starter(
    alphas: jnp.ndarray,
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
        title=rf"SOFA Progression over {std_sym} space @$\alpha=${alphas.mean():.3f}{'\n'}Parenchymal Layer",
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


def viz_progression(
    true_sofa: jnp.ndarray | np.ndarray,
    true_infs: jnp.ndarray | np.ndarray,
    pred_sofa: jnp.ndarray | np.ndarray,
    pred_infs: jnp.ndarray | np.ndarray,
    mask: np.ndarray,
    ax: Axes | None = None,
    figure_dir: str = "figures/model",
    filename: str = "latent_plane",
) -> Axes:
    if ax is None:
        _fig, ax = plt.subplots(1, 1)

    NB, B, T = pred_sofa.shape

    # --- SOFA
    mean_sofa = (pred_sofa - true_sofa).mean(axis=(0, 1), where=mask)
    std_sofa = (pred_sofa - true_sofa).std(axis=(0, 1), where=mask)
    ci95_sofa = 1.96 * std_sofa / np.sqrt(NB * B)

    ax.plot(np.arange(T), mean_sofa, label="SOFA Error")
    ax.plot(np.arange(T), mean_sofa + ci95_sofa, linestyle="--", color="gray")
    ax.plot(np.arange(T), mean_sofa - ci95_sofa, linestyle="--", color="gray")
    ax.set_title("Progression Prediction - True")

    plt.tight_layout()
    ax.grid()
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    return ax


def viz_plane(
    true_sofa: jnp.ndarray | np.ndarray,
    alphas: jnp.ndarray,
    betas: jnp.ndarray,
    sigmas: jnp.ndarray,
    lookup: LatentLookup,
    mask: np.ndarray,
    figax: tuple[Figure, Axes] | None = None,
    figure_dir: str = "figures/model",
    filename: str = "latent_plane",
    *,
    cmaps: bool = True,
) -> Axes:
    if figax is not None:
        fig, ax = figax
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    betas_space = jnp.arange(*BETA_SPACE)
    sigmas_space = jnp.arange(*SIGMA_SPACE)
    beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing="ij")
    param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    metrics = lookup.hard_get_fsq(jnp.asarray(param_grid)).reshape((len(betas_space), len(sigmas_space)))

    # --- SOFA
    std_sym = r"$s^{1}$"
    ax = space_plot(
        metrics,
        xs=np.asarray(betas_space),
        ys=np.asarray(sigmas_space),
        title=rf"SOFA Progression over {std_sym} space @$\alpha=${alphas.mean():.3f}{'\n'}Parenchymal Layer",
        cmap=cmaps,
        filename=filename,
        figure_dir=figure_dir,
        figax=(fig, ax),
    )
    beta_scale = len(betas_space) * (betas[mask] - BETA_SPACE[0]) / (BETA_SPACE[1] - BETA_SPACE[0])
    sigma_scale = len(sigmas_space) * (1 - (sigmas[mask] - SIGMA_SPACE[0]) / (SIGMA_SPACE[1] - SIGMA_SPACE[0]))
    cm = plt.colormaps.get_cmap("copper")
    if cmaps:
        sm = ScalarMappable(cmap=cm)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar2.set_label("Actual SOFA-score")
    ax.scatter(beta_scale, sigma_scale, c=(true_sofa / 24)[mask], cmap=cm)

    # plt.tight_layout()
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    return ax


def viz_heatmap_concepts(
    true_sofa: jnp.ndarray | np.ndarray,
    true_infs: jnp.ndarray | np.ndarray,
    pred_sofa: jnp.ndarray | np.ndarray,
    pred_infs: jnp.ndarray | np.ndarray,
    cmap: bool,
    figure_dir: str = "figures/model",
    filename: str = "prediction.jpg",
    figax: tuple[Figure, tuple[Axes, Axes]] | None = None,
) -> tuple[Axes, Axes]:
    if figax is not None:
        fig, ax = figax
    if ax is None or len(ax) < 2:
        fig, ax = plt.subplots(1, 2)

    sofa_bins = [np.arange(-0.5, 24.5, 1), np.arange(-0.5, 24.5, 1)]  # 1-point bins
    sofa_heatmap, sofa_xedges, sofa_yedges = np.histogram2d(true_sofa, pred_sofa, bins=sofa_bins)

    # Plot the 2D histogram as an image
    # --- SOFA
    im = ax[0].imshow(
        np.log(sofa_heatmap.T + 1),  # transpose so it matches axes orientation
        origin="lower",
        extent=[sofa_xedges[0], sofa_xedges[-1], sofa_yedges[0], sofa_yedges[-1]],
        aspect="auto",
        cmap="hot",
    )
    ax[0].plot(range(24), range(24), color="tab:red", label="optimum")
    ax[0].set_ylabel("Predicted SOFA-score")
    ax[0].set_xlabel("Actual SOFA-score")
    ax[0].legend()
    if cmap:
        fig.colorbar(im, ax=ax[0], label="Count")

    # --- infection
    infection_bins = [np.arange(-0.05, 1.05, 0.1), np.arange(-0.05, 1.05, 0.1)]  # 1-point bins
    infection_heatmap, infection_xedges, infection_yedges = np.histogram2d(true_infs, pred_infs, bins=infection_bins)
    im = ax[1].imshow(
        np.log(infection_heatmap.T + 1),
        origin="lower",
        extent=[infection_xedges[1], infection_xedges[-1], infection_yedges[0], infection_yedges[-1]],
        aspect="auto",
        cmap="hot",
    )
    ax[1].set_ylabel("Predicted infection (probability)")
    ax[1].set_xlabel("Actual infection (bool)")
    # ax[1].legend()
    if cmap:
        fig.colorbar(im, ax=ax[1], label="Count")

    plt.tight_layout()
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}", transparent=True, dpi=800)
    return ax


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

    ax[0].set_title("ROC")
    ax[1].set_title("PRC")
    # ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}", transparent=True, dpi=800)
    return ax
