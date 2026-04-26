import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import Axes, Figure
from scipy import ndimage
from scipy.stats import binned_statistic_2d
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from sepsis_osc.ldm.lookup import LatentLookup, compute_window_bounds
from sepsis_osc.ldm.model_structs import AuxLosses
from sepsis_osc.utils.config import BETA_SPACE, SIGMA_SPACE, plt_params
from sepsis_osc.visualisations.viz_param_space import space_plot

plt.rcParams.update(plt_params)


def viz_space_distribution_countour(
    betas: jnp.ndarray,
    sigmas: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    resolution: float = 0.25,
    show_cmap: bool = True,
    show_inlay_notation: bool = True,
    figax: tuple[Figure, Axes] | None = None,
) -> tuple[Figure, Axes]:
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(1, 1)

    bins = [
        np.arange(BETA_SPACE[0], BETA_SPACE[1], BETA_SPACE[2] * resolution),
        np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1], SIGMA_SPACE[2] * resolution),
    ]
    betas_space = jnp.arange(BETA_SPACE[0], BETA_SPACE[1] + BETA_SPACE[2], BETA_SPACE[2])
    sigmas_space = jnp.arange(SIGMA_SPACE[0], SIGMA_SPACE[1] + SIGMA_SPACE[2], SIGMA_SPACE[2])
    heatmap, *_ = np.histogram2d(betas[mask], sigmas[mask], bins=bins)

    extent = (betas_space.min(), betas_space.max(), sigmas_space.min(), sigmas_space.max())

    log_heatmap = np.log1p(heatmap)
    smoothed_log_density = ndimage.gaussian_filter(log_heatmap.T, sigma=1.2)

    X, Y = np.meshgrid(
        np.linspace(extent[0], extent[1], smoothed_log_density.shape[1]),
        np.linspace(extent[2], extent[3], smoothed_log_density.shape[0]),
    )
    contours = ax.contour(X, Y, smoothed_log_density, levels=5, cmap="OrRd", linewidths=1.2)

    if show_cmap:
        cbar_density = fig.colorbar(contours, ax=ax, location="right", shrink=0.8, pad=0.05)
        for line in cbar_density.ax.collections:
            line.set_linewidth(40.0)
        cbar_density.set_label("log(Density)")

    if show_inlay_notation:
        ax.clabel(contours, inline=True, fontsize=7, fmt="%1.1f")

    return fig, ax


def viz_space_alignment(
    test_y: jnp.ndarray,
    betas: jnp.ndarray,
    sigmas: jnp.ndarray,
    mask: np.ndarray,
    *,
    resolution: float = 0.25,
    show_cmap: bool = True,
    figax: tuple[Figure, Axes] | None = None,
) -> tuple[Figure, Axes]:
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(1, 1)

    bins = [
        np.arange(BETA_SPACE[0], BETA_SPACE[1], BETA_SPACE[2] * resolution),
        np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1], SIGMA_SPACE[2] * resolution),
    ]
    x = betas[mask]
    y = sigmas[mask]
    betas_space = jnp.arange(BETA_SPACE[0], BETA_SPACE[1] + BETA_SPACE[2], BETA_SPACE[2])
    sigmas_space = jnp.arange(SIGMA_SPACE[0], SIGMA_SPACE[1] + SIGMA_SPACE[2], SIGMA_SPACE[2])
    extent = (float(betas_space.min()), float(betas_space.max()), float(sigmas_space.min()), float(sigmas_space.max()))
    values = test_y[0, mask, 0]  # SOFA score

    mean_map, *_ = binned_statistic_2d(x, y, values, statistic="mean", bins=bins)
    mean_map = mean_map[:, ::-1].T

    alpha_mean = ~np.isnan(mean_map)
    mean_map = np.nan_to_num(mean_map, nan=0.0)

    boundaries = np.arange(-0.0, 25.5, 1)
    base_cmap = plt.get_cmap("gray")
    norm = colors.BoundaryNorm(boundaries, ncolors=base_cmap.N, extend="neither")

    im_mean = ax.imshow(
        mean_map, extent=extent, aspect="auto", cmap=base_cmap, norm=norm, alpha=alpha_mean.astype(float)
    )

    if show_cmap:
        cbar = fig.colorbar(im_mean, ax=ax, location="right", shrink=0.8, pad=0.05)
        cbar.set_label("Mean Ground Truth SOFA-score")

    return fig, ax


def viz_trajectories_over_time(
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

    betas_space = np.arange(*BETA_SPACE)
    sigmas_space = np.arange(*SIGMA_SPACE)
    beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing="ij")
    param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    metrics_full = lookup.hard_get_fsq(jnp.asarray(param_grid), jnp.array([0.0, 0.0], dtype=jnp.uint32)).reshape(
        len(betas_space), len(sigmas_space)
    )
    metrics_np = np.asarray(metrics_full)

    # --- SOFA
    std_sym = r"$s^{1}$"
    ax = space_plot(
        metrics_np,
        xs=np.asarray(betas_space),
        ys=np.asarray(sigmas_space),
        title=rf"SOFA Progression over time inside the {std_sym} space{'\n'}Parenchymal Layer",
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

    ax.scatter(betas, sigmas, c=t_idx, cmap="copper", alpha=0.8, s=3.0)

    ax.set_xlabel(r"$\beta / \pi$")
    ax.set_ylabel(r"$\sigma$")
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")
    return ax


def viz_patients_latent(
    true_sofa: np.ndarray,
    betas: np.ndarray,
    sigmas: np.ndarray,
    lookup: LatentLookup,
    mask: np.ndarray,
    figax: tuple[Figure, Axes] | None = None,
    *,
    window_size: int = 5,
    zoom: bool = True,
    cmaps: bool = True,
) -> tuple[Figure, Axes]:
    if figax is not None:
        fig, ax = figax
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    betas_space = np.arange(*BETA_SPACE)
    sigmas_space = np.arange(*SIGMA_SPACE)
    beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing="ij")
    param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    metrics_full = lookup.hard_get_fsq(jnp.asarray(param_grid), jnp.array([0, 0], dtype=jnp.uint32)).reshape(
        len(betas_space), len(sigmas_space)
    )

    metrics_np = np.asarray(metrics_full)

    if zoom:
        beta_min, beta_max, sigma_min, sigma_max = compute_window_bounds(
            betas=betas, sigmas=sigmas, betas_space=betas_space, sigmas_space=sigmas_space, window_size=window_size
        )

        metrics = metrics_np[beta_min:beta_max, sigma_min:sigma_max]
        betas_space = betas_space[beta_min:beta_max]
        sigmas_space = sigmas_space[sigma_min:sigma_max]
    else:
        metrics = metrics_np

    std_sym = r"$s^{1}$"
    ax = space_plot(
        metrics,
        xs=np.asarray(betas_space),
        ys=np.asarray(sigmas_space),
        title=rf"Trajectories over {std_sym} space",
        cmap=cmaps,
        filename="",
        figax=(fig, ax),
        xticklabel_rot=0,
        num_ticks=5,
    )

    boundaries = np.arange(-0.0, 25.5, 1)
    base_cmap = plt.get_cmap("gray")
    norm = colors.BoundaryNorm(boundaries, ncolors=base_cmap.N, extend="neither")

    if cmaps:
        sm = ScalarMappable(cmap=base_cmap, norm=norm)
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

        ax.scatter(betas[i, m], sigmas[i, m], c=true_sofa[i, m], cmap=base_cmap, norm=norm, s=20)
        ax.scatter(betas[i, m][[0, -1]], sigmas[i, m][[0, -1]], c="white", marker="x", s=20)

        ax.annotate(
            rf"P{i + 1}: \textbf{{0}}",
            (betas[i, m][0] - 5 * BETA_SPACE[2], sigmas[i, m][0] - 3 * SIGMA_SPACE[2]),
            color="white",
            weight="heavy",
        )
        ax.annotate(
            rf"P{i + 1}: \textbf{{{betas[i, m].size}}}",
            (betas[i, m][-1] - 4 * BETA_SPACE[2], sigmas[i, m][-1] + 1 * SIGMA_SPACE[2]),
            color="white",
            weight="heavy",
        )

    return fig, ax


def viz_patients(
    patient_idx: list[int],
    test_y: np.ndarray,
    test_m: np.ndarray,
    test_metrics: AuxLosses,
    lookup_table: LatentLookup,
) -> tuple[Figure, Axes]:
    if not isinstance(patient_idx, list):
        patient_idx = [patient_idx]

    fig = plt.figure()
    gs = GridSpec(len(patient_idx), 2, figure=fig, width_ratios=[1.3, 1], wspace=0.2, hspace=0.4)
    axl = fig.add_subplot(gs[:, 0])

    viz_patients_latent(
        true_sofa=test_y[0, patient_idx, :, 0],
        betas=np.asarray(test_metrics.beta[0, patient_idx]),
        sigmas=np.asarray(test_metrics.sigma[0, patient_idx]),
        mask=test_m[0, patient_idx],
        zoom=True,
        window_size=5,
        lookup=lookup_table,
        cmaps=True,
        figax=(fig, axl),
    )

    for i, pidx in enumerate(patient_idx):
        axr = fig.add_subplot(gs[i, 1])
        m = test_m[0, pidx]
        axr.plot(test_y[0, pidx, :, 0][m], label="Ground Truth")
        axr.plot(test_metrics.hists_sofa_score[0, pidx, :][m], label="Prediction")
        axr.set_ylim(0, 24)
        axr.set_ylabel(f"Patient {i + 1}")
        axr.set_ylim(0, 24)
        axr.set_yticks(range(25))

        axr.grid(visible=True, axis="y", which="major", color="gray", linestyle=":", alpha=0.4)
        axr.set_yticks([0, 4, 8, 12, 16, 20, 24], minor=False)
        axr.set_yticks(np.arange(0, 25, 2), minor=True)

        axr.grid(visible=True, axis="y", which="both", alpha=0.8)
        if i == 0:
            axr.set_title("SOFA-score")
        if i == len(patient_idx) - 1:
            axr.set_xlabel("time [hours]")
            axr.legend(ncols=2, bbox_to_anchor=(1, -0.6), frameon=False)

    fig.subplots_adjust(bottom=0.2)
    return fig, axl


def viz_concept_densities(
    true_sofa: jnp.ndarray | np.ndarray,
    true_infs: jnp.ndarray | np.ndarray,
    pred_sofa: jnp.ndarray | np.ndarray,
    pred_infs: jnp.ndarray | np.ndarray,
    *,
    show_inf: str | None = None,
    cmap: bool,
    figax: tuple[Figure, tuple[Axes, Axes, Axes]] | None = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(1, 3 if show_inf is not None else 2)

    sofa_bins = [np.arange(-0.5, 24.5, 1), np.arange(-0.5, 24.5, 1)]
    sofa_heatmap = np.histogram2d(true_sofa, pred_sofa, bins=sofa_bins)
    sofa = (sofa_heatmap, "SOFA-score", ax[0])

    dsofa_bins = [np.arange(-24.5, 24.5, 1), np.arange(-24.5, 24.5, 1)]  # 1-point bins
    dsofa_heatmap = np.histogram2d(np.diff(true_sofa), np.diff(pred_sofa), bins=dsofa_bins)
    dsofa = (dsofa_heatmap, r"$\Delta$SOFA-score", ax[1])

    if show_inf is not None:
        if show_inf == "temporal":
            inf_bins = [np.linspace(0, 169, 40), np.linspace(-169, 169, 40)]
            inf_heatmap = np.histogram2d(true_infs, pred_infs, bins=inf_bins)
            inf = (inf_heatmap, "Temporal Infection Alignment", ax[2])
        elif show_inf == "smoothed":
            inf_bins = [np.arange(-0.05, 1.05, 0.1), np.arange(-0.05, 1.05, 0.1)]
            inf_heatmap = np.histogram2d(true_infs, pred_infs, bins=inf_bins)
            inf = (inf_heatmap, "Suspected Infection", ax[2])
        else:
            raise NotImplementedError("Only 'smoothed' or 'temporal'")

    heats = np.log1p(
        np.concat(
            [
                np.asarray(sofa_heatmap[0]).flatten(),
                np.asarray(dsofa_heatmap[0]).flatten(),
                np.asarray(inf_heatmap[0]).flatten() if show_inf is not None else np.array([0]),
            ]
        )
    )
    norm = colors.Normalize(vmin=np.min(heats), vmax=np.max(heats))

    images = []

    for _i, ((heatmap, xedges, yedges), name, axi) in enumerate(
        [sofa, dsofa, inf] if show_inf is not None else [sofa, dsofa]
    ):
        # Plot the 2D histogram as an image
        masked_heatmap = np.ma.masked_where(heatmap <= 0, heatmap)
        log_data = np.log1p(masked_heatmap.T)
        images.append(
            axi.imshow(
                log_data,  # transpose so it matches axes orientation
                origin="lower",
                extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                aspect="auto",
                cmap="OrRd",  # OrRd
                norm=norm,
            )
        )
        axi.set_box_aspect(1)
        axi.set_xlabel("Ground Truth")
        axi.plot(
            (xedges[0], xedges[-1]),
            (0 if _i == 2 else yedges[0], yedges[-1]),
            color="black",
            label="Optimal",
            linestyle=":",
        )
        axi.set_title(name)
    if cmap:
        pos = ax[-1].get_position()

        # Create colorbar axes outside the plot
        cax = fig.add_axes((pos.x1 + 0.05, pos.y0 + 0.1, 0.02, pos.height))
        fig.colorbar(images[-1], cax=cax, label="log(Count)")

    ax[0].legend(frameon=False)
    ax[0].set_ylabel("Predicted")
    fig.subplots_adjust(wspace=0.2, top=0.9, bottom=0.15)

    ax[0].text(-0.15, 1.0, r"\textbf{A}", transform=ax[0].transAxes, fontsize=14, fontweight="bold", va="top")
    ax[1].text(-0.15, 1.0, r"\textbf{B}", transform=ax[1].transAxes, fontsize=14, fontweight="bold", va="top")
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
    ax[1].legend()

    plt.tight_layout()
    if filename and figure_dir:
        plt.savefig(f"{figure_dir}/{filename}", transparent=True, dpi=800)
    return ax


def viz_curves_sepsis(true_sep3: jnp.ndarray, pred_sep3_risk: jnp.ndarray) -> Figure:
    fig, ax = plt.subplots(1, 2)
    roc = RocCurveDisplay.from_predictions(
        true_sep3, pred_sep3_risk, ax=ax[0], curve_kwargs={"color": "tab:orange"}, plot_chance_level=True
    )
    roc.ax_.plot((0, 0), (1, 1), label="random", color="red", linestyle=":")
    PrecisionRecallDisplay.from_predictions(
        true_sep3, pred_sep3_risk, ax=ax[1], color="tab:orange", plot_chance_level=True
    )

    ax[0].set_title("Receiver Operating Characteristics")
    ax[1].set_title("Precision Recall Curve")
    ax[1].legend(loc="upper right")

    return fig


def viz_losses(tb_df: pd.DataFrame, hparams: pd.DataFrame) -> Figure:
    fig, axs = plt.subplots(3, 3, sharex=True)

    data_train = tb_df.query("tag == 'train_losses/total_loss_mean'")
    data_val = tb_df.query("tag == 'val_losses/total_loss_mean'")
    axs[0, 0].plot(data_train["step"], data_train["value"], label="Training")
    axs[0, 0].plot(data_val["step"], data_val["value"], label="Validation")
    axs[0, 0].legend(ncols=2, bbox_to_anchor=(2.4, 1.75), frameon=False)
    axs[0, 0].set_title(r"log($L_\text{total}$)")
    axs[0, 0].grid(True)
    axs[0, 0].set_yscale("log")
    print((data_train["value"]).to_numpy()[[0, -1]], (data_val["value"]).to_numpy()[[0, -1]])

    data = tb_df.query("tag == 'sepsis_metrics/AUROC_pred_sep'")
    axs[0, 1].plot(data["step"], data["value"], c="tab:orange")
    axs[0, 1].set_title(r"AUROC")
    axs[0, 1].grid(True)

    data = tb_df.query("tag == 'sepsis_metrics/AUPRC_pred_sep'")
    axs[0, 2].plot(data["step"], data["value"], c="tab:orange")
    axs[0, 2].set_title(r"AUPRC")
    axs[0, 2].grid(True)

    losses = ("sepsis-3", "sofa", "infection", "recon_loss", "spreading_loss", "boundary_loss")
    loss_subscripts = ("sepsis", "sofa", "inf", "dec", "spread", "boundary")
    lambdas = ("sep3", "sofa_classification", "inf", "recon", "spreading", "boundary")

    # axs = np.empty((2, 3), dtype=np.object_)
    for i, (loss, subscript, weight) in enumerate(zip(losses, loss_subscripts, lambdas, strict=True)):
        ax = axs[i // 3 + 1, i % 3]
        data_train = tb_df.query(f"tag == 'train_losses/{loss}_mean'")
        data_val = tb_df.query(f"tag == 'val_losses/{loss}_mean'")
        llambda = hparams.loc["value", f"losses_lambda_{weight}"]
        print(
            loss, (data_train["value"] * llambda).to_numpy()[[0, -1]], (data_val["value"] * llambda).to_numpy()[[0, -1]]
        )
        ax.plot(data_train["step"], data_train["value"] * llambda, label="Training")
        ax.plot(data_val["step"], data_val["value"] * llambda, label="Validation")
        ax.set_title(rf"$L_\text{'{'}{subscript}{'}'}\lambda_\text{'{'}{subscript}{'}'}$")
        ax.grid(True)
        if i // 3 == 1:
            ax.set_xlabel("Epoch")

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.3)
    return fig


def viz_loss_mean_std(
    agg_data: dict[str, pd.DataFrame],
    losses: tuple[str],
    loss_subscripts: tuple[str],
    lambdas: tuple[str],
    hparams: pd.DataFrame,
    *,
    alpha: float = 0.4,
) -> Figure:
    fig, axs = plt.subplots(3, 3, sharex=True)

    # Total loss
    total_loss = agg_data["train_losses/total_loss_mean"]
    val_loss = agg_data["val_losses/total_loss_mean"]
    axs[0, 0].plot(total_loss["step"], total_loss["mean"], label="Training")
    axs[0, 0].fill_between(
        total_loss["step"], total_loss["mean"] - total_loss["std"], total_loss["mean"] + total_loss["std"], alpha=alpha
    )
    axs[0, 0].plot(val_loss["step"], val_loss["mean"], label="Validation")
    axs[0, 0].fill_between(
        val_loss["step"], val_loss["mean"] - val_loss["std"], val_loss["mean"] + val_loss["std"], alpha=alpha
    )
    axs[0, 0].legend(ncols=2, bbox_to_anchor=(2.3, -3.5), frameon=False)
    axs[0, 0].set_title(r"log($\cal{L}_{\mathrm{total}}$)")
    axs[0, 0].grid(True)
    axs[0, 0].set_yscale("log")

    # AUROC & AUPRC
    for i, metric in enumerate(["sepsis_metrics/AUROC_pred_sep", "sepsis_metrics/AUPRC_pred_sep"]):
        df = agg_data[metric]
        axs[0, i + 1].plot(df["step"], df["mean"], c="tab:orange")
        axs[0, i + 1].fill_between(
            df["step"], df["mean"] - df["std"], df["mean"] + df["std"], color="tab:orange", alpha=alpha
        )
        axs[0, i + 1].set_title(metric.split("/")[-1].split("_")[0])
        axs[0, i + 1].grid(True)

    # Individual losses
    if losses:
        for i, (loss, subscript, weight) in enumerate(zip(losses, loss_subscripts, lambdas, strict=True)):
            ax = axs[i // 3 + 1, i % 3]
            train_tag = f"train_losses/{loss}_mean"
            val_tag = f"val_losses/{loss}_mean"
            if train_tag not in agg_data:
                continue
            train_df = agg_data[train_tag].copy()
            val_df = agg_data[val_tag].copy()
            llambda = hparams.loc["value", f"losses_lambda_{weight}"] if hparams is not None else 1.0
            train_df["mean"] *= llambda
            train_df["std"] *= llambda
            val_df["mean"] *= llambda
            val_df["std"] *= llambda

            ax.plot(train_df["step"], train_df["mean"], label="Training")
            ax.fill_between(
                train_df["step"], train_df["mean"] - train_df["std"], train_df["mean"] + train_df["std"], alpha=alpha
            )
            ax.plot(val_df["step"], val_df["mean"], label="Validation")
            ax.fill_between(val_df["step"], val_df["mean"] - val_df["std"], val_df["mean"] + val_df["std"], alpha=alpha)

            ax.set_title(rf"$\cal{{L}}_\mathrm{{{subscript}}}\lambda_\mathrm{{{subscript}}}$")
            ax.grid(True)
            if i // 3 == 1:
                ax.set_xlabel("Epoch")

    fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.2, left=0.07, right=0.99)
    return fig
