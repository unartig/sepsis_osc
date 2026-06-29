import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

from sepsis_osc.ldm.analysis_helper import AnalysisResult, RunResult, Scenario
from sepsis_osc.ldm.commons import get_space_vals
from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel
from sepsis_osc.utils.config import AGE_FEATURES, BETA_SPACE, HIGH_SOFA_THRESH, INFL_FEATURES, SIGMA_SPACE, SOFA_FLAT
from sepsis_osc.visualisations.viz_model_results import viz_space_distribution_countour
from sepsis_osc.visualisations.viz_param_space import space_plot

color_map = {
    "SOFA": "#E66101",
    "Inflammation": "#5E3C99",
    "Age/Other": "#FDB863",
    "Default": "#9B3A8C",
}


def get_feat_color(feat: str) -> str:
    if feat in SOFA_FLAT:
        return color_map["SOFA"]
    if feat in INFL_FEATURES:
        return color_map["Inflammation"]
    if feat in AGE_FEATURES:
        return color_map["Age/Other"]
    return color_map["Default"]


colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#111111",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
]


def viz_alignment_and_recon(
    corr_df: pd.DataFrame, val: str | None, title: str = r"Feature alignment with $\beta$ / $\sigma$"
) -> plt.Figure:
    corr_df = corr_df.copy()

    corr_df["_sort_key"] = corr_df.mean(axis=1)
    if val is not None:
        corr_df["_val"] = val
        corr_df = corr_df.sort_values("_sort_key", ascending=False)
        val = corr_df["_val"].to_numpy()
        corr_df = corr_df.drop(columns="_val")
    else:
        corr_df = corr_df.sort_values("_sort_key", ascending=False)

    fig, ax1 = plt.subplots(layout="constrained")
    x = np.arange(len(corr_df)) * 1.5
    w = 0.38

    hatches = [None, "///", "..."]
    metric_legends = []
    for i, col in enumerate(corr_df.columns):
        if col == "_sort_key":
            continue
        bar_colors = [get_feat_color(f) for f in corr_df.index]
        bars = ax1.bar(
            x + (i - 0.5) * w,
            corr_df[col].values,
            width=w,
            color=bar_colors,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=0.5,
        )

        for bar in bars:
            bar.set_hatch(hatches[i])

        metric_legends.append(
            mpatches.Patch(
                facecolor=get_feat_color(""),
                hatch=hatches[i],
                label=rf"$r_\{col}$" if col in ("beta", "sigma") else rf"$r_{{\mathrm{{{col}}}}}$",
            )
        )

    group_legends = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[grp_name], alpha=0.85, label=grp_name)
        for grp_name in ("SOFA", "Inflammation", "Age/Other")
    ]

    ax1.legend(handles=metric_legends + group_legends, loc="upper right", frameon=True)

    ax1.set_xticks(x)
    ax1.set_xticklabels(corr_df.index, rotation=90)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set(xlabel="Feature", ylabel="Pearson r", title=title)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_xlim(-0.7, x.max() + 0.7)

    val_line = None
    if val is not None:
        ax2 = ax1.twinx()
        (val_line,) = ax2.plot(x, val, color="#45b3e7", linestyle="-", marker=".", linewidth=1.2, label=r"Pearson $r$")
        ax2.set_ylabel(r"Pearson $r$", color="#45b3e7")
        ax2.tick_params(axis="y", labelcolor="#45b3e7")

        global_min = min(corr_df.min().min(), val.min())
        global_max = max(corr_df.max().max(), val.max())
        padding = (global_max - global_min) * 0.05
        ymin = global_min - padding
        ymax = global_max + padding
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

    group_legends = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[grp_name], alpha=0.85, label=grp_name)
        for grp_name in ("SOFA", "Inflammation", "Age/Other")
    ]

    all_legends = metric_legends + group_legends + ([val_line] if val_line is not None else [])
    ax1.legend(
        handles=all_legends, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(all_legends), frameon=True
    )

    ax1.set_title(title)
    return fig


def viz_latent_scatter(
    beta_ts: np.ndarray,
    sigma_ts: np.ndarray,
    model: LatentDynamicsModel,
    color_vals: np.ndarray,
    label: str,
    cmap: str = "gray",
    resolution: float = 0.25,
    title: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots()

    bins = [
        np.arange(BETA_SPACE[0], BETA_SPACE[1] - BETA_SPACE[2], BETA_SPACE[2] * resolution),
        np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1] - SIGMA_SPACE[2], SIGMA_SPACE[2] * resolution),
    ]

    _betas_space, _sigmas_space, space_vals = get_space_vals(model.lookup)
    space_plot(
        space_vals,
        xs=np.arange(*BETA_SPACE),
        ys=np.arange(*SIGMA_SPACE),
        title="",
        cmap=False,
        filename="",
        alpha=0.7,
        xticklabel_rot=0,
        num_ticks=5,
        figax=(fig, ax),
    )

    bin_stats, x_edges, y_edges, _binnumber = stats.binned_statistic_2d(
        x=beta_ts, y=sigma_ts, values=color_vals, statistic="mean", bins=bins
    )
    extent = (
        x_edges[0],
        x_edges[-1],
        y_edges[0],
        y_edges[-1],
    )
    im = ax.imshow(bin_stats.T, extent=extent, origin="lower", cmap=cmap, interpolation="nearest", aspect="auto")

    plt.colorbar(im, ax=ax, label=label)
    ax.set(xlabel=r"$\beta$", ylabel=r"$\sigma$", title=title or f"Latent space mean for feature: {label}")

    return fig


def viz_latent_scatter_grid(
    beta_ts: np.ndarray,
    sigma_ts: np.ndarray,
    model: LatentDynamicsModel,
    color_vals_dict: dict,
    cmap: str = "gray",
    resolution: float = 0.25,
    title: str | None = None,
) -> plt.Figure:
    labels = list(color_vals_dict.keys())
    num_plots = len(labels)
    num_cols = 3
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols)

    axes = np.array([axes]) if num_plots == 1 else axes.flatten()

    bins = [
        np.arange(BETA_SPACE[0], BETA_SPACE[1] - BETA_SPACE[2], BETA_SPACE[2] * resolution),
        np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1] - SIGMA_SPACE[2], SIGMA_SPACE[2] * resolution),
    ]

    _betas_space, _sigmas_space, space_vals = get_space_vals(model.lookup)

    for i, label in enumerate(labels):
        ax = axes[i]
        color_vals = color_vals_dict[label]

        space_plot(
            space_vals,
            xs=np.arange(*BETA_SPACE),
            ys=np.arange(*SIGMA_SPACE),
            title="",
            cmap=False,
            filename="",
            alpha=0.7,
            xticklabel_rot=0,
            num_ticks=5,
            figax=(fig, ax),
        )

        bin_stats, x_edges, y_edges, _ = stats.binned_statistic_2d(
            x=beta_ts, y=sigma_ts, values=color_vals, statistic="mean", bins=bins
        )
        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])

        im = ax.imshow(bin_stats.T, extent=extent, origin="lower", cmap=cmap, interpolation="nearest", aspect="auto")

        plt.colorbar(im, ax=ax, label=label)
        ax.set(xlabel=r"$\beta$", ylabel=r"$\sigma$", title=label)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    return fig


def viz_subgroup_separation(
    beta_ts: np.ndarray,
    sigma_ts: np.ndarray,
    model: LatentDynamicsModel,
    sep: np.ndarray,
    inf_: np.ndarray,
    peak_sofa: np.ndarray,
) -> plt.Figure:
    subgroups = [
        {"title": "Sepsis vs Non-Sepsis", "ma": sep == 1.0, "mb": sep != 1.0, "la": "Sepsis", "lb": "No sepsis"},
        {
            "title": "Infection vs No Infection",
            "ma": inf_ > 0,
            "mb": inf_ == 0,
            "la": "Infection",
            "lb": "No infection",
        },
        {
            "title": rf"Peak SOFA $\geq$ {HIGH_SOFA_THRESH}",
            "ma": peak_sofa >= HIGH_SOFA_THRESH,
            "mb": peak_sofa < HIGH_SOFA_THRESH,
            "la": rf"SOFA $\geq$ {HIGH_SOFA_THRESH}",
            "lb": rf"SOFA $<$ {HIGH_SOFA_THRESH}",
        },
    ]
    fig = plt.figure()

    outer_gs = GridSpec(1, len(subgroups), figure=fig, wspace=0.4)

    *_, space_vals = get_space_vals(model.lookup)

    for col, sg in enumerate(subgroups):
        inner_gs = outer_gs[0, col].subgridspec(7, 4, hspace=0.05, wspace=0.05)

        ax_histx = fig.add_subplot(inner_gs[0, 0:3])
        ax_scat = fig.add_subplot(inner_gs[1:, 0:3], sharex=ax_histx)
        ax_histy = fig.add_subplot(inner_gs[1:, 3], sharey=ax_scat)

        space_plot(
            space_vals,
            xs=np.arange(*BETA_SPACE),
            ys=np.arange(*SIGMA_SPACE),
            title="",
            cmap=False,
            filename="",
            alpha=0.7,
            xticklabel_rot=0,
            num_ticks=5,
            figax=(fig, ax_scat),
        )

        group_data = [
            {"mask": sg["mb"], "color": "#378ADD", "label": sg["lb"], "alpha_scat": 0.15},
            {"mask": sg["ma"], "color": "#D85A30", "label": sg["la"], "alpha_scat": 0.25},
        ]

        beta_bins = np.linspace(beta_ts.min(), beta_ts.max(), 40)
        sigma_bins = np.linspace(sigma_ts.min(), sigma_ts.max(), 40)

        for g in group_data:
            m, c, l = g["mask"], g["color"], g["label"]

            m = np.asarray(m)
            c = str(c)
            ax_scat.scatter(beta_ts[m], sigma_ts[m], c=c, alpha=float(g["alpha_scat"]), s=1.5, label=l, rasterized=True)
            ax_scat.scatter(
                beta_ts[m].mean(), sigma_ts[m].mean(), c=c, marker="X", s=120, edgecolors="black", lw=0.8, zorder=5
            )

            ax_histx.hist(beta_ts[m], bins=beta_bins, color=c, alpha=0.5, density=False)
            ax_histy.hist(sigma_ts[m], bins=sigma_bins, color=c, alpha=0.5, density=False, orientation="horizontal")

        ax_histx.set_title(str(sg["title"]), pad=10)
        ax_scat.set(xlabel=r"$\beta$")
        ax_histy.set_xlabel("n Samples")

        if col == 0:
            ax_scat.set_ylabel(r"$\sigma$")
            ax_histx.set_ylabel("n Samples")

        leg = ax_scat.legend(markerscale=4, loc="lower left")
        for lh in leg.legend_handles:
            if lh:
                lh.set_alpha(1.0)

        ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_histy.tick_params(axis="y", labelleft=False, left=False)
        ax_histx.tick_params(axis="y", labelsize=8)
        ax_histy.tick_params(axis="x", labelsize=8)
        ax_histx.set_yticks([ax_histx.get_ylim()[1]])
        ax_histy.set_xticks([ax_histy.get_xlim()[1]])
        ax_histx.grid(axis="y", linestyle="--")
        ax_histy.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
        ax_histx.set_axisbelow(True)
        ax_histy.set_axisbelow(True)

    fig.tight_layout()
    return fig


def viz_per_patient_recon(
    recon_np: np.ndarray,
    actual_np: np.ndarray,
    mask_np: np.ndarray,
    feat_names: list[str],
    feat_order: list[int],
    pat_idx: int = 0,
) -> plt.Figure:
    actual = actual_np[pat_idx, mask_np[pat_idx]]
    recon_p = recon_np[pat_idx, mask_np[pat_idx]]
    n_feat = len(feat_names)

    n_cols = 5
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 1.8), squeeze=False)
    for viz_i, feat_i in enumerate(range(n_feat)):
        ax = axes.ravel()[viz_i]
        ax.plot(actual[:, feat_i], color="#378ADD", lw=1.2, label="actual")
        ax.plot(recon_p[:, feat_i], color="#D85A30", lw=1.2, ls="--", label="recon")
        ax.set_title(f"{feat_names[feat_i]}", pad=2)
        ax.tick_params()
    for ax in axes.ravel()[n_feat:]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True, ncols=2)
    fig.suptitle(f"Patient {pat_idx} : Ground Truth and Reconstructed", y=0.96)
    plt.tight_layout()
    return fig


def viz_per_recon_dist(
    recon_np: np.ndarray, actual_np: np.ndarray, mask_np: np.ndarray, feat_names: list[str]
) -> plt.Figure:
    actual = actual_np[mask_np]
    recon_p = recon_np[mask_np]

    n_feat = len(feat_names)

    n_cols = 5
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 1.8), squeeze=False)

    for feat_i in range(n_feat):
        ax = axes.ravel()[feat_i]

        combined = np.concatenate([actual[:, feat_i], recon_p[:, feat_i]])
        bins = np.histogram_bin_edges(combined, bins=50)

        ax.hist(actual[:, feat_i], bins=bins, alpha=0.5, density=False, label="Ground Truth")
        ax.hist(recon_p[:, feat_i], bins=bins, alpha=0.5, density=False, label="Reconstructed")

        ax.set_title(feat_names[feat_i], pad=2)

    for ax in axes.ravel()[n_feat:]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True, ncols=2)
    fig.suptitle("Distributions: Ground Truth and Reconstructed", y=0.96)

    plt.tight_layout()
    return fig


def viz_per_recon_densities(
    recon_np: np.ndarray, actual_np: np.ndarray, mask_np: np.ndarray, feat_names: list[str]
) -> plt.Figure:
    actual = actual_np[mask_np]
    recon_p = recon_np[mask_np]

    n_feat = len(feat_names)
    n_cols = 5
    n_rows = int(np.ceil(n_feat / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    shared_norm = SymLogNorm(linthresh=0.01, vmin=1, vmax=actual.shape[0] // 10)

    im = None

    for feat_i in range(n_feat):
        ax = axes.ravel()[feat_i]
        x_data = actual[:, feat_i]
        y_data = recon_p[:, feat_i]

        im = ax.hist2d(x_data, y_data, bins=30, norm=shared_norm, cmap="OrRd")

        min_val = min(x_data.min(), y_data.min())
        max_val = max(x_data.max(), y_data.max())

        padding = (max_val - min_val) * 0.05 if max_val != min_val else 1.0
        viz_min = min_val - padding
        viz_max = max_val + padding

        ax.plot(
            [viz_min, viz_max],
            [viz_min, viz_max],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )

        ax.set_xlim(viz_min, viz_max)
        ax.set_ylim(viz_min, viz_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(feat_names[feat_i])

        row_i = feat_i // n_cols
        col_i = feat_i % n_cols
        is_bottom_row = row_i == n_rows - 1
        is_last_in_col = feat_i + n_cols >= n_feat
        if is_bottom_row or is_last_in_col:
            ax.set_xlabel("Actual")

        if col_i == 0:
            ax.set_ylabel("Reconstructed")

    # Hide unused axes
    for ax in axes.ravel()[n_feat:]:
        ax.set_visible(False)

    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    fig.suptitle("Prediction: Ground Truth and Reconstructed", y=0.98)

    cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))  # [left, bottom, width, height]
    if im:
        fig.colorbar(im[3], cax=cbar_ax, label="log(Density)")

    return fig


def viz_s2_latent_stability(results: list[AnalysisResult]) -> plt.Figure:
    beta_stats = {
        "mu": np.array([r.run.beta.mean() for r in results]),
        "sd": np.array([r.run.beta.std() for r in results]),
    }

    sigma_stats = {
        "mu": np.array([r.run.sigma.mean() for r in results]),
        "sd": np.array([r.run.sigma.std() for r in results]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, coord, latent_stats in [
        (axes[0], r"$\beta$", beta_stats),
        (axes[1], r"$\sigma$", sigma_stats),
    ]:
        x = np.arange(len(latent_stats["mu"]))
        ax.errorbar(x, latent_stats["mu"], yerr=latent_stats["sd"], fmt="o", capsize=3, color="#378ADD", lw=1.2)
        ax.axhline(
            latent_stats["mu"].mean(),
            ls="--",
            color="#D85A30",
            lw=1,
            label=f"overall mean={latent_stats['mu'].mean():.3f}",
        )
        ax.set(xlabel="Split", ylabel=coord, title=f"{coord} mean +/- std across splits")
    plt.tight_layout()
    return fig


def viz_cohens_d_distribution(all_stats_dfs: list[pd.DataFrame], title: str = "Cohen's d across splits") -> plt.Figure:
    """all_stats_dfs: list of DataFrames from compute_subgroup_stats."""

    combined = pd.concat(all_stats_dfs, ignore_index=True)
    groups = combined.groupby(["comparison", "coord"])["cohens_d"]

    keys = list(groups.groups.keys())
    means = [groups.get_group(k).mean() for k in keys]
    stds = [groups.get_group(k).std() for k in keys]
    labels = [f"{c}\n{coord}" for c, coord in keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#378ADD", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="Cohen's d", title=title)
    plt.tight_layout()
    return fig


def draw_confidence_ellipse(
    x: np.ndarray, y: np.ndarray, ax: plt.Axes, n_std: float = 1.96, facecolor: str = "none", **kwargs
) -> Patch | None:
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def viz_comparison(all_results: dict[str, Scenario], title: str = "Experiment comparison") -> plt.Figure:
    tasks = [
        {"title": "Sepsis-3", "x": "auroc_sep3", "y": "auprc_sep3"},
        {"title": r"$\Delta$SOFA $\geq$ 2", "x": "auroc_sofa_d2", "y": "auprc_sofa_d2"},
        {"title": "Susp. Infection", "x": "auroc_inf", "y": "auprc_inf"},
    ]
    scenarios = list(all_results.keys())

    fig, axes = plt.subplots(1, len(tasks))
    fig.suptitle(title, fontweight="bold")

    if len(tasks) == 1:
        axes = [axes]

    for col, task in enumerate(tasks):
        ax = axes[col]

        for color, sc in zip(colors, scenarios, strict=False):
            df = all_results[sc].perf_df
            if task["x"] not in df or task["y"] not in df:
                continue

            x_data = df[task["x"]].dropna()
            y_data = df[task["y"]].dropna()

            # 1. Scatter plot
            ax.scatter(
                x_data,
                y_data,
                label=sc,
                color=color,
                alpha=0.4,
                edgecolors="w",
                s=30,
            )

            # 2. 95% Confidence Ellipse (using 1.96 for approx 95% CI bounds)
            draw_confidence_ellipse(
                x_data, y_data, ax, n_std=1.96, edgecolor=color, linestyle="-", linewidth=1.5, alpha=0.8
            )

            # 3. Optional: Plot the mean center point
            ax.scatter(x_data.mean(), y_data.mean(), color=color, edgecolors="black", marker="o", s=60, zorder=5)

        ax.set_title(task["title"])
        ax.set_xlabel("AUROC")
        ax.grid(True, linestyle="--", alpha=0.4)

        if col == 0:
            ax.set_ylabel("AUPRC")

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def viz_alignment_heatmap(
    entries: list[tuple[str, pd.DataFrame]],
    perf_metric: dict[str, np.ndarray] | None = None,
    title: str | None = None,
    row_label_every: int = 5,
) -> plt.Figure:
    feats = list(entries[0][1].index)
    row_labels = [e[0] for e in entries]
    r_beta_mat = np.array([e[1]["beta"].array for e in entries])  # (rows, feats)
    r_sigma_mat = np.array([e[1]["sigma"].array for e in entries])

    # sort features by combined stability score
    score = 0.5 * (r_beta_mat.mean(0) + r_sigma_mat.mean(0))
    sort_idx = np.argsort(score)[::-1]
    feats = [feats[i] for i in sort_idx]
    r_beta_mat = r_beta_mat[:, sort_idx]
    r_sigma_mat = r_sigma_mat[:, sort_idx]

    n_feats = len(feats)
    n_rows = r_beta_mat.shape[0]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("RdBu_r")

    has_fit = perf_metric is not None
    n_panels = 3 + (1 if has_fit else 0)
    height_ratios = ([0.8] if has_fit else []) + [1, 1, 0.12]

    fig, axes = plt.subplots(
        n_panels,
        1,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )
    if has_fit:
        ax_fit, ax_beta, ax_sigma, ax_cat = axes
    else:
        has_fit = None
        ax_beta, ax_sigma, ax_cat = axes

    # optional R^2 panel
    if has_fit:
        fit_mat = np.array([perf_metric[lbl] for lbl in row_labels])[:, sort_idx]  # (rows, feats)
        fit_mean = fit_mat.mean(0)
        fit_std = fit_mat.std(0)
        x = np.arange(n_feats)
        ax_fit.errorbar(
            x,
            fit_mean,
            yerr=fit_std,
            fmt="o",
            color="#45b3e7",
            markersize=3,
            capsize=2,
            elinewidth=1,
            label=r"Pearson $r$ mean ± std",
        )
        ax_fit.set_xlim(-0.5, n_feats - 0.5)
        ax_fit.set_xticks([])
        ax_fit.set_ylabel(rf"Decoder{'\n'}Performance")
        ax_fit.axhline(0, color="gray", lw=0.6, ls="--")
        ax_fit.legend(loc="best")

    # beta / sigma heatmaps
    for ax, mat, label in [(ax_beta, r_beta_mat, r"$\beta$"), (ax_sigma, r_sigma_mat, r"$\sigma$")]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        tick_idx = np.arange(n_rows)[::row_label_every] if n_rows > row_label_every else np.arange(n_rows)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([row_labels[i] for i in tick_idx])
        ax.set_ylabel(label, rotation=0, labelpad=14, va="center")
        ax.set_xticks([])

    # category bar
    cat_colors = [get_feat_color(f) for f in feats]
    ax_cat.imshow(
        np.array([plt.matplotlib.colors.to_rgba(c) for c in cat_colors])[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
    )
    ax_cat.set_yticks([])
    ax_cat.set_xticks(np.arange(n_feats))
    ax_cat.set_xticklabels(feats, rotation=90)
    ax_cat.xaxis.set_tick_params(length=0, pad=2)

    # colorbar
    cax = inset_axes(
        ax_beta,
        width="1.5%",
        height="200%",
        loc="upper right",
        bbox_to_anchor=(0.02, 0, 1, 1),
        bbox_transform=ax_beta.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pearson r")

    handles = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
    fig.legend(
        handles=handles, loc="upper center", ncols=len(entries), frameon=True, bbox_to_anchor=(0.5, 0.965), fontsize=9
    )

    fig.suptitle(title or r"Feature alignment stability: $\beta$ and $\sigma$", y=0.99)
    return fig


def viz_dists_cv(perf_df: pd.DataFrame, results: list[AnalysisResult]) -> plt.Figure:
    cv_fig, _axs = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True)

    _best = perf_df.sort_values(by="auroc_sep3")[-2:].reset_index()
    _worst = perf_df.sort_values(by="auroc_sep3")[:2].reset_index()

    # map (rep, fold) -> AnalysisResult for quick lookup
    by_split = {(r.rep, r.fold): r for r in results}

    for _i, _df in enumerate((_best, _worst)):
        for _ind, _row in _df.iterrows():
            _rep, _fold = int(_row["rep"]), int(_row["fold"])
            _ax = _axs[_ind + _i * len(_best)]
            _result = by_split[(_rep, _fold)]
            _metrics = _result.run.metrics
            _m = _result.run.mask

            betas_space, sigmas_space, metric_np = get_space_vals(_result.run.model.lookup)

            space_plot(
                metric_np,
                xs=np.asarray(betas_space),
                ys=np.asarray(sigmas_space),
                title="",
                cmap=False,
                filename="",
                alpha=0.7,
                xticklabel_rot=0,
                num_ticks=5,
                figax=(cv_fig, _ax),
            )
            viz_space_distribution_countour(
                _metrics.beta, _metrics.sigma, _m, show_cmap=False, show_inlay_notation=False, figax=(cv_fig, _ax)
            )

            _ax.annotate(
                f"Rep.: {_rep + 1}, Fold: {_fold + 1}, \nAUROC: {_row['auroc_sep3']:.2f}, AUPRC: {_row['auprc_sep3']:.2f}",
                (0.41, 0.05),
                color="white",
                weight="bold",
                fontsize=8,
            )

            _ax.set_yticklabels([])
            if _i != 0 or _ind != 0:
                _ax.set_ylabel("")
            _ax.set_xticklabels([])

    _cmap = plt.get_cmap("OrRd")
    _norm = matplotlib.colors.BoundaryNorm([0, 1.5, 3, 4.5, 6], _cmap.N)
    _sm = plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)

    _cbar = cv_fig.colorbar(_sm, ax=_axs, location="right", fraction=0.0175, pad=0.04)
    if _cbar:
        _cbar.solids.set_edgecolor("none")
        _cbar.solids.set_rasterized(False)
        _cbar.set_label("log(Density)", rotation=90, labelpad=15)

    cv_fig.suptitle("Latent Distributions of best and worst performing splits")
    return cv_fig


def viz_decoder_perf_heatmap(
    scenarios: dict[str, Scenario],
    feat_names: list[str],
    title: str = r"Decoder Pearson $r$ scenario comparison",
    exclude_from_scale: tuple[str] = ("standard_no_recon",),
) -> plt.Figure:
    names = list(scenarios.keys())
    perf_means, perf_stds = [], []
    for name in names:
        all_perf = np.array([r.recon_pr for r in scenarios[name].results])  # (splits, feats)
        perf_means.append(all_perf.mean(axis=0))
        perf_stds.append(all_perf.std(axis=0))
    perf_means = np.array(perf_means)  # (scenarios, feats)
    perf_stds = np.array(perf_stds)

    overall = perf_means.mean(axis=0)
    sort_idx = np.argsort(overall)[::-1]
    sorted_feats = [feat_names[i] for i in sort_idx]
    perf_means = perf_means[:, sort_idx]
    perf_stds = perf_stds[:, sort_idx]

    # color scale based only on non-excluded scenarios
    scale_mask = np.array([n not in exclude_from_scale for n in names])
    mean_vmax = np.nanmax(np.abs(perf_means[scale_mask]))
    mean_vmin = -mean_vmax
    std_vmax = np.nanmax(perf_stds[scale_mask])

    fig, axes = plt.subplots(2, 1, sharex=True)

    im0 = axes[0].imshow(perf_means, aspect="auto", cmap="RdBu_r", vmin=mean_vmin, vmax=mean_vmax)
    plt.colorbar(im0, ax=axes[0], label="mean Pearson $r$")
    axes[0].set_yticks(np.arange(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_title(f"{title} mean")

    im1 = axes[1].imshow(perf_stds, aspect="auto", cmap="viridis", vmin=0, vmax=std_vmax)
    plt.colorbar(im1, ax=axes[1], label="std Pearson $r$")
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_xticks(np.arange(len(sorted_feats)))
    axes[1].set_xticklabels(sorted_feats, rotation=90)
    axes[1].set_title(f"{title} std across splits")

    plt.tight_layout()
    return fig
