import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import logging

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import matplotlib.style
    import scipy.ndimage as ndimage
    import numpy as np
    import pandas as pd

    from diffrax import Dopri5, Dopri8, Tsit5
    from matplotlib import colors
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Ellipse
    from matplotlib.pyplot import Figure, Axes
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from scipy import stats
    from scipy.stats import binned_statistic_2d
    from scipy.signal import fftconvolve
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tbparse import SummaryReader

    from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics, DNMState, DynamicNetworkModel
    from sepsis_osc.ldm.checkpoint_utils import load_checkpoint
    from sepsis_osc.ldm.commons import build_lookup_table
    from sepsis_osc.ldm.data_loading import get_data_sets_online
    from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel
    from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
    from sepsis_osc.ldm.model_structs import LoadingConfig, LossesConfig
    from sepsis_osc.ldm.train_online import process_val_epoch
    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed, plt_params
    from sepsis_osc.utils.jax_config import setup_jax
    from sepsis_osc.utils.logger import setup_logging
    from sepsis_osc.utils.model_to_latex import model_to_latex_figure

    from sepsis_osc.ldm.model_structs import AuxLosses
    from sepsis_osc.ldm.lookup import compute_window_bounds

    setup_jax(simulation=False)

    setup_logging()
    logger = logging.getLogger(__name__)

    matplotlib.style.use("default")
    plt.rcParams.update(plt_params)
    return (
        ALPHA,
        BETA_SPACE,
        DNMConfig,
        DNMMetrics,
        Dopri5,
        DynamicNetworkModel,
        Ellipse,
        GridSpec,
        LatentDynamicsModel,
        LoadingConfig,
        LossesConfig,
        SIGMA_SPACE,
        ScalarFormatter,
        Storage,
        SummaryReader,
        as_2d_indices,
        average_precision_score,
        build_lookup_table,
        colors,
        eqx,
        fftconvolve,
        get_data_sets_online,
        inset_axes,
        jax,
        jax_random_seed,
        jnp,
        jr,
        load_checkpoint,
        logger,
        model_to_latex_figure,
        np,
        pd,
        plt,
        plt_params,
        process_val_epoch,
        roc_auc_score,
        stats,
    )


@app.cell
def _():
    from sepsis_osc.visualisations.viz_param_space import pretty_plot, space_plot
    from sepsis_osc.visualisations.viz_single_run import plot_kappa, plot_phase_snapshot, plot_kuramoto
    from sepsis_osc.visualisations.viz_model_results import (
        viz_space_alignment,
        viz_space_distribution_countour,
        viz_concept_densities,
        viz_loss_mean_std,
        viz_patients,
        viz_curves_sepsis,
    )


    return (
        plot_kappa,
        plot_kuramoto,
        plot_phase_snapshot,
        space_plot,
        viz_concept_densities,
        viz_curves_sepsis,
        viz_loss_mean_std,
        viz_patients,
        viz_space_alignment,
        viz_space_distribution_countour,
    )


@app.cell
def _(
    ALPHA,
    BETA_SPACE,
    DNMConfig,
    DNMMetrics,
    LossesConfig,
    SIGMA_SPACE,
    Storage,
    as_2d_indices,
    build_lookup_table,
    np,
):
    db_str = "DaisyFinal"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )

    lookup_table = build_lookup_table(storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)
    large_lookup_table =  build_lookup_table(storage, alpha=ALPHA, beta_space=(0, 1.0, 0.01), sigma_space=SIGMA_SPACE)

    b, r = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
    a = np.ones_like(b) * ALPHA
    params = DNMConfig.batch_as_index(a, b, r, 0.2)
    metrics_3d, _ = storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
    metrics_2d = metrics_3d.to_jax().reshape([1, *metrics_3d.shape["r_1"]]).squeeze()

    lb, lr = as_2d_indices((0, 1.0, 0.01), SIGMA_SPACE)
    la = np.ones_like(lb) * ALPHA
    lparams = DNMConfig.batch_as_index(la, lb, lr, 0.2)
    lmetrics_3d, _ = storage.read_multiple_results(lparams, proto_metric=DNMMetrics, threshold=0.0)
    lmetrics_2d = lmetrics_3d.to_jax().reshape([1, *lmetrics_3d.shape["r_1"]]).squeeze()

    storage.close()

    loss_conf = LossesConfig(
        lambda_sep3=300.0,
        lambda_inf=1.0,
        lambda_sofa_classification=2000.0,
        lambda_spreading=6e-3,
        lambda_boundary=30.0,
        lambda_recon=2.5,
    )    
    return lmetrics_2d, lookup_table, loss_conf, metrics_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Physiological Model
    """)
    return


@app.cell
def _(DNMConfig, Dopri5, DynamicNetworkModel, jax, jnp, jr):
    jax.config.update("jax_enable_x64", True)
    N = 200
    M = 1
    T_init, T_max = 0, 2000
    T_step = 0.01
    A = DNMConfig(N=N, C=0.2, omega_1=0.0, omega_2=0.0, a_1=1.0, epsilon_1=0.03, epsilon_2=0.3, alpha=-0.28, beta=0.5, sigma=1.0)
    B = DNMConfig(N=N, C=0.2, omega_1=0.0, omega_2=0.0, a_1=1.0, epsilon_1=0.03, epsilon_2=0.3, alpha=-0.28, beta=0.58, sigma=1.0)
    C = DNMConfig(N=N, C=0.2, omega_1=0.0, omega_2=0.0, a_1=1.0, epsilon_1=0.03, epsilon_2=0.3, alpha=-0.28, beta=0.7, sigma=1.0)
    D = DNMConfig(N=N, C=0.2, omega_1=0.0, omega_2=0.0, a_1=1.0, epsilon_1=0.03, epsilon_2=0.3, alpha=-0.28, beta=0.5, sigma=0.2)

    rand_keys = [jr.key(2), jr.key(24), jr.key(42), jr.key(2), jr.key(356)]
    run_confs = [A, B, C, C, D]
    _dnm = DynamicNetworkModel(full_save=True, steady_state_check=False, progress_bar=True, full_save_dtype=jnp.float64)
    sols = [
        _dnm.integrate(config=run_conf, M=M, key=rand_key, T_init=T_init, T_max=T_max, T_step=T_step, solver=Dopri5())
        for rand_key, run_conf in zip(rand_keys, run_confs, strict=True)
    ]
    jax.config.update("jax_enable_x64", False)
    return (sols,)


@app.cell
def _(
    GridSpec,
    ScalarFormatter,
    inset_axes,
    jnp,
    np,
    plot_kappa,
    plot_kuramoto,
    plot_phase_snapshot,
    plt,
    plt_params,
    sols,
):
    def plot_init(phis: str = "plot"):
        ys, dys = sols[0].ys
        ys, dys = (ys.astype(jnp.float32).remove_infs().squeeze().enforce_bounds(), dys.astype(jnp.float32).remove_infs().squeeze())
        sorting1 = np.lexsort((dys.phi_1.mean(axis=0), ys.phi_1[-1]))
        sorting2 = np.lexsort((dys.phi_2.mean(axis=0), ys.phi_2[-1]))

        x, y = plt_params["figure.figsize"]

        fig = plt.figure(figsize=(x, y / 1.5))
        polar = phis == "polar"
        wr = [1, 0.8, .01, 0.8, 1, .1] if polar else [1, .01, 1, 1, 1, .1] 
        gs = fig.add_gridspec(1, 4 + 2, width_ratios=wr, wspace=0.4)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[2-polar], projection="polar" if polar else None)
        ax2 = fig.add_subplot(gs[3], projection="polar" if polar else None)
        ax3 = fig.add_subplot(gs[4])


        gs_cbar_left = GridSpec(1, 1, wspace=0.2, hspace=0.3)
        gs_cbar_right = GridSpec(1, 1, wspace=0.3, hspace=0.3)

        cax2 = inset_axes(ax3, width="10%", height="100%", loc='lower left',
                          bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax3.transAxes, borderpad=0)
        plot_kappa(
            np.asarray(ys.kappa_1),
            np.asarray(ys.kappa_2),
            sorting_1=sorting1,
            sorting_2=sorting2,
            t=0,
            figax=(fig, [ax0, ax3]),
            cax = [None, cax2],
            cbar1=False,
            cbar2=True,
            cbar_loc="side",
        )
        if phis=="plot":
            plot_phase_snapshot(np.asarray(ys.phi_1), np.asarray(ys.phi_2), t=0, deriv=False, ax=[ax1, ax2])
            ax1.set_aspect(100, adjustable="box")
            ax2.set_aspect(100, adjustable="box")
            ax1.set_title("$\\phi_i^{1}$ / $\\pi$")
            ax2.set_title("$\\phi_i^{2}$ / $\\pi$")
            ax1.set_xlabel("Index i")
            ax2.set_xlabel("Index i")
            ax2.set_yticklabels([])
            ax2.set_yticks([])
        elif phis=="hist":
            bins = 50
            ax1.hist(ys.phi_1[0, :], range=(0, 2*np.pi), bins=bins, log=True)
            ax2.hist(ys.phi_2[0, :], range=(0, 2*np.pi), bins=bins, log=True, color="tab:green")
            ax1.set_aspect(3, adjustable="box")
            ax2.set_aspect(3, adjustable="box")
            for ax in [ax1, ax2]:
                ax.set_ylim(0.9, 200)
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.set_yticks([1, 10, 100])
                ax.set_xticks([0, np.pi, 2*np.pi])
                ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
            ax1.set_ylabel("Count")
            ax1.set_title("$\\phi_i^{1}$ / $\\pi$")
            ax2.set_title("$\\phi_i^{2}$ / $\\pi$")
            ax1.set_xlabel("rad")
            ax2.set_xlabel("rad")
            ax2.set_yticklabels([])
            ax2.set_yticks([])
        elif phis=="polar":
            ax1 = plot_kuramoto(ys.phi_1, t=0, ax=ax1)
            ax2 = plot_kuramoto(ys.phi_2, t=0, ax=ax2, color="tab:green")
            ax1.set_title("$\\phi_i^{1}$")
            ax2.set_title("$\\phi_i^{2}$")
            pos1 = ax1.get_position()
            pos2 = ax2.get_position()
            ax1.set_position([pos1.x0, pos1.y0 - 0.05, pos1.width, pos1.height])
            ax2.set_position([pos2.x0, pos2.y0 - 0.05, pos2.width, pos2.height])
        else:
            return fig

        ax0.set_xlabel("Index i")
        ax0.set_ylabel("Index j")
        ax3.set_xlabel("Index i")
        ax3.set_ylabel("Index j")

        ax0.set_title("$\\kappa_{ij}^1$")
        ax0.set_xticks([0, 50, 100, 150, 200])
        ax3.set_title("$\\kappa_{ij}^2$")
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.text(-5, -5, "\\{", fontsize=25, rotation=90, fontweight="light")
        ax3.text(-15, -40, r"C=0.2", fontsize=14, fontweight="light")
        ax3.set_xticks([100, 150, 200])
        return fig

    return (plot_init,)


@app.cell
def _(plot_init):
    init_fig = plot_init(phis="plot")
    init_fig
    return (init_fig,)


@app.cell
def _(init_fig):
    init_fig.savefig("typst/images/paper/init.svg")
    init_fig.savefig("typst/images/paper/init.png", dpi=400)
    return


@app.cell
def _(plot_init):
    init_hist_fig = plot_init(phis="hist")
    init_hist_fig
    return (init_hist_fig,)


@app.cell
def _(init_hist_fig):
    init_hist_fig.savefig("typst/images/paper/init_hist.svg")
    init_hist_fig.savefig("typst/images/paper/init_hist.png", dpi=400)
    return


@app.cell
def _(plot_init):
    init_polar_fig = plot_init(phis="polar")
    init_polar_fig
    return (init_polar_fig,)


@app.cell
def _(init_polar_fig):
    init_polar_fig.savefig("typst/images/paper/init_polar.svg")
    init_polar_fig.savefig("typst/images/paper/init_polar.png", dpi=400)
    return


@app.cell
def _(BETA_SPACE, SIGMA_SPACE, lmetrics_2d, np, plt):
    def plot_phase() -> plt.Figure:
        xs = np.arange(*(0.0, 1.0, 0.01),)
        ys = np.arange(*SIGMA_SPACE)
        orig_xs = np.asarray([np.argmin(np.abs(xs - x)) for x in [0.4, 0.7]])
        orig_ys = np.asarray([len(ys) - np.argmin(np.abs(ys - y)) - 1 for y in [0.0, 1.5]])

        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, squeeze=True)

        im0 = axes[0].imshow(lmetrics_2d.s_1.T[::-1, :], aspect="auto", cmap="viridis")
        im1 = axes[1].imshow(lmetrics_2d.s_2.T[::-1, :], aspect="auto", cmap="viridis")
        axes[0].set_ylabel(r"$\sigma$")
        axes[1].set_xlabel(r"$\beta / \pi$")
        axes[0].set_xlabel(r"$\beta / \pi$")
        xtick_positions = np.linspace(0, len(xs) - 1, 6, dtype=int)
        xlabels = xs[xtick_positions] + 0.01
        xlabels[0] = 0
        ytick_positions = np.linspace(0, len(ys) - 1, 6, dtype=int)
        ylabels = np.round(ys[ytick_positions][::-1] + 0.015,4)
        ylabels[-1] = 0
        for a in axes:
            a.set_xticks(xtick_positions)
            a.set_xticklabels(xlabels, rotation=0)
            a.set_yticks(ytick_positions)
            a.set_yticklabels(ylabels)
            a.plot([orig_xs[0], orig_xs[1]], [orig_ys[0], orig_ys[0]], color="white", linewidth=0.5)
            a.plot([orig_xs[0], orig_xs[1]], [orig_ys[1], orig_ys[1]], color="white", linewidth=0.5)
            a.plot([orig_xs[0], orig_xs[0]], [orig_ys[0], orig_ys[1]], color="white", linewidth=0.5)
            a.plot([orig_xs[1], orig_xs[1]], [orig_ys[0], orig_ys[1]], color="white", linewidth=0.5)

        fig.colorbar(im0, label=r"$s^1$", ax=axes[0], location="right",)
        fig.colorbar(im1, label=r"$s^2$", ax=axes[1], location="right",)

        configs = {r"\textbf{A}": (0.5, 1.0), r"\textbf{B}": (0.58, 1.0), r"\textbf{C}": (0.695, 1.0), r"\textbf{D}": (0.5, 0.2)}

        for n, c in configs.items():
            b, s = c
            beta_scale = len(xs) * (b - BETA_SPACE[0]) / (BETA_SPACE[1] - BETA_SPACE[0])
            sigma_scale = len(ys) * (1 - (s - SIGMA_SPACE[0]) / (SIGMA_SPACE[1] - SIGMA_SPACE[0]))
            for i in range(2):
                    axes[i].scatter(beta_scale, sigma_scale, c="white")
                    axes[i].annotate(n, (beta_scale - 8, sigma_scale + 1), c="white", fontweight="bold")
        axes[0].set_title("Parenchymal Layer")
        axes[1].set_title("Immune Layer")
        axes[0].text(-.1, 1.1, r"\textbf{A}", transform=axes[0].transAxes, fontsize=14, fontweight="bold", va="top")
        axes[1].text(-.1, 1.1, r"\textbf{B}", transform=axes[1].transAxes, fontsize=14, fontweight="bold", va="top")

        fig.subplots_adjust(wspace=0.1, hspace=0.25, top=0.9, bottom=0.15)
        return fig

    return (plot_phase,)


@app.cell
def _(plot_phase):
    phase_fig = plot_phase()
    phase_fig
    return (phase_fig,)


@app.cell
def _(phase_fig):
    phase_fig.savefig("typst/images/paper/phase.svg")
    phase_fig.savefig("typst/images/paper/phase.png", dpi=400)
    return


@app.cell
def _(metrics_2d, plt):
    _fig, _ax = plt.subplots(1, 1)
    _ax.imshow(metrics_2d.s_1.T[::-1, :], aspect="auto", cmap="viridis")
    _fig.subplots_adjust(top=1.0, bottom=0, left=0, right=1.0)
    _ax.set_yticks([])
    _ax.set_xticks([])
    _fig.savefig("typst/images/paper/phase_empty.svg")
    return


@app.cell
def _(
    GridSpec,
    ScalarFormatter,
    jnp,
    np,
    plot_kappa,
    plot_phase_snapshot,
    plt,
    sols,
):
    def plot_snapshots(hist: bool = False) -> plt.Figure:
        fig = plt.figure(figsize=(10, 5))
        gs_left_half = GridSpec(
            len(sols), 2,
            width_ratios=[1, 0.145],
            wspace=-0.6,
            hspace=0.3,
        )
        gs_right_half = GridSpec(
            len(sols), 2,
            width_ratios=[0.145, 1],
            wspace=0.-0.6,
            hspace=0.3,
        )
        gs_left_half.update(left=0.08, right=0.48)
        gs_right_half.update(left=0.52, right=0.92)
        # Create colorbar axes on top of first row
        #gs_cbar_left = GridSpec(1, 1, wspace=0.2, hspace=0.3)
        gs_cbar_right = GridSpec(1, 1, wspace=0.3, hspace=0.3)

        for i, sol in enumerate(sols):
            ys, dys = sol.ys
            ys, dys = (
               ys.astype(jnp.float32).remove_infs().squeeze().enforce_bounds(),
               dys.astype(jnp.float32).remove_infs().squeeze(),
            )
            sorting1 = np.lexsort((dys.phi_1.mean(axis=0), ys.phi_1[-1]))
            sorting2 = np.lexsort((dys.phi_2.mean(axis=0), ys.phi_2[-1]))

            axk1  = fig.add_subplot(gs_left_half[i, 0])
            axdp1 = fig.add_subplot(gs_left_half[i, 1])
            axdp2 = fig.add_subplot(gs_right_half[i, 0])
            axk2  = fig.add_subplot(gs_right_half[i, 1])

            # Create colorbar axes for first row
            caxl, caxr = None, None
            if i == 0:
               #caxl = fig.add_subplot(gs_cbar_left[0, 0])
               caxr = fig.add_subplot(gs_cbar_right[0, 0])
               axk1.set_title(r"$\kappa_{ij}^1$")
               axdp1.set_title(r"$\dot{\phi_i}^{1}$")
               axdp2.set_title(r"$\dot{\phi_i}^{2}$")
               axk2.set_title(r"$\kappa_{ij}^2$")
            axs = (axk1, axdp1, axdp2, axk2)

            plot_kappa(
               np.asarray(ys.kappa_1),
               np.asarray(ys.kappa_2),
               sorting_1=sorting1,
               sorting_2=sorting2,
               t=-1,
               figax=(fig, [axk1, axk2]),
               cax=[None, caxr],
               cbar1=False,
               cbar2=i == 0,
               cbar_loc="side",
            )
            if hist:
               bins = 20
               axdp1.hist(dys.phi_1[-1, :], range=(-2., 2.), bins=bins, log=True)
               axdp2.hist(dys.phi_2[-1, :], range=(-2., 2.), bins=bins, log=True, color="tab:green")
               for ax in [axdp1, axdp2]:
                   ax.set_ylim(0.9, 200)
                   ax.set_yscale("log")
                   ax.yaxis.set_major_formatter(ScalarFormatter())
                   ax.set_yticks([1, 10, 100])
            else:
               plot_phase_snapshot(np.asarray(dys.phi_1), np.asarray(dys.phi_2), t=-1, deriv=True, ax=[axdp1, axdp2], sort=True)

            axdp2.set_yticks([])
            axdp2.set_yticklabels([])
            axk1.set_ylabel("Index j")
            axk2.set_ylabel("Index j")
            axk2.yaxis.set_label_position("right")
            axk2.yaxis.tick_right()

            for ax in axs:
               if i != len(sols) - 1:
                   ax.set_xticklabels([])
               else:
                   if hist:
                       axdp1.set_xlabel("rad/time")
                       axdp2.set_xlabel("rad/time")
                       axk1.set_xlabel("Index i")
                       axk2.set_xlabel("Index i")
                   else:
                       for ax in axs:
                           ax.set_xlabel("Index i")

            axk1.text(-1.7, 0.7, (r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{C'}", r"\textbf{D}")[i], transform=axk1.transAxes, fontsize=14, fontweight="bold", va="top")

            pos = axdp1.get_position()
           # axdp1.set_position([pos.x0 - 0.015, pos.y0, pos.width, pos.height])

        ##gs_cbar_left.update(left=0.285, right=0.36, bottom=0.89, top=0.91)
        gs_cbar_right.update(left=0.79, right=0.8025, bottom=0.34, top=0.666)
        return fig


    return (plot_snapshots,)


@app.cell
def _(plot_snapshots):
    snap_fig = plot_snapshots()
    snap_fig
    return (snap_fig,)


@app.cell
def _(snap_fig):
    snap_fig.savefig("typst/images/paper/snapshots.svg")
    snap_fig.savefig("typst/images/paper/snapshots.png", dpi=400)
    return


@app.cell
def _(plot_snapshots):
    snap_hist_fig = plot_snapshots(hist=True)
    snap_hist_fig
    return (snap_hist_fig,)


@app.cell
def _(snap_hist_fig):
    snap_hist_fig.savefig("typst/images/paper/snapshots_hist.svg")
    snap_hist_fig.savefig("typst/images/paper/snapshots_hist.png", dpi=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Latent Dynamics Model
    """)
    return


@app.cell
def _(
    LatentDynamicsModel,
    LoadingConfig,
    SummaryReader,
    eqx,
    load_checkpoint,
    logger,
    pd,
):
    def get_model(
        rep: int, fold: int
    ) -> tuple[
        LatentDynamicsModel,
        pd.DataFrame,
        pd.DataFrame,
        str,
        int,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        _run_name = f"rep{rep:002}_fold{fold:002}"
        run_dir = f"runs/cv3/{_run_name}"
        tb_reader = SummaryReader(run_dir)
        _tb_df = tb_reader.scalars
        hparams = tb_reader.hparams.T
        hparams.columns = hparams.loc["tag"]
        hparams = hparams.drop("tag")
        _auprc = _tb_df.query("tag == 'sepsis_metrics/AUPRC_pred_sep'")
        prc_epoch = _auprc.loc[_auprc["value"].idxmax(), "step"]
        _auroc = _tb_df.query("tag == 'sepsis_metrics/AUROC_pred_sep'")
        roc_epoch = _auroc.loc[_auroc["value"].idxmax(), "step"]
        best_epoch = round((prc_epoch + roc_epoch) / 2)
        logger.info(
            f"Best Epoch: {best_epoch} with between AUPRC={_auprc['value'].max():.3f}@{prc_epoch} AUROC={_auroc['value'].max():.3f}@{roc_epoch}"
        )
        _load_epoch = best_epoch
        load_conf = LoadingConfig(from_dir=run_dir, epoch=_load_epoch)
        if load_conf.from_dir:
            model, _ = load_checkpoint(load_conf.from_dir + "/checkpoints", load_conf.epoch, None)
        model = eqx.nn.inference_mode(model)
        assert model
        return (model, _tb_df, hparams, _run_name, _load_epoch, _auroc, _auprc)

    return (get_model,)


@app.cell
def _(get_model, model_to_latex_figure):
    _print_model, _tb_df, _print_hparams, _run_name, _load_epoch, _auroc, _auprc = get_model(0, 0)
    print(model_to_latex_figure(_print_model))
    return


@app.cell
def _(
    average_precision_score,
    get_data_sets_online,
    get_model,
    jax_random_seed,
    jnp,
    jr,
    logger,
    lookup_table,
    loss_conf,
    np,
    pd,
    process_val_epoch,
    roc_auc_score,
):
    from jax.tree_util import Partial
    repetitions = 5
    n_folds = 5
    rows = []
    all_cv_metrics = {}
    dtype = jnp.float32
    key = jr.PRNGKey(jax_random_seed)

    process_val_epoch._cached.clear_cache()

    for rep in range(repetitions):
        for fold in range(n_folds):
            try:
                model, _tb_df, hparams, _run_name, _load_epoch, _auroc, _auprc = get_model(rep, fold)
            except ValueError as e:
                logger.warning(f"Summary not found {rep}-{repetitions - 1} {fold}-{n_folds - 1}")
                continue
            except FileNotFoundError as e:
                logger.warning(f"Failed loading checkpoint: {e}.")
                continue
            except Exception as e:
                logger.warning(f"Failed to load {rep}-{repetitions - 1} {fold}-{n_folds - 1} sequences. {e}")
                continue
            logger.info(f"Loaded {rep}-{repetitions} {fold}-{n_folds}.")
            _data = get_data_sets_online(
                swapaxes_y=(1, 2, 0),
                dtype=dtype,
                cv_repetitions=repetitions,
                repetition_index=rep,
                cv_folds=n_folds,
                fold_index=fold,
                sequence_files="data/cv/sequence_",
            )
            *_, test_x, test_y, test_m = _data
            test_m = test_m.astype(np.bool)
            test_x, test_y, test_m = (test_x[None], test_y[None], test_m[None])
            test_metrics = process_val_epoch(
                model,
                x_data=test_x,
                y_data=test_y,
                mask_data=test_m,
                step=jnp.array(1000000.0, dtype=jnp.int32),
                key=key,
                lookup_func=Partial(lookup_table.soft_get_local),
                loss_params=loss_conf,
            )
            true_sofa = np.asarray(test_y[..., 0])[test_m]
            _true_sofa_d2 = np.concat([np.zeros(test_y.shape[:-2])[..., None], np.asarray(jnp.diff(test_y[..., 0], axis=-1) > 0)], axis=-1)[
                test_m
            ]
            true_inf = np.asarray(test_y[..., 1])[test_m]
            true_sep3 = np.asarray(test_y[..., 2] == 1.0)[test_m]
            pred_sep3_risk = np.asarray(test_metrics.sep3_risk)[test_m]
            _pred_sofa_d2_risk = np.asarray(test_metrics.sofa_d2_risk)[test_m]
            pred_susp_inf_p = np.asarray(test_metrics.susp_inf_p)[test_m]
            pred_sofa_score = np.asarray(test_metrics.hists_sofa_score)[test_m]
            row = {
                "repetition": rep,
                "fold": fold,
                "run_name": _run_name,
                "best_epoch": int(_load_epoch),
                "val_auprc_max": float(_auprc["value"].max()),
                "val_auroc_max": float(_auroc["value"].max()),
                "test_auroc_sep3": roc_auc_score(true_sep3, pred_sep3_risk),
                "test_auprc_sep3": average_precision_score(true_sep3, pred_sep3_risk),
                "test_auroc_sofa_d2": roc_auc_score(_true_sofa_d2, _pred_sofa_d2_risk),
                "test_auprc_sofa_d2": average_precision_score(_true_sofa_d2, _pred_sofa_d2_risk),
                "test_auroc_susp_inf": roc_auc_score(true_inf > 0, pred_susp_inf_p),
                "test_auprc_susp_inf": average_precision_score(true_inf > 0, pred_susp_inf_p),
            }
            rows.append(row)
            all_cv_metrics[(rep, fold)] = test_metrics, test_x, test_y, test_m

    results_df = pd.DataFrame(rows)  # validation info from tensorboard
    return (
        all_cv_metrics,
        dtype,
        hparams,
        key,
        n_folds,
        repetitions,
        results_df,
    )


@app.cell
def _(results_df):
    c = results_df.copy()
    return


@app.cell
def _(np, results_df):
    print(len(results_df))
    _summary = results_df.agg({'test_auroc_sep3': ['mean', 'std'], 'test_auprc_sep3': ['mean', 'std']})
    print(np.round(_summary * 100, 4))
    _summary_inf = results_df.agg({'test_auroc_susp_inf': ['mean', 'std'], 'test_auprc_susp_inf': ['mean', 'std']})
    print(np.round(_summary_inf * 100, 4))
    return


@app.cell
def _(
    BETA_SPACE,
    SIGMA_SPACE,
    all_cv_metrics,
    colors,
    jnp,
    lookup_table,
    np,
    plt,
    results_df,
    space_plot,
    viz_space_alignment,
    viz_space_distribution_countour,
):
    betas_space = jnp.arange(BETA_SPACE[0], BETA_SPACE[1] + BETA_SPACE[2], BETA_SPACE[2])
    sigmas_space = jnp.arange(SIGMA_SPACE[0], SIGMA_SPACE[1] + SIGMA_SPACE[2], SIGMA_SPACE[2])
    beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing="ij")
    param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)
    metrics = lookup_table.hard_get_fsq(jnp.asarray(param_grid)).reshape(len(betas_space), len(sigmas_space))

    cv_fig_combined, _axs = plt.subplots(1, 4, sharex=True, sharey=True)

    _best = results_df.sort_values(by="test_auroc_sep3")[-2:].reset_index()
    _worst = results_df.sort_values(by="test_auroc_sep3")[:2].reset_index()

    for _i, _df in enumerate((_best, _worst)):
        for _ind, _row in _df.iterrows():
            _rep = _row["repetition"]
            _fold = _row["fold"]
            _ax = _axs[_ind + _i * len(_best)]
            _metrics, _x, _y, _m = all_cv_metrics[(_rep, _fold)]

            space_plot(
                metrics,
                xs=np.asarray(betas_space),
                ys=np.asarray(sigmas_space),
                title="",
                cmap=False,
                filename="",
                alpha=0.7,
                xticklabel_rot=0,
                num_ticks=5,
                figax=(cv_fig_combined, _ax),
            )
            viz_space_distribution_countour(
                _metrics.beta, _metrics.sigma, _m, show_cmap=False, show_inlay_notation=False, figax=(cv_fig_combined, _ax)
            )
            viz_space_alignment(_y[None], _metrics.beta, _metrics.sigma, _m, show_cmap=False, figax=(cv_fig_combined, _ax))

            if False:
                # Move viridis cmap
                _cbar_ax = _ax.images[0].colorbar.ax
                _pos = _cbar_ax.get_position()
                _cbar_ax.set_position([_pos.x0 - 0.04, _pos.y0, _pos.width, _pos.height])

            _ax.annotate(
                f"Rep.: {_rep + 1}, Fold: {_fold + 1}, \nAUROC: {_row['test_auroc_sep3']:.2f}, AUPRC: {_row['test_auprc_sep3']:.2f}",
                (0.41, 0.05),
                color="white",
                weight="bold",
                fontsize=8,
            )

            _ax.set_yticklabels([])
            if _ind != 0:
                _ax.set_ylabel("")

            _ax.set_xticklabels([])
            if _i != 1:
                _ax.set_xlabel("")

    _cmap = plt.get_cmap("OrRd")
    _norm = colors.BoundaryNorm([0, 1.5, 3, 4.5, 6], _cmap.N)
    _sm = plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)

    _cbar = cv_fig_combined.colorbar(_sm, ax=_axs, location='right', fraction=0.0175, pad=0.04)
    _cbar.set_label('log(Density)', rotation=270, labelpad=15)

    cv_fig_combined.subplots_adjust(wspace=0.1, hspace=0.0, bottom=0.15, top=0.9, left=0.05, right=0.85)
    cv_fig_combined
    return betas_space, cv_fig_combined, metrics, sigmas_space


@app.cell
def _(cv_fig_combined):
    cv_fig_combined.savefig("typst/images/paper/cv_comparison_combined.svg")
    cv_fig_combined.savefig("typst/images/paper/cv_comparison_combined.png", dpi=400)
    return


@app.cell
def _(
    all_cv_metrics,
    betas_space,
    colors,
    metrics,
    np,
    plt,
    results_df,
    sigmas_space,
    space_plot,
    viz_space_distribution_countour,
):
    cv_fig, _axs = plt.subplots(1, 4, sharex=True, sharey=True)

    _best = results_df.sort_values(by="test_auroc_sep3")[-2:].reset_index()
    _worst = results_df.sort_values(by="test_auroc_sep3")[:2].reset_index()

    for _i, _df in enumerate((_best, _worst)):
        for _ind, _row in _df.iterrows():
            _rep = _row["repetition"]
            _fold = _row["fold"]
            _ax = _axs[_ind + _i * len(_best)]
            _metrics, _x, _y, _m = all_cv_metrics[(_rep, _fold)]

            space_plot(
                metrics,
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
                f"Rep.: {_rep + 1}, Fold: {_fold + 1}, \nAUROC: {_row['test_auroc_sep3']:.2f}, AUPRC: {_row['test_auprc_sep3']:.2f}",
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
    _norm = colors.BoundaryNorm([0, 1.5, 3, 4.5, 6], _cmap.N)
    _sm = plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)

    _cbar = cv_fig.colorbar(_sm, ax=_axs, location='right', fraction=0.0175, pad=0.04)
    _cbar.set_label('log(Density)', rotation=90, labelpad=15)

    cv_fig.subplots_adjust(wspace=0.1, hspace=0.0, bottom=0.15, top=0.9, left=0.05, right=0.85)

    cv_fig
    return (cv_fig,)


@app.cell
def _(cv_fig):
    cv_fig.savefig("typst/images/paper/cv_comparison.svg")
    cv_fig.savefig("typst/images/paper/cv_comparison.png", dpi=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tests
    """)
    return


@app.cell
def _(np, stats):
    def welch_t(mean1, std1, n1, mean2, std2, n2):
        # t statistic
        se = np.sqrt((std1**2)/n1 + (std2**2)/n2)
        t = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        df_num = (std1**2/n1 + std2**2/n2) ** 2
        df_den = ((std1**2/n1)**2)/(n1-1) + ((std2**2/n2)**2)/(n2-1)
        df = df_num / df_den

        # two-sided p-value
        p = 2 * stats.t.sf(abs(t), df)
        p_greater = stats.t.sf(t, df)

        return t, df, p, p_greater

    return (welch_t,)


@app.cell
def _(results_df, welch_t):
    yaib_auroc_mean = 83.6 / 100 
    yaib_auroc_std = 0.3 / 100 
    ldm_auroc_mean = results_df["test_auroc_sep3"].mean() 
    ldm_auroc_std = results_df["test_auroc_sep3"].std() 
    yaib_auprc_mean = 9.1 / 100
    yaib_auprc_std = 0.3 / 100 
    ldm_auprc_mean = results_df["test_auprc_sep3"].mean() 
    ldm_auprc_std = results_df["test_auprc_sep3"].std()

    baselines = {
        "Reg. Logistic Regression": (77.1, 0.4, 4.6, 0.1),
        "LightGBM":                 (77.5, 0.3, 5.9, 0.2),
        "Transformer":              (80.0, 0.8, 6.6, 0.2),
        "LSTM":                     (82.0, 0.3, 8.0, 0.2),
        "TCN":                      (82.7, 0.3, 8.8, 0.2),
        "GRU":                      (83.6, 0.3, 9.1, 0.3),
    }

    ldm_auroc = (ldm_auroc_mean * 100, ldm_auroc_std * 100)
    ldm_auprc = (ldm_auprc_mean * 100, ldm_auprc_std * 100)
    n_ldm, n_base = len(results_df), 25

    header = f"{'Baseline':<24} | {'AUROC [t, df, p]':^28} | {'AUPRC [t, df, p]':^28}"
    print(header)
    print("-" * len(header))

    for name, (b_au_m, b_au_s, b_pr_m, b_pr_s) in baselines.items():
        # AUROC Stats
        t_au, df_au, _, p_au = welch_t(ldm_auroc[0], ldm_auroc[1], n_ldm, b_au_m, b_au_s, n_base)
        p_au_str = f"{p_au:.3f}" if p_au >= 0.001 else "<0.001"

        # AUPRC Stats
        t_pr, df_pr, _, p_pr = welch_t(ldm_auprc[0], ldm_auprc[1], n_ldm, b_pr_m, b_pr_s, n_base)
        p_pr_str = f"{p_pr:.3f}" if p_pr >= 0.001 else "<0.001"

        print(f"{name:<24} | {t_au:6.3f}, {df_au:6.3f}, {p_au_str:>6} | {t_pr:6.3f}, {df_pr:6.3f}, {p_pr_str:>6}")
    return yaib_auprc_mean, yaib_auprc_std, yaib_auroc_mean, yaib_auroc_std


@app.cell
def _(plt, results_df, yaib_auprc_mean, yaib_auroc_mean):
    _fig, _ax = plt.subplots(1, 2)
    _ax[0].hist(results_df['test_auroc_sep3'])
    _ax[0].vlines(results_df['test_auroc_sep3'].mean(), 0, 5, color='tab:green')
    _ax[0].vlines(yaib_auroc_mean, 0, 5, color='tab:orange')
    _ax[1].hist(results_df['test_auprc_sep3'])
    _ax[1].vlines(results_df['test_auprc_sep3'].mean(), 0, 5, color='tab:green')
    _ax[1].vlines(yaib_auprc_mean, 0, 5, color='tab:orange')
    return


@app.cell
def _(Ellipse, np):
    def plot_cov_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
        """
        Draw a covariance ellipse centered at `mean`.
        n_std=2 -> ~95% ellipse if data ~ normal.
        """
        vals, vecs = np.linalg.eigh(cov)  # Eigen-decomposition
        order = vals.argsort()[::-1]
        vals, vecs = (vals[order], vecs[:, order])
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False, **kwargs)  # Angle of rotation
        ax.add_patch(ellipse)  # Width & height = 2 * n_std * sqrt(eigenvalues)

    return (plot_cov_ellipse,)


@app.cell
def _(
    np,
    plot_cov_ellipse,
    plt,
    results_df,
    yaib_auprc_mean,
    yaib_auprc_std,
    yaib_auroc_mean,
    yaib_auroc_std,
):
    x = results_df['test_auroc_sep3'].values
    y = results_df['test_auprc_sep3'].values
    mean_ldm = [x.mean(), y.mean()]
    cov_ldm = np.cov(x, y)
    _fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.6, label='LDM runs')
    ax.scatter(*mean_ldm, marker='x', s=100, label='LDM mean')
    plot_cov_ellipse(mean_ldm, cov_ldm, ax, n_std=2, label='LDM 95% ellipse')
    mean_yaib = [yaib_auroc_mean, yaib_auprc_mean]
    cov_yaib = np.array([[yaib_auroc_std ** 2, 0], [0, yaib_auprc_std ** 2]])
    ax.scatter(*mean_yaib, marker='o', s=100, label='YAIB mean')
    plot_cov_ellipse(mean_yaib, cov_yaib, ax, n_std=2, linestyle='--', label='YAIB 95% ellipse')
    ax.set_xlabel('AUROC')
    ax.set_ylabel('AUPRC')
    ax.grid(True)
    ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Other Vizualisations
    """)
    return


@app.cell
def _(SummaryReader, np, pd):
    from collections import defaultdict

    def load_tb_scalars(run_dirs, tags):
        """
        Load scalars from multiple runs.

        Args:
            run_dirs (list[str]): Paths to tensorboard runs.
            tags (list[str]): Tags to extract, e.g., ['train_losses/total_loss_mean', ...]

        Returns:
            dict[tag] -> list of DataFrames (one per run)
        """
        all_data = defaultdict(list)
        for run_dir in run_dirs:
            try:
                tb_reader = SummaryReader(run_dir)
                _tb_df = tb_reader.scalars
                for tag in tags:
                    df_tag = _tb_df.query(f"tag == '{tag}'")[['step', 'value']].copy()
                    all_data[tag].append(df_tag)
            except Exception as e:
                print(f'Failed to read {run_dir}: {e}')
                continue
        return all_data

    def aggregate_runs(all_data):
        """
        Given a dict of tag -> list of dfs, return tag -> df with mean/std
        """
        agg_data = {}
        for tag, dfs in all_data.items():
            all_steps = sorted(set().union(*(df['step'].to_numpy() for df in dfs)))
            mean_vals = []
            std_vals = []  # Get common steps
            for step in all_steps:
                vals = [df.loc[df['step'] == step, 'value'].values[0] for df in dfs if step in df['step'].values]
                mean_vals.append(np.mean(vals))
                std_vals.append(np.std(vals))
            agg_data[tag] = pd.DataFrame({'step': all_steps, 'mean': mean_vals, 'std': std_vals})
        return agg_data

    return aggregate_runs, load_tb_scalars


@app.cell
def _(aggregate_runs, load_tb_scalars, n_folds, repetitions):
    # Prepare run directories
    run_dirs = [f'runs/cv3/rep{rep:02}_fold{fold:02}' for rep in range(repetitions) for fold in range(n_folds)]
    losses = ['total_loss', 'sepsis-3', 'sofa', 'infection', 'recon_loss', 'spreading_loss', 'boundary_loss']
    # Define tags you want
    eval_metrics = ['AUROC_pred_sep', 'AUPRC_pred_sep']
    eval_tags = [f'sepsis_metrics/{m}' for m in eval_metrics]
    for loss in losses:
    # Dynamically build the list
        eval_tags.extend([f'train_losses/{loss}_mean', f'val_losses/{loss}_mean'])
    all_data = load_tb_scalars(run_dirs, eval_tags)
    agg_data = aggregate_runs(all_data)
    losses = ('sepsis-3', 'sofa', 'infection', 'recon_loss', 'spreading_loss', 'boundary_loss')
    loss_subscripts = ('sepsis', 'sofa', 'inf', 'dec', 'spread', 'boundary')
    # Plot
    lambdas = ('sep3', 'sofa_classification', 'inf', 'recon', 'spreading', 'boundary')
    return agg_data, lambdas, loss_subscripts, losses


@app.cell
def _(agg_data, hparams, lambdas, loss_subscripts, losses, viz_loss_mean_std):
    loss_fig = viz_loss_mean_std(agg_data, losses, loss_subscripts, lambdas, hparams)
    loss_fig
    return (loss_fig,)


@app.cell
def _(loss_fig):
    loss_fig.savefig("typst/images/paper/losses_std.svg")
    loss_fig.savefig("typst/images/paper/losses_std.png", dpi=400)
    return


@app.cell
def _(np, plt, results_df):
    # Get most mediocre run
    mean_auprc = results_df["val_auprc_max"].mean()
    mean_auroc = results_df["val_auroc_max"].mean()
    print(mean_auroc)
    print(mean_auprc)
    results_df["dist_from_mean"] = np.sqrt((results_df["val_auprc_max"] - mean_auprc) ** 2 + (results_df["val_auroc_max"] - mean_auroc) ** 2)
    # Compute Euclidean distance from the mean
    avg_run_row = results_df.loc[results_df["dist_from_mean"].idxmin()]
    avg_run_name = avg_run_row["run_name"]

    plt.scatter(
        results_df["val_auroc_max"], results_df["val_auprc_max"], alpha=0.6, label="Individual Runs", color="steelblue", edgecolor="white"
    )

    # Plot the Mean Point
    plt.scatter(mean_auroc, mean_auprc, color="red", marker="X", s=100, label="Mean Point", zorder=5)

    # Highlight the Representative Run
    plt.scatter(
        avg_run_row["val_auroc_max"],
        avg_run_row["val_auprc_max"],
        facecolors="none",
        edgecolors="green",
        s=150,
        linewidths=2,
        label=f"Representative ({avg_run_row['run_name']})",
        zorder=6,
    )

    # Add a dashed line showing the distance
    plt.plot([mean_auroc, avg_run_row["val_auroc_max"]], [mean_auprc, avg_run_row["val_auprc_max"]], "g--", alpha=0.5)

    plt.xlabel("Validation AUROC (Max)")
    plt.ylabel("Validation AUPRC (Max)")
    plt.title("Identifying the Representative Run (Closest to Mean)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Average / representative run:", avg_run_name)
    print(avg_run_row[["val_auprc_max", "val_auroc_max"]])
    rep_avg, fold_avg = avg_run_name.split("_")
    # Pick run with minimal distance
    rep_avg, fold_avg = (int(rep_avg[-2:]), int(fold_avg[-2:]))
    return fold_avg, rep_avg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single Run Analysis
    """)
    return


@app.cell
def _(
    dtype,
    fold_avg,
    get_data_sets_online,
    get_model,
    n_folds,
    np,
    rep_avg,
    repetitions,
):
    single_model, _tb_df, single_hparams, _run_name, _load_epoch, _auroc, _auprc = get_model(rep_avg, fold_avg)
    _data = get_data_sets_online(swapaxes_y=(1, 2, 0), dtype=dtype, cv_repetitions=repetitions, repetition_index=rep_avg, cv_folds=n_folds, fold_index=fold_avg, sequence_files='data/cv/sequence_')
    *_, single_test_x, single_test_y, single_test_m = _data
    single_test_m = single_test_m.astype(np.bool)
    single_test_x, single_test_y, single_test_m = (single_test_x[None], single_test_y[None], single_test_m[None])
    return (
        single_hparams,
        single_model,
        single_test_m,
        single_test_x,
        single_test_y,
    )


@app.cell
def _(single_hparams):
    single_hparams.T
    return


@app.cell
def _(
    jnp,
    key,
    lookup_table,
    loss_conf,
    process_val_epoch,
    single_model,
    single_test_m,
    single_test_x,
    single_test_y,
):
    single_test_metrics = process_val_epoch(
        single_model,
        x_data=single_test_x,
        y_data=single_test_y,
        mask_data=single_test_m,
        step=jnp.array(1000000.0, dtype=jnp.int32),
        key=key,
        lookup_func=lookup_table.soft_get_local,
        loss_params=loss_conf,
    )
    return (single_test_metrics,)


@app.cell
def _(
    average_precision_score,
    jnp,
    np,
    roc_auc_score,
    single_test_m,
    single_test_metrics,
    single_test_y,
):
    true_sofa_1 = np.asarray(single_test_y[..., 0])[single_test_m]
    _true_sofa_d2 = np.concat(
        [np.zeros(single_test_y.shape[:-2])[..., None], np.asarray(jnp.diff(single_test_y[..., 0], axis=-1) > 0)], axis=-1
    )[single_test_m]
    true_inf_1 = np.asarray(single_test_y[..., 1])[single_test_m]
    true_sep3_1 = np.asarray(single_test_y[..., 2] == 1.0)[single_test_m]
    pred_sep3_risk_1 = np.asarray(single_test_metrics.sep3_risk)[single_test_m]
    _pred_sofa_d2_risk = np.asarray(single_test_metrics.sofa_d2_risk)[single_test_m]
    pred_susp_inf_p_1 = np.asarray(single_test_metrics.susp_inf_p)[single_test_m]
    pred_sofa_score_1 = np.asarray(single_test_metrics.hists_sofa_score)[single_test_m]
    print(f"AUROC sepis-3  {roc_auc_score(true_sep3_1, pred_sep3_risk_1) * 100:.3f}")
    print(f"AUPRC sepis-3  {average_precision_score(true_sep3_1, pred_sep3_risk_1) * 100:.3f}")
    print(f"AUROC sofa d2  {roc_auc_score(_true_sofa_d2, _pred_sofa_d2_risk) * 100:.2f}")
    print(f"AUPRC sofa d2  {average_precision_score(_true_sofa_d2, _pred_sofa_d2_risk) * 100:.2f}")
    print(f"AUROC susp inf {roc_auc_score(true_inf_1 > 0, pred_susp_inf_p_1) * 100:.2f}")
    print(f"AUPRC susp inf {average_precision_score(true_inf_1 > 0, pred_susp_inf_p_1) * 100:.2f}")
    print(f"RMSE SOFA      {np.sqrt(np.mean((pred_sofa_score_1 - true_sofa_1) ** 2)):.2f}")
    return pred_sep3_risk_1, pred_sofa_score_1, true_sep3_1, true_sofa_1


@app.cell
def _(fftconvolve, np, single_test_m, single_test_metrics, single_test_y):
    def get_event_times(true_infs, pred_infs, mask, threshold=0.1):
        T = np.asarray(true_infs) * mask
        P = np.asarray(pred_infs) * mask

        # normalize
        t_mean = T.mean(axis=1, keepdims=True)
        t_std = T.std(axis=1, keepdims=True) + 1e-6
        p_mean = P.mean(axis=1, keepdims=True)
        p_std = P.std(axis=1, keepdims=True) + 1e-6

        T_norm = (T - t_mean) / t_std
        P_norm = (P - p_mean) / p_std

        # Correlation(f, g) = Conv(f, reverse(g))
        corr = fftconvolve(T_norm, P_norm[:, ::-1], mode='full', axes=1)

        num_time_steps = T.shape[1]
        lags = np.arange(-num_time_steps + 1, num_time_steps)
        best_lags = lags[np.argmax(corr, axis=1)]

        has_event = T.max(axis=1) > 0.1 

        gt_times = np.argmax(T[has_event], axis=1)
        pred_times = gt_times - best_lags[has_event]
        return gt_times, pred_times

    _mask = np.asarray(single_test_m)[0]
    _true_infs = np.asarray(single_test_y[..., 1])[0]
    _pred_infs = np.asarray(single_test_metrics.susp_inf_p)[0]

    gt_times, pred_times = get_event_times(_true_infs, _pred_infs, _mask)
    return gt_times, pred_times


@app.cell
def _(
    gt_times,
    pred_sofa_score_1,
    pred_times,
    true_sofa_1,
    viz_concept_densities,
):
    heat_fig, heat_ax = viz_concept_densities(true_sofa_1, gt_times, pred_sofa_score_1, pred_times, cmap=True)
    heat_ax[0].set_title("SOFA-score")
    heat_ax[1].set_title(r"$\Delta$SOFA-score")
    heat_fig
    return (heat_fig,)


@app.cell
def _(heat_fig):
    heat_fig.savefig("typst/images/paper/heat.svg")
    heat_fig.savefig("typst/images/paper/heat.png", dpi=400)
    return


@app.cell
def _(np, single_test_metrics, single_test_y):
    _p_idx = np.argsort(single_test_y[0, :, :, 0].std(axis=-1))[-10:]
    print(_p_idx)
    _p_idx = np.argsort(single_test_y[0, :, :, 0].max(axis=-1))[-10:]
    print(_p_idx)
    _p_idx = np.argsort(np.diff(single_test_y[0, :, :, 0], axis=-1).sum(axis=-1))[-10:]
    print(_p_idx)
    _p_idx = np.argsort(np.square(single_test_y[0, :, :, 0] - single_test_metrics.hists_sofa_score[0, :]).mean(axis=-1))[:10]
    print(_p_idx)
    _p_idx = np.argsort(np.square(single_test_y[0, :, :, 0] - single_test_metrics.hists_sofa_score[0, :]).mean(axis=-1))[-10:]
    print(_p_idx)
    return


@app.cell
def _(
    lookup_table,
    single_test_m,
    single_test_metrics,
    single_test_y,
    viz_patients,
):
    _p_idx = [3638, 12, 4154]
    patient_fig, _axes = viz_patients(_p_idx, single_test_y, single_test_m, single_test_metrics, lookup_table)
    _axes.text(-.1, 1.1, r"\textbf{A}", transform=_axes.transAxes, fontsize=14, fontweight="bold", va="top")
    _axes.text(1.6, 1.1, r"\textbf{B}", transform=_axes.transAxes, fontsize=14, fontweight="bold", va="top")
    patient_fig
    return (patient_fig,)


@app.cell
def _(patient_fig):
    patient_fig.savefig("typst/images/paper/trajectory.svg")
    patient_fig.savefig("typst/images/paper/trajectory.png", dpi=400)
    return


@app.cell
def _(
    betas_space,
    metrics,
    np,
    plt,
    sigmas_space,
    single_test_m,
    single_test_metrics,
    single_test_y,
    space_plot,
    viz_space_alignment,
    viz_space_distribution_countour,
):
    heat_space_fig, _axs = plt.subplots(1, 2, sharey=True, width_ratios=[1.0, 1.2])
    for _i, _ax in enumerate(_axs):
        space_plot(
            metrics,
            xs=np.asarray(betas_space),
            ys=np.asarray(sigmas_space),
            title="",
            cmap=_i,
            filename="",
            alpha=1.0,
            xticklabel_rot=0,
            num_ticks=5,
            figax=(heat_space_fig, _ax),
        )
    viz_space_distribution_countour(
        single_test_metrics.beta,
        single_test_metrics.sigma,
        single_test_m,
        show_cmap=True,
        show_inlay_notation=True,
        figax=(heat_space_fig, _axs[0]),
    )
    viz_space_alignment(
        single_test_y[None], single_test_metrics.beta, single_test_metrics.sigma, single_test_m, show_cmap=True, figax=(heat_space_fig, _axs[1])
    )

    # Move viridis cmap
    _cbar_ax = _ax.images[0].colorbar.ax
    _pos = _cbar_ax.get_position()
    _cbar_ax.set_position([_pos.x0 - 0.04, _pos.y0, _pos.width, _pos.height])

    heat_space_fig.subplots_adjust(top=0.95, bottom=0.15, wspace=0.1, left=0.07, right=0.9)
    heat_space_fig
    return (heat_space_fig,)


@app.cell
def _(heat_space_fig):
    heat_space_fig.savefig("typst/images/paper/heat_space.svg")
    heat_space_fig.savefig("typst/images/paper/heat_space.png", dpi=400)
    return


@app.cell
def _(
    betas_space,
    metrics,
    np,
    plt,
    sigmas_space,
    single_test_m,
    single_test_metrics,
    single_test_y,
    space_plot,
    viz_space_alignment,
    viz_space_distribution_countour,
):
    heat_space_combined_fig, _ax = plt.subplots(1, 1)
    space_plot(
        metrics,
        xs=np.asarray(betas_space),
        ys=np.asarray(sigmas_space),
        title="",
        cmap=True,
        filename="",
        alpha=1.0,
        xticklabel_rot=0,
        num_ticks=5,
        figax=(heat_space_combined_fig, _ax),
    )
    viz_space_alignment(
        single_test_y[None], single_test_metrics.beta, single_test_metrics.sigma, single_test_m, show_cmap=True, figax=(heat_space_combined_fig, _ax)
    )
    viz_space_distribution_countour(
        single_test_metrics.beta, single_test_metrics.sigma, single_test_m, show_cmap=True, show_inlay_notation=True, figax=(heat_space_combined_fig, _ax)
    )

    heat_space_combined_fig.subplots_adjust(top=0.95, bottom=0.15, wspace=-1, left=0.07, right=1.1)
    _cbar_ax = _ax.images[0].colorbar.ax
    _pos = _cbar_ax.get_position()
    _cbar_ax.set_position([_pos.x0-.05, _pos.y0, _pos.width, _pos.height])
    # Move viridis cmap

    heat_space_combined_fig
    return (heat_space_combined_fig,)


@app.cell
def _(heat_space_combined_fig):
    heat_space_combined_fig.savefig("typst/images/paper/heat_space_combined.png", dpi=400)
    heat_space_combined_fig.savefig("typst/images/paper/heat_space_combined.svg")
    return


@app.cell
def _(pred_sep3_risk_1, true_sep3_1, viz_curves_sepsis):
    area_fig = viz_curves_sepsis(true_sep3_1, pred_sep3_risk_1)
    area_fig
    return (area_fig,)


@app.cell
def _(area_fig):
    area_fig.savefig("typst/images/paper/areas.png", dpi=400)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
