import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from sepsis_osc.utils.jax_config import setup_jax
    setup_jax(simulation=True)
    import jax.numpy as jnp
    import jax.random as jr
    from diffrax import Dopri5, Dopri8

    from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DynamicNetworkModel
    from sepsis_osc.utils.config import jax_random_seed
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from sepsis_osc.dnm.dynamic_network_model import DNMState, DNMMetrics
    from sepsis_osc.utils.config import plt_params

    from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.visualisations.viz_param_space import pretty_plot, space_plot

    import matplotlib.style

    matplotlib.style.use(
        "default"
    )
    plt.rcParams.update(plt_params)
    return (
        DNMConfig,
        DNMMetrics,
        DNMState,
        Dopri5,
        Dopri8,
        DynamicNetworkModel,
        FFMpegWriter,
        FuncAnimation,
        LatentLookup,
        Storage,
        as_2d_indices,
        jax_random_seed,
        jnp,
        jr,
        np,
        plt,
        setup_jax,
        space_plot,
    )


@app.cell
def _(DNMConfig, DynamicNetworkModel, jax_random_seed, jnp, jr):
    rand_key = jr.key(jax_random_seed + 1)
    M = 1
    rand_keys = jr.split(rand_key, M)
    _N = 200
    #### Parameters
    conf_sync = DNMConfig(N=_N, C=0.2, omega_1=0.0, omega_2=0.0, a_1=1.0, epsilon_1=0.03, epsilon_2=0.3, alpha=-0.28, beta=0.5, sigma=1.0)
    conf_cluster = DNMConfig(N=_N, C=0.2, omega_1=0.0, omega_2=0.0, a_1=1.0, epsilon_1=0.03, epsilon_2=0.3, alpha=-0.28, beta=0.7, sigma=1.0)
    T_init, T_max = (0, 200)
    T_step = 0.5  # local infection
    dnm = DynamicNetworkModel(full_save=True, steady_state_check=False, full_save_dtype=jnp.float32)  # adaption rate  # phase lage  # age parameter  # local infection  # adaption rate  # phase lage  # age parameter
    return M, T_init, T_max, T_step, conf_cluster, conf_sync, dnm, rand_key


@app.cell
def _(
    Dopri8,
    M,
    T_init,
    T_max,
    T_step,
    conf_cluster,
    conf_sync,
    dnm,
    jnp,
    np,
    rand_key,
):
    sol_sync = dnm.integrate(
        config=conf_sync,
        M=M,
        key=rand_key,
        T_init=0,
        T_max=T_max,
        T_step=T_step,
        ts=np.arange(T_init, T_max, T_step),
        solver=Dopri8(),
    )
    sol_cluster = dnm.integrate(
        config=conf_cluster,
        M=M,
        key=rand_key,
        T_init=0,
        T_max=T_max,
        T_step=T_step,
        ts=np.arange(T_init, T_max, T_step),
        solver=Dopri8(),
    )

    ys_sync, dys_sync = sol_sync.ys
    print(ys_sync.shape, dys_sync.shape)
    ys_sync, dys_sync= ys_sync.remove_infs().squeeze().enforce_bounds(), dys_sync.remove_infs().squeeze()
    ts_sync = np.asarray(sol_sync.ts).squeeze()
    ts_sync = ts_sync[~jnp.isinf(ts_sync)]

    ys_cluster, dys_cluster = sol_cluster.ys
    ys_cluster, dys_cluster = ys_cluster.remove_infs().squeeze().enforce_bounds(), dys_cluster.remove_infs().squeeze()
    ts_cluster = np.asarray(sol_cluster.ts).squeeze()
    ts_cluster = ts_cluster[~jnp.isinf(ts_cluster)]
    return dys_cluster, dys_sync, ts_sync, ys_cluster, ys_sync


@app.cell
def _(DNMState, FFMpegWriter, Figure, FuncAnimation, np, plt):
    def gif_both(ys_1: DNMState, dys_1: DNMState, ys_2: DNMState, dys_2: DNMState, ts: np.ndarray, *, fps: int=10, filename: str='figures/kuramoto.gif', single: int | None=None) -> Figure:
        T, _N = ys_1.phi_1.shape
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.0025, 1, 1], wspace=0.9)
        axs = np.asarray([[None] * 4 for _ in range(2)])
        axs[0, 0] = fig.add_subplot(gs[0, 0], projection='polar')
        axs[0, 1] = fig.add_subplot(gs[0, 1], projection='polar')
        axs[0, 2] = fig.add_subplot(gs[0, 3], projection='polar')
        axs[0, 3] = fig.add_subplot(gs[0, 4], projection='polar')
        axs[1, 0] = fig.add_subplot(gs[1, 0])
        axs[1, 1] = fig.add_subplot(gs[1, 1])
        axs[1, 2] = fig.add_subplot(gs[1, 3])
        axs[1, 3] = fig.add_subplot(gs[1, 4])
        radii = np.ones(_N)
        N_bins = 36  # 2 rows, 5 columns → column 2 is a spacer
        bin_edges = np.linspace(0, 2 * np.pi, N_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bottom = 0.1
        range_n = np.arange(_N)
        pscatters = tuple()
        dscatters = tuple()
        bars = tuple()  # Top row (polar)
        for i, (p, dp) in enumerate(((ys_1, dys_1), (ys_2, dys_2))):
            ii = i * 2
            pscatter1 = axs[0, 0 + ii].scatter(p.phi_1[single or 0], radii, c='tab:blue', zorder=2)
            pscatter2 = axs[0, 1 + ii].scatter(p.phi_2[single or 0], radii, c='tab:green', zorder=2)
            pscatters = pscatters + (pscatter1, pscatter2)
            dscatter1 = axs[1, 0 + ii].scatter(range_n, dp.phi_1[single or 0], c='tab:blue')  # Bottom row (normal axes, or also polar if you want)
            dscatter2 = axs[1, 1 + ii].scatter(range_n, dp.phi_2[single or 0], c='tab:green')
            dscatters = dscatters + (dscatter1, dscatter2)
            cmap1 = plt.colormaps['Blues']
            cmap2 = plt.colormaps['Greens']
            cmaps = (cmap1, cmap2)
            bars1 = axs[0, 0 + ii].bar(bin_centers, np.zeros(N_bins), width=2 * np.pi / N_bins, bottom=bottom, zorder=0)
            bars2 = axs[0, 1 + ii].bar(bin_centers, np.zeros(N_bins), width=2 * np.pi / N_bins, bottom=bottom, zorder=0)
            bars = bars + (bars1, bars2)
            for j, pp in enumerate((p.phi_1[single or 0], p.phi_2[single or 0])):
                z = np.exp(1j * pp)
                R = np.abs(np.mean(z))
                counts, _ = np.histogram(pp, bins=N_bins, range=(0, 2 * np.pi), density=True)
                max_count = counts.max() if counts.max() > 0 else 1
                for count, bar in zip(counts, bars[j + ii], strict=True):
                    bar.set_height(count)
                    bar.set_facecolor(cmaps[j](count / max_count))
                    bar.set_alpha(0.8)
        fig.subplots_adjust(wspace=10)
        axs[0, 0].set_title('Parenchymal')
        axs[0, 1].set_title('Immune')
        axs[0, 2].set_title('Parenchymal')
        axs[0, 3].set_title('Immune')
        axs[0, 0].set_ylabel('Phase')
        axs[0, 0].yaxis.set_label_coords(-0.45, 0.5)
        axs[1, 0].set_ylabel('Angle Velocity')
        for i in range(4):
            axs[1, i].grid()
            axs[1, i].set_ylim(-1.5, 1.5)
            axs[0, i].set_yticklabels([])
        time_text = axs[1, 1].text(1.6, -0.2, f'Time step: {ts[single or 0]:.4f}', transform=axs[1, 1].transAxes, ha='center', va='top', color='black')
        axs[0, 0].text(-0.6, 1.3, 'A', transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold', va='top')
        axs[0, 0].text(0.8, 1.825, '$\\beta=0.5\\pi,\\sigma=1.0$', transform=axs[0, 0].transAxes, fontsize=12, va='top')
        axs[0, 2].text(-0.6, 1.3, 'B', transform=axs[0, 2].transAxes, fontsize=14, fontweight='bold', va='top')
        axs[0, 2].text(0.8, 1.825, '$\\beta=0.7\\pi,\\sigma=1.0$', transform=axs[0, 2].transAxes, fontsize=12, va='top')
        if single is not None:
            fig.savefig(filename)
            return

        def update(t: int) -> list:
            for i, (p, dp) in enumerate(((ys_1, dys_1), (ys_2, dys_2))):
                ii = i * 2
                phis_1, phis_2 = (p.phi_1, p.phi_2)
                dphis_1, dphis_2 = (dp.phi_1, dp.phi_2)
                for j, p in enumerate((phis_1, phis_2)):
                    phis_t = p[t]
                    pscatters[j + ii].set_offsets(np.c_[phis_t, radii])
                    z = np.exp(1j * phis_t)
                    R = np.abs(np.mean(z))
                    counts, _ = np.histogram(phis_t, bins=N_bins, range=(0, 2 * np.pi), density=True)
                    max_count = counts.max() if counts.max() > 0 else 1
                    for count, bar in zip(counts, bars[j + ii], strict=True):
                        bar.set_height(count)
                        bar.set_facecolor(cmaps[j](count / max_count))
                        bar.set_alpha(0.8)
                for j, d in enumerate((dphis_1, dphis_2)):
                    dphis_t = d[t]
                    dscatters[j + ii].set_offsets(np.c_[range_n, dphis_t])
                time_text.set_text(f'Time step: {ts[t]:.4f}')
            return [*pscatters, *dscatters, *bars[0], *bars[1], time_text]
        ani = FuncAnimation(fig, update, frames=T, blit=True)
        ani.save(filename, writer=FFMpegWriter(fps=fps))
        return fig

    return (gif_both,)


@app.cell
def _(dys_cluster, dys_sync, gif_both, ts_sync, ys_cluster, ys_sync):
    fig = gif_both(
        ys_sync, dys_sync, ys_cluster, dys_cluster,
        ts_sync,
        filename="typst/images/presentation/simulation.gif",
        fps=9,
    )
    fig
    return


@app.cell
def _(dys_cluster, dys_sync, gif_both, ts_sync, ys_cluster, ys_sync):
    gif_both(
        ys_sync, dys_sync, ys_cluster, dys_cluster,
        ts_sync,
        filename="typst/images/presentation/simulation0.png",
        single=0,
    )
    gif_both(
        ys_sync, dys_sync, ys_cluster, dys_cluster,
        ts_sync,
        filename="typst/images/presentation/simulation1.png",
        single=-1,
    )
    return


@app.cell
def _(Dopri5, DynamicNetworkModel, T_init, conf_cluster, conf_sync, jnp, jr):
    _N = 200
    M_1 = 50
    run_confs = [conf_sync, conf_cluster]
    step_size = 10
    T_max_1 = 1000
    T_step_1 = 10
    dnm_1 = DynamicNetworkModel(full_save=False, steady_state_check=False, progress_bar=True)
    ensemble_sols = [dnm_1.integrate(config=run_conf, M=M_1, key=jr.key(0), T_init=T_init, T_max=T_max_1, T_step=T_step_1, solver=Dopri5(), ts=jnp.arange(T_init, T_max_1, step_size)) for run_conf in run_confs]
    return T_step_1, ensemble_sols


@app.cell
def _(T_step_1, ensemble_sols, np, plt):
    def plot_ensembles() -> None:
        fig, axs = plt.subplots(len(ensemble_sols), 1, figsize=(6, 5), sharex=True)
        ysmax = 0.0
        for i, sol in enumerate(ensemble_sols):
            t = np.arange(sol.ys.r_1.shape[0]) * T_step_1
            ysmax = max(sol.ys.s_1.max(), sol.ys.s_2.max(), ysmax)
            ax0 = axs[i]
            ax0.plot(t, sol.ys.s_1, color='tab:blue', alpha=0.8)
            ax0.plot((0, 0), (0, 0), label='Single Ensemble Member', color='tab:blue', rasterized=True)
            ax0.plot(t, sol.ys.s_1.mean(axis=-1), c='tab:orange', label='Ensemble Mean')
            ax0.text(-0.16, 0.5, f"{('A', 'B')[i]}", transform=ax0.transAxes, fontsize=14, fontweight='bold', va='top')
            if i > 0:
                ax0.set_xlabel('t')
            ax0.set_ylim(0, ysmax * 1.05)
        axs[0].set_title('Standard Deviation of the Phase Velocity ($s^1$)' + '\n' + 'Parenchymal Layer')
        axs[0].legend(ncols=2, bbox_to_anchor=(1.0, 1.5), frameon=False)
        fig.subplots_adjust(wspace=0.4)
        return fig

    return (plot_ensembles,)


@app.cell
def _(plot_ensembles):
    ensemble_fig = plot_ensembles()
    ensemble_fig
    return (ensemble_fig,)


@app.cell
def _(ensemble_fig):
    ensemble_fig.savefig("../typst/images/presentation/ensembles.svg")
    return


@app.cell
def _(DNMConfig, DNMMetrics, Storage, as_2d_indices, jnp, np):
    ALPHA = -0.28
    BETA_SPACE = (0.4, 0.7, 0.005)  # only original area
    SIGMA_SPACE = (0.0, 1.5, 0.015)  # only original area
    _db_str = 'DaisyFinal'
    sim_storage = Storage(key_dim=9, metrics_kv_name=f'../data/{_db_str}SepsisMetrics.db/', parameter_k_name=f'../data/{_db_str}SepsisParameters_index.bin', use_mem_cache=True)
    sim_storage.close()
    b, r = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
    a = np.ones_like(b) * ALPHA
    indices_3d = jnp.concatenate([a[..., np.newaxis], b[..., np.newaxis], r[..., np.newaxis]], axis=-1)[np.newaxis, ...]
    spacing_3d = jnp.array([0.0, BETA_SPACE[2], SIGMA_SPACE[2]])
    params = DNMConfig.batch_as_index(a, b, r, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
    metrics_3d = metrics_3d.to_jax().reshape([1, *metrics_3d.shape['r_1']])
    metrics_2d = metrics_3d.squeeze()
    return ALPHA, BETA_SPACE, SIGMA_SPACE, metrics_2d


@app.cell
def _(BETA_SPACE, SIGMA_SPACE, metrics_2d, np, plt, space_plot):
    def plot_phase() -> plt.Figure:
        xs = np.arange(*BETA_SPACE)
        ys = np.arange(*SIGMA_SPACE)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        space_plot(
            metrics_2d.s_1, title=r"Standard Deviation of the Phase Velocity ($s^1$)" + "\n" + "Parenchymal Layer", xs=xs, ys=ys, figax=(fig, ax),
        )

        configs = {"A": (0.5, 1.0), "B": (0.697, 1.0)}
        for n, c in configs.items():
            b, s = c
            beta_scale = len(xs) * (b - BETA_SPACE[0]) / (BETA_SPACE[1] - BETA_SPACE[0])
            sigma_scale = len(ys) * (1 - (s - SIGMA_SPACE[0]) / (SIGMA_SPACE[1] - SIGMA_SPACE[0]))
            ax.scatter(beta_scale, sigma_scale, c="tab:orange")
            ax.annotate(n, (beta_scale -5, sigma_scale + 1), c="tab:orange", fontsize=18)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.25)
        return fig

    return (plot_phase,)


@app.cell
def _(plot_phase):
    phase_fig = plot_phase()
    phase_fig
    return (phase_fig,)


@app.cell
def _(phase_fig):
    phase_fig.savefig("../typst/images/presentation/phase.svg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LDM
    """)
    return


@app.cell
def _(setup_jax):
    from tbparse import SummaryReader
    import equinox as eqx
    from matplotlib.pyplot import Figure, Axes
    from matplotlib import colors
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sepsis_osc.ldm.train_online import process_val_epoch
    from sepsis_osc.ldm.checkpoint_utils import load_checkpoint
    from sepsis_osc.ldm.commons import build_lookup_table
    from sepsis_osc.ldm.data_loading import get_data_sets_online, prepare_batches_mask
    from sepsis_osc.ldm.model_structs import LossesConfig, LoadingConfig
    from scipy.stats import binned_statistic_2d
    setup_jax(simulation=False)
    return (
        Axes,
        Figure,
        LoadingConfig,
        LossesConfig,
        SummaryReader,
        average_precision_score,
        binned_statistic_2d,
        build_lookup_table,
        colors,
        eqx,
        get_data_sets_online,
        load_checkpoint,
        prepare_batches_mask,
        process_val_epoch,
        roc_auc_score,
    )


@app.cell
def _(
    ALPHA,
    BETA_SPACE,
    LossesConfig,
    SIGMA_SPACE,
    Storage,
    build_lookup_table,
    get_data_sets_online,
    jax_random_seed,
    jnp,
    jr,
    prepare_batches_mask,
):
    train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m = get_data_sets_online(swapaxes_y=(1, 2, 0), dtype=jnp.float32, path_prefix='..', cv_repetitions=2, repetition_index=0, cv_folds=2, fold_index=0)
    for y, m, s in ((train_y, train_m, 'Train'), (val_y, val_m, 'Val'), (test_y, test_m, 'Test')):
        print(f'Prevalence {s} Set {((y * m[..., None]).max(axis=1) == 1.0).mean(axis=0) * 100}%')
    key = jr.PRNGKey(jax_random_seed)
    test_x, test_y, test_m, _ = prepare_batches_mask(test_x, test_y, test_m, test_x.shape[0], key=key)
    test_x, test_y, test_m = (test_x[0], test_y[0], test_m[0])
    loss_conf = LossesConfig(lambda_sep3=300.0, lambda_inf=1.0, lambda_sofa_classification=2000.0, lambda_spreading=0.006, lambda_boundary=30.0, lambda_recon=2.5)
    _db_str = 'DaisyFinal'
    storage = Storage(key_dim=9, metrics_kv_name=f'../data/{_db_str}SepsisMetrics.db/', parameter_k_name=f'../data/{_db_str}SepsisParameters_index.bin', use_mem_cache=True)
    storage.close()
    lookup_table = build_lookup_table(storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)  # strip mini-epoch dimension
    return lookup_table, loss_conf, test_m, test_x, test_y


@app.cell
def _(
    LoadingConfig,
    SummaryReader,
    eqx,
    jax_random_seed,
    jr,
    load_checkpoint,
    logger,
):
    run_name = 'best'
    RUN_DIR = f'../runs/{run_name}'
    tb_reader = SummaryReader(RUN_DIR)
    tb_df = tb_reader.scalars
    hparams = tb_reader.hparams.T
    hparams.columns = hparams.loc['tag']
    hparams = hparams.drop('tag')
    auprc = tb_df.query("tag == 'sepsis_metrics/AUPRC_pred_sep'")
    prc_epoch = auprc.loc[auprc['value'].idxmax(), 'step']
    auroc = tb_df.query("tag == 'sepsis_metrics/AUROC_pred_sep'")
    roc_epoch = auroc.loc[auroc['value'].idxmax(), 'step']
    best_epoch = round((prc_epoch + roc_epoch) / 2)
    print(f"Best Epoch: {best_epoch} with between AUPRC={auprc['value'].max():.3f}@{prc_epoch} AUROC={auroc['value'].max():.3f}@{roc_epoch}")
    LOAD_EPOCH = best_epoch
    key_1 = jr.PRNGKey(jax_random_seed)
    load_conf = LoadingConfig(from_dir=RUN_DIR, epoch=LOAD_EPOCH)
    if load_conf.from_dir:
        try:
            model, _ = load_checkpoint(load_conf.from_dir + '/checkpoints', load_conf.epoch, None)
        except FileNotFoundError as e:
            logger.warning(f'Error loading checkpoint: {e}.')
            load_conf.epoch = 0
            load_conf.from_dir = ''
    model = eqx.nn.inference_mode(model)
    assert model
    return key_1, model


@app.cell
def _(
    jnp,
    key_1,
    lookup_table,
    loss_conf,
    model,
    process_val_epoch,
    test_m,
    test_x,
    test_y,
):
    test_metrics = process_val_epoch(model, x_data=test_x, y_data=test_y, mask_data=test_m, step=jnp.array(1000000.0, dtype=jnp.int32), key=key_1, lookup_func=lookup_table.soft_get_local, loss_params=loss_conf)
    return (test_metrics,)


@app.cell
def _(
    average_precision_score,
    jnp,
    np,
    roc_auc_score,
    test_m,
    test_metrics,
    test_y,
):
    true_sofa = np.asarray(test_y[..., 0])[test_m]
    true_sofa_d2 = np.concat(
        [np.zeros(test_y.shape[:-2])[..., None], np.asarray(jnp.diff(test_y[..., 0], axis=-1) > 0)], axis=-1
    )[test_m]
    true_inf = np.asarray(test_y[..., 1])[test_m]
    true_sep3 = np.asarray(test_y[..., 2] == 1.0)[test_m]
    pred_sep3_risk = np.asarray(test_metrics.sep3_risk)[test_m]
    pred_sofa_d2_risk = np.asarray(test_metrics.sofa_d2_risk)[test_m]
    pred_susp_inf_p = np.asarray(test_metrics.susp_inf_p)[test_m]
    pred_sofa_score = np.asarray(test_metrics.hists_sofa_score)[test_m]

    print(f"AUROC sepis-3  {roc_auc_score(true_sep3, pred_sep3_risk) * 100:.3f}")
    print(f"AUPRC sepis-3  {average_precision_score(true_sep3, pred_sep3_risk) * 100:.3f}")
    print(f"AUROC sofa d2  {roc_auc_score(true_sofa_d2, pred_sofa_d2_risk) * 100:.2f}")
    print(f"AUPRC sofa d2  {average_precision_score(true_sofa_d2, pred_sofa_d2_risk) * 100:.2f}")
    print(f"AUROC susp inf {roc_auc_score(true_inf > 0, pred_susp_inf_p) * 100:.2f}")
    print(f"AUPRC susp inf {average_precision_score(true_inf > 0, pred_susp_inf_p) * 100:.2f}")
    return


@app.cell
def _(heat_fig):
    heat_fig.savefig("../typst/images/presentation/heat.svg")
    return


@app.cell
def _(
    Axes,
    BETA_SPACE,
    Figure,
    LatentLookup,
    SIGMA_SPACE,
    binned_statistic_2d,
    colors,
    jnp,
    np,
    plt,
    space_plot,
    test_y,
):
    def viz_space_heatmap(betas: jnp.ndarray, sigmas: jnp.ndarray, lookup: LatentLookup, *, mask: np.ndarray, cmaps: bool=True, figax: tuple[Figure, Axes] | None=None) -> tuple[Figure, Axes]:
        if figax is not None:
            fig, ax = figax
        else:
            fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(top=0.95, bottom=0.2)
        betas_space = jnp.arange(*BETA_SPACE)
        sigmas_space = jnp.arange(*SIGMA_SPACE)
        beta_grid, sigma_grid = np.meshgrid(betas_space, sigmas_space, indexing='ij')
        param_grid = np.stack([beta_grid.ravel(), sigma_grid.ravel()], axis=1)
        metrics = lookup.hard_get_fsq(jnp.asarray(param_grid)).reshape(len(betas_space), len(sigmas_space))
        ax = space_plot(metrics, xs=np.asarray(betas_space), ys=np.asarray(sigmas_space), title=f'', cmap=True, figax=(fig, ax), filename='')
        _N, T = betas.shape
        reso = 0.25
        bins = [np.arange(BETA_SPACE[0], BETA_SPACE[1], BETA_SPACE[2] * reso), np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1], SIGMA_SPACE[2] * reso)]
        heatmap, xedges, yedges = np.histogram2d(betas[mask], sigmas[mask], bins=bins)
        heatmap = heatmap[:, ::-1].T
        alpha = np.ones_like(heatmap, dtype=np.float32)
        alpha[heatmap == 0] = 0.0
        im = ax.images[0]
        extent = im.get_extent()
        cbar = ax.images[0].colorbar
        cbar_ax = cbar.ax
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])
        x = betas[mask]
        y = sigmas[mask]
        values = test_y[0, mask, 0]
        mean_map, xedges, yedges, _ = binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
        mean_map = mean_map[:, ::-1].T
        alpha_mean = ~np.isnan(mean_map)
        mean_map = np.nan_to_num(mean_map, nan=0.0)
        cm = plt.colormaps.get_cmap('copper')
        norm = colors.Normalize(vmin=0, vmax=24)
        im_mean = ax.imshow(mean_map, extent=extent, aspect='auto', cmap=cm, norm=norm, alpha=alpha_mean.astype(float))
        cbar = fig.colorbar(im_mean, ax=ax, location='right', shrink=0.8)
        cbar.set_label('Mean of the ground truth SOFA-score')
        ax.set_xlabel('$\\beta / \\pi$')
        ax.set_ylabel('$\\sigma$')
        return (fig, ax)

    return (viz_space_heatmap,)


@app.cell
def _(lookup_table, test_m, test_metrics, viz_space_heatmap):
    heat_space_fig, ax = viz_space_heatmap(
        test_metrics.beta[0],
        test_metrics.sigma[0],
        lookup=lookup_table,
        mask=test_m[0],
    )

    heat_space_fig
    return (heat_space_fig,)


@app.cell
def _(heat_space_fig):
    heat_space_fig.savefig("../typst/images/presentation/heat_space.svg")
    return


if __name__ == "__main__":
    app.run()
