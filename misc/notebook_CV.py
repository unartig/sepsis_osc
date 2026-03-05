import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import logging

    import pandas as pd
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tbparse import SummaryReader
    from scipy import stats

    from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel
    from sepsis_osc.ldm.data_loading import get_data_sets_online
    from sepsis_osc.ldm.model_structs import LossesConfig
    from sepsis_osc.ldm.train_online import process_val_epoch
    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed, plt_params
    from sepsis_osc.utils.jax_config import setup_jax

    from sepsis_osc.ldm.checkpoint_utils import load_checkpoint
    from sepsis_osc.ldm.model_structs import LoadingConfig
    from sepsis_osc.ldm.commons import build_lookup_table
    setup_jax(simulation=False)
    from sepsis_osc.utils.logger import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)

    from matplotlib.patches import Ellipse


    plt.rcParams.update(plt_params)
    return (
        ALPHA,
        BETA_SPACE,
        Ellipse,
        LatentDynamicsModel,
        LoadingConfig,
        LossesConfig,
        SIGMA_SPACE,
        Storage,
        SummaryReader,
        average_precision_score,
        build_lookup_table,
        eqx,
        get_data_sets_online,
        jax_random_seed,
        jnp,
        jr,
        load_checkpoint,
        logger,
        np,
        pd,
        plt,
        process_val_epoch,
        roc_auc_score,
        stats,
    )


@app.cell
def _(
    ALPHA,
    BETA_SPACE,
    LossesConfig,
    SIGMA_SPACE,
    Storage,
    build_lookup_table,
):
    db_str = "DaisyFinal"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"../data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"../data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    storage.close()

    loss_conf = LossesConfig(
        lambda_sep3=300.0,
        lambda_inf=1.0,
        lambda_sofa_classification=2000.0,
        lambda_spreading=6e-3,
        lambda_boundary=30.0,
        lambda_recon=2.5,
    )    

    lookup_table = build_lookup_table(storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)
    return lookup_table, loss_conf


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
    def get_model(rep: int, fold: int) -> tuple[LatentDynamicsModel, pd.DataFrame, pd.DataFrame, str, int, pd.DataFrame, pd.DataFrame]:
        _run_name = f'rep{rep:002}_fold{fold:002}'
        run_dir = f'../runs/cv3/{_run_name}'
        tb_reader = SummaryReader(run_dir)
        _tb_df = tb_reader.scalars
        hparams = tb_reader.hparams.T
        hparams.columns = hparams.loc['tag']
        hparams = hparams.drop('tag')
        _auprc = _tb_df.query("tag == 'sepsis_metrics/AUPRC_pred_sep'")
        prc_epoch = _auprc.loc[_auprc['value'].idxmax(), 'step']
        _auroc = _tb_df.query("tag == 'sepsis_metrics/AUROC_pred_sep'")
        roc_epoch = _auroc.loc[_auroc['value'].idxmax(), 'step']
        best_epoch = round((prc_epoch + roc_epoch) / 2)
        logger.info(f"Best Epoch: {best_epoch} with between AUPRC={_auprc['value'].max():.3f}@{prc_epoch} AUROC={_auroc['value'].max():.3f}@{roc_epoch}")
        _load_epoch = best_epoch
        load_conf = LoadingConfig(from_dir=run_dir, epoch=_load_epoch)
        if load_conf.from_dir:
            model, _ = load_checkpoint(load_conf.from_dir + '/checkpoints', load_conf.epoch, None)
        model = eqx.nn.inference_mode(model)
        assert model
        return (model, _tb_df, hparams, _run_name, _load_epoch, _auroc, _auprc)

    return (get_model,)


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
    repetitions = 5
    n_folds = 5
    rows = []
    dtype = jnp.float32
    key = jr.PRNGKey(jax_random_seed)
    for rep in range(repetitions):
        for fold in range(n_folds):
            try:
                model, _tb_df, hparams, _run_name, _load_epoch, _auroc, _auprc = get_model(rep, fold)
            except ValueError as e:
                logger.warning(f'Summary not found {rep}-{repetitions - 1} {fold}-{n_folds - 1}')
                continue
            except FileNotFoundError as e:
                logger.warning(f'Failed loading checkpoint: {e}.')
                continue
            except Exception as e:
                logger.warning(f'Failed to load {rep}-{repetitions - 1} {fold}-{n_folds - 1} sequences. {e}')
                continue
            logger.info(f'Loaded {rep}-{repetitions} {fold}-{n_folds}.')
            _data = get_data_sets_online(swapaxes_y=(1, 2, 0), dtype=dtype, cv_repetitions=repetitions, repetition_index=rep, cv_folds=n_folds, fold_index=fold, sequence_files='../data/cv/sequence_')
            *_, test_x, test_y, test_m = _data
            test_m = test_m.astype(np.bool)
            test_x, test_y, test_m = (test_x[None], test_y[None], test_m[None])
            test_metrics = process_val_epoch(model, x_data=test_x, y_data=test_y, mask_data=test_m, step=jnp.array(1000000.0, dtype=jnp.int32), key=key, lookup_func=lookup_table.soft_get_local, loss_params=loss_conf)
            true_sofa = np.asarray(test_y[..., 0])[test_m]
            _true_sofa_d2 = np.concat([np.zeros(test_y.shape[:-2])[..., None], np.asarray(jnp.diff(test_y[..., 0], axis=-1) > 0)], axis=-1)[test_m]
            true_inf = np.asarray(test_y[..., 1])[test_m]
            true_sep3 = np.asarray(test_y[..., 2] == 1.0)[test_m]
            pred_sep3_risk = np.asarray(test_metrics.sep3_risk)[test_m]
            _pred_sofa_d2_risk = np.asarray(test_metrics.sofa_d2_risk)[test_m]
            pred_susp_inf_p = np.asarray(test_metrics.susp_inf_p)[test_m]
            pred_sofa_score = np.asarray(test_metrics.hists_sofa_score)[test_m]
            row = {'repetition': rep, 'fold': fold, 'run_name': _run_name, 'best_epoch': int(_load_epoch), 'val_auprc_max': float(_auprc['value'].max()), 'val_auroc_max': float(_auroc['value'].max()), 'test_auroc_sep3': roc_auc_score(true_sep3, pred_sep3_risk), 'test_auprc_sep3': average_precision_score(true_sep3, pred_sep3_risk), 'test_auroc_sofa_d2': roc_auc_score(_true_sofa_d2, _pred_sofa_d2_risk), 'test_auprc_sofa_d2': average_precision_score(_true_sofa_d2, _pred_sofa_d2_risk), 'test_auroc_susp_inf': roc_auc_score(true_inf > 0, pred_susp_inf_p), 'test_auprc_susp_inf': average_precision_score(true_inf > 0, pred_susp_inf_p)}
            rows.append(row)
    results_df = pd.DataFrame(rows)  # validation info from tensorboard  # test metrics
    return dtype, hparams, key, results_df


@app.cell
def _(results_df):
    c = results_df.copy()
    return (c,)


@app.cell
def _(c):
    results_df_1 = c.copy()
    print(len(results_df_1))
    return (results_df_1,)


@app.cell
def _(np, results_df_1):
    summary = results_df_1.agg({'test_auroc_sep3': ['mean', 'std'], 'test_auprc_sep3': ['mean', 'std']})
    print(np.round(summary * 100, 4))
    return


@app.cell
def _(np, stats):
    def welch_t_from_stats(mean1, std1, n1, mean2, std2, n2):
        # t statistic
        se = np.sqrt((std1**2)/n1 + (std2**2)/n2)
        t = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        df_num = (std1**2/n1 + std2**2/n2) ** 2
        df_den = ((std1**2/n1)**2)/(n1-1) + ((std2**2/n2)**2)/(n2-1)
        df = df_num / df_den

        # two-sided p-value
        p = 2 * stats.t.sf(abs(t), df)

        return t, df, p

    return (welch_t_from_stats,)


@app.cell
def _(results_df_1, welch_t_from_stats):
    yaib_auroc_mean = 83.6 / 100
    yaib_auroc_std = 0.3 / 100
    ldm_auroc_mean = results_df_1['test_auroc_sep3'].mean()
    ldm_auroc_std = results_df_1['test_auroc_sep3'].std()
    yaib_auprc_mean = 9.1 / 100
    yaib_auprc_std = 0.3 / 100
    ldm_auprc_mean = results_df_1['test_auprc_sep3'].mean()
    ldm_auprc_std = results_df_1['test_auprc_sep3'].std()
    n1 = len(results_df_1)
    n2 = 25
    t_auroc, df_auroc, p_auroc = welch_t_from_stats(ldm_auroc_mean, ldm_auroc_std, n1, yaib_auroc_mean, yaib_auroc_std, n2)
    print('AUROC comparison')
    print('t =', t_auroc)
    print('df =', df_auroc)
    print('p =', p_auroc)
    t_auprc, df_auprc, p_auprc = welch_t_from_stats(ldm_auprc_mean, ldm_auprc_std, n1, yaib_auprc_mean, yaib_auprc_std, n2)
    print('\nAUPRC comparison')
    print('t =', t_auprc)
    print('df =', df_auprc)
    print('p =', p_auprc)
    return (
        ldm_auprc_mean,
        ldm_auprc_std,
        ldm_auroc_mean,
        ldm_auroc_std,
        n1,
        n2,
        yaib_auprc_mean,
        yaib_auprc_std,
        yaib_auroc_mean,
        yaib_auroc_std,
    )


@app.cell
def _(
    ldm_auprc_mean,
    ldm_auprc_std,
    ldm_auroc_mean,
    ldm_auroc_std,
    n1,
    n2,
    stats,
    yaib_auprc_mean,
    yaib_auprc_std,
    yaib_auroc_mean,
    yaib_auroc_std,
):
    "One sided, H0 - same mean, H1 - LDM mean is greater"
    rroc = stats.ttest_ind_from_stats(ldm_auroc_mean, ldm_auroc_std, n1, yaib_auroc_mean, yaib_auroc_std, n2, alternative="greater", equal_var=False)
    rprc = stats.ttest_ind_from_stats(ldm_auprc_mean, ldm_auprc_std, n1, yaib_auprc_mean, yaib_auprc_std, n2, alternative="greater", equal_var=False)
    print(rroc)
    print(rprc)
    return


@app.cell
def _(plt, results_df_1, yaib_auprc_mean, yaib_auroc_mean):
    plt.scatter(results_df_1['test_auroc_sep3'], results_df_1['test_auprc_sep3'])
    plt.scatter(results_df_1['test_auroc_sep3'].mean(), results_df_1['test_auprc_sep3'].mean())
    plt.scatter(yaib_auroc_mean, yaib_auprc_mean)
    plt.grid()
    return


@app.cell
def _(plt, results_df_1, yaib_auprc_mean, yaib_auroc_mean):
    _fig, _ax = plt.subplots(1, 2)
    _ax[0].hist(results_df_1['test_auroc_sep3'])
    _ax[0].vlines(results_df_1['test_auroc_sep3'].mean(), 0, 5, color='tab:green')
    _ax[0].vlines(yaib_auroc_mean, 0, 5, color='tab:orange')
    _ax[1].hist(results_df_1['test_auprc_sep3'])
    _ax[1].vlines(results_df_1['test_auprc_sep3'].mean(), 0, 5, color='tab:green')
    _ax[1].vlines(yaib_auprc_mean, 0, 5, color='tab:orange')
    return


@app.cell
def _(Ellipse, np):
    def plot_cov_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
        """
        Draw a covariance ellipse centered at `mean`.
        n_std=2 → ~95% ellipse if data ~ normal.
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
    results_df_1,
    yaib_auprc_mean,
    yaib_auprc_std,
    yaib_auroc_mean,
    yaib_auroc_std,
):
    x = results_df_1['test_auroc_sep3'].values
    y = results_df_1['test_auprc_sep3'].values
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
def _():
    from sepsis_osc.visualisations.viz_model_results import (viz_curves_sepsis, viz_loss_mean_std, viz_patients, 
                                                             viz_concept_densities, viz_space_heatmap)

    return (
        viz_concept_densities,
        viz_curves_sepsis,
        viz_loss_mean_std,
        viz_patients,
        viz_space_heatmap,
    )


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
def _(aggregate_runs, load_tb_scalars):
    # Prepare run directories
    repetitions_1 = 5
    n_folds_1 = 5
    run_dirs = [f'../runs/cv3/rep{rep:02}_fold{fold:02}' for rep in range(repetitions_1) for fold in range(n_folds_1)]
    losses = ['total_loss', 'sepsis-3', 'sofa', 'infection', 'recon_loss', 'spreading_loss', 'boundary_loss']
    # Define tags you want
    metrics = ['AUROC_pred_sep', 'AUPRC_pred_sep']
    tags = [f'sepsis_metrics/{m}' for m in metrics]
    for loss in losses:
    # Dynamically build the list
        tags.extend([f'train_losses/{loss}_mean', f'val_losses/{loss}_mean'])
    all_data = load_tb_scalars(run_dirs, tags)
    agg_data = aggregate_runs(all_data)
    losses = ('sepsis-3', 'sofa', 'infection', 'recon_loss', 'spreading_loss', 'boundary_loss')
    loss_subscripts = ('sepsis', 'sofa', 'inf', 'dec', 'spread', 'boundary')
    # Plot
    lambdas = ('sep3', 'sofa_classification', 'inf', 'recon', 'spreading', 'boundary')
    return agg_data, lambdas, loss_subscripts, losses, n_folds_1, repetitions_1


@app.cell
def _(agg_data, hparams, lambdas, loss_subscripts, losses, viz_loss_mean_std):
    loss_fig = viz_loss_mean_std(agg_data, losses, loss_subscripts, lambdas, hparams)

    loss_fig
    return (loss_fig,)


@app.cell
def _(loss_fig):
    loss_fig.savefig("../typst/images/losses_std.svg")
    loss_fig.savefig("../typst/images/paper/losses_std.png", dpi=400)
    return


@app.cell
def _(np, results_df_1):
    mean_auprc = results_df_1['val_auprc_max'].mean()
    mean_auroc = results_df_1['val_auroc_max'].mean()
    print(mean_auroc)
    print(mean_auprc)
    results_df_1['dist_from_mean'] = np.sqrt((results_df_1['val_auprc_max'] - mean_auprc) ** 2 + (results_df_1['val_auroc_max'] - mean_auroc) ** 2)
    # Compute Euclidean distance from the mean
    avg_run_row = results_df_1.loc[results_df_1['dist_from_mean'].idxmin()]
    avg_run_name = avg_run_row['run_name']
    print('Average / representative run:', avg_run_name)
    print(avg_run_row[['val_auprc_max', 'val_auroc_max']])
    rep_1, fold_1 = avg_run_name.split('_')
    # Pick run with minimal distance
    rep_1, fold_1 = (int(rep_1[-2:]), int(fold_1[-2:]))
    return fold_1, rep_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single Run Analysis
    """)
    return


@app.cell
def _(
    dtype,
    fold_1,
    get_data_sets_online,
    get_model,
    n_folds_1,
    np,
    rep_1,
    repetitions_1,
):
    model_1, _tb_df, hparams_1, _run_name, _load_epoch, _auroc, _auprc = get_model(rep_1, fold_1)
    _data = get_data_sets_online(swapaxes_y=(1, 2, 0), dtype=dtype, cv_repetitions=repetitions_1, repetition_index=rep_1, cv_folds=n_folds_1, fold_index=fold_1, sequence_files='../data/cv/sequence_')
    *_, test_x_1, test_y_1, test_m_1 = _data
    test_m_1 = test_m_1.astype(np.bool)
    test_x_1, test_y_1, test_m_1 = (test_x_1[None], test_y_1[None], test_m_1[None])
    return hparams_1, model_1, test_m_1, test_x_1, test_y_1


@app.cell
def _(hparams_1):
    hparams_1.T
    return


@app.cell
def _(
    jnp,
    key,
    lookup_table,
    loss_conf,
    model_1,
    process_val_epoch,
    test_m_1,
    test_x_1,
    test_y_1,
):
    test_metrics_1 = process_val_epoch(model_1, x_data=test_x_1, y_data=test_y_1, mask_data=test_m_1, step=jnp.array(1000000.0, dtype=jnp.int32), key=key, lookup_func=lookup_table.soft_get_local, loss_params=loss_conf)
    return (test_metrics_1,)


@app.cell
def _(
    average_precision_score,
    jnp,
    np,
    roc_auc_score,
    test_m_1,
    test_metrics_1,
    test_y_1,
):
    true_sofa_1 = np.asarray(test_y_1[..., 0])[test_m_1]
    _true_sofa_d2 = np.concat([np.zeros(test_y_1.shape[:-2])[..., None], np.asarray(jnp.diff(test_y_1[..., 0], axis=-1) > 0)], axis=-1)[test_m_1]
    true_inf_1 = np.asarray(test_y_1[..., 1])[test_m_1]
    true_sep3_1 = np.asarray(test_y_1[..., 2] == 1.0)[test_m_1]
    pred_sep3_risk_1 = np.asarray(test_metrics_1.sep3_risk)[test_m_1]
    _pred_sofa_d2_risk = np.asarray(test_metrics_1.sofa_d2_risk)[test_m_1]
    pred_susp_inf_p_1 = np.asarray(test_metrics_1.susp_inf_p)[test_m_1]
    pred_sofa_score_1 = np.asarray(test_metrics_1.hists_sofa_score)[test_m_1]
    print(f'AUROC sepis-3  {roc_auc_score(true_sep3_1, pred_sep3_risk_1) * 100:.3f}')
    print(f'AUPRC sepis-3  {average_precision_score(true_sep3_1, pred_sep3_risk_1) * 100:.3f}')
    print(f'AUROC sofa d2  {roc_auc_score(_true_sofa_d2, _pred_sofa_d2_risk) * 100:.2f}')
    print(f'AUPRC sofa d2  {average_precision_score(_true_sofa_d2, _pred_sofa_d2_risk) * 100:.2f}')
    print(f'AUROC susp inf {roc_auc_score(true_inf_1 > 0, pred_susp_inf_p_1) * 100:.2f}')
    print(f'AUPRC susp inf {average_precision_score(true_inf_1 > 0, pred_susp_inf_p_1) * 100:.2f}')
    return (
        pred_sep3_risk_1,
        pred_sofa_score_1,
        pred_susp_inf_p_1,
        true_inf_1,
        true_sep3_1,
        true_sofa_1,
    )


@app.cell
def _(
    pred_sofa_score_1,
    pred_susp_inf_p_1,
    true_inf_1,
    true_sofa_1,
    viz_concept_densities,
):
    heat_fig, heat_ax = viz_concept_densities(true_sofa_1, true_inf_1, pred_sofa_score_1, pred_susp_inf_p_1, cmap=True)
    heat_fig
    return (heat_fig,)


@app.cell
def _(heat_fig):
    heat_fig.savefig("../typst/images/paper/heat.png", dpi=400)
    return


@app.cell
def _(np, test_metrics_1, test_y_1):
    _p_idx = np.argsort(test_y_1[0, :, :, 0].std(axis=-1))[-10:]
    print(_p_idx)
    _p_idx = np.argsort(test_y_1[0, :, :, 0].max(axis=-1))[-10:]
    print(_p_idx)
    _p_idx = np.argsort(np.diff(test_y_1[0, :, :, 0], axis=-1).sum(axis=-1))[-10:]
    print(_p_idx)
    _p_idx = np.argsort(np.square(test_y_1[0, :, :, 0] - test_metrics_1.hists_sofa_score[0, :]).mean(axis=-1))[:10]
    print(_p_idx)
    _p_idx = np.argsort(np.square(test_y_1[0, :, :, 0] - test_metrics_1.hists_sofa_score[0, :]).mean(axis=-1))[-10:]
    print(_p_idx)
    return


@app.cell
def _(lookup_table, test_m_1, test_metrics_1, test_y_1, viz_patients):
    _p_idx = [3638, 12, 4154]
    patient_fig = viz_patients(_p_idx, test_y_1, test_m_1, test_metrics_1, lookup_table)
    patient_fig
    return (patient_fig,)


@app.cell
def _(patient_fig):
    patient_fig.savefig("../typst/images/paper/trajectory.png", dpi=400)
    return


@app.cell
def _(lookup_table, test_m_1, test_metrics_1, test_y_1, viz_space_heatmap):
    heat_space_fig, _ax = viz_space_heatmap(test_y_1, test_metrics_1.beta[0], test_metrics_1.sigma[0], lookup=lookup_table, mask=test_m_1[0])
    heat_space_fig
    return (heat_space_fig,)


@app.cell
def _(heat_space_fig):
    heat_space_fig.savefig("../typst/images/paper/heat_space.png", dpi=400)
    return


@app.cell
def _(pred_sep3_risk_1, true_sep3_1, viz_curves_sepsis):
    area_fig = viz_curves_sepsis(true_sep3_1, pred_sep3_risk_1)
    area_fig
    return (area_fig,)


@app.cell
def _(area_fig):
    area_fig.savefig("../typst/images/paper/areas.png", dpi=400)
    return


if __name__ == "__main__":
    app.run()
