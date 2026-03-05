import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import logging
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr
    import jax.tree_util as jtu
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tbparse import SummaryReader
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    from scipy.stats import binned_statistic_2d
    from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
    from matplotlib.cm import ScalarMappable
    from sepsis_osc.ldm.data_loading import get_data_sets_online, prepare_batches_mask
    from sepsis_osc.ldm.lookup import as_2d_indices, compute_window_bounds
    from sepsis_osc.ldm.model_structs import LossesConfig
    from sepsis_osc.ldm.train_online import process_val_epoch
    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.utils.config import ALPHA, ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE, jax_random_seed, plt_params
    from sepsis_osc.utils.jax_config import setup_jax
    from sepsis_osc.visualisations.viz_model_results import viz_concept_densities, viz_patients_latent, viz_patients, viz_space_heatmap, viz_losses, viz_curves_sepsis
    from sepsis_osc.ldm.commons import build_lookup_table
    from sepsis_osc.ldm.checkpoint_utils import load_checkpoint
    from sepsis_osc.ldm.model_structs import LoadingConfig
    setup_jax(simulation=False)
    from sepsis_osc.utils.logger import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
    from matplotlib.pyplot import Axes, Figure
    from matplotlib import colors
    from matplotlib.gridspec import GridSpec
    from sepsis_osc.visualisations.viz_param_space import space_plot
    from sepsis_osc.ldm.lookup import LatentLookup
    plt.rcParams.update(plt_params)
    return (
        ALPHA,
        ALPHA_SPACE,
        BETA_SPACE,
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
        jtu,
        load_checkpoint,
        logger,
        np,
        prepare_batches_mask,
        process_val_epoch,
        roc_auc_score,
        viz_concept_densities,
        viz_curves_sepsis,
        viz_losses,
        viz_patients,
        viz_space_heatmap,
    )


@app.cell
def _(
    LossesConfig,
    get_data_sets_online,
    jax_random_seed,
    jnp,
    jr,
    prepare_batches_mask,
):
    (
        train_x,
        train_y,
        train_m,
        val_x,
        val_y,
        val_m,
        test_x,
        test_y,
        test_m,
    ) = get_data_sets_online(swapaxes_y=(1, 2, 0), dtype=jnp.float32, path_prefix="..", cv_repetitions=2, repetition_index=0, cv_folds=2, fold_index=0)
    for y, m, s in ((train_y, train_m, "Train"), (val_y, val_m, "Val"), (test_y, test_m, "Test")):
        print(f"Prevalence {s} Set {((y * m[..., None]).max(axis=1) == 1.0).mean(axis=0) * 100}%")

    key = jr.PRNGKey(jax_random_seed)
    test_x, test_y, test_m, _ = prepare_batches_mask(test_x, test_y, test_m, test_x.shape[0], key=key)
    test_x, test_y, test_m = test_x[0], test_y[0], test_m[0]  # strip mini-epoch dimension

    loss_conf = LossesConfig(
        lambda_sep3=300.0,
        lambda_inf=1.0,
        lambda_sofa_classification=2000.0,
        lambda_spreading=6e-3,
        lambda_boundary=30.0,
        lambda_recon=2.5,
    )
    return loss_conf, test_m, test_x, test_y


@app.cell
def _(
    ALPHA,
    ALPHA_SPACE,
    BETA_SPACE,
    SIGMA_SPACE,
    Storage,
    build_lookup_table,
    np,
):
    betas = np.arange(BETA_SPACE[0], BETA_SPACE[1], BETA_SPACE[2])
    sigmas = np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1], SIGMA_SPACE[2])
    alphas = np.arange(ALPHA_SPACE[0], ALPHA_SPACE[1], ALPHA_SPACE[2])

    # DATA
    db_str = "DaisyFinal"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"../data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"../data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    sim_storage.close()
    lookup_table = build_lookup_table(sim_storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)
    return (lookup_table,)


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
    return hparams, key_1, model, tb_df


@app.cell
def _(eqx, jtu, model):
    def print_param_shapes(model):
        leaves_with_paths = jtu.tree_leaves_with_path(eqx.filter(model, eqx.is_array))
        total = 0
        print(f"{'Name':30} {'Shape':12}Parameter Count")
        for path, value in leaves_with_paths:
            name = '.'.join((str(p) for p in path))  # Convert path (tuple of keys) into readable string
            print(f"{name.strip('.'):30} {str(value.shape):12}{value.size}")
            total = total + value.size
        print(f'Total: {total}')
    print('+++ Infection Module Parameter +++')
    print_param_shapes({'GRU-Cell': model.inf_encoder, 'Proj': model.inf_proj_out})
    print('+++ SOFA Module Parameter +++')
    print_param_shapes(model.latent_pre_encoder)
    print_param_shapes({'GRU-Cell': model.latent_encoder, 'Proj': model.latent_proj_out})
    print('+++ Decoder +++')
    print_param_shapes(model.decoder)
    return


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
    return (
        pred_sep3_risk,
        pred_sofa_score,
        pred_susp_inf_p,
        true_inf,
        true_sep3,
        true_sofa,
    )


@app.cell
def _(
    pred_sofa_score,
    pred_susp_inf_p,
    true_inf,
    true_sofa,
    viz_concept_densities,
):
    heat_fig, heat_ax = viz_concept_densities(true_sofa, true_inf, pred_sofa_score, pred_susp_inf_p, cmap=True)
    heat_fig
    return (heat_fig,)


@app.cell
def _(heat_fig):
    heat_fig.savefig("../typst/images/heat.svg")
    heat_fig.savefig("../typst/images/paper/heat.png", dpi=400)
    return


@app.cell
def _(lookup_table, np, test_m, test_metrics, test_y, viz_patients):
    # 4494 3908

    # highest sofa std in first batch
    p_idx = np.argsort(test_y[0, :, :, 0].std(axis=-1))[-10:]
    print(p_idx)
    # highest sofa std in first batch
    p_idx = np.argsort(test_y[0, :, :, 0].max(axis=-1))[-10:]
    print(p_idx)
    # largest increase
    p_idx = np.argsort((np.diff(test_y[0, :, :, 0], axis=-1).sum(axis=-1) == 1))[-20:]
    print(p_idx)
    # best predictions
    p_idx = np.argsort(np.square(test_y[0, :, :, 0] - test_metrics.hists_sofa_score[0, :]).mean(axis=-1))[:20]
    print(p_idx)
    # highest sofa decrease
    # 10 24 23 33 54 61 70
    # 70

    # worsening 5332, 4608, 5814
    # recover 16 29, 84, 107
    # healthy 18, 62 5114, 4199, 1251, 137, 141, 147, 150, 157
    p_idx = [5814, 107, 5114]
    patient_fig = viz_patients(p_idx, test_y, test_m, test_metrics, lookup_table)
    patient_fig
    return (patient_fig,)


@app.cell
def _(patient_fig):
    patient_fig.savefig(f"../typst/images/trajectory.svg")
    patient_fig.savefig(f"../typst/images/paper/trajectory.png", dpi=400)
    return


@app.cell
def _(lookup_table, test_m, test_metrics, test_y, viz_space_heatmap):
    heat_space_fig, ax = viz_space_heatmap(
        test_y,
        test_metrics.beta[0],
        test_metrics.sigma[0],
        lookup=lookup_table,
        mask=test_m[0],
    )

    heat_space_fig
    return (heat_space_fig,)


@app.cell
def _(heat_space_fig):
    heat_space_fig.savefig("../typst/images/heat_space.svg")
    heat_space_fig.savefig("../typst/images/paper/heat_space.png", dpi=400)
    heat_space_fig.savefig("../typst/images/heat_space.png", dpi=400)
    return


@app.cell
def _():
    """
    def viz_loss() -> Figure:
        fig, axs = plt.subplots(3, 3, sharex=True)

        data_train = tb_df.query(f"tag == 'train_losses/total_loss_mean'")
        data_val = tb_df.query(f"tag == 'val_losses/total_loss_mean'")
        axs[0, 0].plot(data_train["step"], data_train["value"], label="Training")
        axs[0, 0].plot(data_val["step"], data_val["value"], label="Validation")
        axs[0, 0].legend(ncols=2, bbox_to_anchor=(2.4, 1.75), frameon=False)
        axs[0, 0].set_title(r"log($L_\text{total}$)")
        axs[0, 0].grid(True)
        axs[0, 0].set_yscale("log")
        print((data_train["value"]).to_numpy()[[0, -1]], (data_val["value"]).to_numpy()[[0, -1]])

        data = tb_df.query(f"tag == 'sepsis_metrics/AUROC_pred_sep'")
        axs[0, 1].plot(data["step"], data["value"], c="tab:orange")
        axs[0, 1].set_title(r"AUROC")
        axs[0, 1].grid(True)

        data = tb_df.query(f"tag == 'sepsis_metrics/AUPRC_pred_sep'")
        axs[0, 2].plot(data["step"], data["value"], c="tab:orange")
        axs[0, 2].set_title(r"AUPRC")
        axs[0, 2].grid(True)

        losses = ("sepsis-3", "sofa", "infection", "recon_loss", "spreading_loss", "boundary_loss")
        loss_subscripts = ("sepsis", "sofa", "inf", "dec", "spread", "boundary")
        lambdas = ("sep3", "sofa_classification", "inf", "recon", "spreading", "boundary")

        #axs = np.empty((2, 3), dtype=np.object_)
        for i, (loss, subscript, weight) in enumerate(zip(losses, loss_subscripts, lambdas, strict=True)):
            ax = axs[i//3 + 1, i%3]
            data_train = tb_df.query(f"tag == 'train_losses/{loss}_mean'")
            data_val = tb_df.query(f"tag == 'val_losses/{loss}_mean'")
            llambda = hparams.loc["value", f"losses_lambda_{weight}"]
            print(loss,(data_train["value"]*llambda).to_numpy()[[0, -1]], (data_val["value"]*llambda).to_numpy()[[0, -1]])
            ax.plot(data_train["step"], data_train["value"]*llambda, label="Training")
            ax.plot(data_val["step"], data_val["value"]*llambda, label="Validation")
            ax.set_title(fr"$L_\text{'{'}{subscript}{'}'}\lambda_\text{'{'}{subscript}{'}'}$")
            ax.grid(True)
            if i//3 == 1:
                ax.set_xlabel("Epoch")

        fig.subplots_adjust(hspace=0.5)
        fig.subplots_adjust(wspace=0.3)
        return fig
    """
    return


@app.cell
def _(hparams, tb_df, viz_losses):
    loss_fig = viz_losses(tb_df, hparams)
    loss_fig
    return (loss_fig,)


@app.cell
def _(loss_fig):
    loss_fig.savefig("../typst/images/losses.svg")
    loss_fig.savefig("../typst/images/paper/losses.png", dpi=400)
    return


@app.cell
def _():
    """
    def viz_areas() -> Figure:
        fig, ax = plt.subplots(1, 2)
        roc = RocCurveDisplay.from_predictions(
            true_sep3, pred_sep3_risk, ax=ax[0], curve_kwargs={"color": "tab:orange"}, plot_chance_level=True
        )
        roc.ax_.plot((0, 0), (1,1), label="random", color="red", linestyle=":")
        PrecisionRecallDisplay.from_predictions(true_sep3, pred_sep3_risk, ax=ax[1], color="tab:orange", plot_chance_level=True)

        ax[0].set_title("Receiver Operating Characteristics")
        ax[1].set_title("Precision Recall Curve")
        ax[1].legend(loc="upper right")

        return fig
    """
    return


@app.cell
def _(pred_sep3_risk, true_sep3, viz_curves_sepsis):
    area_fig = viz_curves_sepsis(true_sep3, pred_sep3_risk)
    area_fig
    return (area_fig,)


@app.cell
def _(area_fig):
    area_fig.savefig("../typst/images/areas.svg")
    area_fig.savefig("../typst/images/paper/areas.png", dpi=400)
    return


@app.cell
def _(hparams):
    hparams.T
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
