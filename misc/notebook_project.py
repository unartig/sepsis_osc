import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⚠️ Warning ⚠️
    As the data loading will be performed in parallel, it is advised to run this on the CPU.
    And this uses *a lot* of RAM, be warned! (If you run into memory issues reduce the number of max_workers)
    """)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import logging
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.style
    import numpy as np
    import pandas as pd

    from sepsis_osc.ldm.analysis_helper import (
        Scenario,
        build_scenario,
        compute_alignment,
        compute_recon,
        extract_inputs,
        extract_labels,
        performance_table,
        pre_load_all_data_parallel,
    )
    from sepsis_osc.ldm.commons import get_space_vals
    from sepsis_osc.ldm.data_loading import get_raw_data
    from sepsis_osc.ldm.lookup import LatentLookup
    from sepsis_osc.ldm.model_structs import LossesConfig
    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.utils.config import (
        ALPHA,
        BETA_SPACE,
        CV_FOLDS,
        CV_REPETITIONS,
        SIGMA_SPACE,
        plt_params,
    )
    from sepsis_osc.utils.jax_config import setup_jax
    from sepsis_osc.utils.logger import setup_logging
    from sepsis_osc.visualisations.viz_model_results import (
        viz_concept_densities,
        viz_loss_mean_std,
    )
    from sepsis_osc.visualisations.viz_param_space import space_plot
    from sepsis_osc.visualisations.viz_scenarios import (
        viz_alignment_and_recon,
        viz_alignment_heatmap,
        viz_cohens_d_distribution,
        viz_comparison,
        viz_decoder_perf_heatmap,
        viz_dists_cv,
        viz_latent_scatter,
        viz_latent_scatter_grid,
        viz_per_patient_recon,
        viz_per_recon_densities,
        viz_per_recon_dist,
        viz_s2_latent_stability,
        viz_subgroup_separation,
    )

    setup_jax(simulation=False)
    setup_logging()
    logger = logging.getLogger(__name__)

    matplotlib.style.use("default")
    plt.rcParams.update(plt_params)
    return (
        ALPHA,
        BETA_SPACE,
        CV_FOLDS,
        CV_REPETITIONS,
        LatentLookup,
        LossesConfig,
        Path,
        SIGMA_SPACE,
        Scenario,
        Storage,
        build_scenario,
        compute_alignment,
        compute_recon,
        defaultdict,
        extract_inputs,
        extract_labels,
        get_raw_data,
        get_space_vals,
        np,
        pd,
        performance_table,
        plt,
        pre_load_all_data_parallel,
        space_plot,
        viz_alignment_and_recon,
        viz_alignment_heatmap,
        viz_cohens_d_distribution,
        viz_comparison,
        viz_concept_densities,
        viz_decoder_perf_heatmap,
        viz_dists_cv,
        viz_latent_scatter,
        viz_latent_scatter_grid,
        viz_loss_mean_std,
        viz_per_patient_recon,
        viz_per_recon_densities,
        viz_per_recon_dist,
        viz_s2_latent_stability,
        viz_subgroup_separation,
    )


@app.cell
def _(Path):
    ABLATIONS = [
        "standard",
        "standard_no_sep",
        "standard_no_sofa",
        "standard_no_recon",
        "standard_no_spread",
        "standard_no_boundary",
    ]

    VARIATIONS = [
        "standard",
        "mlp",
        "surrogate",
        "approx",
        "linear_smooth",
        "linear_step",
        "radial_modest",
        "radial_modest_latent_lookup",
    ]

    RUN_BASE        = "runs/cv_clean/miiv"
    DB_NAME         = "miiv"
    TARGET_NAME     = "sep3_alt"
    COHORT_NAME     = f"{TARGET_NAME}_with_marginals_ramp"
    YAIB_DATA_DIR   = f"/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/{COHORT_NAME}/{DB_NAME}"
    SEQUENCE_FILES  = f"data/cv/{DB_NAME}/sequence_"
    SIM_DB_STR      = "DaisyFinal"

    REFERENCE_SCENARIO = "standard"

    LOSS_CONF = dict(
        lambda_sep3=300.0,
        lambda_inf=1.0,
        lambda_sofa_classification=2000.0,
        lambda_spreading=6e-3,
        lambda_boundary=30.0,
        lambda_recon=2.5,
    )

    OUTPUT_DIR = Path("figures")
    return (
        ABLATIONS,
        LOSS_CONF,
        OUTPUT_DIR,
        REFERENCE_SCENARIO,
        SIM_DB_STR,
        VARIATIONS,
        YAIB_DATA_DIR,
    )


@app.cell
def _(
    ALPHA,
    BETA_SPACE,
    LOSS_CONF,
    LatentLookup,
    LossesConfig,
    SIGMA_SPACE,
    SIM_DB_STR,
    Storage,
):
    """Lookup table and loss config. Built once, shared everywhere."""
    _storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{SIM_DB_STR}SepsisMetrics.db/",
        parameter_k_name=f"data/{SIM_DB_STR}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    _storage.close()

    lookup_table = LatentLookup.build(
        _storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE
    )
    loss_conf = LossesConfig(**LOSS_CONF)


@app.cell
def _(Path, YAIB_DATA_DIR, get_raw_data, pd):
    """Feature name → column index mapping (single source of truth)."""
    raw_data = get_raw_data(_data_dir=Path(YAIB_DATA_DIR))
    _col_names = raw_data["test"]["FEATURES"].columns
    features = pd.Series({
        col: i - 2
        for i, col in enumerate(_col_names)
        if col not in ("stay_id", "time")
        and not col.startswith("MissingIndicator_")
    })
    print(f"Features: {len(features)}\n{list(features.index)}")
    return features, raw_data


@app.cell
def _():
    return


@app.cell
def _(CV_FOLDS, CV_REPETITIONS, mo, pre_load_all_data_parallel):
    with mo.status.spinner(title="Pre-loading all data splits in parallel..."):
        shared_data = pre_load_all_data_parallel("data/cv/miiv/sequence_")

    print(f"Loaded {len(shared_data)} / {CV_REPETITIONS * CV_FOLDS} splits")
    return (shared_data,)


@app.cell
def _(CV_FOLDS, CV_REPETITIONS, build_scenario, features, shared_data):
    standard = build_scenario("standard", CV_FOLDS, CV_REPETITIONS, "runs/cv_clean/miiv", shared_data, features)
    return (standard,)


@app.cell
def _(standard):
    standard_repr = standard.representative()
    return (standard_repr,)


@app.cell
def _(
    ABLATIONS,
    CV_FOLDS,
    CV_REPETITIONS,
    VARIATIONS,
    build_scenario,
    features,
    mo,
    shared_data,
    standard,
):
    def build_scenarios(names, repetitions, folds, cache=None):
        cache = cache or {}
        return {
            name: cache[name] if name in cache else build_scenario(name, repetitions, folds, "runs/cv_clean/miiv", shared_data, features)
            for name in mo.status.progress_bar(names, title="Processing scenarios")
        }

    s3_ablations = build_scenarios(ABLATIONS, CV_REPETITIONS, CV_FOLDS, cache={"standard": standard})
    s3_variations = build_scenarios(VARIATIONS, CV_REPETITIONS, CV_FOLDS, cache={"standard": standard})
    return s3_ablations, s3_variations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Stage 0: Single Split · Single Scenario
    Features.
    """)


@app.cell
def _(compute_alignment, features, mo, pd, raw_data, viz_alignment_and_recon):
    _full_features = pd.concat([v["FEATURES"].to_pandas() for v in raw_data.values()])
    _full_outcomes = pd.concat([v["OUTCOME"].to_pandas() for v in raw_data.values()])

    corr = compute_alignment(
        {"Sepsis-3": _full_outcomes["sep3_alt"], "Inflammation": _full_outcomes["susp_inf_alt"], "SOFA": _full_outcomes["sofa"]}, _full_features[features.index]
    )


    mo.mpl.interactive(
        viz_alignment_and_recon(
            corr,
            None,
            title=r"Feature alignment with the labels Sepsis-3, Inflammation and SOFA-score",
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Stage 1: Single Split · Single Scenario
    Deep inspection of the representative run of `standard`.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Predictive performance
    """)


@app.cell
def _(
    BETA_SPACE,
    SIGMA_SPACE,
    get_space_vals,
    mo,
    np,
    space_plot,
    standard_repr,
):
    _betas_space, _sigmas_space, _space_vals = get_space_vals(standard_repr.run.model.lookup)

    mo.mpl.interactive(
        space_plot(
            _space_vals,
            xs=np.arange(*BETA_SPACE),
            ys=np.arange(*SIGMA_SPACE),
            title=f"Latent Space for scenario '{standard_repr.scenario}'",
            cmap=True,
            filename="",
            xticklabel_rot=0,
            num_ticks=3,
        )
    )


@app.cell
def _(mo, pd, standard_repr):
    mo.ui.table(
        pd.DataFrame([standard_repr.perf])
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parameter alignment
    """)


@app.cell
def _(extract_inputs, features, mo, pd, standard_repr):
    test_x, test_y, test_m = standard_repr.run.data
    _in = extract_inputs(test_x, test_m, features)
    _out = pd.DataFrame(test_y[test_m].reshape(-1, 3), columns=["SOFA", "Inflammation", "Sepsis-3"])
    in_n_out = pd.concat([_in, _out], axis=1)

    s1_feature_dd = mo.ui.dropdown(options=list(in_n_out.columns), value=in_n_out.columns[0], label="Colour feature")
    s1_feature_dd
    return in_n_out, s1_feature_dd, test_m, test_x, test_y


@app.cell
def _(in_n_out, mo, s1_feature_dd, standard_repr, viz_latent_scatter):
    mo.mpl.interactive(
        viz_latent_scatter(
            standard_repr.run.beta,
            standard_repr.run.sigma,
            standard_repr.run.model,
            in_n_out[s1_feature_dd.value].values,
            label=s1_feature_dd.value,
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Subgroup separation
    """)


@app.cell
def _(
    extract_labels,
    mo,
    standard_repr,
    test_m,
    test_y,
    viz_subgroup_separation,
):
    sep, inf, sofa, peak_sofa = extract_labels(test_y, test_m)
    mo.mpl.interactive(
        viz_subgroup_separation(
            standard_repr.run.beta,
            standard_repr.run.sigma,
            standard_repr.run.model,
            sep,
            inf,
            peak_sofa,
        )
    )


@app.cell
def _(mo, standard_repr):
    mo.ui.table(standard_repr.subgroup_stats)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decoder reconstruction
    """)


@app.cell
def _(mo, test_x):
    s1_pat_dd = mo.ui.dropdown(range(test_x.shape[1]), value=0, label="Patient")
    s1_sort_dd = mo.ui.dropdown(["name", "mse", "var"], value="name", label="Sort by")
    mo.hstack([s1_pat_dd, s1_sort_dd])


@app.cell
def _(REFERENCE_SCENARIO, mo, standard_repr, viz_alignment_and_recon):
    # TODO viz statistic of missing
    # TODO no inflammation coloring
    mo.mpl.interactive(
        viz_alignment_and_recon(
            standard_repr.correlation,
            standard_repr.recon_pr,
            title=rf"Feature alignment with $\beta$, $\sigma$ for scenario '{REFERENCE_SCENARIO}'",
        )
    )


@app.cell
def _(
    compute_recon,
    features,
    mo,
    standard_repr,
    test_m,
    test_x,
    viz_per_patient_recon,
):
    # TODO unnormalized?
    sort_key = standard_repr.recon_pr.argsort(axis=-1)[..., -10:][::-1]
    best_recon_feats = [features[features == i].index[0] for i in sort_key]
    recon = compute_recon(standard_repr.run.model, standard_repr.run.metrics.beta, standard_repr.run.metrics.sigma)

    mo.mpl.interactive(
        viz_per_patient_recon(
            recon[0][..., sort_key],
            test_x[0][..., sort_key],
            test_m[0],
            feat_names=best_recon_feats,
            feat_order=sort_key,
            pat_idx=123,
        )
    )
    return best_recon_feats, recon, sort_key


@app.cell
def _(
    best_recon_feats,
    mo,
    recon,
    sort_key,
    test_m,
    test_x,
    viz_per_recon_dist,
):
    mo.mpl.interactive(
        viz_per_recon_dist(
            recon[0][..., sort_key],
            test_x[0][..., sort_key],
            test_m[0],
            feat_names=best_recon_feats
        )
    )


@app.cell
def _(
    best_recon_feats,
    mo,
    recon,
    sort_key,
    test_m,
    test_x,
    viz_per_recon_densities,
):
    mo.mpl.interactive(
        viz_per_recon_densities(
            recon[0][..., sort_key],
            test_x[0][..., sort_key],
            test_m[0],
            feat_names=best_recon_feats
        )
    )


@app.cell
def _(best_recon_feats, in_n_out, mo, standard_repr, viz_latent_scatter_grid):

    feature_vals = {feature: in_n_out[feature].values for feature in best_recon_feats[:6]}
    mo.mpl.interactive(
        viz_latent_scatter_grid(
            standard_repr.run.beta,
            standard_repr.run.sigma,
            standard_repr.run.model,
            feature_vals
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paper Figures
    """)


@app.cell
def _(mo, np, standard_repr, test_m, test_y, viz_concept_densities):
    heat_fig, heat_ax = viz_concept_densities(
        test_y[test_m, 0],
        np.array([0]),
        standard_repr.run.metrics.hists_sofa_score[0][test_m[0]],
        np.array([0]),
        cmap=True,
    )
    heat_ax[0].set_title("SOFA-score")
    heat_ax[1].set_title(r"$\Delta$SOFA-score")
    mo.mpl.interactive(heat_fig)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    # Stage 2: All Splits · Single Scenario
    Cross-split stability of the reference scenario. Loads one split at a time.
    """)


@app.cell
def _(standard):
    standard.perf_df


@app.cell
def _(defaultdict, mo, np, pd, standard, viz_loss_mean_std):
    losses = ["total_loss", "sepsis-3", "sofa", "infection", "recon_loss", "spreading_loss", "boundary_loss"]
    eval_metrics = ["AUROC_pred_sep", "AUPRC_pred_sep"]
    eval_tags = [f"sepsis_metrics/{m}" for m in eval_metrics]
    for loss in losses:
        eval_tags.extend([f"train_losses/{loss}_mean", f"val_losses/{loss}_mean"])

    def load_tb_scalars(dfs, tags):
        all_data = defaultdict(list)
        for _tb_df in dfs:
            for tag in tags:
                df_tag = _tb_df.query(f"tag == '{tag}'")[["step", "value"]].copy()
                all_data[tag].append(df_tag)
        return all_data

    def aggregate_runs(all_data):
        agg_data = {}
        for tag, dfs in all_data.items():
            all_steps = sorted(set().union(*(df["step"].to_numpy() for df in dfs)))
            mean_vals, std_vals = [], []
            for step in all_steps:
                vals = [df.loc[df["step"] == step, "value"].values[0] for df in dfs if step in df["step"].values]
                mean_vals.append(np.mean(vals))
                std_vals.append(np.std(vals))
            agg_data[tag] = pd.DataFrame({"step": all_steps, "mean": mean_vals, "std": std_vals})
        return agg_data

    # usage
    tb_dfs = [r.run.tb_df for r in standard.results]
    hparams_list = [r.run.hparams for r in standard.results]

    all_data = load_tb_scalars(tb_dfs, eval_tags)
    agg_data = aggregate_runs(all_data)

    loss_names = ("sepsis-3", "sofa", "infection", "recon_loss", "spreading_loss", "boundary_loss")
    loss_subscripts = ("sepsis", "sofa", "inf", "dec", "spread", "boundary")
    lambdas = ("sep3", "sofa_classification", "inf", "recon", "spreading", "boundary")

    loss_fig = viz_loss_mean_std(agg_data, loss_names, loss_subscripts, lambdas, hparams_list[0])

    mo.mpl.interactive(loss_fig)
    return aggregate_runs, eval_tags, load_tb_scalars


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-split subgroup stability
    """)


@app.cell
def _(mo, standard, viz_cohens_d_distribution):
    mo.mpl.interactive(
        viz_cohens_d_distribution(
            [r.subgroup_stats for r in standard.results],
            title="Cohen's d across splits for scenario 'standard'",
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-split latent distribution stability
    """)


@app.cell
def _(mo, standard, viz_s2_latent_stability):
    mo.mpl.interactive(viz_s2_latent_stability(standard.results))


@app.cell
def _(mo, standard, viz_dists_cv):
    mo.mpl.interactive(viz_dists_cv(standard.perf_df, standard.results))


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-split alignment stability
    """)


@app.cell
def _(mo, standard, viz_alignment_heatmap):
    entries = [(str(i), r.correlation) for i, r in enumerate(standard.results)]
    pr = {str(i): r.recon_pr for i, r in enumerate(standard.results)}
    mo.mpl.interactive(
        viz_alignment_heatmap(entries, perf_metric=pr, title=r"Feature alignment stability across splits ($\beta$, $\sigma$, Pearson $r$) for 'standard'")
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Stage 3: All Splits · All Scenarios
    Comparative summaries. Run for both `ABLATIONS` and `VARIATIONS`.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Performance comparison
    """)


@app.cell
def _(
    Scenario,
    mo,
    performance_table,
    s3_ablations,
    s3_variations,
    viz_comparison,
):
    def _summarise(scenarios: dict[str, Scenario]):
        metrics = ["auroc_sep3", "auprc_sep3", "auroc_sofa_d2", "auprc_sofa_d2", "auroc_inf", "auprc_inf"]
        out = {}
        for name, sc in scenarios.items():
            df = sc.perf_df
            out[name] = {m: (df[m].mean(), df[m].std()) for m in metrics if m in df}
        return out

    abl_perf_tbl = performance_table(_summarise(s3_ablations))
    var_perf_tbl = performance_table(_summarise(s3_variations))


    mo.vstack([
        mo.md("### Ablations"),
        mo.mpl.interactive(
            viz_comparison(s3_ablations, title="Ablation experiment comparison")
        ),
        mo.ui.table(abl_perf_tbl.reset_index()),
        mo.md("### Variations"),
        mo.mpl.interactive(
            viz_comparison(s3_variations, title="Variation experiment comparison")
        ),
        mo.ui.table(var_perf_tbl.reset_index()),
    ])
    return abl_perf_tbl, var_perf_tbl


@app.cell
def _(mo):
    mo.md("""
    ## Feature alignment heatmaps
    """)


@app.cell
def _(Scenario, mo, s3_ablations, s3_variations, viz_alignment_heatmap):
    def _scenario_corrs(scenarios: dict[str, Scenario]):
        return {name: sc.corr_mean for name, sc in scenarios.items() if sc.corr_mean is not None}


    entries_ablation = [(name, sc.corr_mean) for name, sc in s3_ablations.items() if sc.corr_mean is not None]
    entries_variations = [(name, sc.corr_mean) for name, sc in s3_variations.items() if sc.corr_mean is not None]

    mo.vstack(
        [
            mo.md("### Ablations"),
            mo.mpl.interactive(
                viz_alignment_heatmap(entries_ablation, title=r"Feature alignment r_$\beta$ and $r_$\sigma$: Ablations", row_label_every=1)
            ),
            mo.md("### Variations"),
            mo.mpl.interactive(
                viz_alignment_heatmap(entries_variations, title=r"Feature alignment r_$\beta$ and $r_$\sigma$: Ablations", row_label_every=1)
            ),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decoder MSE heatmaps
    """)


@app.cell
def _(features, mo, s3_ablations, s3_variations, viz_decoder_perf_heatmap):
    mo.vstack(
        [
            mo.mpl.interactive(viz_decoder_perf_heatmap(s3_ablations, features.index, title=r"Decoder Pearson $r$: Ablations")),
            mo.mpl.interactive(viz_decoder_perf_heatmap(s3_variations, features.index, title="Decoder Pearson $r$: Variations")),
        ]
    )


@app.cell
def _(pd):
    name_mapping = {
        "sep3": "Sepsis-3",
        "sofa_d2": 'Delta #acr("SOFA") $>=$ 2',
        "inf": "Infection",
        "sep3_sofa_only": "Sepsis-3 using Organ branch only",
        "sep3_inf_only": "Sepsis-3 using Infection branch only",
    }

    def build_pivoted_perf(df):
        drop_cols = ["rep", "fold", "rmse_sofa", "auroc_sep3_sofa_only_gt", "auprc_sep3_sofa_only_gt", "auroc_sep3_inf_only_gt", "auprc_sep3_inf_only_gt"]
        valid_drop_cols = [c for c in drop_cols if c in df.columns]
        df_metrics = df.drop(columns=valid_drop_cols)

        means = df_metrics.mean()
        stds = df_metrics.std()

        records = []
        for col in df_metrics.columns:
            parts = col.split("_")
            metric = parts[0].upper()
            setting = "_".join(parts[1:])

            m_val = means[col] * 100
            s_val = stds[col] * 100

            combined_str = f"${m_val:.2f} plus.minus {s_val:.2f}$" if pd.notna(s_val) else f"${m_val:.2f}$"

            records.append({
                "Setting": setting,
                "Metric": metric,
                "Value": combined_str
            })

        pivot_df = pd.DataFrame(records).pivot(index="Setting", columns="Metric", values="Value")

        pivot_df.index = pivot_df.index.map(name_mapping)
        pivot_df = pivot_df.reindex(index=list(name_mapping.values()), columns=["AUROC", "AUPRC"])
        pivot_df.index.name = None

        return pivot_df

    def build_pivoted_cohens(df):
        if "mean" in df.columns:
            col = "mean_std"
            df[col] = df.apply(lambda r: f"${r['mean']:.2f} plus.minus {r['std']:.2f}$", axis=1)
        else:
            col = "cohens d"
            df[col] = df.apply(lambda r: f"${r['cohens_d']:.2f}$", axis=1)
        pivot_df = df.pivot(index="comparison", columns="coord", values=col)
        pivot_df.index.name = "Comparison"
        pivot_df.columns.name = None
        pivot_df.columns = pivot_df.columns.str.replace("\\","")
        pivot_df.index.name = None
        return pivot_df

    return build_pivoted_cohens, build_pivoted_perf


@app.cell
def _(
    BETA_SPACE,
    OUTPUT_DIR,
    SIGMA_SPACE,
    Scenario,
    aggregate_runs,
    build_pivoted_cohens,
    build_pivoted_perf,
    compute_recon,
    eval_tags,
    extract_inputs,
    extract_labels,
    features,
    get_space_vals,
    load_tb_scalars,
    np,
    pd,
    plt,
    s3_ablations,
    s3_variations,
    space_plot,
    standard_repr,
    viz_alignment_and_recon,
    viz_alignment_heatmap,
    viz_cohens_d_distribution,
    viz_concept_densities,
    viz_dists_cv,
    viz_latent_scatter_grid,
    viz_loss_mean_std,
    viz_per_patient_recon,
    viz_per_recon_densities,
    viz_per_recon_dist,
    viz_s2_latent_stability,
    viz_subgroup_separation,
):
    def save_scenario_figures(scenario: Scenario, experiment: str):
        """Generate and save the standard figure set for one scenario."""

        out_dir = OUTPUT_DIR / experiment / scenario.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # representative split for per-split deep-dive plots
        rep_result = scenario.representative()
        test_x, test_y, test_m = rep_result.run.data
        _in = extract_inputs(test_x, test_m, features)
        _out = pd.DataFrame(test_y[test_m].reshape(-1, 3), columns=["SOFA", "Inflammation", "Sepsis-3"])
        in_n_out = pd.concat([_in, _out], axis=1)
        sep, inf_, _, peak_sofa = extract_labels(test_y, test_m)

        print("#"*10, scenario.name, "#"*10)
        print("Perf Repr")
        perf_df = build_pivoted_perf(pd.DataFrame([rep_result.perf]))
        print(perf_df)
        print("Subgroup Stats Repr")
        single_cohens_pivot_df = build_pivoted_cohens(rep_result.subgroup_stats[["comparison", "cohens_d", "coord"]])
        print(single_cohens_pivot_df)
        print("Perf CV")
        pivot_df = build_pivoted_perf(scenario.perf_df)
        print(pivot_df.style.to_typst())
        print("Cohens CV")
        cohens_pivot_df = build_pivoted_cohens(scenario.subgroup_stats_mean)
        print(cohens_pivot_df.style.to_typst())

        figs = {}

        _betas_space, _sigmas_space, _space_vals = get_space_vals(rep_result.run.model.lookup)
        fig_space, ax = plt.subplots()
        _ = space_plot(
            _space_vals,
            xs=np.arange(*BETA_SPACE),
            ys=np.arange(*SIGMA_SPACE),
            title=f"Latent Space of scenario '{scenario.name}'",
            cmap=True,
            filename="",
            xticklabel_rot=0,
            num_ticks=3,
            figax=(fig_space, ax)
        )
        figs["latent_space"] = fig_space

        figs["alignment_recon"] = viz_alignment_and_recon(
            rep_result.correlation,
            rep_result.recon_pr,
            title=rf"Feature alignment with $\beta$, $\sigma$ of scenario '{scenario.name}'",
        )

        x_df = extract_inputs(test_x, test_m, features)

        figs["subgroup_separation"] = viz_subgroup_separation(
            rep_result.run.beta,
            rep_result.run.sigma,
            rep_result.run.model,
            sep,
            inf_,
            peak_sofa,
        )

        figs["cohens_d_splits"] = viz_cohens_d_distribution(
            [r.subgroup_stats for r in scenario.results],
            title=f"Cohen's d across splits of scenario '{scenario.name}'",
        )

        entries = [(str(i), r.correlation) for i, r in enumerate(scenario.results)]
        pr = {str(i): r.recon_pr for i, r in enumerate(scenario.results)}
        figs["alignment_heatmap_splits"] = viz_alignment_heatmap(
            entries,
            perf_metric=pr,
            title=rf"Feature alignment stability ($\beta$, $\sigma$, Pearson $r$) of scenario '{scenario.name}'",
        )

        sort_key = rep_result.recon_pr.argsort(axis=-1)[..., -10:][::-1]
        best_recon_feats = [features[features == i].index[0] for i in sort_key]
        recon = compute_recon(rep_result.run.model, rep_result.run.metrics.beta, rep_result.run.metrics.sigma)
        figs["per_patient_recon"] = viz_per_patient_recon(
            recon[0][..., sort_key],
            test_x[0][..., sort_key],
            test_m[0],
            feat_names=best_recon_feats,
            feat_order=sort_key,
            pat_idx=123,
        )
        figs["recon_dist"] = viz_per_recon_dist(
            recon[0][..., sort_key],
            test_x[0][..., sort_key],
            test_m[0],
            feat_names=best_recon_feats,
        )

        figs["recon_scatter"] = viz_per_recon_densities(
            recon[0][..., sort_key], test_x[0][..., sort_key], test_m[0], feat_names=best_recon_feats
        )

        feature_vals = {feature: in_n_out[feature].values for feature in best_recon_feats[:6]}
        figs["latent_scatter"] = viz_latent_scatter_grid(rep_result.run.beta, rep_result.run.sigma, rep_result.run.model, feature_vals)

        heat_fig, heat_ax = viz_concept_densities(
            test_y[test_m, 0],
            np.array([0]),
            standard_repr.run.metrics.hists_sofa_score[0][test_m[0]],
            np.array([0]),
            cmap=True,
        )
        heat_ax[0].set_title("SOFA-score")
        heat_ax[1].set_title(r"$\Delta$SOFA-score")
        figs["heat"] = heat_fig

        figs["latent_stability"] = viz_s2_latent_stability(scenario.results)

        figs["dists_cv"] = viz_dists_cv(scenario.perf_df, scenario.results)


        tb_dfs = [r.run.tb_df for r in scenario.results]
        hparams_list = [r.run.hparams for r in scenario.results]

        all_data = load_tb_scalars(tb_dfs, eval_tags)
        agg_data = aggregate_runs(all_data)

        loss_names = ("sepsis-3", "sofa", "infection", "recon_loss", "spreading_loss", "boundary_loss")
        loss_subscripts = ("sepsis", "sofa", "inf", "dec", "spread", "boundary")
        lambdas = ("sep3", "sofa_classification", "inf", "recon", "spreading", "boundary")

        figs["losses"] = viz_loss_mean_std(agg_data, loss_names, loss_subscripts, lambdas, hparams_list[0])

        for fig_name, fig in figs.items():
            print(fig_name)
            tight = "tight" if  fig_name != "alignment_heatmap_splits" else None
            fig.savefig(out_dir / f"png_{fig_name}.png", dpi=150, bbox_inches=tight)
            fig.savefig(out_dir / f"svg_{fig_name}.svg", bbox_inches=tight)
            plt.close(fig)

        rep_result.subgroup_stats.to_csv(out_dir / "subgroup_stats.csv")
        pd.DataFrame([rep_result.perf]).to_csv(out_dir / "representative_perf.csv")
        scenario.perf_df.to_csv(out_dir / "cv_perf.csv")

        return figs


    for _name, _sc in s3_ablations.items():
        save_scenario_figures(_sc, "ablations")

    for _name, _sc in s3_variations.items():
        save_scenario_figures(_sc, "variations")

    print(f"Saved figures to {OUTPUT_DIR.resolve()}")


@app.cell
def _(standard):
    for i, s in enumerate(standard.results):
        print(i, s.run.best_epoch, s.run.rep, s.run.fold)


@app.cell
def _(
    OUTPUT_DIR,
    compute_alignment,
    features,
    pd,
    raw_data,
    viz_alignment_and_recon,
):
    _full_features = pd.concat([v["FEATURES"].to_pandas() for v in raw_data.values()])
    _full_outcomes = pd.concat([v["OUTCOME"].to_pandas() for v in raw_data.values()])

    corr = compute_alignment(
        {"Sepsis-3": _full_outcomes["sep3_alt"], "Inflammation": _full_outcomes["susp_inf_alt"], "SOFA": _full_outcomes["sofa"]}, _full_features[features.index]
    )

    _fig_alignment = viz_alignment_and_recon(
        corr,
        None,
        title=r"Feature alignment with the labels Sepsis-3, Inflammation and SOFA-score",
    )
    _fig_alignment.savefig(OUTPUT_DIR / "label_alignment.png")
    _fig_alignment.savefig(OUTPUT_DIR / "label_alignment.svg")


@app.cell
def _(
    OUTPUT_DIR,
    Scenario,
    abl_perf_tbl,
    features,
    performance_table,
    s3_ablations,
    s3_variations,
    var_perf_tbl,
    viz_alignment_heatmap,
    viz_comparison,
    viz_decoder_perf_heatmap,
):
    def _scenario_corrs(scenarios: dict[str, Scenario]):
        return {name: sc.corr_mean for name, sc in scenarios.items() if sc.corr_mean is not None}


    _entries_ablation = [(name, sc.corr_mean) for name, sc in s3_ablations.items() if sc.corr_mean is not None]
    _entries_variations = [(name, sc.corr_mean) for name, sc in s3_variations.items() if sc.corr_mean is not None]

    def _summarise(scenarios: dict[str, Scenario]):
        metrics = ["auroc_sep3", "auprc_sep3", "auroc_sofa_d2", "auprc_sofa_d2", "auroc_inf", "auprc_inf"]
        out = {}
        for name, sc in scenarios.items():
            df = sc.perf_df
            out[name] = {m: (df[m].mean(), df[m].std()) for m in metrics if m in df}
        return out

    _abl_perf_tbl = performance_table(_summarise(s3_ablations), title="Performance: Ablations")
    _var_perf_tbl = performance_table(_summarise(s3_variations), title="Performance: Variations")

    _abl_heat_fig = viz_alignment_heatmap(_entries_ablation, title=r"Feature alignment $r_\beta$ and $r_\sigma$: Ablations", row_label_every=1)
    _abl_heat_fig.savefig(OUTPUT_DIR / "ablation_alignment_summary.png")
    _abl_heat_fig.savefig(OUTPUT_DIR / "ablation_alignment_summary.svg")
    _var_heat_fig = viz_alignment_heatmap(_entries_variations, title=r"Feature alignment $r_\beta$ and $r_\sigma$: Ablations", row_label_every=1)
    _var_heat_fig.savefig(OUTPUT_DIR / "variation_alignment_summary.png")
    _var_heat_fig.savefig(OUTPUT_DIR / "variation_alignment_summary.svg")

    _abl_dec_fig = viz_decoder_perf_heatmap(s3_ablations, features.index, title=r"Decoder Pearson $r$: Ablations")
    _abl_dec_fig.savefig(OUTPUT_DIR / "ablation_decoder_summary.png")
    _abl_dec_fig.savefig(OUTPUT_DIR / "ablation_decoder_summary.svg")
    _var_dec_fig = viz_decoder_perf_heatmap(s3_variations, features.index, title=r"Decoder Pearson $r$: Variations")
    _var_dec_fig.savefig(OUTPUT_DIR / "variation_decoder_summary.png")
    _var_dec_fig.savefig(OUTPUT_DIR / "variation_decoder_summary.svg")



    _abl_comp_fig = viz_comparison(s3_ablations, title="Ablation experiment comparison")
    _abl_comp_fig.savefig(OUTPUT_DIR / "ablation_comparison.png")
    _abl_comp_fig.savefig(OUTPUT_DIR / "ablation_comparison.svg")
    print(abl_perf_tbl.reset_index().style.to_typst())
    _var_comp_fig = viz_comparison(s3_variations, title="Variation experiment comparison")
    _var_comp_fig.savefig(OUTPUT_DIR / "variation_comparison.png")
    _var_comp_fig.savefig(OUTPUT_DIR / "variation_comparison.svg")
    print(var_perf_tbl.reset_index().style.to_typst())


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
