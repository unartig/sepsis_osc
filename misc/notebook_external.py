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
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.style
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    from sepsis_osc.ldm.analysis_helper import (
        Scenario,
        build_scenario,
        get_data_and_features,
        performance_table,
        pre_load_all_data_parallel,
    )
    from sepsis_osc.ldm.lookup import LatentLookup
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
    from sepsis_osc.visualisations.viz_scenarios import (
        viz_alignment_and_recon,
        viz_alignment_heatmap,
        viz_comparison,
        viz_decoder_perf_heatmap,
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
        Path,
        SIGMA_SPACE,
        Scenario,
        Storage,
        build_scenario,
        get_data_and_features,
        multipletests,
        np,
        pd,
        performance_table,
        pre_load_all_data_parallel,
        stats,
        viz_alignment_and_recon,
        viz_alignment_heatmap,
        viz_comparison,
        viz_decoder_perf_heatmap,
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

    TARGET_NAME     = "sep3_alt"
    COHORT_NAME     = f"{TARGET_NAME}_with_marginals_ramp"
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
    return COHORT_NAME, OUTPUT_DIR, SIM_DB_STR


@app.cell
def _(ALPHA, BETA_SPACE, LatentLookup, SIGMA_SPACE, SIM_DB_STR, Storage):
    """Lookup table and loss config — built once, shared everywhere."""
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


@app.cell
def _(COHORT_NAME, get_data_and_features):
    miiv_data, miiv_features = get_data_and_features(f"/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/{COHORT_NAME}/miiv")
    eicu_data, eicu_features = get_data_and_features(f"/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/{COHORT_NAME}/eicu")
    return eicu_features, miiv_features


@app.cell
def _(eicu_features, miiv_features):
    features = miiv_features | eicu_features
    return (features,)


@app.cell
def _(CV_FOLDS, CV_REPETITIONS, mo, pre_load_all_data_parallel):
    with mo.status.spinner(title="Pre-loading all data splits in parallel..."):
        shared_eicu = pre_load_all_data_parallel("data/cv/eicu/sequence_")
        print("NEXT")
        shared_miiv = pre_load_all_data_parallel("data/cv/miiv/sequence_")

    print(f"Loaded MIIV {len(shared_miiv)} / {CV_REPETITIONS * CV_FOLDS} splits")
    print(f"Loaded eICU {len(shared_eicu)} / {CV_REPETITIONS * CV_FOLDS} splits")
    return shared_eicu, shared_miiv


@app.cell
def _(
    CV_FOLDS,
    CV_REPETITIONS,
    build_scenario,
    features,
    shared_eicu,
    shared_miiv,
):
    miiv_scenario = build_scenario("standard", CV_FOLDS, CV_REPETITIONS, "runs/cv_clean/miiv", shared_miiv, features)
    eicu_scenario = build_scenario("standard", CV_FOLDS, CV_REPETITIONS, "runs/cv_clean/eicu", shared_eicu, features)

    miiv_trained_eicu_tested = build_scenario("standard", CV_FOLDS, CV_REPETITIONS, "runs/cv_clean/miiv", shared_eicu, features)
    eicu_trained_miiv_tested = build_scenario("standard", CV_FOLDS, CV_REPETITIONS, "runs/cv_clean/eicu", shared_miiv, features)
    return (
        eicu_scenario,
        eicu_trained_miiv_tested,
        miiv_scenario,
        miiv_trained_eicu_tested,
    )


@app.cell
def _(eicu_scenario):
    standard_repr = eicu_scenario.representative()


@app.cell
def _(miiv_scenario):
    miiv_scenario.perf_df.aggregate(["mean", "std"]) * 100


@app.cell
def _(eicu_scenario):
    eicu_scenario.perf_df.aggregate(["mean", "std"]) * 100


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Performance comparison
    """)


@app.cell
def _(
    Scenario,
    eicu_scenario,
    eicu_trained_miiv_tested,
    miiv_scenario,
    miiv_trained_eicu_tested,
    mo,
    performance_table,
    viz_comparison,
):

    def _summarise(scenarios: dict[str, Scenario]):
        metrics = ["auroc_sep3", "auprc_sep3", "auroc_sofa_d2", "auprc_sofa_d2", "auroc_inf", "auprc_inf"]
        out = {}
        for name, sc in scenarios.items():
            df = sc.perf_df
            out[name] = {m: (df[m].mean(), df[m].std()) for m in metrics if m in df}
        return out
    external_vals = {"MIIV": miiv_scenario, "eICU": eicu_scenario, "MIIV-eICU": miiv_trained_eicu_tested, "eICU-MIIV": eicu_trained_miiv_tested}
    ext_val_perf_tabl = performance_table(_summarise(external_vals))


    mo.vstack([
        mo.mpl.interactive(
            viz_comparison(external_vals, title="External validation comparison")
        ),
        mo.ui.table(ext_val_perf_tabl.reset_index())
    ])
    return (external_vals,)


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
def _(Scenario, eicu_scenario, miiv_scenario, multipletests, pd, welch_t):

    def calc_tests(scenario: Scenario, baselines):
        ldm_auroc_mean = scenario.perf_df["auroc_sep3"].mean()
        ldm_auroc_std = scenario.perf_df["auroc_sep3"].std()
        ldm_auprc_mean = scenario.perf_df["auprc_sep3"].mean()
        ldm_auprc_std = scenario.perf_df["auprc_sep3"].std()

        ldm_auroc = (ldm_auroc_mean * 100, ldm_auroc_std * 100)
        ldm_auprc = (ldm_auprc_mean * 100, ldm_auprc_std * 100)
        n_ldm, n_base = len(scenario.perf_df), 25

        results = []
        rows =[]
        for name, (b_au_m, b_au_s, b_pr_m, b_pr_s) in baselines.items():
            # AUROC Stats
            t_au, df_au, p_au, _ = welch_t(ldm_auroc[0], ldm_auroc[1], n_ldm, b_au_m, b_au_s, n_base)
            p_au_str = f"${p_au:.3f}$" if p_au >= 0.001 else "$<0.001$"

            # AUPRC Stats
            t_pr, df_pr, p_pr, _ = welch_t(ldm_auprc[0], ldm_auprc[1], n_ldm, b_pr_m, b_pr_s, n_base)
            p_pr_str = f"${p_pr:.3f}$" if p_pr >= 0.001 else "$<0.001$"

            rows.append({"Model": name, "*AUROC* $plus.minus$": f"${b_au_m} plus.minus {b_au_s}$", '$t_"AUROC"$': f"{t_au:6.4f}", '$p_"AUROC"$': f"{p_au_str}",
                                       "*AUPRC* $plus.minus$": f"${b_pr_m} plus.minus {b_pr_s}$", '$t_"AUPRC"$': f"{t_pr:6.4f}", '$p_"AUPRC"$': f"{p_pr_str}"})
            results.append((name, t_au, df_au, p_au, t_pr, df_pr, p_pr))

        rows.append({"Model": "LDM", "*AUROC* $plus.minus$": f"${ldm_auroc_mean * 100:6.3f} plus.minus {ldm_auroc_std * 100:6.3f}$", '$t_"AUROC"$': "--", '$p_"AUROC"$': "--",
                                     "*AUPRC* $plus.minus$": f"${ldm_auprc_mean * 100:6.3f} plus.minus {ldm_auprc_std * 100:6.3f}$", '$t_"AUPRC"$': "--", '$p_"AUPRC"$': "--"})

        _summary = pd.DataFrame(rows)
        _summary.index = _summary["Model"]
        _summary = _summary.drop(columns=["Model"])
        print(_summary.style.to_typst())
        # Correct all 12 p-values together
        all_pvals = [r[3] for r in results] + [r[6] for r in results]  # 6 AUROC + 6 AUPRC
        _, pvals_corrected, _, _ = multipletests(all_pvals, method="holm")
        p_au_corrected = pvals_corrected[:6]
        p_pr_corrected = pvals_corrected[6:]

        print("\n\nCorrected")

        # Print with corrected p-values
        rows =[]
        for _i, ((_name, _t_au, _df_au, _, _t_pr, _df_pr, _), (name, (b_au_m, b_au_s, b_pr_m, b_pr_s))) in enumerate(zip(results, baselines.items())):
            _p_au_str = f"${p_au_corrected[_i]:.3f}$" if p_au_corrected[_i] >= 0.001 else "$<0.001$"
            _p_pr_str = f"${p_pr_corrected[_i]:.3f}$" if p_pr_corrected[_i] >= 0.001 else "$<0.001$"
            rows.append({"Model": name, "*AUROC* $plus.minus$": f"${b_au_m} plus.minus {b_au_s}$", '$t_"AUROC"$': f"{_t_au:6.4f}", '$p_"AUROC"$': f"{_p_au_str}",
                                       "*AUPRC* $plus.minus$": f"${b_pr_m} plus.minus {b_pr_s}$", '$t_"AUPRC"$': f"{_t_pr:6.4f}", '$p_"AUPRC"$': f"{_p_pr_str}"})
        rows.append({"Model": "LDM", "*AUROC* $plus.minus$": f"${ldm_auroc_mean * 100:6.3f} plus.minus {ldm_auroc_std * 100:6.3f}$", '$t_"AUROC"$': "--", '$p_"AUROC"$': "--",
                                     "*AUPRC* $plus.minus$": f"${ldm_auprc_mean * 100:6.3f} plus.minus {ldm_auprc_std * 100:6.3f}$", '$t_"AUPRC"$': "--", '$p_"AUPRC"$': "--"})
        _summary = pd.DataFrame(rows)
        _summary.index = _summary["Model"]
        _summary = _summary.drop(columns=["Model"])
        print(_summary.style.to_typst())


    miiv_baselines = {
        "Reg. Logistic Regression": (77.1, 0.4, 4.6, 0.1),
        "LightGBM":                 (77.5, 0.3, 5.9, 0.2),
        "Transformer":              (80.0, 0.8, 6.6, 0.2),
        "LSTM":                     (82.0, 0.3, 8.0, 0.2),
        "TCN":                      (82.7, 0.3, 8.8, 0.2),
        "GRU":                      (83.6, 0.3, 9.1, 0.3),
    }

    calc_tests(miiv_scenario, miiv_baselines)

    eicu_baselines = {
        "Reg. Logistic Regression": (71.8, 0.3, 2.9, 0.1),
        "LightGBM":                 (69.1, 0.3, 3.3, 0.1),
        "Transformer":              (77.4, 0.2, 5.1, 0.1),
        "LSTM":                     (74.0, 0.2, 4.0, 0.1),
        "TCN":                      (76.7, 0.1, 4.9, 0.1),
        "GRU":                      (76.2, 0.1, 4.6, 0.1),
    }

    calc_tests(eicu_scenario, eicu_baselines)


@app.cell
def _(mo):
    mo.md("""
    ## Feature alignment heatmaps
    """)


@app.cell
def _(Scenario, external_vals, mo, viz_alignment_heatmap):
    def _scenario_corrs(scenarios: dict[str, Scenario]):
        return {name: sc.corr_mean for name, sc in scenarios.items() if sc.corr_mean is not None}


    entries_vals = [(name, sc.corr_mean) for name, sc in external_vals.items() if sc.corr_mean is not None]


    mo.mpl.interactive(
        viz_alignment_heatmap(entries_vals, title=r"Feature alignment r_$\beta$ and $r_$\sigma$: Ablations", row_label_every=1)
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decoder MSE heatmaps
    """)


@app.cell
def _(external_vals, features, mo, viz_decoder_perf_heatmap):
    mo.vstack(
        [
            mo.mpl.interactive(viz_decoder_perf_heatmap(external_vals, features.index, title=r"Decoder Pearson $r$: External Validations")),
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
    external_vals,
    features,
    performance_table,
    viz_alignment_heatmap,
    viz_comparison,
    viz_decoder_perf_heatmap,
):
    def _scenario_corrs(scenarios: dict[str, Scenario]):
        return {name: sc.corr_mean for name, sc in scenarios.items() if sc.corr_mean is not None}


    _entries_vals = [(name, sc.corr_mean) for name, sc in external_vals.items() if sc.corr_mean is not None]

    def _summarise(scenarios: dict[str, Scenario]):
        metrics = ["auroc_sep3", "auprc_sep3", "auroc_sofa_d2", "auprc_sofa_d2", "auroc_inf", "auprc_inf"]
        out = {}
        for name, sc in scenarios.items():
            df = sc.perf_df
            out[name] = {m: (df[m].mean(), df[m].std()) for m in metrics if m in df}
        return out

    _abl_perf_tbl = performance_table(_summarise(external_vals), title="Performance: External Validation")

    _abl_heat_fig = viz_alignment_heatmap(_entries_vals, title=r"Feature alignment $r_\beta$ and $r_\sigma$: External Validation", row_label_every=1)
    _abl_heat_fig.savefig(OUTPUT_DIR / "external_alignment_summary.png")
    _abl_heat_fig.savefig(OUTPUT_DIR / "external_alignment_summary.svg")

    _abl_dec_fig = viz_decoder_perf_heatmap(external_vals, features.index, title=r"Decoder Pearson $r$: External Validation")
    _abl_dec_fig.savefig(OUTPUT_DIR / "external_decoder_summary.png")
    _abl_dec_fig.savefig(OUTPUT_DIR / "external_decoder_summary.svg")


    _abl_comp_fig = viz_comparison(external_vals, title="External Validation comparison")
    _abl_comp_fig.savefig(OUTPUT_DIR / "external_comparison.png")
    _abl_comp_fig.savefig(OUTPUT_DIR / "external_comparison.svg")
    print(_abl_perf_tbl.reset_index().style.to_typst())


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
