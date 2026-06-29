import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    from pathlib import Path

    import json

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    # from icu_benchmarks.data.split_process_data import preprocess_data
    from icu_benchmarks.constants import RunMode

    # from icu_benchmarks.data.preprocessor import PolarsRegressionPreprocessor
    import os

    from sepsis_osc.ldm.gin_configs import file_names, new_vars, paper_vars, modality_mapping

    LABEL_COL = "sep3_alt"


    def get_data(db_name):
        sep3_path = Path(f"/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/{LABEL_COL}_with_marginals_ramp/{db_name}")
        cohort_demo_path = Path(f"misc/cohort_stats_{db_name}.csv")

        sep3_data = {f: pl.read_parquet(sep3_path / file_names[f]) for f in file_names.keys() if os.path.exists(sep3_path / file_names[f])}
        cohort_demo_data = pd.read_csv(cohort_demo_path)

        print(sep3_data["OUTCOME"].columns)
        print("Samples:", len(sep3_data["OUTCOME"]))
        print("Patients:", sep3_data["OUTCOME"]["stay_id"].unique().len())
        df_outcome = sep3_data["OUTCOME"].to_pandas().set_index(["stay_id", "time"])
        df_dynamic = sep3_data["DYNAMIC"].to_pandas()
        df_static = sep3_data["STATIC"].to_pandas().set_index(["stay_id"])
        df_merged_temp = pd.merge(df_dynamic, df_static, on='stay_id', how='left').set_index(['stay_id', 'time'])
        df = pd.merge(df_merged_temp, df_outcome, left_index=True, right_index=True, how='left')
        del df_outcome, df_dynamic, df_static, sep3_data
        df["sex"] = np.where(df["sex"] == "Female", 0, 1)
    
        cohort_demo_data.index = cohort_demo_data["stay_id"]
        return df, cohort_demo_data

    return LABEL_COL, get_data, json, pd


@app.cell
def _(get_data):
    miiv_df, miiv_demo = get_data("miiv")
    eicu_df, eicu_demo = get_data("eicu")

    dfs = {"MIIV": miiv_df, "eICU": eicu_df}
    return dfs, eicu_demo, eicu_df, miiv_demo, miiv_df


@app.cell
def _(df, label_col, pd):
    with pd.option_context("display.max_rows", None):
        print(df.loc[30000484, [label_col, "sofa", "susp_inf_alt", "susp_inf_ramp", "yaib_label", "death"]])
    return


@app.cell
def _(dfs, json, miiv_df, pd):
    dir = "."
    file = "misc/concept-dict.json"

    with open(f"{dir}/{file}") as f:
        _json = json.load(f)

    columns = miiv_df.columns
    conc_list = []
    for col in columns:
        if col not in ("stay_id", "time") and not col.startswith("Missing") and col in _json.keys():
            concept = _json[col]
            short = {"name": col}
            short["unit"] = concept["unit"] if "unit" in concept else ""
            short["min"] = concept["min"] if "min" in concept else ""
            short["max"] = concept["max"] if "max" in concept else ""
            short["description"] = concept["description"] if "description" in concept else ""

            for name, df in dfs.items():  
                short[f"{name} % missing"] = f"{df[col].isna().sum()/len(df[col])*100:.2f}"
            conc_list.append(short)
    conc = pd.DataFrame(conc_list)
    conc.index = conc["name"]
    conc = conc.drop(columns=["name"])
    conc
    return conc, df


@app.cell
def _(conc):
    print(conc.style.to_typst())

    return


@app.cell
def _(LABEL_COL, eicu_demo, eicu_df, miiv_demo, miiv_df):
    def get_cohort_stats(df, demo):
        cdf = df.copy().reset_index()
    
        patient_sep = (
            cdf.groupby("stay_id")[LABEL_COL]
              .max()
              .rename("sep3")
        )
        sep_onset = (
            cdf[cdf[LABEL_COL] == 1]
            .groupby("stay_id")["time"]
            .min()
            .rename("sep_onset_time")
        )
        death_icu_label = (
            cdf.groupby("stay_id")["death_icu"]
            .max()
            .rename("death_icu")
        )
        death_label = (
            cdf.groupby("stay_id")["death"]
            .max()
            .rename("death")
        )
    
    
        patient_level = (
            cdf.groupby("stay_id")
              .agg(
                  sex=("sex", "first"),
                  age=("age", "first"),
                  weight=("weight", "first"),
                  sofa_median=("sofa", "median"),
                  sofa_max=("sofa", "max"),
                  los=("time", "max"),
                  #death=("death", "max"),
                  death_icu=("death_icu", "max"),
              )
              .join(patient_sep)
              .join(sep_onset)
              .join(death_label)
              .join(demo)
        )
    
        sep_pos = patient_level[patient_level["sep3"] == 1]
        sep_neg = patient_level[patient_level["sep3"] == 0]
        return patient_level, sep_pos, sep_neg, len(patient_level)

    def cohort_summary(df, n, include_onset=False):
        summary = {
            "N": f"{len(df)} ({len(df)/n * 100:.1f}%)",
            "Male n (%)": f"{(df.sex == 1).sum()} ({100 * (df.sex == 1).mean():.1f}%)",
            "Age at admission, median (IQR)": f"{df.age.median():.1f} ({df.age.quantile(0.25):.1f}–{df.age.quantile(0.75):.1f})",
            "Weight at admission, median (IQR)": f"{df.weight.median():.1f} ({df.weight.quantile(0.25):.1f}–{df.weight.quantile(0.75):.1f})",
            "SOFA median, median (IQR)": f"{df.sofa_median.median():.1f} ({df.sofa_median.quantile(0.25):.1f}–{df.sofa_median.quantile(0.75):.1f})",
            "SOFA max, median (IQR)": f"{df.sofa_max.median():.1f} ({df.sofa_max.quantile(0.25):.1f}–{df.sofa_max.quantile(0.75):.1f})",
            "hospital LOS hours, median (IQR)": f"{df.hospital_los_hours.median():.1f} ({df.hospital_los_hours.quantile(0.25):.1f}–{df.hospital_los_hours.quantile(0.75):.1f})",
            "Hospital Mortality, (%)": f"{(df.hospital_expire_flag == 1).sum()} ({100 * (df.hospital_expire_flag == 1).mean():.1f}%)",
        }

        # Ethnicity percentages
        for cat in df['ethnicity_group'].unique():
            count = (df['ethnicity_group'] == cat).sum()
            pct = 100 * count / len(df)
            summary[f"Ethnicity {cat}, (%)"] = f"{count} ({pct:.1f}%)"

        # Admission type percentages
        for cat in df['admission_group'].unique():
            count = (df['admission_group'] == cat).sum()
            pct = 100 * count / len(df)
            summary[f"Admission {cat}, (%)"] = f"{count} ({pct:.1f}%)"

        # Include SEP onset time if requested
        if include_onset and 'sep_onset_time' in df.columns:
            summary["SEP-3 onset time, median (IQR)"] = (
                f"{df.sep_onset_time.median():.1f} "
                f"({df.sep_onset_time.quantile(0.25):.1f}–{df.sep_onset_time.quantile(0.75):.1f})"
            )

        return summary

    miiv_patient_level, miiv_sep_pos_level, miiv_sep_neg_level, miiv_n = get_cohort_stats(miiv_df, miiv_demo)
    eicu_patient_level, eicu_sep_pos_level, eicu_sep_neg_level, eicu_n = get_cohort_stats(eicu_df, eicu_demo)
    return (
        cohort_summary,
        eicu_n,
        eicu_patient_level,
        eicu_sep_neg_level,
        eicu_sep_pos_level,
        miiv_n,
        miiv_patient_level,
        miiv_sep_neg_level,
        miiv_sep_pos_level,
    )


@app.cell
def _(
    cohort_summary,
    miiv_n,
    miiv_patient_level,
    miiv_sep_neg_level,
    miiv_sep_pos_level,
    pd,
):
    miiv_table = pd.DataFrame.from_dict(
        {
            "All patients": cohort_summary(miiv_patient_level, miiv_n),
            "SEP-3 positive": cohort_summary(miiv_sep_pos_level, miiv_n, include_onset=True),
            "SEP-3 negative": cohort_summary(miiv_sep_neg_level, miiv_n),
        },
        orient="columns"
    )

    print(miiv_table.style.to_typst())
    return


@app.cell
def _(
    cohort_summary,
    eicu_n,
    eicu_patient_level,
    eicu_sep_neg_level,
    eicu_sep_pos_level,
    pd,
):
    eicu_table = pd.DataFrame.from_dict(
        {
            "All patients": cohort_summary(eicu_patient_level, eicu_n),
            "SEP-3 positive": cohort_summary(eicu_sep_pos_level, eicu_n, include_onset=True),
            "SEP-3 negative": cohort_summary(eicu_sep_neg_level, eicu_n),
        },
        orient="columns"
    )

    print(eicu_table.style.to_typst())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
