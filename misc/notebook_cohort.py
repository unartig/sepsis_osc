import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import pandas as pd
    from pathlib import Path

    import json

    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    from icu_benchmarks.data.split_process_data import preprocess_data
    from icu_benchmarks.constants import RunMode
    from icu_benchmarks.data.preprocessor import PolarsRegressionPreprocessor
    import os

    from sepsis_osc.ldm.gin_configs import file_names, new_vars, paper_vars, modality_mapping

    label_col = "sep3_alt_first"
    sep3_path = Path(f"/home/unartig/Desktop/uni/ResearchProject/yaib_docker/YAIB-cohorts/data/{label_col}_with_marginals_ramp/miiv")
    cohort_demo_path = Path("cohort_stats.csv")

    def get_data(p):
        return {
            f: pl.read_parquet(p / file_names[f])
            for f in file_names.keys()
            if os.path.exists(p / file_names[f])
        }

    sep3_data = get_data(sep3_path)
    cohort_demo = pd.read_csv(cohort_demo_path)


    print(sep3_data["OUTCOME"].columns)
    print("Samples:", len(sep3_data["OUTCOME"]))
    print("Patients:", sep3_data["OUTCOME"]["stay_id"].unique().len())
    return cohort_demo, json, label_col, np, pd, plt, sep3_data


@app.cell
def _(cohort_demo, np, pd, sep3_data):
    df_outcome = sep3_data["OUTCOME"].to_pandas().set_index(["stay_id", "time"])
    df_dynamic = sep3_data["DYNAMIC"].to_pandas()
    df_static = sep3_data["STATIC"].to_pandas().set_index(["stay_id"])
    df_merged_temp = pd.merge(df_dynamic, df_static, on='stay_id', how='left').set_index(['stay_id', 'time'])
    df = pd.merge(df_merged_temp, df_outcome, left_index=True, right_index=True, how='left')
    del df_outcome, df_dynamic, df_static, sep3_data
    df["sex"] = np.where(df["sex"] == "Female", 0, 1)

    cohort_demo.index = cohort_demo["stay_id"]
    return (df,)


@app.cell
def _(df, label_col, pd):
    with pd.option_context("display.max_rows", None):
        print(df.loc[30000484, [label_col, "sofa", "susp_inf_alt", "susp_inf_ramp", "yaib_label", "death"]])
    return


@app.cell
def _(df, json, pd):
    dir = "."
    file = "concept-dict.json"

    with open(f"{dir}/{file}") as f:
        _json = json.load(f)

    columns = df.columns
    conc_list = []
    for col in columns:
        if col not in ("stay_id", "time") and not col.startswith("Missing") and col in _json.keys():
            concept = _json[col]
            short = {"name": col}
            short["unit"] = concept["unit"] if "unit" in concept else ""
            short["min"] = concept["min"] if "min" in concept else ""
            short["max"] = concept["max"] if "max" in concept else ""
            short["description"] = concept["description"] if "description" in concept else ""
            conc_list.append(short)
    conc = pd.DataFrame(conc_list)
    conc
    return


@app.cell
def _(cohort_demo):
    print(cohort_demo.columns)

    print(cohort_demo["ethnicity_group"].unique())
    print(cohort_demo["admission_group"].unique())
    print(cohort_demo["hospital_expire_flag"].unique())
    return


@app.cell
def _(cohort_demo, df, label_col, pd):
    cdf = df.copy().reset_index()

    patient_sep = (
        cdf.groupby("stay_id")[label_col]
          .max()
          .rename("sep3")
    )
    sep_onset = (
        cdf[cdf[label_col] == 1]
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
          .join(cohort_demo)
    )

    sep_pos = patient_level[patient_level["sep3"] == 1]
    sep_neg = patient_level[patient_level["sep3"] == 0]

    def cohort_summary(df, include_onset=False):
        summary = {
            "N": f"{len(df)} ({len(df)/len(patient_level) * 100:.1f}%)",
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


    table1 = pd.DataFrame.from_dict(
        {
            "All patients": cohort_summary(patient_level),
            "SEP-3 positive": cohort_summary(sep_pos, include_onset=True),
            "SEP-3 negative": cohort_summary(sep_neg),
        },
        orient="columns"
    )

    table1
    return patient_level, table1


@app.cell
def _(table1):
    for row in table1.iterrows():
        print(f"[{row[0].replace(", (%)", "").replace(", median (IQR)", "").replace("Ethnicity ", "").replace("Admission", "")}],")
        print(f"[{row[1]["All patients"]}],")
        print(f"[{row[1]["SEP-3 positive"]}],")
        print(f"[{row[1]["SEP-3 negative"]}],")
        print()
    return


@app.cell
def _(df, np, plt):
    def single_feature(y, log=False):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))
        ax0.scatter(df.index.get_level_values("stay_id"), df[y], s=0.1, alpha=0.1)
        ax0.set_title(f"{y}")
        ax1.hist(df[y], bins=100, density=True);
        if log:
            ax0.set_yscale("log")
            ax1.set_yscale("log")
        ax1.set_title(f"{y} Histogram")
        return fig

    def single_stay(sid, x_list, df=df):
        stay = df.loc[sid]
        n = len(x_list)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))  # dynamic width

        if n == 1:
            axes = [axes]

        for ax, x in zip(axes, x_list):
            ax.scatter(stay.index, stay[x])
            ax.set_title(x)

        plt.tight_layout()
    
        return fig


    def feature_hist(y, log=False, bs=50, df=df):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        uniques = np.unique(df[y])
        ax.hist(df[y], bins=bs, density=True)
        plt.tight_layout()
        if log:
            ax.set_yscale("log")
        return fig

    def feature_pie(y, df=df):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        uniques, counts = np.unique(df[y], return_counts=True)
        ax.pie(x=counts, labels=uniques)
        plt.tight_layout()
        return fig

    return feature_hist, feature_pie


@app.cell
def _(feature_hist, patient_level, plt):
    feature_hist("weight", df=patient_level)
    plt.show()
    return


@app.cell
def _(feature_hist, patient_level, plt):
    feature_hist("age", bs=8, df=patient_level)

    plt.show()
    return


@app.cell
def _(df, feature_hist):
    feature_hist("sofa", bs=24, df=df)
    return


@app.cell
def _(df, label_col, np, plt):
    plt.hist(df.loc[df[label_col] == 1.0, "sofa"], bins=np.arange(24));
    return


@app.cell
def _(df, feature_pie):
    feature_pie("sofa", df=df)
    return


@app.cell
def _(feature_pie, patient_level):
    feature_pie("sex", df=patient_level)
    return


@app.cell
def _(feature_pie, patient_level):
    feature_pie("sep3", df=patient_level)
    return


@app.cell
def _(df, label_col, np):
    from matplotlib_venn import venn3
    p_inf = np.asarray(df["susp_inf_ramp"] > 0.0).astype(np.float32)
    df["sofa_diff_direct"] = df.groupby(level="stay_id", group_keys=False)["sofa"].diff().fillna(0)
    p_sofa = np.asarray(df["sofa_diff_direct"] > 0.0).astype(np.float32)
    p_sep3 = np.asarray(df[label_col] == 1.0).astype(np.float32)
    return p_inf, p_sep3, p_sofa, venn3


@app.cell
def _(np, p_inf, p_sep3, p_sofa, plt, venn3):
    A = p_inf.astype(bool)
    B = p_sofa.astype(bool)
    C = p_sep3.astype(bool)

    A_only = np.sum(A & ~B & ~C)
    B_only = np.sum(~A & B & ~C)
    AB_only = np.sum(A & B & ~C)
    C_only = np.sum(~A & ~B & C)
    AC_only = np.sum(A & ~B & C)  # for some reason here is one :^?
    BC_only = np.sum(~A & B & C)
    ABC_overlap = np.sum(A & B & C)

    print("total sep", np.sum(C))
    print("AB", AB_only)
    print("AC", AC_only)
    print("BC", BC_only)
    print("C", C_only)
    print("ABC", ABC_overlap)

    subset_sizes = (
        A_only,      # 100: A & ~B & ~C
        B_only,      # 010: ~A & B & ~C
        AB_only,     # 110: A & B & ~C
        C_only,      # 001: ~A & ~B & C
        AC_only,     # 101: A & ~B & C
        BC_only,     # 011: ~A & B & C
        ABC_overlap  # 111: A & B & C
    )

    set_labels = ('Suspected Infection', 'SOFA increase', 'Sepsis-Label')

    # --- percentages ---
    total = np.array(A.size)
    subset_percentages = subset_sizes / total * 100

    plt.figure(figsize=(8, 8))
    v = venn3(subsets=subset_percentages, set_labels=set_labels)

    # format labels as percentages
    for idx, label in enumerate(v.subset_labels):
        if label:
            label.set_text(f"{subset_percentages[idx]:.1f}%")

    plt.title("Venn Diagram (Percentages)")
    plt.show()

    plt.savefig("../typst/images/yaib_sets.svg")

    # TODO percentages
    print(f"Total Observations (Union of Sets): {sum(subset_sizes)}")
    print(f"Total Observations (DataFrame/Array Length): {len(p_inf)}")
    return


@app.cell
def _(df, pd):
    stay_ids = df.index.get_level_values("stay_id").unique()
    print(stay_ids)
    pd.DataFrame(stay_ids).to_csv("cohort_ids.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
