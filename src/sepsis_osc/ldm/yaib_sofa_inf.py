import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame, Series

from src.cohort import Cohort, SelectionCriterion
from src.ricu import stay_windows
from src.ricu_utils import (
    longest_rle,
    make_grid_mapper,
    make_outcome_windower,
    make_patient_mapper,
    make_prevalence_calculator,
    n_obs_per_row,
    stop_window_at,
)
from src.steps import (
    AggStep,
    CombineStep,
    CustomStep,
    DropStep,
    FilterStep,
    InputStep,
    LoadStep,
    Pipeline,
    TransformStep,
)

# https://eth-mds.github.io/ricu/reference/callback_sofa.html
outc_vars = ["sofa", "susp_inf_alt", "sep3_alt"]
static_vars = ["age", "sex", "height", "weight"]
dynamic_vars = [
    "alb",
    "alp",
    "alt",
    "ast",
    "be",
    "bicar",
    "bili",
    "bili_dir",
    "bnd",
    "bun",
    "ca",
    "cai",
    "ck",
    "ckmb",
    "cl",
    "crea",
    "crp",
    "dbp",
    "fgn",
    "fio2",
    "glu",
    "hgb",
    "hr",
    "inr_pt",
    "k",
    "lact",
    "lymph",
    "map",
    "mch",
    "mchc",
    "mcv",
    "methb",
    "mg",
    "na",
    "neut",
    "o2sat",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "resp",
    "sbp",
    "temp",
    "tnt",
    "urine",
    "wbc",
]


def create_sofa_task(args):
    print("Start creating the SOFA task.")
    print("   Preload variables")
    sepsis_args = {}
    if args.src in ["eicu", "eicu_demo", "hirid"]:
        sepsis_args["si_mode"] = "abx"

    # Load the concepts SOFA, suspected infection and Sepsis3
    sofa = LoadStep(outc_vars, args.src, cache=True, **sepsis_args).perform()

    static = LoadStep(static_vars, args.src, cache=True).perform()
    dynamic = LoadStep(dynamic_vars, args.src, cache=True).perform()
    assert isinstance(sofa, DataFrame), "Could not get SOFA DataFrame"
    assert isinstance(dynamic, DataFrame), "Could not get Dynamic DataFrame"
    assert isinstance(static, DataFrame), "Could not get Static DataFrame"

    sofa.loc[sofa.sep3_alt == True, "susp_inf_alt"] = True

    print("   Define observation times")
    patients = stay_windows(args.src)
    patients = stop_window_at(patients, end=24 * 7)

    print("   Define exclusion criteria")
    # General exclusion criteria
    excl1 = SelectionCriterion("Invalid length of stay")
    excl1.add_step([InputStep(patients), FilterStep("end", lambda x: x < 0)])

    excl2 = SelectionCriterion("Length of stay < 6h")
    excl2.add_step([LoadStep("los_icu", args.src), FilterStep("los_icu", lambda x: x < 6 / 24)])
    # excl2.add_step([InputStep(sofa), FilterStep("los_icu", lambda x: x < 6 / 24)])

    excl3 = SelectionCriterion("Less than 4 hours with any measurement")
    excl3.add_step([InputStep(dynamic), AggStep("stay_id", "count"), FilterStep("time", lambda x: x < 4)])

    excl4 = SelectionCriterion("More than 12 hour gap between measurements")
    excl4.add_step([
        InputStep(dynamic),
        CustomStep(make_grid_mapper(patients, step_size=1)),
        CustomStep(n_obs_per_row),
        TransformStep("n", lambda x: x > 0),
        AggStep("stay_id", longest_rle, "n"),
        FilterStep("n", lambda x: x > 12),
    ])

    excl5 = SelectionCriterion("Aged < 18 years")
    # excl5.add_step([LoadStep("age", args.src), FilterStep("age", lambda x: x < 18)])
    excl5.add_step([InputStep(static), FilterStep("age", lambda x: x < 18)])

    # Task-specific exclusion criteria
    # TODO
    # criteria
    # sepsis_add6 = sofa[sofa["time"] >= 0].copy()
    # sepsis_add6["time"] += 6
    # patients = stop_window_at(patients, end=sepsis_add6)

    get_first_sepsis = Pipeline("Get patients")
    get_first_sepsis.add_step([InputStep(sofa), AggStep("stay_id", "max")])
    load_hospital_id = LoadStep("hospital_id", src=args.src)
    excl6 = SelectionCriterion("Low sepsis prevalence")
    excl6.add_step([
        CombineStep(steps=[get_first_sepsis, load_hospital_id], func=make_prevalence_calculator("sep3_alt")),
        FilterStep("prevalence", lambda x: x == 0),
    ])

    # excl7 = SelectionCriterion("Sepsis onset before 6h in the ICU")
    # excl7.add_step([InputStep(sofa), AggStep(["stay_id"], lambda x: x.iloc[0]), FilterStep("time", lambda x: x < 6)])

    print("   Select cohort\n")
    cohort = Cohort(patients)
    cohort.add_criterion(
        [excl1, excl2, excl3, excl4, excl5, excl6]
        if args.src in ["eicu", "eicu_demo"]
        else [excl1, excl2, excl3, excl4, excl5]
    )
    print(cohort.criteria)
    patients, attrition = cohort.select()
    print("\n")

    print("   Load and format input data")
    outc_formatting = Pipeline("Prepare length of stay")
    outc_formatting.add_step([
        InputStep(sofa),
        CustomStep(make_grid_mapper(patients)),
        CustomStep(lambda x: x.dropna().astype(int), "remove nans and intify"),
        CustomStep(make_outcome_windower(1, "susp_inf_alt")),
        CustomStep(make_outcome_windower(1, "sofa")),
        CustomStep(make_outcome_windower(1, "sep3_alt")),
    ])
    outc = outc_formatting.apply()

    dyn_formatting = Pipeline("Prepare dynamic variables")
    # Filter dyn based on whether its (stay_id, time) pairs are in sofa_matches
    sofa_index = outc.set_index(["stay_id", "time"]).index
    dyn_formatting.add_step([
        InputStep(dynamic),
        CustomStep(lambda x: x.set_index(["stay_id", "time"])),  # Set MultiIndex
        CustomStep(lambda x: x.loc[sofa_index]),  # Filter using the MultiIndex
        # CustomStep(lambda x: x.reset_index()),
        DropStep("__index_level_0__"),
        CustomStep(make_grid_mapper(patients, step_size=1)),
    ])
    dyn = dyn_formatting.apply()

    sta_formatting = Pipeline("Prepare static variables")
    sta_formatting.add_step([InputStep(static), CustomStep(make_patient_mapper(patients))])
    sta = sta_formatting.apply()

    return (outc, dyn, sta), attrition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="mimic_demo",
        help="name of datasource",
        choices=["aumc", "eicu", "eicu_demo", "hirid", "mimic", "mimic_demo", "miiv"],
    )
    parser.add_argument("--out_dir", default="../data/sofa", help="path where to store extracted data")
    args = parser.parse_known_args()[0]

    (outc, dyn, sta), attrition = create_sofa_task(args)

    save_dir = os.path.join(args.out_dir, args.src)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Data Shapes: {outc.shape}, {dyn.shape}, {sta.shape}")
    pq.write_table(pa.Table.from_pandas(outc), os.path.join(save_dir, "outc.parquet"))
    pq.write_table(pa.Table.from_pandas(dyn), os.path.join(save_dir, "dyn.parquet"))
    pq.write_table(pa.Table.from_pandas(sta), os.path.join(save_dir, "sta.parquet"))

    attrition.to_csv(os.path.join(save_dir, "attrition.csv"))
