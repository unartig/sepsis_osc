from numpy.random import default_rng

# Random seeds
jax_random_seed = 123
np_random_seed = 123
np_rng = default_rng(np_random_seed)
random_seed = 123

# Workers
max_workers = 22 * 2

# DB names
db_parameter_keys = "storage/SepsisParameters_index.bin"
db_metrics_key_value = "storage/SepsisMetrics.db"

# YAIB data
target_name = "sep3_alt_first"
cohort_name = f"{target_name}_with_marginals_ramp"
yaib_data_dir = f"/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/{cohort_name}/miiv"


# Log Config
cfg_log_level = "info"
log_file = "sepsis.log"

sequence_files = "data/sequence_"

ALPHA_SPACE = (-0.28 - 0.001, -0.28 + 0.001, 0.001)  # only original slice
ALPHA = -0.28
BETA_SPACE = (0.4, 0.7, 0.005)
SIGMA_SPACE = (0.0, 1.5, 0.015)  # only original area

# PLT config
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt_params = {
    "figure.figsize": (10, 4),  # inches
    "font.size": SMALL_SIZE,
    "axes.titlesize": BIGGER_SIZE,
    "axes.labelsize": MEDIUM_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "legend.fontsize": SMALL_SIZE,
    "figure.dpi": 100,
}
