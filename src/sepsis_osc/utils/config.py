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
yaib_data_dir = "/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/inf_sofa/mimic"


# Log Config
cfg_log_level = "info"
log_file = "sepsis.log"

sequence_files = "data/sequence_"

# DAISY 1
# ALPHA_SPACE = (-1.0, 1.0, 0.04)
# BETA_SPACE = (0.2, 1.0, 0.02)
# SIGMA_SPACE = (0.0, 1.5, 0.04)

ALPHA_SPACE = (-0.28 -0.001, -0.28 + 0.001, 0.001)  # only original slice
ALPHA = -0.28
# BETA_SPACE = (0.4, 0.7, 0.01)  # only original area
# BETA_SPACE = (0.0, 1.0, 0.01)
# SIGMA_SPACE = (0.0, 1.5, 0.01)
# SIGMA_SPACE = (0.0, 1.0, 0.01)  # only original area

# DAISY 2
# ALPHA_SPACE = (-0.84, 0.84+0.28, 0.28)  # full with mirror
# ALPHA_SPACE = (-0.84, -0.84+3*0.28, 0.28)  # only negative slice
BETA_SPACE = (0.0, 1.0, 0.01)
SIGMA_SPACE = (0.0, 1.5, 0.015)

# DAISY HALF
# BETA_SPACE = (0.0, 1.0, 0.01 * 2)
# SIGMA_SPACE = (0.0, 1.5, 0.015 * 2)

