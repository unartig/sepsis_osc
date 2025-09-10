# Random seeds
jax_random_seed = 12
np_random_seed = 12
random_seed = 12

# Workers
max_workers = 22 * 2

# DB names
db_parameter_keys = "storage/SepsisParameters_index.bin"
db_metrics_key_value = "storage/SepsisMetrics.db"

# YAIB data
yaib_data_dir = "/home/unartig/Desktop/uni/ResearchProject/YAIB-cohorts/data/inf_sofa_short/mimic"


# Log Config
cfg_log_level = "info"
log_file = "sepsis.log"

sequence_files = "data/sequence_"

# ALPHA_SPACE = (-1.0, 1.0, 0.04)
# BETA_SPACE = (0.2, 1.0, 0.02)
# SIGMA_SPACE = (0.0, 1.5, 0.04)
ALPHA_SPACE = (-0.28, -0.27, 0.01)
BETA_SPACE = (0.0, 1.0, 0.01)
SIGMA_SPACE = (0.0, 1.5, 0.01)
