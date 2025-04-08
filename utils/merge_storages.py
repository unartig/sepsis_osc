import logging
from utils.logger import setup_logging
from storage.storage_interface import Storage
import numpy as np

setup_logging()
logger = logging.getLogger(__name__)


base = Storage()
other = Storage(
    key_dim=9,
    parameter_k_name="storage/other/ColabSepsisParameters_index.bin",
    metrics_kv_name="storage/other/ColabSepsisMetrics.db",
)
print(other.parameter_k_name)
print(base.current_idx, other.current_idx)
# base.merge(other)
# base.close()
