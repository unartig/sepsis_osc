import logging

from storage.storage_interface import Storage
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


base_str = "Tiny"
base = Storage(
    parameter_k_name=f"storage/other/{base_str}SepsisParameters_index.bin",
    metrics_kv_name=f"storage/other/{base_str}SepsisMetrics.db",
)

other_str = "Big"
other = Storage(
    key_dim=9,
    parameter_k_name=f"storage/other/{other_str}SepsisParameters_index.bin",
    metrics_kv_name=f"storage/other/{other_str}SepsisMetrics.db",
    use_mem_cache=False,
)
print(base.current_idx, other.current_idx)
base.merge(other)
base.close()
