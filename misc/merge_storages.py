import logging

from sepsis_osc.dnm.dynamic_network_model import DNMMetrics
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


base_str = "test"
base = Storage(
    parameter_k_name=f"data/{base_str}SepsisParameters_index.bin",
    metrics_kv_name=f"data/{base_str}SepsisMetrics.db",
    use_mem_cache=True,
)

other_str = "DaisyFinal"
other = Storage(
    key_dim=9,
    parameter_k_name=f"data/{other_str}SepsisParameters_index.bin",
    metrics_kv_name=f"data/{other_str}SepsisMetrics.db",
    use_mem_cache=True,
)
print(base.current_idx, other.current_idx)
base.merge(other, DNMMetrics, overwrite=True)
other.close()
base.close()
