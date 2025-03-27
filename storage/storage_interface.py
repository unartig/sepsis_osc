import logging

import faiss
import plyvel
import json
import numpy as np


from utils.config import db_parameter_keys, db_metrics_key_value
from simulation import SystemState, SystemMetrics

logger = logging.getLogger(__name__)


class Storage:
    def __init__(
        self,
        key_dim: int = 9,
        parameter_k_name: str = db_parameter_keys,
        metrics_kv_name: str = db_metrics_key_value,
    ):
        self.parameter_k_name = parameter_k_name
        self.metrics_kv_name = metrics_kv_name
        # FAISS stores the Parameter vectors for fast NN retrieval
        self.key_dim = key_dim
        self.__db_keys = self.__setup_faiss(self.parameter_k_name)

        # RocksDB stores the actual metric-arrays,
        # key: FAISS-index, value: npz compressed SystemMetric
        self.__db_metric = plyvel.DB(self.metrics_kv_name, create_if_missing=True)
        last_id_bytes = self.__db_metric.get(b"faiss_last_id")
        self.last_id = int(last_id_bytes.decode()) if last_id_bytes else 0

    def __setup_faiss(self, db_name: str):
        index = faiss.IndexFlatL2(self.key_dim)  # L2 (Euclidean) distance
        faiss_index_file = db_parameter_keys
        try:
            index = faiss.read_index(faiss_index_file)
            logger.info(f"FAISS index {db_name} loaded from disk.")
        except Exception as _:
            logger.info("No existing FAISS index, starting fresh.")
        return index

    def add_faiss(self, key: np.ndarray):
        logger.info(f"Adding key {key[0]} to FAISS index as [{self.last_id}] ")
        self.__db_keys.add(key)
        self.__db_metric.put(b"faiss_last_id", str(self.last_id).encode())

    def write_faiss(self):
        logger.info(f"Writing FAISS index to {self.parameter_k_name}")
        faiss.write_index(self.__db_keys, db_parameter_keys)

    def find_faiss(self, query_key: np.ndarray, k: int = 1):
        logger.info(f"Searching {query_key} in FAISS index")
        distances, indices = self.__db_keys.search(query_key, k=k)
        logger.info(f"Found vectors have distance {distances}")
        return indices

    def add_rocks_metric(self, metrics: SystemMetrics):
        logger.info(f"Adding SystemMetrics with key {self.last_id} to RocksDB")
        with open("storage/tmp.npz", "wb") as f:
            np.savez_compressed(
                f,
                r_1=metrics.r_1,
                r_2=metrics.r_2,
                s_1=metrics.s_1,
                s_2=metrics.s_2,
                ns_1=metrics.ns_1,
                ns_2=metrics.ns_2,
                f_1=metrics.f_1,
                f_2=metrics.f_2,
            )
        with open("storage/tmp.npz", "rb") as f:
            self.__db_metric.put(f"metrics_{str(self.last_id)}".encode(), f.read())

    def read_rocks_metric(self, key: str) -> SystemMetrics:
        logger.info(f"Reading SystemMetrics with key {key} from RocksDB")

        data_bytes = self.__db_metric.get(f"metrics_{key}".encode())

        with open("storage/tmp.npz", "wb") as f:
            f.write(data_bytes)

        data = np.load("storage/tmp.npz")
        return SystemMetrics(
            r_1=data["r_1"],
            r_2=data["r_2"],
            s_1=data["s_1"],
            s_2=data["s_2"],
            ns_1=data["ns_1"],
            ns_2=data["ns_2"],
            f_1=data["f_1"],
            f_2=data["f_2"],
        )

    def add_result(self, params: tuple[int | float, ...], metrics: SystemMetrics):
        logger.info(f"Adding new Results for {params}, starting Pipeline")
        np_params = np.array([params], dtype=np.float32)
        self.add_faiss(np_params)
        self.add_rocks_metric(metrics)
        self.last_id += 1

    def read_result(self, params: tuple[int|float, ...]) -> None | SystemMetrics:
        logger.info(f"Getting Results for {params}, starting Pipeline")
        np_params = np.array([params], dtype=np.float32)
        index = self.find_faiss(np_params)
        return self.read_rocks_metric(str(index[0][0]))

    def close(self):
        self.write_faiss()
        logger.info(f"Writing RocksDB to {self.metrics_kv_name}")
        self.__db_metric.close()
