import logging

import faiss
import msgpack
import msgpack_numpy as mnp
import numpy as np
import rocksdbpy as rock

from simulation.data_classes import SystemMetrics
from utils.config import db_metrics_key_value, db_parameter_keys

logger = logging.getLogger(__name__)


def pprint_key(key: list[float] | tuple[float, ...] | np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(x), 4) for x in key)


class Storage:
    def __init__(
        self,
        key_dim: int = 9,
        parameter_k_name: str = "",
        metrics_kv_name: str = "",
        use_mem_cache: bool = True,
    ):
        parameter_k_name = db_parameter_keys if not parameter_k_name else parameter_k_name
        metrics_kv_name = db_metrics_key_value if not metrics_kv_name else metrics_kv_name
        logger.info(f"Got {parameter_k_name} and {metrics_kv_name}")

        # FAISS stores the Parameter vectors for fast NN retrieval
        self.parameter_k_name = parameter_k_name
        self.key_dim = key_dim
        self.__db_keys = self.__setup_faiss(self.parameter_k_name)

        # RocksDB stores the actual metric-arrays,
        # key: FAISS-index, value: Serialized SystemMetric
        self.metrics_kv_name = metrics_kv_name
        self.__db_metric, self.current_idx = self.__setup_rocksdb(self.metrics_kv_name)

        # Memory Cache for faster lookups
        self.use_mem_cache = use_mem_cache
        self.__memory_cache: dict[str, bytes] = {}
        if self.use_mem_cache:
            self.__setup_memory_cache()

    def __setup_faiss(
        self,
        db_name: str,
    ):
        key_index = faiss.IndexFlatL2(self.key_dim)  # L2 (Euclidean) distance
        try:
            key_index = faiss.read_index(self.parameter_k_name)
            logger.info(f"FAISS index {db_name} loaded from disk.")
        except Exception as _:
            logger.info("No existing FAISS index, starting fresh.")
        return key_index

    def __setup_rocksdb(self, db_name: str):
        opts = rock.Option()
        opts.create_if_missing(True)
        opts.set_allow_mmap_reads(True)

        rocksdb = rock.open(db_name, opts=opts)
        last_id_bytes = rocksdb.get(b"faiss_last_id")
        current_idx = int(last_id_bytes.decode()) if last_id_bytes else 0
        return rocksdb, current_idx

    def __setup_memory_cache(self):
        for key, data_bytes in self.__db_metric.iterator():
            if key.startswith(b"metrics_"):
                self.add_metric(
                    None,
                    data_bytes,
                    str(key.decode()).lstrip("metrics_"),
                )

        logger.info(f"Successfully loaded {len(self.__memory_cache)} metrics into memory.")

    def add_faiss(
        self,
        key: np.ndarray,
    ):
        logger.info(f"Adding key {key[0]} to FAISS index as [{self.current_idx}] ")
        self.__db_keys.add(key)

    def find_faiss(
        self,
        query_key: np.ndarray,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Searching {pprint_key(query_key[0])} in FAISS index")
        distances, indices = self.__db_keys.search(query_key, k=k)
        logger.info(f"Found vectors with distance {distances}, index {indices}")
        return indices, distances

    def add_metric(
        self,
        metrics: SystemMetrics | None,
        packed_data: bytes | None,
        index: str = "",
    ):
        if not index:
            logger.error("Cannot save empty key in MemoryCache")
            return
        if metrics and not packed_data:
            packed_data = msgpack.packb(
                {
                    "r_1": np.asarray(metrics.r_1),
                    "r_2": np.asarray(metrics.r_2),
                    "m_1": np.asarray(metrics.m_1),
                    "m_2": np.asarray(metrics.m_2),
                    "s_1": np.asarray(metrics.s_1),
                    "s_2": np.asarray(metrics.s_2),
                    "q_1": np.asarray(metrics.q_1),
                    "q_2": np.asarray(metrics.q_2),
                    "f_1": np.asarray(metrics.f_1),
                    "f_2": np.asarray(metrics.f_2),
                },
                default=mnp.encode,
            )

        if packed_data:
            if self.use_mem_cache:
                logger.info(f"Adding SystemMetrics with key {index} to MemoryCache")
                self.__memory_cache[index] = packed_data
            else:
                logger.info(f"Adding SystemMetrics with key {index} to RocksDB")
                self.__db_metric.set(f"metrics_{index}".encode(), packed_data)

    def read_metric(
        self,
        key: str,
    ) -> SystemMetrics | None:

        if self.use_mem_cache:
            logger.info(f"Reading SystemMetrics with key {key} from MemoryCache")
            if key in self.__memory_cache:
                data_bytes = self.__memory_cache[key]
            else:
                logger.info(f"Could not find SystemMetrics for {key} in MemoryCache")
                return None
        else:
            logger.info(f"Reading SystemMetrics with key {key} from RocksDB")
            data_bytes = self.__db_metric.get(f"metrics_{key}".encode())
            if not data_bytes:
                logger.info(f"Could not find SystemMetrics for {key} in RocksDB")

        unpacked = msgpack.unpackb(data_bytes, object_hook=mnp.decode)

        return SystemMetrics(
            r_1=unpacked["r_1"],
            r_2=unpacked["r_2"],
            m_1=unpacked["m_1"],
            m_2=unpacked["m_2"],
            s_1=unpacked["s_1"],
            s_2=unpacked["s_2"],
            q_1=unpacked["q_1"],
            q_2=unpacked["q_2"],
            f_1=unpacked["f_1"],
            f_2=unpacked["f_2"],
        )

    def add_result(
        self,
        params: tuple[int | float, ...] | np.ndarray,
        metrics: SystemMetrics,
        overwrite: bool = False,
    ) -> bool:
        np_params = np.array([params], dtype=np.float32)
        index, distance = self.find_faiss(np_params)
        if distance != 0.0:
            logger.info(f"Adding new Results for {pprint_key(params)}, starting Pipeline")
            np_params = np.array([params], dtype=np.float32)
            self.add_faiss(np_params)
            self.add_metric(metrics, None, str(self.current_idx))
            self.current_idx += 1
            return True
        if overwrite:
            logger.info(f"Adding new Results for {pprint_key(params)}, starting Pipeline")
            np_params = np.array([params], dtype=np.float32)
            self.add_metric(metrics, None, index=str(int(index[0][0])))
            return True

        logger.info("Will not overwrite")
        return False

    def read_result(
        self,
        params: tuple[int | float, ...] | np.ndarray,
        threshold=np.inf,
    ) -> None | SystemMetrics:
        logger.info(f"Getting Metrics for {pprint_key(params)}")
        np_params = np.array([params], dtype=np.float32)
        index, distance = self.find_faiss(np_params)
        if distance[0][0] <= threshold:
            logger.info(f"Using parameter-set with distance {distance[0][0]} <= threshold {threshold}")
            return self.read_metric(str(index[0][0]))
        logger.info(f"Will not use parameter-set with distance {distance[0][0]} <= threshold {threshold}")
        return None

    def read_multiple_results(
        self,
        params: np.ndarray,
        threshold=0.0,  # for multi read we dont want to miss
    ) -> None | SystemMetrics:
        logger.info(f"Getting Metrics for multiple queries with shape {params.shape}")

        # flatten for faiss lookup
        original_shape = params.shape[:-1]
        np_params = np.asarray(params, dtype=np.float32).reshape(-1, params.shape[-1])

        indices, distances = self.find_faiss(np_params)

        # back to original shape
        indices = indices.reshape(original_shape + (indices.shape[-1],))
        distances = distances.reshape(original_shape + (distances.shape[-1],))

        # function to be vectorized
        def fetch_metric(index, distance):
            if distance <= threshold:
                # other = Storage(self.key_dim, "", self.metrics_kv_name)
                return self.read_metric(str(index))
            return None

        vectorized_fetch = np.vectorize(fetch_metric, otypes=[object])
        results = vectorized_fetch(indices, distances)

        # Filter out None values and extract attributes into separate arrays
        valid_results = [r for r in results.flatten() if r is not None]

        if not valid_results:
            return None

        merged_metrics = SystemMetrics(
            r_1=np.stack([r.r_1 for r in valid_results]).reshape(original_shape + (-1,)),
            r_2=np.stack([r.r_2 for r in valid_results]).reshape(original_shape + (-1,)),
            m_1=np.stack([r.m_1 for r in valid_results]).reshape(original_shape + (-1,)),
            m_2=np.stack([r.m_2 for r in valid_results]).reshape(original_shape + (-1,)),
            s_1=np.stack([r.s_1 for r in valid_results]).reshape(original_shape + (-1,)),
            s_2=np.stack([r.s_2 for r in valid_results]).reshape(original_shape + (-1,)),
            q_1=np.stack([r.q_1 for r in valid_results]).reshape(original_shape + (-1,)),
            q_2=np.stack([r.q_2 for r in valid_results]).reshape(original_shape + (-1,)),
            f_1=np.stack([r.f_1 for r in valid_results]).reshape(original_shape + (-1,)),
            f_2=np.stack([r.f_2 for r in valid_results]).reshape(original_shape + (-1,)),
        )
        return merged_metrics

    def write(self):
        logger.info(f"Writing FAISS index to {self.parameter_k_name}")
        faiss.write_index(self.__db_keys, self.parameter_k_name)
        logger.info(f"Writing RocksDB to {self.metrics_kv_name}")
        if self.use_mem_cache:
            for index, data in self.__memory_cache.items():
                self.__db_metric.set(f"metrics_{index}".encode(), data)
        self.__db_metric.set(b"faiss_last_id", str(self.current_idx - 1).encode())
        logger.info("All in-memory data has been saved to RocksDB")

    def close(
        self,
    ):
        self.write()
        self.__db_metric.close()

    def merge(
        self,
        other: "Storage",
        overwrite: bool = False,
    ) -> "None | Storage":
        # Merges other into self
        if self.key_dim != other.key_dim:
            logger.error(f"Cannot merge Storages with different Key dimension (got {self.key_dim} and {other.key_dim})")
            return None
        if self.parameter_k_name == other.parameter_k_name or self.metrics_kv_name == other.metrics_kv_name:
            logger.error("Cannot merge Storages with same database files")
            return None
        logger.info("Sarting to merge")
        other_keys = other.__db_keys.reconstruct_n(0, other.__db_keys.ntotal)
        added = 0
        for key in other_keys:
            logger.info(f"Checking for other parameter-set {key}")
            metrics = other.read_result(key, threshold=0.0)
            if metrics:
                success = self.add_result(key, metrics, overwrite=overwrite)
                added += 1 if success else 0
        logger.info(f"Added {added} keys by merging")
        return self
