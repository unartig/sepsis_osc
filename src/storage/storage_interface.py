import concurrent.futures
import logging
from functools import wraps
from time import time
from typing import Optional

import faiss
import msgpack
import msgpack_numpy as mnp
import numpy as np
import rocksdbpy as rock


from simulation.data_classes import SystemMetrics
from utils.config import db_metrics_key_value, db_parameter_keys, max_workers

mnp.patch()
logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("func:%r took: %2.6f sec" % (f.__name__, te - ts))
        return result

    return wrap


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
        self._starting_index = self.current_idx

        # Memory Cache for faster lookups
        self.use_mem_cache = use_mem_cache
        self.__memory_cache: dict[str, bytes] = {}
        if self.use_mem_cache:
            self.__setup_memory_cache()
            self.__new_indices = set()

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

    @timing
    def __setup_memory_cache(self):
        def insert_metric(kv):
            if kv:
                key, data_bytes = kv
                key = str(key.decode()).lstrip("metrics_")
                self.__memory_cache[key] = data_bytes

        # Split keysinto chunks for each thread
        logger.info("Preloading memory cache from RocksDB...")
        kv_pairs = [kv for kv in self.__db_metric.iterator() if kv[0].startswith(b"metrics_")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(insert_metric, kv_pairs)

        logger.info(f"Successfully loaded {len(self.__memory_cache)} metrics into memory.")

    def add_faiss(
        self,
        key: np.ndarray,
    ):
        logger.info(f"Adding key {pprint_key(key[0])} to FAISS index as [{self.current_idx}] ")
        self.__db_keys.add(key)

    def find_faiss(
        self,
        query_key: np.ndarray,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"Searching {pprint_key(query_key[0]) if query_key.shape[0] == 1 else str(len(query_key)) + ' keys'} in FAISS index"
        )
        distances, indices = self.__db_keys.search(query_key, k=k)
        logger.info(
            f"Found vectors with distance {distances[0][0] if query_key.size == self.key_dim else np.sum(distances)}, index {indices[0][0] if query_key.size == self.key_dim else (int(indices.min()), int(indices.max()))}"
        )
        return indices, distances

    def add_metric(
        self,
        metrics: Optional[SystemMetrics],
        packed_data: Optional[bytes],
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
                    "sr_1": np.asarray(metrics.sr_1),
                    "sr_2": np.asarray(metrics.sr_2),
                    "tt": np.asarray(metrics.tt),
                },
                default=mnp.encode,
            )

        if packed_data:
            if self.use_mem_cache:
                logger.info(f"Adding SystemMetrics with key {index} to MemoryCache")
                self.__memory_cache[index] = packed_data
                self.__new_indices.add(index)
            else:
                logger.info(f"Adding SystemMetrics with key {index} to RocksDB")
                self.__db_metric.set(f"metrics_{index}".encode(), packed_data)

    def read_metric(
        self,
        key: str,
    ) -> Optional[SystemMetrics]:
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
            sr_1=unpacked["sr_1"],
            sr_2=unpacked["sr_2"],
            tt=unpacked["tt"],
        )

    def add_result(
        self,
        params: tuple[int | float, ...] | np.ndarray,
        metrics: SystemMetrics,
        overwrite: bool = False,
    ) -> bool:
        single_metrics = metrics.copy().as_single()
        np_params = np.array([params], dtype=np.float32)
        index, distance = self.find_faiss(np_params)
        if distance != 0.0:
            logger.info(f"Adding new Results for {pprint_key(params)}, starting Pipeline")
            np_params = np.array([params], dtype=np.float32)
            self.add_faiss(np_params)
            self.add_metric(single_metrics, None, str(self.current_idx))
            self.current_idx += 1
            return True
        if overwrite:
            logger.info(f"Adding new Results for {pprint_key(params)}, starting Pipeline")
            np_params = np.array([params], dtype=np.float32)
            self.add_metric(single_metrics, None, index=str(int(index[0][0])))
            return True

        logger.info("Will not overwrite")
        return False

    def read_result(
        self,
        params: tuple[int | float, ...] | np.ndarray,
        threshold=np.inf,
    ) -> Optional[SystemMetrics]:
        logger.info(f"Getting Metrics for {pprint_key(params)}")
        np_params = np.array([params], dtype=np.float32)
        index, distance = self.find_faiss(np_params)
        if distance[0][0] <= threshold:
            logger.info(f"Using parameter-set with distance {distance[0][0]} <= threshold {threshold}")
            return self.read_metric(str(index[0][0]))
        logger.info(f"Will not use parameter-set with distance {distance[0][0]} <= threshold {threshold}")
        return None

    @timing
    def read_multiple_results(
        self,
        params: np.ndarray,
    ) -> Optional[SystemMetrics]:
        logger.info(f"Getting Metrics for multiple queries with shape {params.shape}")

        original_shape = params.shape[:-1]
        np_params = np.asarray(params, dtype=np.float32).reshape(-1, params.shape[-1])

        indices, distances = self.find_faiss(np_params)
        indices = indices.reshape(original_shape)

        if np.any(distances != 0.0):
            logger.error("Could not match bulk query")
            return None

        inds = np.unravel_index(np.arange(np.prod(original_shape)), original_shape)
        inds = list(zip(*inds))

        metrics_shape = original_shape + (1,)
        res = SystemMetrics(
            r_1=np.empty(metrics_shape, dtype=np.float32),
            r_2=np.empty(metrics_shape, dtype=np.float32),
            m_1=np.empty(metrics_shape, dtype=np.float32),
            m_2=np.empty(metrics_shape, dtype=np.float32),
            s_1=np.empty(metrics_shape, dtype=np.float32),
            s_2=np.empty(metrics_shape, dtype=np.float32),
            q_1=np.empty(metrics_shape, dtype=np.float32),
            q_2=np.empty(metrics_shape, dtype=np.float32),
            f_1=np.empty(metrics_shape, dtype=np.float32),
            f_2=np.empty(metrics_shape, dtype=np.float32),
            sr_1=np.empty(metrics_shape, dtype=np.float32),
            sr_2=np.empty(metrics_shape, dtype=np.float32),
            tt=np.empty(metrics_shape, dtype=np.float32),
        )

        # batch read the data from RocksDB or Cache
        def fetch_and_unpack_bulk(ind_chunk: list[tuple[int, ...]]) -> bool:
            if self.use_mem_cache:
                raw_list = [self.__memory_cache[str(ind)] for ind in ind_chunk]
            else:
                # RocksDB multi_get returns a list of raw bytes
                keys = [f"metrics_{str(indices[ind])}".encode() for ind in ind_chunk]
                raw_list = self.__db_metric.multi_get(keys)

            for ind, raw in zip(ind_chunk, raw_list):
                if not raw:
                    logger.error(f"Failed to fetch data for index {ind}")
                    return False

                unpacked = msgpack.unpackb(raw, object_hook=mnp.decode)

                res.r_1[ind] = unpacked["r_1"]
                res.r_2[ind] = unpacked["r_2"]
                res.m_1[ind] = unpacked["m_1"]
                res.m_2[ind] = unpacked["m_2"]
                res.s_1[ind] = unpacked["s_1"]
                res.s_2[ind] = unpacked["s_2"]
                res.q_1[ind] = unpacked["q_1"]
                res.q_2[ind] = unpacked["q_2"]
                res.f_1[ind] = unpacked["f_1"]
                res.f_2[ind] = unpacked["f_2"]
                if res.sr_1 is not None and res.sr_2 is not None and res.tt is not None:
                    res.sr_1[ind] = unpacked["sr_1"]
                    res.sr_2[ind] = unpacked["sr_2"]
                    res.tt[ind] = unpacked["tt"]
            return True

        # Split indices into chunks for each thread
        chunk_size = len(inds) // max_workers + 1
        chunks = [inds[i : i + chunk_size] for i in range(0, len(inds), chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            valid_results = list(executor.map(fetch_and_unpack_bulk, chunks))

        if not any(valid_results):
            logger.error("Retrieved invalid metrics")
            return None

        logger.info("Successfuly fetched bulk metrics")
        return res

    def write(self):
        logger.info(f"Writing FAISS index to {self.parameter_k_name}")
        faiss.write_index(self.__db_keys, self.parameter_k_name)
        logger.info(f"Writing RocksDB to {self.metrics_kv_name}")
        if self.use_mem_cache:
            for index in self.__new_indices:
                self.__db_metric.set(f"metrics_{index}".encode(), self.__memory_cache[index])
            self.__new_indices = set()
        self.__db_metric.set(b"faiss_last_id", str(self.current_idx).encode())
        logger.info("All in-memory data has been saved to RocksDB")

    def close(
        self,
    ):
        if self.use_mem_cache and len(self.__new_indices) != 0:
            self.write()
        self.__db_metric.close()

    def merge(
        self,
        other: "Storage",
        overwrite: bool = False,
    ) -> "Optional[Storage]":
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
            logger.info(f"Checking for other parameter-set {pprint_key(key)}")
            metrics = other.read_result(key, threshold=0.0)
            if metrics:
                success = self.add_result(key, metrics, overwrite=overwrite)
                added += 1 if success else 0
        logger.info(f"Added {added} keys by merging")
        return self
