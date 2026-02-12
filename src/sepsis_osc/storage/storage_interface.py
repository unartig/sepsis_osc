import concurrent.futures
import logging
from typing import Optional

import faiss
import numpy as np
import rocksdbpy as rock

from sepsis_osc.dnm.abstract_ode import MetricBase, MetricT
from sepsis_osc.utils.config import db_metrics_key_value, db_parameter_keys, max_workers
from sepsis_osc.utils.utils import timing

logger = logging.getLogger(__name__)


def pprint_key(key: list[float] | tuple[float, ...] | np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(x), 4) for x in key)


class Storage:
    def __init__(
        self,
        key_dim: int = 9,
        parameter_k_name: str = "",
        metrics_kv_name: str = "",
        *,
        use_mem_cache: bool = True,
    ) -> None:
        parameter_k_name = parameter_k_name or db_parameter_keys
        metrics_kv_name = metrics_kv_name or db_metrics_key_value
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
    ) -> faiss.IndexFlatL2:
        key_index = faiss.IndexFlatL2(self.key_dim)  # L2 (Euclidean) distance
        try:
            key_index = faiss.read_index(self.parameter_k_name)
            logger.info(f"FAISS index {db_name} loaded from disk.")
        except Exception as _:
            logger.info("No existing FAISS index, starting fresh.")
        return key_index

    def __setup_rocksdb(self, db_name: str) -> tuple[rock.RocksDB, int]:
        opts = rock.Option()
        opts.create_if_missing(create_if_missing=True)
        opts.set_allow_mmap_reads(True)

        rocksdb = rock.open(db_name, opts=opts)
        last_id_bytes = rocksdb.get(b"faiss_last_id")
        current_idx = int(last_id_bytes.decode()) if last_id_bytes else 0
        return rocksdb, current_idx

    @timing
    def __setup_memory_cache(self) -> None:
        def insert_metric(kv) -> None:
            if kv:
                key, data_bytes = kv
                key = str(key.decode()).lstrip("metrics_")
                self.__memory_cache[key] = data_bytes

        logger.info("Preloading memory cache from RocksDB...")
        kv_pairs = [kv for kv in self.__db_metric.iterator() if kv[0].startswith(b"metrics_")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(insert_metric, kv_pairs)

        logger.info(f"Successfully loaded {len(self.__memory_cache)} metrics into memory.")

    def add_faiss(
        self,
        key: np.ndarray,
    ) -> None:
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
        packed_data: bytes | None,
        index: str = "",
    ) -> None:
        if not index:
            logger.error("Cannot save empty key in MemoryCache")
            return

        if packed_data:
            if self.use_mem_cache:
                logger.info(f"Adding SystemMetrics with key {index} to MemoryCache")
                self.__memory_cache[index] = packed_data
                self.__new_indices.add(index)
            else:
                logger.info(f"Adding SystemMetrics with key {index} to RocksDB")
                self.__db_metric.set(f"metrics_{index}".encode(), packed_data)
        else:
            logger.error("Something went wrong, no packed data...")

    def read_metric(
        self,
        key: str,
    ) -> bytes | None:
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
        return data_bytes

    def add_result(
        self,
        params: tuple[int | float, ...] | np.ndarray,
        packed_metric: bytes,
        *,
        overwrite: bool = False,
    ) -> bool:
        np_params = np.array([params], dtype=np.float32)
        index, distance = self.find_faiss(np_params)
        if distance != 0.0:
            logger.info(f"Adding new Results for {pprint_key(params)}, starting Pipeline")
            np_params = np.array([params], dtype=np.float32)
            self.add_faiss(np_params)
            self.add_metric(packed_metric, index=str(self.current_idx))
            self.current_idx += 1
            return True
        if overwrite:
            logger.info(f"Adding new Results for {pprint_key(params)}, starting Pipeline")
            np_params = np.array([params], dtype=np.float32)
            self.add_metric(packed_metric, index=str(int(index[0][0])))
            return True

        logger.info("Will not overwrite")
        return False

    def read_result(
        self,
        params: tuple[int | float, ...] | np.ndarray,
        proto_metric: MetricT,
        threshold: np.float32 = np.float32(np.inf),
    ) -> MetricT | None:
        logger.info(f"Getting Metrics for {pprint_key(params)}")
        np_params = np.array([params], dtype=np.float32)
        index, distance = self.find_faiss(np_params)
        if distance[0][0] <= threshold:
            logger.info(f"Using parameter-set with distance {distance[0][0]} <= threshold {threshold}")
            return proto_metric.deserialise(self.read_metric(str(index[0][0])))
        logger.info(f"Will not use parameter-set with distance {distance[0][0]} <= threshold {threshold}")
        return None

    @timing
    def read_multiple_results(self, params: np.ndarray, proto_metric: MetricT, threshold: float = 0.0) -> tuple[MetricBase, np.ndarray] | tuple[None, None]:
        logger.info(f"Getting Metrics for multiple queries with shape {params.shape}")

        original_shape = params.shape[:-1]
        np_params = np.asarray(params, dtype=np.float32).reshape(-1, params.shape[-1])
        indices, distances = self.find_faiss(np_params, k=1)
        indices = indices.reshape(original_shape)
        distances = distances.reshape(original_shape)

        if np.any(distances.max() > threshold):
            logger.error("Could not match bulk query")
            return None, None

        inds = np.unravel_index(np.arange(np.prod(original_shape)), original_shape)
        inds = list(zip(*inds))

        metrics_shape = (*original_shape, 1)
        res = proto_metric.np_empty(metrics_shape)

        def _get_value_or_key(ind: tuple[int, ...]) -> bytes:
            index = int(indices[ind] if len(ind) else indices[ind[0]])
            if self.use_mem_cache:
                return self.__memory_cache[str(index)]
            else:
                return f"metrics_{index}".encode()

        # batch read the data from RocksDB or Cache
        def fetch_and_unpack_bulk(ind_chunk: list[tuple[int, ...]]) -> bool:
            if self.use_mem_cache:
                raw_list = [_get_value_or_key(ind) for ind in ind_chunk]
            else:
                keys = [_get_value_or_key(ind) for ind in ind_chunk]
                raw_list = self.__db_metric.multi_get(keys)

            for ind, raw in zip(ind_chunk, raw_list, strict=True):
                if not raw:
                    logger.error(f"Failed to fetch data for index {ind}")
                    return False

                metric = proto_metric.deserialise(raw)
                res.insert_at(ind, metric)
            return True

        # Split indices into chunks for each thread
        chunk_size = len(inds) // max_workers + 1
        chunks = [inds[i : i + chunk_size] for i in range(0, len(inds), chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            valid_results = list(executor.map(fetch_and_unpack_bulk, chunks))

        if not any(valid_results):
            logger.error("Retrieved invalid metrics")
            return None, None

        logger.info("Successfuly retrieved bulk metrics")
        return res, distances

    def write(self) -> None:
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
    ) -> None:
        if self.use_mem_cache and len(self.__new_indices) != 0:
            self.write()
        self.__db_metric.close()

    def merge(
        self,
        other: "Storage",
        proto_metric: MetricT,
        *,
        overwrite: bool = False,
    ) -> "Storage | None":
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
            metrics = other.read_result(key, proto_metric, threshold=np.float32(0.0))
            if metrics:
                success = self.add_result(key, metrics.serialise(), overwrite=overwrite)
                added += 1 if success else 0
        logger.info(f"Added {added} keys by merging")
        return self

