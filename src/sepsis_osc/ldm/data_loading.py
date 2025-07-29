import logging
from pathlib import Path

from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.preprocessor import PolarsRegressionPreprocessor
import polars as pl
from jaxtyping import Array, Float
import jax.numpy as jnp
import numpy as np
import jax.random as jr

from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.config import yaib_data_dir, sequence_files
from sepsis_osc.ldm.gin_configs import file_names, vars, modality_mapping
import os

logger = logging.getLogger(__name__)

data_dir = Path(yaib_data_dir)


def get_raw_data():
    data = preprocess_data(
        data_dir=data_dir,
        file_names=file_names,
        preprocessor=PolarsRegressionPreprocessor,
        seed=666,
        debug=False,
        generate_cache=True,
        load_cache=True,
        cv_repetitions=2,
        repetition_index=0,
        train_size=None,
        cv_folds=2,
        fold_index=0,
        pretrained_imputation_model=None,
        runmode=RunMode.regression,
        vars=vars,
        modality_mapping=modality_mapping,
        complete_train=False,
    )

    for si, s in data.items():
        for k, df in s.items():
            data[si][k] = data[si][k].sort(by=["stay_id", "time"])
    return data


def prepare_batches(
    x_data: Float[Array, "nsamples dim"],
    y_data: Float[Array, "nsamples dim"],
    batch_size: int,
    key: jnp.ndarray,
    perc: float = 1.0,
    shuffle=True,
) -> tuple[Float[Array, "nbatches batch dim"], Float[Array, "nbatches batch dim"], int]:
    num_samples = int(perc * x_data.shape[0])
    num_features = x_data.shape[1:]
    num_targets = y_data.shape[1:]

    # Shuffle data
    if shuffle:
        perm = jr.permutation(key, num_samples)
        x_shuffled = x_data[perm]
        y_shuffled = y_data[perm]
    else:
        x_shuffled = x_data[:num_samples]
        y_shuffled = y_data[:num_samples]

    # Ensure full batches only
    num_full_batches = num_samples // batch_size
    x_truncated = x_shuffled[: num_full_batches * batch_size]
    y_truncated = y_shuffled[: num_full_batches * batch_size]

    # Reshape into batches
    x_batched = x_truncated.reshape(num_full_batches, batch_size, *num_features)
    y_batched = y_truncated.reshape(num_full_batches, batch_size, *num_targets)

    return x_batched, y_batched, num_full_batches


def prepare_sequences(
    x_df: pl.DataFrame,
    y_df: pl.DataFrame,
    window_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Sort and group only once
    x_df = x_df.sort(["stay_id", "time"])
    y_df = y_df.sort(["stay_id", "time"])

    x_seqs = []
    y_seqs = []

    for pid in x_df["stay_id"].unique():
        x_group = x_df.filter(pl.col("stay_id") == pid).drop(["stay_id", "time"])
        y_group = y_df.filter(pl.col("stay_id") == pid).drop(["stay_id", "time"])

        x_np = x_group.to_numpy()
        y_np = y_group.to_numpy()
        n = len(x_np)

        if n < window_len:
            continue

        for i in range(n - window_len + 1):
            x_seqs.append(x_np[i : i + window_len])
            y_seqs.append(y_np[i : i + window_len])

    x_out = np.array(x_seqs)
    y_out = np.array(y_seqs)
    return x_out, y_out


def get_data_sets(
    window_len=6, dtype=jnp.float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(sequence_files + f"len_{window_len}.npz"):
        logger.info("Processed sequence files found. Loading data from disk...")
        loaded = np.load(sequence_files + f"len_{window_len}.npz")
        train_x, train_y = loaded["train_x"], loaded["train_y"]
        val_x, val_y = loaded["val_x"], loaded["val_y"]
        test_x, test_y = loaded["test_x"], loaded["test_y"]
        logger.info("Data loaded successfully.")
    else:
        logger.warning("Processed sequence files not found. Preparing sequences and saving data...")
        train_y, train_x, val_y, val_x, test_y, test_x = [
            v.drop([
                col
                for col in v.columns
                if col.startswith("Missing") or col in {"sep3_alt", "__index_level_0__", "los_icu"}
            ])
            for inner in get_raw_data().values()
            for v in inner.values()
        ]
        train_x, train_y = prepare_sequences(train_x, train_y, window_len)
        val_x, val_y = prepare_sequences(val_x, val_y, window_len)
        test_x, test_y = prepare_sequences(test_x, test_y, window_len)

        np.savez_compressed(
            sequence_files + f"len_{window_len}",
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
        )
        logger.info("Data prepared and saved sequences successfully.")

    return (
        train_x.astype(dtype),
        train_y.astype(dtype),
        val_x.astype(dtype),
        val_y.astype(dtype),
        test_x.astype(dtype),
        test_y.astype(dtype),
    )


if __name__ == "__main__":
    setup_logging()
    # print(os.listdir(data_dir))
    print(len(get_raw_data()["train"]["OUTCOME"]))
