import copy
import logging
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import polars as pl
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.preprocessor import PolarsRegressionPreprocessor
from icu_benchmarks.data.split_process_data import preprocess_data
from tqdm import trange

from sepsis_osc.ldm.gin_configs import file_names, modality_mapping, new_vars
from sepsis_osc.utils.config import cohort_name, np_rng, sequence_files, target_name, yaib_data_dir
from sepsis_osc.utils.logger import setup_logging

logger = logging.getLogger(__name__)

data_dir = Path(yaib_data_dir)

vars_copy = new_vars.copy()


def get_raw_data(
    _data_dir: Path = data_dir, yaib_vars: dict[str, str | list[str]] = vars_copy
) -> dict[str, pl.DataFrame]:
    yaib_vars = copy.deepcopy(yaib_vars)
    yaib_vars["LABEL"][0] = target_name
    logger.info(f"Searching for sequence_files in {_data_dir}")
    data = preprocess_data(
        data_dir=_data_dir,
        file_names=file_names,
        preprocessor=PolarsRegressionPreprocessor,
        seed=666,
        debug=False,
        generate_cache=True,
        load_cache=True,
        cv_repetitions=2,
        repetition_index=0,
        train_size=0.75,
        cv_folds=2,
        fold_index=0,
        pretrained_imputation_model=None,
        runmode=RunMode.classification,
        vars=yaib_vars,
        modality_mapping=modality_mapping,
        complete_train=True,
        # label=target_name
    )

    for si, s in data.items():
        for k in s:
            by = []
            if "stay_id" in s[k].columns:
                by.append("stay_id")
            if "time" in s[k].columns:
                by.append("time")
            data[si][k] = s[k].sort(by=by)
    return data


def prepare_batches(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
    key: jnp.ndarray,
    perc: float = 1.0,
    *,
    shuffle: bool = True,
    pos_fraction: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    num_samples = int(perc * x_data.shape[0])
    pos_mask = y_data.sum(axis=-1) > 0
    pos_idx, neg_idx = np.where(pos_mask)[0], np.where(~pos_mask)[0]

    if pos_fraction == -1.0:
        idx = jr.permutation(key, num_samples) if shuffle else np.arange(num_samples)
    else:
        n_pos = int(num_samples * pos_fraction)
        n_neg = num_samples - n_pos
        pos_idx = np_rng.choice(pos_idx, n_pos, replace=True)
        neg_idx = np_rng.choice(neg_idx, n_neg, replace=n_neg > len(neg_idx))
        idx = np.concatenate([pos_idx, neg_idx])
        if shuffle:
            np_rng.shuffle(idx)

    x, y = x_data[idx], y_data[idx]
    n_batches = x.shape[0] // batch_size
    x = x[: n_batches * batch_size].reshape(n_batches, batch_size, *x.shape[1:])
    y = y[: n_batches * batch_size].reshape(n_batches, batch_size, *y.shape[1:])
    return x, y, n_batches


def prepare_sequences(
    x_df: pl.DataFrame, y_df: pl.DataFrame, window_len: int, time_step: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    m_index = ["stay_id", "time"]

    cols = x_df.columns
    front = sorted([c for c in cols if not c.startswith("MissingIndicator_") and c not in m_index])
    back = sorted([c for c in cols if c.startswith("MissingIndicator_") and c not in m_index])
    x_df = x_df[m_index + front + back]

    # Sort and group only once
    x_df = x_df.sort(m_index)
    y_df = y_df.sort(m_index)

    x_seqs = []
    y_seqs = []

    uniques = x_df["stay_id"].unique()
    for i in trange(len(uniques)):
        sid = uniques[i]
        times = x_df.filter(pl.col("stay_id") == sid)["time"].to_numpy()
        x_group = x_df.filter(pl.col("stay_id") == sid).drop(m_index)
        y_group = y_df.filter(pl.col("stay_id") == sid).drop(m_index)

        x_np = x_group.to_numpy()
        y_np = y_group.to_numpy()
        n = len(x_np)

        if n < window_len:
            continue

        # only fully consecutive sequences
        for j in range(n - window_len + 1):
            if np.all(np.diff(times[j : j + window_len]) == time_step):
                x_seqs.append(x_np[j : j + window_len])
                y_seqs.append(y_np[j : j + window_len])

    x_out = np.array(x_seqs)
    y_out = np.array(y_seqs)
    return x_out, y_out


def get_data_sets_prediction(
    window_len: int = 6,
    dtype: jnp.dtype = jnp.float32,
    swapaxes_y: tuple[int, int, int] = (0, 1, 2),
    *,
    full: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_name = f"len_{window_len}_{cohort_name}{'_full' if full else ''}"
    if Path(sequence_files + f"{file_name}.npz").exists():
        logger.info("Processed sequence files found. Loading data from disk...")
        loaded = np.load(sequence_files + f"{file_name}.npz")
        train_x, train_y = loaded["train_x"], loaded["train_y"]
        val_x, val_y = loaded["val_x"], loaded["val_y"]
        test_x, test_y = loaded["test_x"], loaded["test_y"]
        logger.info("Data loaded successfully.")
    else:
        logger.warning("Processed sequence files not found, reading YAIB-data.")
        train_y, train_x, val_y, val_x, test_y, test_x = [
            v.drop(
                [
                    col
                    for col in v.columns
                    if col.startswith("Missing" if not full else "WWW")
                    or col in {"__index_level_0__", "los_icu", "susp_inf_alt", "sep3_alt", "yaib_label"}
                ]
            )
            for inner in get_raw_data().values()
            for v in inner.values()
        ]
        logger.info("Preparing sequences and saving data...")
        train_x, train_y = prepare_sequences(train_x, train_y, window_len)
        val_x, val_y = prepare_sequences(val_x, val_y, window_len)
        test_x, test_y = prepare_sequences(test_x, test_y, window_len)

        np.savez_compressed(
            sequence_files + f"{file_name}",
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
        )
        logger.info("Data prepared and saved sequences successfully.")

    # reorder for (sofa, susp_inf_ramp, sep3_alt)
    return (
        train_x.astype(dtype),
        train_y.astype(dtype)[..., swapaxes_y],
        val_x.astype(dtype),
        val_y.astype(dtype)[..., swapaxes_y],
        test_x.astype(dtype),
        test_y.astype(dtype)[..., swapaxes_y],
    )


def get_all(data: PredictionPolarsDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xl, yl, ml = [], [], []
    for i in trange(data.num_stays):
        xs, ys, ms = data[i]
        xl.append(xs)
        yl.append(ys)
        ml.append(ms)
    return np.asarray(xl), np.asarray(yl), np.asarray(ml)


def get_data_sets_detection(
    dtype: jnp.dtype = jnp.float32, swapaxes_y: tuple[int, int, int] = (0, 1, 2), *, use_yaib_label: bool = True
) -> tuple[np.ndarray, ...]:
    file_name = f"{'yaib_' if use_yaib_label else ''}{cohort_name}_detection"
    if Path(sequence_files + f"{file_name}.npz").exists():
        logger.info("Processed sequence files found. Loading data from disk...")
        loaded = np.load(sequence_files + f"{file_name}.npz")
        train_x, train_y, train_m = loaded["train_x"], loaded["train_y"], loaded["train_m"]
        val_x, val_y, val_m = loaded["val_x"], loaded["val_y"], loaded["val_m"]
        test_x, test_y, test_m = loaded["test_x"], loaded["test_y"], loaded["test_m"]
        logger.info("Data loaded successfully.")
    else:
        logger.warning("Processed sequence files not found, reading YAIB-data.")
        detect_vars = copy.deepcopy(vars_copy)
        label = "yaib_label" if use_yaib_label else target_name
        detect_vars["LABEL"] = [label, "sofa", "susp_inf_ramp"]
        data = get_raw_data(yaib_vars=detect_vars)
        (
            train_x,
            train_y,
            train_m,
        ) = get_all(
            PredictionPolarsDataset(data, split="train", vars=detect_vars, ram_cache=False, name=f"{cohort_name}_train")
        )
        (
            val_x,
            val_y,
            val_m,
        ) = get_all(
            PredictionPolarsDataset(data, split="val", vars=detect_vars, ram_cache=False, name=f"{cohort_name}_val")
        )
        (
            test_x,
            test_y,
            test_m,
        ) = get_all(
            PredictionPolarsDataset(data, split="test", vars=detect_vars, ram_cache=False, name=f"{cohort_name}_test")
        )
        np.savez_compressed(
            sequence_files + f"{file_name}",
            train_x=train_x,
            train_y=train_y,
            train_m=train_m,
            val_x=val_x,
            val_y=val_y,
            val_m=val_m,
            test_x=test_x,
            test_y=test_y,
            test_m=test_m,
        )
        logger.info("Data prepared and saved sequences successfully.")

    # reorder for (sofa, susp_inf_ramp, sep3_alt)
    return (
        train_x.astype(dtype),
        train_y.astype(dtype)[..., swapaxes_y],
        train_m.astype(dtype),
        val_x.astype(dtype),
        val_y.astype(dtype)[..., swapaxes_y],
        val_m.astype(dtype),
        test_x.astype(dtype),
        test_y.astype(dtype)[..., swapaxes_y],
        test_m.astype(dtype),
    )


def prepare_batches_mask(
    x_data: np.ndarray,
    y_data: np.ndarray,
    mask_data: np.ndarray,
    batch_size: int,
    key: jnp.ndarray,
    perc: float = 1.0,
    *,
    shuffle: bool = True,
    pos_fraction: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    num_samples = int(perc * x_data.shape[0])
    pos_mask = y_data.sum(axis=-1) > 0
    pos_idx, neg_idx = np.where(pos_mask)[0], np.where(~pos_mask)[0]

    if pos_fraction == -1.0:
        idx = jr.permutation(key, num_samples) if shuffle else np.arange(num_samples)
    else:
        n_pos = int(num_samples * pos_fraction)
        n_neg = num_samples - n_pos
        pos_idx = np_rng.choice(pos_idx, n_pos, replace=True)
        neg_idx = np_rng.choice(neg_idx, n_neg, replace=n_neg > len(neg_idx))
        idx = np.concatenate([pos_idx, neg_idx])
        if shuffle:
            np_rng.shuffle(idx)

    x, y, m = x_data[idx], y_data[idx], mask_data[idx]
    n_batches = x.shape[0] // batch_size
    x = x[: n_batches * batch_size].reshape(n_batches, batch_size, *x.shape[1:])
    y = y[: n_batches * batch_size].reshape(n_batches, batch_size, *y.shape[1:])
    m = m[: n_batches * batch_size].reshape(n_batches, batch_size, *m.shape[1:])
    return x, y, m, n_batches


if __name__ == "__main__":
    setup_logging()
    # print(os.listdir(data_dir))
    # print(len(get_raw_data()["train"]["OUTCOME"]))
    # tx, ty, tm, vx, vy, vm, _x, _y, _m = get_data_sets_detection()
    (
        t_x,
        t_y,
        t_m,
        v_x,
        v_y,
        v_m,
        _x,
        _y,
        _m,
    ) = get_data_sets_detection(swapaxes_y=(1, 2, 0), dtype=jnp.float32)
    print(t_y[1, :, 2])
