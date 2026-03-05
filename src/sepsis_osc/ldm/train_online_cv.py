import gc
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from jax._src import dispatch
from jaxtyping import PRNGKeyArray, PyTree
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorboardX import SummaryWriter

from sepsis_osc.ldm.calibration_model import TemperatureScaling
from sepsis_osc.ldm.checkpoint_utils import load_checkpoint, save_checkpoint
from sepsis_osc.ldm.commons import build_lookup_table, custom_warmup_cosine
from sepsis_osc.ldm.data_loading import get_data_sets_online, prepare_batches_mask
from sepsis_osc.ldm.early_stopping import EarlyStopping
from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel, init_ldm_weights
from sepsis_osc.ldm.logging_utils import flatten_dict, log_train_metrics, log_val_metrics
from sepsis_osc.ldm.lookup import LatentLookup
from sepsis_osc.ldm.model_structs import LoadingConfig, LossesConfig, LRConfig, SaveConfig, TrainingConfig
from sepsis_osc.ldm.train_online import process_train_epoch, process_val_epoch
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging


@dataclass
class ExperimentConfig:
    train: TrainingConfig
    lr: LRConfig
    loss: LossesConfig
    load: LoadingConfig
    save: SaveConfig
    dtype: jnp.dtype = jnp.float32


def load_fold_data(
    dtype: jnp.dtype, cv_rep: int, rep_idx: int, cv_folds: int, fold_idx: int
) -> dict[str, tuple[np.ndarray, ...]]:
    data = get_data_sets_online(
        swapaxes_y=(1, 2, 0),
        dtype=dtype,
        cv_repetitions=cv_rep,
        repetition_index=rep_idx,
        cv_folds=cv_folds,
        fold_index=fold_idx,
        sequence_files="data/cv/sequence_",
    )

    train_x, train_y, train_m, val_x, val_y, val_m, *_ = data
    train_m = train_m.astype(np.bool)
    val_m = val_m.astype(np.bool)

    return {
        "train": (train_x, train_y, train_m),
        "val": (val_x, val_y, val_m),
    }


def init_model_and_optimizer(
    config: ExperimentConfig,
    train_y: np.ndarray,
    train_m: np.ndarray,
    key: PRNGKeyArray,
) -> tuple:
    counts = np.bincount(train_y[train_m][..., 0].astype(int), minlength=24)
    logger.error(counts)
    sofa_dist = counts / counts.sum()

    key_model, key_weights = jr.split(key)

    model = LatentDynamicsModel(
        key=key_model,
        input_dim=52,
        latent_enc_hidden_dim=64,
        latent_hidden_dim=8,
        latent_dim=2,
        inf_hidden_dim=64,
        inf_dim=1,
        dec_hidden_dim=32,
        lookup_kernel_size=7,
    )

    model = init_ldm_weights(model, key_weights, scale=1e-4)

    steps_per_epoch = config.train.batch_size
    schedule = custom_warmup_cosine(
        init_value=config.lr.init,
        peak_value=config.lr.peak,
        warmup_steps=config.lr.warmup_epochs * steps_per_epoch,
        cycles=[(steps_per_epoch * 200, 1.0)],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.lr.enc_wd),
    )

    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    return model, static, optimizer, opt_state, schedule, jnp.asarray(sofa_dist)


def train_one_run(
    config: ExperimentConfig,
    model: LatentDynamicsModel,
    static_model: PyTree,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    lr_schedule: Callable,
    lookup_table: LatentLookup,
    train_data: tuple[np.ndarray, ...],
    val_data: tuple[np.ndarray, ...],
    key: PRNGKeyArray,
    writer: SummaryWriter,
) -> tuple[LatentDynamicsModel, dict[str, float]]:
    hyper_ldm = model.hypers_dict()
    hyper_params = flatten_dict(
        {
            "model": hyper_ldm,
            "losses": asdict(loss_conf),
            "train": asdict(train_conf),
            "lr": asdict(lr_conf),
        }
    )
    writer.add_hparams(hyper_params, metric_dict={}, name="")

    train_x, train_y, train_m = train_data
    val_x, val_y, val_m = val_data

    del train_data, val_data

    auroc, auprc = 0, 0

    filter_spec = jtu.tree_map(lambda _: eqx.is_inexact_array, model)

    stop = False
    early_stop_auprc = EarlyStopping(direction=1, patience=10, min_steps=20)
    early_stop_auroc = EarlyStopping(direction=1, patience=10, min_steps=20)
    for epoch in range(config.train.epochs):
        # === TRAIN ===
        if epoch > 0:
            shuffle_key = jr.fold_in(key, epoch)
            x_shuffled, y_shuffled, m_shuffled, ntrain_batches = prepare_batches_mask(
                train_x,
                train_y,
                train_m,
                key=shuffle_key,
                batch_size=train_conf.batch_size,
                mini_epochs=train_conf.mini_epochs,
                pos_fraction=-1,
            )
            config.loss.steps_per_epoch = ntrain_batches
            train_metrics = []
            for mini_epoch in range(train_conf.mini_epochs):
                params_model, opt_state, mini_train_metrics, key = process_train_epoch(
                    model,
                    opt_state=opt_state,
                    x_data=x_shuffled[mini_epoch],
                    y_data=y_shuffled[mini_epoch],
                    mask_data=m_shuffled[mini_epoch],
                    update=optimizer.update,
                    key=key,
                    lookup_func=lookup_table.soft_get_local,
                    loss_params=loss_conf,
                    param_filter=filter_spec,
                )
                train_metrics.append(mini_train_metrics)
                model = eqx.combine(params_model, static_model)
            train_metrics = jtu.tree_map(lambda *xs: jnp.stack(xs), *train_metrics)
            log_msg = log_train_metrics(train_metrics.to_np(), model.params_to_dict(), epoch, writer, mask=m_shuffled)
            writer.add_scalar(
                "lr/learning_rate",
                np.asarray(lr_schedule(jnp.asarray(epoch * train_x.shape[0]).astype(jnp.int32))),
                epoch,
            )
            logger.info(log_msg)
            del shuffle_key, train_metrics, mini_train_metrics, params_model

        # === VALIDATE ===
        if epoch % train_conf.validate_every == 0:
            val_model = eqx.nn.inference_mode(model, value=True)
            val_key = jr.fold_in(key, epoch + 2)
            val_metrics = process_val_epoch(
                val_model,
                x_data=val_x[None],
                y_data=val_y[None],
                mask_data=val_m[None],
                step=opt_state[1][0].count,
                key=val_key,
                lookup_func=lookup_table.hard_get_fsq,
                loss_params=loss_conf,
            )

            if train_conf.calibrate and epoch > 0:
                cal = TemperatureScaling().calibrate(
                    train_metrics.sofa_d2_p[m_shuffled],
                    train_metrics.susp_inf_p[m_shuffled],
                    y_shuffled[m_shuffled, 2],
                )
                val_metrics.sep3_risk = np.zeros_like(val_metrics.sep3_risk)
                val_metrics.sep3_risk[val_m[None]] = cal.get_sepsis_probs(
                    val_metrics.sofa_d2_risk[val_m[None]], val_metrics.susp_inf_p[val_m[None]]
                )
                logger.error(cal.coeffs)

            if train_conf.early_stop:
                auprc = average_precision_score((val_y[val_m, 2] == 1.0), val_metrics.to_np().sep3_risk[val_m[None]])
                auroc = roc_auc_score((val_y[val_m, 2] == 1.0), val_metrics.to_np().sep3_risk[val_m[None]])
                auprc_stop = early_stop_auprc.step(auprc)
                auroc_stop = early_stop_auroc.step(auroc)
                stop = auprc_stop & auroc_stop
                logger.info(
                    f"Bad Steps|Stopped: AUROC {early_stop_auroc.bad_steps}|{early_stop_auroc.stopped} and "
                    f"AUPRC {early_stop_auprc.bad_steps}|{early_stop_auprc.stopped}"
                )

            log_msg = log_val_metrics(val_metrics.to_np(), val_y[None], lookup_table, epoch, writer, mask=val_m[None])
            logger.warning(log_msg)

            model = eqx.nn.inference_mode(val_model, value=False)

        if epoch > 0:
            del x_shuffled, y_shuffled, m_shuffled, val_key, val_metrics, val_model
            gc.collect()

            # --- Save checkpoint ---
        if ((epoch + 1) % save_conf.save_every == 0 or stop) and save_conf.perform:
            save_dir = writer.logdir if not load_conf.from_dir else load_conf.from_dir
            save_checkpoint(
                save_dir + "/checkpoints",
                epoch,
                model,
                opt_state,
                hyper_ldm,
            )

        if stop:
            break

    return model, {"auprc": auprc, "auroc": auroc}


def run_cross_validation(config: ExperimentConfig, n_folds: int = 5, repetitions: int = 1) -> list:
    results = []

    storage = Storage(
        key_dim=9,
        metrics_kv_name="data/DaisyFinalSepsisMetrics.db/",
        parameter_k_name="data/DaisyFinalSepsisParameters_index.bin",
        use_mem_cache=True,
    )

    lookup_table = build_lookup_table(storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)

    for rep in range(repetitions):
        for fold in range(n_folds):
            writer = SummaryWriter(logdir=f"runs/cv3/rep{rep:002}_fold{fold:002}")
            if Path(Path(writer.logdir) / "checkpoints").exists():
                logger.warning(f"Skipping {writer.logdir}")
                continue

            data = load_fold_data(config.dtype, repetitions, rep, n_folds, fold)

            key = jr.PRNGKey(jax_random_seed + rep * 100 + fold)

            model, static, optimizer, opt_state, schedule, sofa_dist = init_model_and_optimizer(
                config,
                data["train"][1],
                data["train"][2],
                key,
            )

            config.loss.sofa_dist = sofa_dist

            model, metrics = train_one_run(
                config,
                model,
                static,
                optimizer,
                opt_state,
                schedule,
                lookup_table,
                data["train"],
                data["val"],
                key,
                writer,
            )

            results.append(metrics)
            writer.close()

            del model, static, optimizer, opt_state, schedule, sofa_dist, data, writer
            gc.collect()
            eqx.clear_caches()
            jax.clear_caches()
            # jax.clear_backends()

    return results


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info")
    dtype = jnp.float32
    logger = logging.getLogger(__name__)

    train_conf = TrainingConfig(
        batch_size=64,
        epochs=int(1e3),
        mini_epochs=4,
        validate_every=1,
        calibrate=False,
        early_stop=True,
    )
    lr_conf = LRConfig(
        init=0.0,
        warmup_epochs=1,
        peak=5e-5,
        enc_wd=2e-1,
    )
    loss_conf = LossesConfig(
        lambda_sep3=600.0,
        lambda_inf=1.0,
        lambda_sofa_classification=2000.0,
        lambda_spreading=6e-3,
        lambda_boundary=30.0,
        lambda_recon=2.5,
    )
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=1, perform=True)

    config = ExperimentConfig(
        train=train_conf,
        lr=lr_conf,
        loss=loss_conf,
        load=load_conf,
        save=save_conf,
    )

    results = run_cross_validation(config, n_folds=5, repetitions=5)
    print(results)
