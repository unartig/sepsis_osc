import logging
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import optuna
from optuna.trial import Trial
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import LatentEncoder, Decoder, InfectionEncoder, init_decoder_weights, init_latent_encoder_weights
from sepsis_osc.ldm.data_loading import get_data_sets_offline, prepare_batches
from sepsis_osc.ldm.early_stopping import EarlyStopping
from sepsis_osc.ldm.gru import GRUPredictor, init_gru_weights
from sepsis_osc.ldm.helper_structs import (
    LossesConfig,
    LRConfig,
    ModelConfig,
    TrainingConfig,
)
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel
from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
from sepsis_osc.ldm.train import (
    ALPHA,
    BETA_SPACE,
    SIGMA_SPACE,
    custom_warmup_cosine,
    process_train_epoch,
    process_val_epoch,
)
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging


def empty():
    return None

def make_objective(
    train_x_inner: np.ndarray,
    train_y_inner: np.ndarray,
    val_x_inner: np.ndarray,
    val_y_inner: np.ndarray,
    sofa_dist: jnp.ndarray,
    deltas: jnp.ndarray,
) -> Callable:
    def objective(trial: Trial) -> float:
        jkey = jr.PRNGKey(jax_random_seed)
        # --- Tune hyperparameters ---
        train_conf = TrainingConfig(
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024]),
            epochs=500,
            window_len=6 + 1,
            perc_train_set=0.111,
            validate_every=1,
        )
        loss_conf = LossesConfig(
            w_recon=1.0,
            w_sequence=0.0,
            w_tc=0.0,
            w_acceleration=0.0,
            w_velocity=0.0,
            w_diff=0.0,
            w_sofa_direction=20.0,
            w_sofa_classification=30.0,
            w_sofa_d2=1.0,
            w_inf=trial.suggest_float("w_inf", 1e-8, 10, log=True),
            w_sep3=0.0,
            w_thresh=0.0,
            w_matching=0.0,
            w_spreading=trial.suggest_float("w_spread", 1e-8, 10, log=True),
            w_distr=0.0,
            anneal_threshs_iter=5 * 1 / train_conf.perc_train_set,
        )

        model_conf = ModelConfig(
            z_latent_dim=2,
            var_features=48,
            stat_features=4,
            w_indicator=True,
            enc_hidden=trial.suggest_categorical("enc_size", [32, 64, 128, 256, 512, 1024]),
            inf_pred_hidden=trial.suggest_categorical("inf_size", [32, 64, 128, 256, 512, 1024]),
            dec_hidden=32,
            predictor_z_hidden=4,
            dropout_rate=trial.suggest_float("drop", 0.1, 0.9, log=False),
        )
        lr_conf = LRConfig(
            init=0.0,
            peak=trial.suggest_float("lr_peak", 1e-5, 1e-1, log=True),
            peak_decay=0.9,
            end=1e-10,
            warmup_epochs=15,
            enc_wd=trial.suggest_float("lr_wd", 1e-8, 1e-1, log=True),
            grad_norm=trial.suggest_float("lr_gn", 0.1, 1.0, log=False),
        )

        key_enc, key_inf_predictor, key_dec, key_predictor = jr.split(jkey, 4)
        full = 2 if model_conf.w_indicator else 1
        encoder = LatentEncoder(
            key_enc,
            full * (model_conf.var_features + model_conf.stat_features),
            model_conf.enc_hidden,
            model_conf.predictor_z_hidden,
            dropout_rate=model_conf.dropout_rate,
            dtype=dtype,
        )
        inf_predictor = InfectionEncoder(
            key_inf_predictor,
            full * (model_conf.var_features + model_conf.stat_features),
            model_conf.inf_pred_hidden,
            dropout_rate=model_conf.dropout_rate,
            dtype=dtype,
        )
        gru = GRUPredictor(
            key=key_predictor,
            z_dim=model_conf.z_latent_dim,
            z_hidden_dim=model_conf.predictor_z_hidden,
            dtype=dtype,
        )
        decoder = Decoder(
            key_dec,
            model_conf.var_features + model_conf.stat_features,
            model_conf.z_latent_dim,
            model_conf.dec_hidden,
            dtype=dtype,
        )
        early_prune = EarlyStopping(patience=20, direction=-1)
        key_enc_weights, key_dec_weights, key_gru_weights = jr.split(jkey, 3)
        encoder = init_latent_encoder_weights(encoder, key_enc_weights)
        decoder = init_decoder_weights(decoder, key_dec_weights)
        gru = init_gru_weights(gru, key_gru_weights)
        model = LatentDynamicsModel(
            encoder=encoder,
            inf_predictor=inf_predictor,
            predictor=gru,
            decoder=decoder,
            alpha=ALPHA,
            ordinal_deltas=deltas,
            sofa_dist=sofa_dist,
        )
        params_model, static_model = eqx.partition(model, eqx.is_inexact_array)
        filter_spec = jtu.tree_map(lambda _: eqx.is_inexact_array, model)

        shuffle_train_key, key = jr.split(jkey, 2)
        steps_per_epoch = (train_x_inner.shape[0] // train_conf.batch_size) * train_conf.batch_size
        shuffle_val_key, key = jr.split(key, 2)
        val_x, val_y, _nval_batches = prepare_batches(val_x_inner, val_y_inner, train_conf.batch_size, shuffle_val_key)

        schedule = custom_warmup_cosine(
            init_value=lr_conf.init,
            peak_value=lr_conf.peak,
            warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
            steps_per_cycle=[steps_per_epoch * n for n in [100]],  # cycle durations
            end_value=lr_conf.end,
            peak_decay=lr_conf.peak_decay,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(lr_conf.grad_norm),
            optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
        )
        opt_state = optimizer.init(params_model)

        epoch = 0
        best_cost = float("inf")
        while epoch < train_conf.epochs:
            shuffle_train_key, key = jr.split(key, 2)
            x_epoch, y_epoch, nbatches = prepare_batches(
                train_x_inner, train_y_inner, train_conf.batch_size, shuffle_train_key, perc=train_conf.perc_train_set
            )
            loss_conf.steps_per_epoch = nbatches
            params_model, opt_state, _, key = process_train_epoch(
                model,
                opt_state=opt_state,
                x_data=x_epoch,
                y_data=y_epoch,
                update=optimizer.update,
                key=key,
                lookup_func=lookup_table.soft_get_local,
                loss_params=loss_conf,
                param_filter=filter_spec,
                calibrate_probs_func=empty,
            )
            model = eqx.combine(params_model, static_model)

            val_model = eqx.nn.inference_mode(model, value=True)
            val_metrics = process_val_epoch(
                val_model,
                x_data=val_x,
                y_data=val_y,
                step=opt_state[1][0].count,
                key=key,
                lookup_func=lookup_table.hard_get_fsq,
                loss_params=loss_conf,
                calibrate_probs_func=empty,
            )
            model = eqx.nn.inference_mode(val_model, value=False)
            sep3 = np.asarray(val_y[..., 2]).any(axis=-1).flatten()
            pred_sep3_p = np.asarray(val_metrics.sep3_p).flatten()
            del val_metrics, val_model

            roc = roc_auc_score(sep3, pred_sep3_p)
            prc = average_precision_score(sep3, pred_sep3_p)
            cost = 1 - (roc)
            logger.info(f"AUROC {roc:.4f} AUPRC {prc:.4f}, epoch {epoch}")

            trial.report(cost, step=epoch)

            if trial.should_prune() or early_prune.step(cost):
                raise optuna.TrialPruned

            best_cost = min(best_cost, cost)

            epoch += 1
        return best_cost

    return objective


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info", console_log=True)
    logger = logging.getLogger(__name__)
    dtype = jnp.float32
    TOTAL_TRIALS = 1000

    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    ) = get_data_sets_offline(window_len=6 + 1, swapaxes_y=(0, 2, 1), dtype=jnp.float32, full=True)

    db_str = "Daisy2"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )

    b, s = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
    a = np.ones_like(b) * ALPHA
    indices_3d = jnp.concatenate([a[..., np.newaxis], b[..., np.newaxis], s[..., np.newaxis]], axis=-1)[np.newaxis, ...]
    spacing_3d = jnp.array([0.0, BETA_SPACE[2], SIGMA_SPACE[2]])
    params = DNMConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
    metrics_3d = metrics_3d.to_jax().reshape([1, *metrics_3d.shape["r_1"]])
    lookup_table = LatentLookup(
        metrics=metrics_3d.reshape((-1, 1)),
        indices=indices_3d.reshape((-1, 3)),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=jnp.float32,
    )
    metrics = jnp.sort(lookup_table.relevant_metrics)
    # TODO dont do this for the windowed
    _sofas, counts = np.unique(test_y[..., 0], return_counts=True, sorted=True)
    counts = np.pad(counts, 24 - counts.size, constant_values=0)
    sofa_dist = counts / counts.sum()
    deltas = metrics[jnp.round(jnp.cumsum(sofa_dist) * len(metrics)).astype(dtype=jnp.int32)]
    deltas = jnp.ones_like(deltas)  # NOTE
    deltas = deltas / jnp.sum(deltas)
    deltas = jnp.asarray(deltas)


    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///data/db.sqlite3",  # storage URL
        study_name="step_by_step",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_warmup_steps=50), patience=15),
    )
    study.optimize(make_objective(train_x, train_y, val_x, val_y, sofa_dist[:-1], deltas), n_trials=TOTAL_TRIALS)
