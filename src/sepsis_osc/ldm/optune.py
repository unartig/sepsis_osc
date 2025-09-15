import logging
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import optuna
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score, average_precision_score

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics
from sepsis_osc.ldm.ae import Decoder, Encoder, init_decoder_weights, init_encoder_weights
from sepsis_osc.ldm.data_loading import get_data_sets, prepare_batches
from sepsis_osc.ldm.gru import GRUPredictor
from sepsis_osc.ldm.helper_structs import (
    LossesConfig,
    LRConfig,
    ModelConfig,
    TrainingConfig,
)
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel
from sepsis_osc.ldm.lookup import LatentLookup, as_3d_indices
from sepsis_osc.ldm.train import (
    ALPHA_SPACE,
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
        # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        train_conf = TrainingConfig(batch_size=1024, epochs=1000, window_len=6, perc_train_set=0.222, validate_every=1)
        loss_conf = LossesConfig(
            w_recon=0.1,  # trial.suggest_float("w_recon", 1e-5, 100, log=False),
            w_tc=0.1,  # trial.suggest_float("w_tc", 1e-5, 100, log=False),
            w_accel=0.1,  # trial.suggest_float("w_accel", 0, 100, log=False),
            w_diff=0.1,  # trial.suggest_float("w_diff", 0, 100, log=False),
            w_sofa_direction=trial.suggest_float("w_dir", 0, 100, log=False),
            w_sofa_classification=trial.suggest_float("w_sofa", 0, 100, log=False),
            w_inf=trial.suggest_float("w_inf", 0, 100, log=False),
            w_inf_alpha=trial.suggest_float("w_inf_alpha", 0, 1, log=False),
            w_inf_gamma=trial.suggest_float("w_inf_gamma", 0, 100, log=False),
            w_sep3=0.0,  # trial.suggest_float("w_sep3", 0, 100, log=False),
            w_thresh=0.0,  # trial.suggest_float("w_thresh", 0, 100, log=False),
            anneal_threshs_iter=0.0,  # trial.suggest_float("anneal_threshs", 1, 100, log=False) / train_conf.perc_train_set,
        )

        model_conf = ModelConfig(
            latent_dim=3,
            input_dim=52,
            enc_hidden=512,  # trial.suggest_categorical("enc_hidden", [16, 32, 64, 128, 256]),
            dec_hidden=64,  # trial.suggest_categorical("dec_hidden", [16, 32, 64, 128, 256]),
            predictor_hidden=4,  # trial.suggest_categorical("pred_hidden", [3, 4, 5]),
            dropout_rate=0.3,
        )
        lr_conf = LRConfig(
            init=0.0,
            peak=5e-3,  # trial.suggest_float("lr_peak", 1e-5, 1e-1, log=True),
            peak_decay=0.9,
            end=1e-10,
            warmup_epochs=15,
            enc_wd=1e-4,  # trial.suggest_float("lr_wd", 1e-5, 1e-1, log=True),
            grad_norm=0.7,  # trial.suggest_float("lr_gn", 0.2, 1.0, log=False),
        )

        key_enc, key_dec, key_predictor = jr.split(jkey, 3)
        encoder = Encoder(
            key_enc,
            model_conf.input_dim,
            model_conf.latent_dim,
            model_conf.enc_hidden,
            model_conf.predictor_hidden,
            dtype=jnp.float32,
        )
        decoder = Decoder(
            key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden, dtype=jnp.float32
        )
        encoder = init_encoder_weights(encoder, key_enc)
        decoder = init_decoder_weights(decoder, key_dec)
        gru = GRUPredictor(key=key_predictor, dim=model_conf.latent_dim - 1, hidden_dim=model_conf.predictor_hidden)
        model = LatentDynamicsModel(
            encoder=encoder, predictor=gru, decoder=decoder, ordinal_deltas=deltas, sofa_dist=sofa_dist
        )
        params_model, static_model = eqx.partition(model, eqx.is_inexact_array)

        shuffle_train_key, key = jr.split(jkey, 2)
        steps_per_epoch = (train_x_inner.shape[0] // train_conf.batch_size) * train_conf.batch_size
        shuffle_val_key, key = jr.split(key, 2)
        val_x, val_y, _nval_batches = prepare_batches(val_x_inner, val_y_inner, train_conf.batch_size, shuffle_val_key)

        schedule = custom_warmup_cosine(
            init_value=lr_conf.init,
            peak_value=lr_conf.peak,
            warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
            steps_per_cycle=[steps_per_epoch * n for n in [trial.suggest_int("len_cycle", 10, 500)]],  # cycle durations
            end_value=lr_conf.end,
            peak_decay=lr_conf.peak_decay,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(lr_conf.grad_norm),
            optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
        )
        opt_state = optimizer.init(params_model)

        best_cost = float("inf")
        prev_cost = float("inf")
        man_patience = 0
        epoch = 0
        while (epoch < 200 or man_patience < 15) and epoch < train_conf.epochs:
            shuffle_train_key, key = jr.split(key, 2)
            x_epoch, y_epoch, nbatches = prepare_batches(
                train_x_inner, train_y_inner, train_conf.batch_size, shuffle_train_key, perc=train_conf.perc_train_set
            )
            loss_conf.steps_per_epoch = nbatches
            params_model, opt_state, _train_metrics, key = process_train_epoch(
                model,
                opt_state,
                x_epoch,
                y_epoch,
                optimizer.update,
                key=key,
                lookup_func=lookup_table.soft_get_local,
                loss_params=loss_conf,
            )
            model = eqx.combine(params_model, static_model)

            val_model = eqx.nn.inference_mode(model, value=True)
            _, val_metrics = process_val_epoch(
                val_model,
                x_data=val_x,
                y_data=val_y,
                step=opt_state[1][0].count,
                key=key,
                lookup_func=lookup_table.hard_get_local,
                loss_params=loss_conf,
            )
            model = eqx.nn.inference_mode(val_model, value=False)
            sep3 = val_y[..., 2].any(axis=-1).flatten()
            pred_sep3_p = val_metrics.sep3_p.flatten()

            roc= roc_auc_score(sep3, pred_sep3_p)
            prc= average_precision_score(sep3, pred_sep3_p)
            cost = 1 - (roc * prc)
            logger.info(f"AUROC {roc:.4f} AUPRC {prc:.4f}, epoch {epoch}")

            trial.report(cost, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if cost < best_cost:
                best_cost = cost
            if cost < prev_cost:
                man_patience += 0
            else:
                man_patience += 1
            prev_cost = cost

            epoch += 1
        return best_cost

    return objective


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info", console_log=True)
    logger = logging.getLogger(__name__)
    dtype = jnp.float32
    TOTAL_TRIALS = 1000

    db_str = "Small"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )

    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = DNMConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params), proto_metric=DNMMetrics, threshold=jnp.inf)
    sim_storage.close()
    lookup_table = LatentLookup(
        metrics=metrics_3d.reshape((-1, 1)),
        indices=jnp.asarray(indices_3d.reshape((-1, 3))),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=dtype,
    )

    train_x, train_y, val_x, val_y, test_x, test_y = get_data_sets(window_len=6, dtype=jnp.float32)
    metrics = jnp.sort(lookup_table.relevant_metrics[..., 0])
    sofa_dist, _ = jnp.histogram(train_y[..., 0], bins=25, density=False)
    sofa_dist = sofa_dist / jnp.sum(sofa_dist)
    deltas = metrics[jnp.round(jnp.cumsum(sofa_dist) * len(metrics)).astype(dtype=jnp.int32)]
    deltas = jnp.ones_like(deltas)  # NOTE
    deltas = deltas / jnp.sum(deltas)
    deltas = jnp.asarray(deltas)

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///data/db.sqlite3",  # storage URL
        study_name="mini",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_warmup_steps=100), patience=15),
    )
    # dataset_fraction_callback = DatasetFractionCallback(total_trials=TOTAL_TRIALS)
    study.optimize(make_objective(train_x, train_y, val_x, val_y, sofa_dist[:-1], deltas), n_trials=TOTAL_TRIALS)
