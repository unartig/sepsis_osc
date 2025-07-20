import logging
from dataclasses import asdict

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import os
import optuna
from optuna.trial import Trial

from sepsis_osc.model.data_loading import prepare_batches, prepare_sequences
from sepsis_osc.model.model_utils import (
    ConceptLossConfig,
    LossesConfig,
    LRConfig,
    ModelConfig,
    TrainingConfig,
    as_3d_indices,
)
from sepsis_osc.model.train import (
    ALPHA_SPACE,
    BETA_SPACE,
    SIGMA_SPACE,
    custom_warmup_cosine,
    process_train_epoch,
    process_val_epoch,
)
from sepsis_osc.model.ae import Decoder, Encoder, init_decoder_weights, init_encoder_weights
from sepsis_osc.model.gru import GRUPredictor
from sepsis_osc.model.latent_dynamics  import LatentDynamicsModel
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed, sequence_files
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging


def make_objective(train_x_inner, train_y_inner, val_x_inner, val_y_inner, sofa_dist):
    def objective(trial: Trial) -> float:
        jkey = jr.PRNGKey(jax_random_seed)
        # --- Tune hyperparameters ---
        w_concept = trial.suggest_float("w_concept", 0, 200, log=False)
        w_recon = trial.suggest_float("w_recon", 0, 50, log=False)
        w_tc = trial.suggest_float("w_tc", 0, 100, log=False)
        w_kl = trial.suggest_float("w_kl", 1e-4, 1e1, log=True)
        enc_hidden = trial.suggest_categorical("enc_hidden", [32, 64, 128, 256])
        pred_hidden = trial.suggest_categorical("pred_hidden", [32, 64, 128, 256])
        dec_hidden = trial.suggest_categorical("dec_hidden", [32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        lr_peak = trial.suggest_float("lr_peak", 1e-5, 1e-2, log=True)
        lr_end = trial.suggest_float("lr_end", 1e-7, 1e-3, log=True)
        anneal_kl = trial.suggest_float("anneal_kl", 20, 100, log=False)
        anneal_concept = trial.suggest_float("anneal_concept", 1, 5, log=False)

        train_conf = TrainingConfig(batch_size=batch_size, epochs=200, window_len=6, perc_train_set=0.2, validate_every=1)
        loss_conf = LossesConfig(
            w_concept=w_concept,
            w_recon=w_recon,
            w_tc=w_tc,
            w_kl=w_kl,
            anneal_kl_iter=anneal_kl * 1/train_conf.perc_train_set,
            anneal_concept_iter=anneal_concept * 1/train_conf.perc_train_set,

            concept=ConceptLossConfig(w_sofa=1.0, w_inf=0.0),
        )

        model_conf = ModelConfig(latent_dim=3, input_dim=52, enc_hidden=enc_hidden, dec_hidden=dec_hidden, predictor_hidden=pred_hidden)
        lr_conf = LRConfig(init=0.0, peak=lr_peak, peak_decay=0.9, end=lr_end, warmup_epochs=15, enc_wd=1e-3, grad_norm=0.1)

        key_enc, key_dec, key_predictor = jr.split(jkey, 3)
        encoder = Encoder(key_enc, model_conf.input_dim, model_conf.latent_dim, model_conf.enc_hidden, dtype=jnp.float32)
        decoder = Decoder(key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden, dtype=jnp.float32)
        encoder = init_encoder_weights(encoder, key_enc)
        decoder = init_decoder_weights(decoder, key_dec)
        gru = GRUPredictor(key=key_predictor, dim=model_conf.latent_dim * 2, hidden_dim=model_conf.predictor_hidden)
        model = LatentDynamicsModel(encoder=encoder, predictor=gru, decoder=decoder, sofa_dist=sofa_dist)
        params_model, static_model = eqx.partition(model, eqx.is_inexact_array)

        shuffle_train_key, key = jr.split(jkey, 2)
        steps_per_epoch = (train_x_inner.shape[0] // train_conf.batch_size) * train_conf.batch_size
        shuffle_val_key, key = jr.split(key, 2)
        val_x, val_y, nval_batches = prepare_batches(val_x_inner, val_y_inner, train_conf.batch_size, shuffle_val_key)

        schedule = custom_warmup_cosine(
            init_value=lr_conf.init,
            peak_value=lr_conf.peak,
            warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
            steps_per_cycle=[steps_per_epoch * n for n in [25, 25, 35, 50]],  # cycle durations
            end_value=lr_conf.end,
            peak_decay=lr_conf.peak_decay,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(lr_conf.grad_norm),
            optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
        )
        opt_state = optimizer.init(params_model)

        best_loss_val = float("inf")
        bad_val_epochs = 0
        bad_train_epochs = 0

        for epoch in range(train_conf.epochs):
            shuffle_train_key, key = jr.split(key, 2)
            x_epoch, y_epoch, nbatches = prepare_batches(
                train_x_inner, train_y_inner, train_conf.batch_size, shuffle_train_key, perc=train_conf.perc_train_set
            )
            loss_conf.steps_per_epoch = nbatches
            params_model, opt_state, train_metrics, key = process_train_epoch(
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

            key, val_metrics = process_val_epoch(
                model,
                x_data=val_x,
                y_data=val_y,
                key=key,
                lookup_func=lookup_table.hard_get_local,
                loss_params=loss_conf,
            )
            concept_loss_val = float(val_metrics.sofa_loss.mean())
            logger.info(
                f"Loss {concept_loss_val:.4f}, epoch {epoch}, bad val epochs {bad_val_epochs}, bad train epochs {bad_train_epochs}"
            )

            trial.report(concept_loss_val, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if concept_loss_val < best_loss_val:
                best_loss_val = concept_loss_val

        return best_loss_val

    return objective


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info", console_log=True)
    logger = logging.getLogger(__name__)
    dtype=jnp.float32
    TOTAL_TRIALS = 1000

    db_str = "Daisy"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )

    if (
        os.path.exists(sequence_files + "test_x.npy")
        and os.path.exists(sequence_files + "test_y.npy")
        and os.path.exists(sequence_files + "val_x.npy")
        and os.path.exists(sequence_files + "val_y.npy")
    ):
        logger.info("Processed sequence files found. Loading data from disk...")
        train_x = np.load(sequence_files + "test_x.npy")
        train_y = np.load(sequence_files + "test_y.npy")
        val_x = np.load(sequence_files + "val_x.npy")
        val_y = np.load(sequence_files + "val_y.npy")
        logger.info("Data loaded successfully.")
    else:
        logger.error("No Sequence Files found")
        exit(0)
    train_x, train_y, val_x, val_y = (
        train_x.astype(jnp.bfloat16),
        train_y.astype(jnp.bfloat16),
        val_x.astype(jnp.bfloat16),
        val_y.astype(jnp.bfloat16),
    )
    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params))
    sofa_dist, _ = jnp.histogram(train_y[..., 0], bins=25, density=False)
    sofa_dist = sofa_dist / jnp.sum(sofa_dist)
    train_sofa_dist = jnp.asarray(sofa_dist)
    lookup_table = JAXLookup(
        metrics=metrics_3d.copy().reshape((-1, 1)),
        indices=jnp.asarray(indices_3d.copy().reshape((-1, 3))),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=dtype
    )
    sim_storage.close()

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///data/db.sqlite3",  # Specify the storage URL here.
        study_name="latent_dynamics_stat_delta",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=25)
    )
    # dataset_fraction_callback = DatasetFractionCallback(total_trials=TOTAL_TRIALS)
    study.optimize(make_objective(train_x, train_y, val_x, val_y, train_sofa_dist), n_trials=TOTAL_TRIALS)
