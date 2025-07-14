import logging
from dataclasses import asdict

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import optuna
from optuna.trial import Trial

from sepsis_osc.model.data_loading import data
from sepsis_osc.model.model_utils import (
    ConceptLossConfig,
    LossesConfig,
    LRConfig,
    ModelConfig,
    TrainingConfig,
    as_3d_indices,
    prepare_batches,
)
from sepsis_osc.model.train import (
    ALPHA_SPACE,
    BETA_SPACE,
    SIGMA_SPACE,
    custom_warmup_cosine,
    process_train_epoch,
    process_val_epoch,
)
from sepsis_osc.model.vae import Decoder, Encoder
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging


def make_objective(train_x_inner, train_y_inner, val_x_inner, val_y_inner, sofa_dist):
    def objective(trial: Trial) -> float:
        jkey = jr.PRNGKey(jax_random_seed)
        # --- Tune hyperparameters ---
        w_concept = trial.suggest_float("w_concept", 0, 100, log=False)
        w_recon = trial.suggest_float("w_recon", 0, 100, log=False)
        w_tc = trial.suggest_float("w_tc", 0, 100, log=False)
        lvd = trial.suggest_float("lookup_vs_direct", 0.5, 1.0)
        enc_hidden = trial.suggest_categorical("enc_hidden", [32, 64, 128, 256])
        dec_hidden = trial.suggest_categorical("dec_hidden", [32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024, 2048])
        lr_peak = trial.suggest_float("lr_peak", 1e-5, 1e-2, log=True)
        lr_end = trial.suggest_float("lr_end", 1e-7, 1e-3, log=True)
        grad_norm = trial.suggest_float("grad_norm", 1e-3, 1, log=True)
        lr_wd = trial.suggest_float("lr_wd", 1e-7, 1, log=True)

        train_conf = TrainingConfig(batch_size=batch_size, epochs=450, perc_train_set=0.5, validate_every=1)
        loss_conf = LossesConfig(
            w_concept=w_concept,
            w_recon=w_recon,
            w_tc=w_tc,
            lookup_vs_direct=lvd,
            concept=ConceptLossConfig(w_sofa=1.0, w_inf=0.0),
        )
        model_conf = ModelConfig(latent_dim=3, input_dim=52, enc_hidden=enc_hidden, dec_hidden=dec_hidden)
        lr_conf = LRConfig(init=0.0, peak=lr_peak, peak_decay=0.9, end=lr_end, warmup_epochs=15, enc_wd=lr_wd, grad_norm=grad_norm)

        key_enc, key_dec = jr.split(jkey)
        encoder = Encoder(key_enc, model_conf.input_dim, model_conf.latent_dim, model_conf.enc_hidden, sofa_dist, dtype=jnp.float32)
        decoder = Decoder(key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden, dtype=jnp.float32)
        params_enc, static_enc = eqx.partition(encoder, eqx.is_inexact_array)
        params_dec, static_dec = eqx.partition(decoder, eqx.is_inexact_array)
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
        opt_enc = optax.chain(
            optax.clip_by_global_norm(lr_conf.grad_norm),
            optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
        )
        opt_dec = optax.adamw(lr_conf.peak)
        opt_state_enc = opt_enc.init(params_enc)
        opt_state_dec = opt_dec.init(params_dec)

        best_loss_val = float("inf")
        bad_val_epochs = 0
        bad_train_epochs = 0

        for epoch in range(train_conf.epochs):
            shuffle_train_key, key = jr.split(key, 2)
            x_epoch, y_epoch, nbatches = prepare_batches(
                train_x_inner, train_y_inner, train_conf.batch_size, shuffle_train_key, perc=train_conf.perc_train_set
            )
            params_enc, params_dec, opt_state_enc, opt_state_dec, _, key = process_train_epoch(
                encoder,
                decoder,
                opt_state_enc,
                opt_state_dec,
                x_epoch,
                y_epoch,
                opt_enc.update,
                opt_dec.update,
                key=key,
                lookup_func=lookup_table.soft_get_local,
                loss_params=loss_conf,
            )
            encoder = eqx.combine(params_enc, static_enc)
            decoder = eqx.combine(params_dec, static_dec)

            key, val_metrics = process_val_epoch(
                encoder,
                decoder,
                x_data=val_x,
                y_data=val_y,
                key=key,
                lookup_func=lookup_table.hard_get_local,
                loss_params=loss_conf,
            )
            concept_loss_val = float(val_metrics.sofa_lookup.mean())
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

    train_y, train_x, val_y, val_x, test_y, test_x = [
        jnp.array(
            v.drop([
                col
                for col in v.columns
                if col.startswith("Missing") or col in {"stay_id", "time", "sep3_alt", "__index_level_0__", "los_icu"}
            ]),
            dtype=dtype,
        )
        for inner in data.values()
        for v in inner.values()
    ]
    db_str = "Daisy"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )

    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params))
    train_sofa_dist, _ = jnp.histogram(train_y, bins=25, density=True)
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
        study_name="s1_fast_half_alpha",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=25)
    )
    # dataset_fraction_callback = DatasetFractionCallback(total_trials=TOTAL_TRIALS)
    study.optimize(make_objective(train_x, train_y, val_x, val_y, train_sofa_dist), n_trials=TOTAL_TRIALS)
