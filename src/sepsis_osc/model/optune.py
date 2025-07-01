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
    LocalityLossConfig,
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


def make_objective(train_x_inner, train_y_inner, val_x_inner, val_y_inner, jkey):
    def objective(trial: Trial) -> float:

        # --- Tune hyperparameters ---
        sigma_input = trial.suggest_float("sigma_input", 0.1, 5.0)
        sigma_sofa = trial.suggest_float("sigma_sofa", 0.1, 5.0)
        w_input = trial.suggest_float("w_input", 0.1, 5.0)
        w_sofa = trial.suggest_float("w_sofa", 0.1, 5.0)
        temperature = trial.suggest_float("temperature_locality", 0.01, 50.0)
        w_recon = trial.suggest_float("w_recon", 1e-2, 1e2, log=True)
        w_concept = trial.suggest_float("w_concept", 1e-2, 1e3, log=True)
        w_locality = trial.suggest_float("w_locality", 1e-2, 1e3, log=True)
        z_alpha= trial.suggest_float("z_alpha", 1e-2, 1e2, log=True)
        z_beta= trial.suggest_float("z_beta", 1e-2, 1e2, log=True)
        z_sigma= trial.suggest_float("z_sigma", 1e-2, 1e2, log=True)
        w_tc = trial.suggest_float("w_tc", 1e-12, 1e1, log=True)

        train_conf = TrainingConfig(batch_size=1024, epochs=150, perc_train_set=1.0, validate_every=1)
        loss_conf = LossesConfig(
            w_recon=w_recon,
            w_concept=w_concept,
            w_locality=w_locality,
            w_tc=w_tc,
            concept=ConceptLossConfig(w_sofa=1.0, w_inf=0.0),
            locality=LocalityLossConfig(
                sigma_input=sigma_input,
                sigma_sofa=sigma_sofa,
                w_input=w_input,
                w_sofa=w_sofa,
                z_scale=jnp.array([z_alpha, z_beta, z_sigma]),
                temperature=temperature,
            ),
        )

        model_conf = ModelConfig(latent_dim=3, input_dim=52, enc_hidden=32, dec_hidden=16)
        key_enc, key_dec = jr.split(jkey)
        encoder = Encoder(key_enc, model_conf.input_dim, model_conf.latent_dim, model_conf.enc_hidden)
        decoder = Decoder(key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden)
        params_enc, static_enc = eqx.partition(encoder, eqx.is_inexact_array)
        params_dec, static_dec = eqx.partition(decoder, eqx.is_inexact_array)
        lr_conf = LRConfig(init=0.0, peak=5e-3, peak_decay=0.9, end=7e-5, warmup_epochs=25, enc_wd=1e-3, grad_norm=0.7)
        shuffle_train_key, key = jr.split(jkey, 2)
        steps_per_epoch = (train_x_inner.shape[0] // train_conf.batch_size) * train_conf.batch_size
        shuffle_val_key, key = jr.split(key, 2)
        val_x, val_y, nval_batches = prepare_batches(val_x_inner, val_y_inner, train_conf.batch_size, shuffle_val_key)
        schedule = custom_warmup_cosine(
            init_value=lr_conf.init,
            peak_value=lr_conf.peak,
            warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
            steps_per_cycle=[steps_per_epoch * n for n in [50, 50, 25]],  # cycle durations
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
                lookup_func=lookup_table.hard_get,
                loss_params=loss_conf,
            )
            concept_loss_val = float(val_metrics.losses_concept_loss.mean())
            logger.info(f"Loss {concept_loss_val:.4f}, epoch {epoch}, bad val epochs {bad_val_epochs}, bad train epochs {bad_train_epochs}")

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
    key = jr.PRNGKey(jax_random_seed)
    TOTAL_TRIALS = 200

    train_y, train_x, val_y, val_x, test_y, test_x = [
        jnp.array(
            v.drop([
                col
                for col in v.columns
                if col.startswith("Missing") or col in {"stay_id", "time", "sep3_alt", "__index_level_0__", "los_icu"}
            ]),
            dtype=jnp.float32,
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
    indices, np_metrics = sim_storage.get_np_lookup()

    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params))
    lookup_table = JAXLookup(
        metrics=np_metrics.to_jax(),
        indices=jnp.asarray(indices[:, 5:8]),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
    )

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///data/db.sqlite3",  # Specify the storage URL here.
        study_name="f1_lookup_local",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )
    # dataset_fraction_callback = DatasetFractionCallback(total_trials=TOTAL_TRIALS)
    study.optimize(make_objective(train_x, train_y, val_x, val_y, key), n_trials=TOTAL_TRIALS)
