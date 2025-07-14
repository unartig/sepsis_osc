import logging
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import os
import optax
from jaxtyping import Array, Float, PyTree
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.model.data_loading import prepare_sequences, prepare_batches, data
from sepsis_osc.model.model_utils import (
    AuxLosses,
    ConceptLossConfig,
    LoadingConfig,
    LossesConfig,
    LRConfig,
    ModelConfig,
    SaveConfig,
    TrainingConfig,
    as_3d_indices,
    load_checkpoint,
    save_checkpoint,
    timing,
)
from sepsis_osc.model.ae import Decoder, Encoder, init_encoder_weights, init_decoder_weights
from sepsis_osc.model.transformer import TransformerForecaster
from sepsis_osc.model.latent_dynamics import LatentDynamicsModel
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed, sequence_files
from sepsis_osc.utils.jax_config import setup_jax
from sepsis_osc.utils.logger import setup_logging

ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE = (-0.52, 0.52, 0.04), (0.2, 1.5, 0.02), (0.0, 1.5, 0.04)
# ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE = (-1.0, 1.0, 0.04), (0.2, 1.5, 0.02), (0.0, 1.5, 0.04)


def ordinal_logits(
    z: Float[Array, "batch 1"], thresholds: Float[Array, "n_classes_minus_1"], temperature: Float[Array, "batch 1"]
) -> Float[Array, "batch n_classes_minus_1"]:
    logits = (z.squeeze()[:, None] - thresholds) / temperature
    return logits


def ordinal_loss(
    predicted_sofa: Float[Array, "batch 1"],
    true_sofa: Float[Array, "batch 1"],
    thresholds: Float[Array, "n_classes_minus_1"],
    label_temperature: Float[Array, "batch 1"],
) -> tuple[Float[Array, "batch 1"], Float[Array, "batch 1"]]:
    # https://arxiv.org/abs/1901.07884
    # CORN (Cumulative Ordinal Regression)
    n_classes = thresholds.shape[-1] + 1

    logits = ordinal_logits(predicted_sofa, thresholds, label_temperature)

    pred_sofa = jnp.sum(predicted_sofa[:, None] > thresholds, axis=-1)

    targets = jnp.arange(n_classes - 1)  # [0, 1, ..., 22]
    true_sofa = true_sofa.squeeze()
    true_ord_targets = true_sofa[:, None] > targets[None, :]  # [B, 23]
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_ord_targets)), pred_sofa


def binary_logits(probs: Float[Array, "batch 1"], eps: float = 1e-6) -> Float[Array, "batch 1"]:
    probs = jnp.clip(probs, eps, 1 - eps)
    return jnp.log(probs) - jnp.log1p(-probs)


def binary_loss(
    predicted_infection: Float[Array, "batch 1"], true_infection: Float[Array, "batch 1"]
) -> Float[Array, "batch 1"]:
    logits = binary_logits(predicted_infection)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, true_infection))


def calc_concept_loss(
    prediction: Float[Array, "batch 2"],
    true_sofa: Float[Array, "batch 1"],
    true_infection: Float[Array, "batch 1"],
    thresholds: Float[Array, "batch n_classes_minus_1"],
    label_temp: Float[Array, "batch 1"],
    params: ConceptLossConfig,
) -> tuple[Float[Array, "batch 1"], Float[Array, "batch 1"], Float[Array, "batch 1"]]:
    loss_sofa, pred_sofa = ordinal_loss(prediction[:, 0], true_sofa, thresholds, label_temp)
    loss_infection = binary_loss(prediction[:, 1], true_infection)

    return params.w_sofa * loss_sofa, params.w_inf * loss_infection, pred_sofa


# === Loss Function ===
@eqx.filter_jit  # (donate="all")
def loss(
    model: LatentDynamicsModel,
    x: Float[Array, "batch time input_dim"],
    true_concepts: Float[Array, "batch time 2"],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    batch_size, T, input_dim = x.shape
    aux_losses = AuxLosses.empty()

    key, *drop_keys = jr.split(key, batch_size + 1)
    drop_keys = jnp.array(drop_keys)

    alpha0, beta0, sigma0 = jax.vmap(model.encoder)(x[:, 0], key=drop_keys)
    alpha0 = alpha0 * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0]
    beta0 = beta0 * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma0 = sigma0 * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]
    aux_losses.alpha, aux_losses.beta, aux_losses.sigma = (
        alpha0.mean(),
        beta0.mean(),
        sigma0.mean(),
    )

    z0 = jnp.concatenate([alpha0, beta0, sigma0], axis=-1)

    def predict_next(carry, _):
        z_seq_so_far, mask_so_far, step, key = carry

        key, *drop_keys = jr.split(key, batch_size + 1)
        drop_keys = jnp.array(drop_keys)

        z_next = jax.vmap(model.forecaster)(z_seq_so_far, jnp.array(drop_keys), mask=mask_so_far)

        # Write next step into padded tensor
        z_seq_so_far = z_seq_so_far.at[:, step + 1].set(z_next)
        mask_so_far = mask_so_far.at[:, step + 1].set(True)

        return (z_seq_so_far, mask_so_far, step + 1, key), z_next

    subkey, key = jr.split(key)
    z_dim = z0.shape[-1]

    z_init = jnp.zeros((batch_size, T, z_dim))
    mask_init = jnp.zeros((batch_size, T), dtype=bool)

    # Insert z0 as first timestep
    z_init = z_init.at[:, 0].set(z0)
    mask_init = mask_init.at[:, 0].set(True)
    carry_init = (z_init, mask_init, 0, subkey)

    (z_seq, mask_final, _, _), z_preds = jax.lax.scan(predict_next, carry_init, xs=None, length=T - 1)

    # Recon Loss
    x_recon = jax.vmap(jax.vmap(model.decoder))(z_seq)
    aux_losses.recon_loss = jnp.mean((x - x_recon) ** 2, axis=-1)

    # Lookup / concept loss
    lookup_temp, label_temp, ordinal_thresholds = model.get_parameters()
    concepts = jax.vmap(lookup_func, in_axes=(1, None))(z_seq, lookup_temp).swapaxes(0, 1)
    print(concepts.shape)
    sofa_true, infection_true = true_concepts[..., 0], true_concepts[..., 1]

    sofa_loss, infection_loss, hists = jax.vmap(calc_concept_loss, in_axes=(1, 1, 1, None, None, None))(
        concepts,
        sofa_true,
        infection_true,
        ordinal_thresholds,
        label_temp,
        params.concept,
    )
    print(sofa_loss.shape, infection_loss.shape, hists.shape)
    aux_losses.concept_loss = sofa_loss + infection_loss


    # Total correlation loss (one per batch)
    z_flat = z_seq.reshape(-1, z_seq.shape[-1])
    z_centered = z_flat - jnp.mean(z_flat, axis=0, keepdims=True)
    cov = z_centered.T @ z_centered / z_flat.shape[0]
    aux_losses.tc_loss = jnp.sum(jnp.abs(cov - jnp.diag(jnp.diag(cov))) ** 2)

    aux_losses.total_loss = (
        aux_losses.recon_loss * params.w_recon
        + aux_losses.concept_loss * params.w_concept
        + aux_losses.tc_loss * params.w_tc
        + 1 / concepts[..., 0].std()
    )

    aux_losses.label_temperature = label_temp.mean(axis=-1)
    aux_losses.lookup_temperature = lookup_temp
    aux_losses.sofa_loss = sofa_loss
    aux_losses.infection_loss = infection_loss
    aux_losses = jax.tree_util.tree_map(jnp.mean, aux_losses)
    aux_losses.hists_sofa_metric = concepts[..., 0].reshape(-1)
    aux_losses.hists_sofa_score = hists.reshape(-1)
    print(sofa_loss.shape, infection_loss.shape, hists.shape)
    aux_losses.sofa_t = sofa_loss
    aux_losses.infection_t = infection_loss
    return aux_losses.total_loss, aux_losses


@eqx.filter_jit
def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    model: LatentDynamicsModel,
    opt_state: optax.OptState,
    update: Callable,
    *,
    key,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[LatentDynamicsModel, optax.OptState, tuple[Array, AuxLosses]]:
    (total_loss, aux_losses), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        model, x_batch, true_c_batch, key=key, lookup_func=lookup_func, params=loss_params
    )

    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    updates, opt_state = update(grads, opt_state, model_params)
    model_params = eqx.apply_updates(model_params, updates)
    model = eqx.combine(model_params, model_static)

    return model, opt_state, (total_loss, aux_losses)


@timing
@eqx.filter_jit
def process_train_epoch(
    model: LatentDynamicsModel,
    opt_state: optax.OptState,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    update: Callable,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[PyTree, PyTree, optax.OptState, optax.OptState, AuxLosses, jnp.ndarray]:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    key, _ = jr.split(key)
    step_losses = []

    def scan_step(carry, batch):
        model_flat, opt_state, key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        model_full = eqx.combine(model_flat, model_static)

        model_new, opt_state, (_, aux_losses) = step_model(
            batch_x,
            batch_true_c,
            model_full,
            opt_state,
            update,
            key=batch_key,
            lookup_func=lookup_func,
            loss_params=loss_params,
        )

        model_flat = eqx.filter(model_new, eqx.is_inexact_array)

        return (
            model_flat,
            opt_state,
            key,
        ), aux_losses

    carry = (
        model_params,
        opt_state,
        key,
    )
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    model_params, opt_state, key = carry

    return (
        model_params,
        opt_state,
        step_losses,
        key,
    )


@timing
@eqx.filter_jit
def process_val_epoch(
    model: LatentDynamicsModel,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[jnp.ndarray, AuxLosses]:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

    key, _ = jr.split(key)

    def scan_step(carry, batch):
        model_flat, key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        model_full = eqx.combine(model_flat, model_static)

        _, aux_losses = loss(
            model_full,
            batch_x,
            batch_true_c,
            key=batch_key,
            lookup_func=lookup_func,
            params=loss_params,
        )

        return (model_flat, key), aux_losses

    carry = (model_params, key)
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    model_params, key = carry

    return (key, step_losses)


def custom_warmup_cosine(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    steps_per_cycle: list[int],
    end_value: float,
    peak_decay: float = 0.5,
):
    cycle_boundaries = jnp.array([warmup_steps + sum(steps_per_cycle[:i]) for i in range(len(steps_per_cycle))])
    cycle_lengths = jnp.array(steps_per_cycle)
    num_cycles = len(steps_per_cycle)

    def schedule(step):
        step = jnp.asarray(step)

        # --- Warmup ---
        def in_warmup_fn(step):
            frac = step / jnp.maximum(warmup_steps, 1)
            return init_value + frac * (peak_value - init_value)

        # --- Cosine Decay Cycles ---
        def in_decay_fn(step):
            # Which cycle are we in?
            rel_step = step - warmup_steps
            cycle_idx = jnp.sum(rel_step >= cycle_boundaries - warmup_steps) - 1
            cycle_idx = jnp.clip(cycle_idx, 0, num_cycles - 1)

            cycle_start = cycle_boundaries[cycle_idx]
            cycle_len = cycle_lengths[cycle_idx]
            step_in_cycle = step - cycle_start

            cycle_frac = jnp.clip(step_in_cycle / jnp.maximum(cycle_len, 1), 0.0, 1.0)
            peak = peak_value * (peak_decay**cycle_idx)
            return end_value + 0.5 * (peak - end_value) * (1 + jnp.cos(jnp.pi * cycle_frac))

        # Select warmup vs decay
        return jax.lax.cond(step < warmup_steps, in_warmup_fn, in_decay_fn, step)

    return schedule


if __name__ == "__main__":
    setup_jax(simulation=False)
    setup_logging("info")
    dtype = jnp.float32
    logger = logging.getLogger(__name__)

    model_conf = ModelConfig(latent_dim=3, input_dim=52, enc_hidden=64, dec_hidden=32)
    train_conf = TrainingConfig(batch_size=1024, window_len=6, epochs=1000, perc_train_set=0.5, validate_every=5)
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=10, perform=True)
    lr_conf = LRConfig(init=0.0, peak=0.002, peak_decay=0.9, end=5e-7, warmup_epochs=15, enc_wd=1e-5, grad_norm=0.1)
    loss_conf = LossesConfig(
        w_concept=1e2,
        w_recon=3,
        w_tc=45,
        concept=ConceptLossConfig(w_sofa=1.0, w_inf=0.0),
    )

    # === Data ===
    train_y, train_x, val_y, val_x, test_y, test_x = [
        v.drop([
            col for col in v.columns if col.startswith("Missing") or col in {"sep3_alt", "__index_level_0__", "los_icu"}
        ])
        for inner in data.values()
        for v in inner.values()
    ]

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
        logger.info("Processed sequence files not found. Preparing sequences and saving data...")
        # Prepare sequences
        train_x, train_y = prepare_sequences(train_x, train_y, train_conf.window_len)
        val_x, val_y = prepare_sequences(val_x, val_y, train_conf.window_len)

        # Save the processed arrays to files
        np.save(sequence_files + "test_x", train_x)
        np.save(sequence_files + "test_y", train_y)
        np.save(sequence_files + "val_x", val_x)
        np.save(sequence_files + "val_y", val_y)
        logger.info("Data prepared and saved sequences successfully.")

    logger.info(f"Train shape      - X {train_y.shape}, Y {train_x.shape}")
    logger.info(f"Validation shape - X {val_y.shape},  Y {val_x.shape}")
    logger.info(f"Test shape       - X {test_y.shape},  Y {test_x.shape}")

    db_str = "Daisy"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    sim_storage.close()

    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(np.asarray(params))
    lookup_table = JAXLookup(
        metrics=metrics_3d.copy().reshape((-1, 1)),
        indices=indices_3d.copy().reshape((-1, 3)),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
        dtype=dtype,
    )
    steps_per_epoch = (train_x.shape[0] // train_conf.batch_size) * train_conf.batch_size
    schedule = custom_warmup_cosine(
        init_value=lr_conf.init,
        peak_value=lr_conf.peak,
        warmup_steps=lr_conf.warmup_epochs * steps_per_epoch,
        steps_per_cycle=[steps_per_epoch * n for n in [25, 25, 35, 50]],  # cycle durations
        end_value=lr_conf.end,
        peak_decay=lr_conf.peak_decay,
    )

    # === Initialization ===
    key = jr.PRNGKey(jax_random_seed)
    key_enc, key_dec, key_transformer = jr.split(key, 3)
    train_sofa_dist, _ = jnp.histogram(train_y[..., 0], bins=25, density=True)
    encoder = Encoder(
        key_enc, model_conf.input_dim, model_conf.latent_dim, model_conf.enc_hidden, train_sofa_dist[:-1], dtype=dtype
    )
    decoder = Decoder(key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden, dtype=dtype)
    key_enc_weights, key_dec_weights = jr.split(key, 2)
    encoder = init_encoder_weights(encoder, key_enc_weights)
    decoder = init_decoder_weights(decoder, key_dec_weights)
    transformer = TransformerForecaster(
        key=key_transformer,
        dim=model_conf.latent_dim,
        depth=2,
        num_heads=3,
        hidden_dim=32,
    )

    model = LatentDynamicsModel(encoder=encoder, forecaster=transformer, decoder=decoder, sofa_dist=train_sofa_dist)

    optimizer = optax.chain(
        optax.clip_by_global_norm(lr_conf.grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
    )

    hyper_enc = {
        "input_dim": encoder.input_dim,
        "latent_dim": encoder.latent_dim,
        "enc_hidden": encoder.enc_hidden,
        "dropout_rate": encoder.dropout_rate,
    }
    hyper_dec = {
        "input_dim": decoder.input_dim,
        "latent_dim": decoder.latent_dim,
        "dec_hidden": decoder.dec_hidden,
    }
    hyper_trans = {
        "latent_dim": transformer.dim,
        "depth": transformer.depth,
        "num_heads": transformer.num_heads,
        "hidden_dim": transformer.hidden_dim,
        "dropout_rate": transformer.dropout_rate,
    }
    params_model, static_model = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params_model)
    if load_conf.from_dir:
        try:
            params_model, static_model, opt_state = load_checkpoint(
                load_conf.from_dir + "/checkpoints", load_conf.epoch, optimizer
            )

            model = eqx.combine(params_model, static_model)
            logger.info(f"Resuming training from epoch {load_conf.epoch + 1}")
        except FileNotFoundError as e:
            logger.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
            load_conf.from_dir = ""

    # === Training Loop ===
    writer = SummaryWriter()
    shuffle_val_key, key = jr.split(key, 2)
    val_x, val_y, nval_batches = prepare_batches(val_x, val_y, train_conf.batch_size, key=shuffle_val_key)
    print(val_x.shape, nval_batches)
    for epoch in range(load_conf.epoch, train_conf.epochs):
        shuffle_key, key = jr.split(key)
        x_shuffled, y_shuffled, ntrain_batches = prepare_batches(
            train_x,
            train_y,
            key=shuffle_key,
            batch_size=train_conf.batch_size,
            perc=train_conf.perc_train_set,
        )

        params_model, opt_state, train_metrics, key = process_train_epoch(
            model,
            opt_state=opt_state,
            x_data=x_shuffled,
            y_data=y_shuffled,
            update=optimizer.update,
            key=key,
            lookup_func=lookup_table.soft_get_local,
            loss_params=loss_conf,
        )
        model = eqx.combine(params_model, static_model)

        log_msg = f"Epoch {epoch} Training Metrics "
        train_metrics = train_metrics.to_dict()
        for group_key, metrics_group in train_metrics.items():
            if group_key in ("hists", "mult"):
                continue
            for metric_name, metric_values in metrics_group.items():
                metric_values = np.asarray(metric_values)
                if metric_values.ndim == 0 or metric_values.size == 0:
                    continue  # Skip scalars or empty metrics
                nsamples = metric_values.shape[0]
                # Log mean/std
                if group_key == "losses":
                    log_msg += f"{metric_name} = {float(metric_values.mean()):.4f} ({float(metric_values.std()):.4f}), "
                # Epoch 0: log every step
                if epoch == 0:
                    for step in range(nsamples):
                        writer.add_scalar(
                            f"train_{group_key}/{metric_name}_step", metric_values[step], epoch * nsamples + step
                        )
                else:
                    chunk_size = max(1, nsamples // 20)
                    for i in range(0, nsamples, chunk_size):
                        start = i
                        end = min(i + chunk_size, nsamples)
                        chunk = metric_values[start:end]
                        if len(chunk) > 0:
                            mean_value = chunk.mean()
                            writer.add_scalar(
                                f"train_{group_key}/{metric_name}_step",
                                mean_value,
                                epoch * nsamples + end - 1,  # Log at last step of chunk
                            )

        logger.info(log_msg)
        del x_shuffled, y_shuffled

        if epoch % train_conf.validate_every == 0:
            key, _ = jr.split(key)
            key, val_metrics = process_val_epoch(
                model,
                x_data=val_x,
                y_data=val_y,
                key=key,
                lookup_func=lookup_table.hard_get_local,
                loss_params=loss_conf,
            )
            log_msg = f"Epoch {epoch} Valdation Metrics "
            val_metrics = val_metrics.to_dict()
            print(val_metrics["mult"]["infection_t"].shape)
            for k in val_metrics.keys():
                if k == "hists":
                    writer.add_histogram(
                        "SOFA Score", np.asarray(val_metrics["hists"]["sofa_score"].flatten()), epoch, bins=24
                    )
                    writer.add_histogram(
                        "SOFA metric", np.asarray(val_metrics["hists"]["sofa_metric"].flatten()), epoch, bins=24
                    )
                elif k == "mult":
                    for t, v in enumerate(np.asarray(val_metrics["mult"]["infection_t"]).mean(axis=0)):
                        writer.add_scalar(f"infection_per_timestep/t{t}", np.asarray(v), epoch)
                    for t, v in enumerate(np.asarray(val_metrics["mult"]["sofa_t"]).mean(axis=0)):
                        writer.add_scalar(f"sofa_per_timestep/t{t}", np.asarray(v), epoch)
                        

                else:
                    for metric_name, metric_values in val_metrics[k].items():
                        log_msg += (
                            f"{metric_name} = {float(metric_values.mean()):.4f} ({float(metric_values.std()):.4f}), "
                            if k == "losses"
                            else ""
                        )
                        writer.add_scalar(f"val_{k}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
                        writer.add_scalar(f"val_{k}/{metric_name}_std", np.asarray(metric_values.std()), epoch)
            logger.warning(log_msg)

        writer.add_scalar(
            "lr/learning_rate", np.asarray(schedule(np.float64(epoch) * np.float64(train_x.shape[0]))), epoch
        )

        # --- Save checkpoint ---
        # if (epoch + 1) % save_conf.save_every == 0 and save_conf.perform:
        #     save_dir = writer.get_logdir() if not load_conf.from_dir else load_conf.from_dir
        #     save_checkpoint(
        #         save_dir + "/checkpoints",
        #         epoch,
        #         params_model,
        #         static_model,
        #         opt_state,
        #         hyper_enc,
        #         hyper_dec,
        #         hyper_trans,
        #     )
    writer.close()
