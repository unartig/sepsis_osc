import logging
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.model.data_loading import data
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
    prepare_batches,
    save_checkpoint,
    timing,
)
from sepsis_osc.model.ae import (
    Decoder,
    Encoder,
    init_decoder_weights,
    init_encoder_weights,
)
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
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
    models: tuple[Encoder, Decoder],
    x: jnp.ndarray,
    true_concepts: jnp.ndarray,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    params: LossesConfig,
) -> tuple[Array, AuxLosses]:
    encoder, decoder = models

    aux_losses = AuxLosses.empty()

    key, *drop_keys = jr.split(key, x.shape[0] + 1)
    drop_keys = jnp.array(drop_keys)

    alpha, beta, sigma, sofa, infection = jax.vmap(encoder)(x, key=drop_keys)
    alpha = alpha * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0]
    beta = beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0]
    sigma = sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0]
    aux_losses.alpha, aux_losses.beta, aux_losses.sigma = (
        alpha.mean(),
        beta.mean(),
        sigma.mean(),
    )
    lookup_temp, label_temp, ordinal_thresholds = (
        encoder.get_parameters()
    )

    concepts_direct = jnp.concatenate([sofa, infection], axis=-1)
    # lookup_indices = SystemConfig.batch_as_index(alpha=alpha, beta=beta, sigma=sigma, C=0.2)
    z_lookup = jnp.concatenate([alpha, beta, sigma], axis=-1)
    concepts_lookup = lookup_func(
        z_lookup,
        temperatures=lookup_temp,
    )

    aux_losses.sofa_lookup, aux_losses.infection_lookup, hists_sofa = calc_concept_loss(
        concepts_lookup, true_concepts[:, 0], true_concepts[:, 1], ordinal_thresholds, label_temp, params.concept
    )
    (
        aux_losses.sofa_direct,
        aux_losses.infection_direct,
        _,
    ) = calc_concept_loss(
        concepts_direct, true_concepts[:, 0], true_concepts[:, 1], ordinal_thresholds, label_temp, params.concept
    )
    lookup_loss = aux_losses.sofa_lookup + aux_losses.infection_lookup
    direct_loss = aux_losses.sofa_direct + aux_losses.infection_direct
    aux_losses.concept_loss = lookup_loss * params.lookup_vs_direct + direct_loss * (1 - params.lookup_vs_direct)

    z_centered = z_lookup - jnp.mean(z_lookup, axis=0, keepdims=True)
    cov = z_centered.T @ z_centered / z_lookup.shape[0]
    aux_losses.tc_loss = jnp.sum(jnp.abs(cov - jnp.diag(jnp.diag(cov))) ** 2)

    x_recon = jax.vmap(decoder)(z_lookup)
    aux_losses.recon_loss = jnp.mean((x - x_recon) ** 2, axis=-1)

    aux_losses.total_loss = (
        aux_losses.recon_loss * params.w_recon + aux_losses.sofa_lookup * params.w_concept + aux_losses.tc_loss * params.w_tc
    )

    aux_losses.label_temperature = label_temp.mean(axis=-1)
    aux_losses.lookup_temperature = lookup_temp
    aux_losses = jax.tree_util.tree_map(jnp.mean, aux_losses)
    aux_losses.hists_sofa_score = hists_sofa
    return aux_losses.total_loss, aux_losses


@eqx.filter_jit
def step_model(
    x_batch: jnp.ndarray,
    true_c_batch: jnp.ndarray,
    models: tuple[Encoder, Decoder],
    opt_state_enc: optax.OptState,
    opt_state_dec: optax.OptState,
    update_enc: Callable,
    update_dec: Callable,
    *,
    key,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[tuple[Encoder, Decoder], tuple[optax.OptState, optax.OptState], tuple[Array, AuxLosses]]:
    encoder, decoder = models
    (total_loss, aux_losses), (grads_enc, grads_dec) = eqx.filter_value_and_grad(loss, has_aux=True)(
        models, x_batch, true_c_batch, key=key, lookup_func=lookup_func, params=loss_params
    )

    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    updates_enc, opt_state_enc = update_enc(grads_enc, opt_state_enc, encoder_params)
    encoder_params = eqx.apply_updates(encoder_params, updates_enc)
    encoder = eqx.combine(encoder_params, encoder_static)

    updates_dec, opt_state_dec = update_dec(grads_dec, opt_state_dec, decoder_params)
    decoder_params = eqx.apply_updates(decoder_params, updates_dec)
    decoder = eqx.combine(decoder_params, decoder_static)

    return (encoder, decoder), (opt_state_enc, opt_state_dec), (total_loss, aux_losses)


@timing
@eqx.filter_jit
def process_train_epoch(
    encoder: Encoder,
    decoder: Decoder,
    opt_state_enc,
    opt_state_dec,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    update_enc: Callable,
    update_dec: Callable,
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[PyTree, PyTree, optax.OptState, optax.OptState, AuxLosses, jnp.ndarray]:
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    key, _ = jr.split(key)
    step_losses = []

    def scan_step(carry, batch):
        (encoder_f, decoder_f), (opt_state_enc, opt_state_dec), key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        encoder_full = eqx.combine(encoder_f, encoder_static)
        decoder_full = eqx.combine(decoder_f, decoder_static)

        (encoder_new, decoder_new), (opt_state_enc, opt_state_dec), (_, aux_losses) = step_model(
            batch_x,
            batch_true_c,
            (encoder_full, decoder_full),
            opt_state_enc,
            opt_state_dec,
            update_enc,
            update_dec,
            key=batch_key,
            lookup_func=lookup_func,
            loss_params=loss_params,
        )

        encoder_f = eqx.filter(encoder_new, eqx.is_inexact_array)
        decoder_f = eqx.filter(decoder_new, eqx.is_inexact_array)

        return (
            (encoder_f, decoder_f),
            (opt_state_enc, opt_state_dec),
            key,
        ), aux_losses

    carry = (
        (encoder_params, decoder_params),
        (opt_state_enc, opt_state_dec),
        key,
    )
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    (encoder_params, decoder_params), (opt_state_enc, opt_state_dec), key = carry

    return (
        encoder_params,
        decoder_params,
        opt_state_enc,
        opt_state_dec,
        step_losses,
        key,
    )


@timing
@eqx.filter_jit
def process_val_epoch(
    encoder: Encoder,
    decoder: Decoder,
    x_data: Float[Array, "num_samples input_dim"],
    y_data: Float[Array, "num_samples 2"],
    *,
    key: jnp.ndarray,
    lookup_func: Callable,
    loss_params: LossesConfig,
) -> tuple[jnp.ndarray, AuxLosses]:
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_inexact_array)
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_inexact_array)

    key, _ = jr.split(key)

    def scan_step(carry, batch):
        (encoder_f, decoder_f), key = carry
        batch_x, batch_true_c = batch

        key, batch_key = jr.split(key)

        encoder_full = eqx.combine(encoder_f, encoder_static)
        decoder_full = eqx.combine(decoder_f, decoder_static)

        _, aux_losses = loss(
            (encoder_full, decoder_full),
            batch_x,
            batch_true_c,
            key=batch_key,
            lookup_func=lookup_func,
            params=loss_params,
        )

        return ((encoder_f, decoder_f), key), aux_losses

    carry = ((encoder_params, decoder_params), key)
    batches = (x_data, y_data)
    carry, step_losses = jax.lax.scan(scan_step, carry, batches)

    (encoder_params, decoder_params), key = carry

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
    
    model_conf = ModelConfig(latent_dim=3, input_dim=52, enc_hidden=32, dec_hidden=64)
    train_conf = TrainingConfig(batch_size=1024, epochs=1000, perc_train_set=1.0, validate_every=5)
    load_conf = LoadingConfig(from_dir="", epoch=0)
    save_conf = SaveConfig(save_every=10, perform=True)
    lr_conf = LRConfig(init=0.0, peak=0.0024157, peak_decay=0.9, end=2.6e-6, warmup_epochs=15, enc_wd=2.36e-6, grad_norm=0.128625)
    loss_conf = LossesConfig(
        w_concept=1e2,
        w_recon=3,
        w_tc=45,
        lookup_vs_direct=0.94,
        concept=ConceptLossConfig(w_sofa=1.0, w_inf=0.0),
    )

    # === Data ===
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
        dtype=dtype
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
    key_enc, key_dec = jr.split(key)
    train_sofa_dist, _ = jnp.histogram(train_y, bins=25, density=True)
    encoder = Encoder(key_enc, model_conf.input_dim, model_conf.latent_dim, model_conf.enc_hidden, train_sofa_dist[:-1], dtype=dtype)
    decoder = Decoder(key_dec, model_conf.input_dim, model_conf.latent_dim, model_conf.dec_hidden, dtype=dtype)

    opt_enc = optax.chain(
        # optax.clip_by_global_norm(lr_conf.grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=lr_conf.enc_wd),
    )
    opt_dec = optax.adamw(lr_conf.peak)

    key_enc_weights, key_dec_weights = jr.split(key, 2)
    encoder = init_encoder_weights(encoder, key_enc_weights)
    decoder = init_decoder_weights(decoder, key_dec_weights)

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
    params_enc, static_enc = eqx.partition(encoder, eqx.is_inexact_array)
    params_dec, static_dec = eqx.partition(decoder, eqx.is_inexact_array)
    opt_state_enc = opt_enc.init(params_enc)
    opt_state_dec = opt_dec.init(params_dec)
    if load_conf.from_dir:
        try:
            params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec = load_checkpoint(
                load_conf.from_dir + "/checkpoints", load_conf.epoch, opt_enc, opt_dec
            )

            encoder = eqx.combine(params_enc, static_enc)
            decoder = eqx.combine(params_dec, static_dec)
            logger.info(f"Resuming training from epoch {load_conf.epoch + 1}")
        except FileNotFoundError as e:
            logger.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
            load_conf.from_dir = ""

    # === Training Loop ===
    writer = SummaryWriter()
    shuffle_val_key, key = jr.split(key, 2)
    val_x, val_y, nval_batches = prepare_batches(val_x, val_y, train_conf.batch_size, shuffle_val_key)
    for epoch in range(load_conf.epoch, train_conf.epochs):
        shuffle_key, key = jr.split(key)
        x_shuffled, y_shuffled, ntrain_batches = prepare_batches(
            train_x, train_y, train_conf.batch_size, shuffle_key, train_conf.perc_train_set
        )

        params_enc, params_dec, opt_state_enc, opt_state_dec, train_metrics, key = process_train_epoch(
            encoder,
            decoder,
            opt_state_enc=opt_state_enc,
            opt_state_dec=opt_state_dec,
            x_data=x_shuffled,
            y_data=y_shuffled,
            update_enc=opt_enc.update,
            update_dec=opt_dec.update,
            key=key,
            lookup_func=lookup_table.soft_get_local,
            loss_params=loss_conf,
        )
        # loss_conf.w_locality = loss_conf.w_locality * (1 - 0.015)
        encoder = eqx.combine(params_enc, static_enc)
        decoder = eqx.combine(params_dec, static_dec)

        log_msg = f"Epoch {epoch} Training Metrics "
        train_metrics = train_metrics.to_dict()
        for group_key, metrics_group in train_metrics.items():
            if group_key == "hists":
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
                encoder,
                decoder,
                x_data=val_x,
                y_data=val_y,
                key=key,
                lookup_func=lookup_table.hard_get_local,
                loss_params=loss_conf,
            )
            log_msg = f"Epoch {epoch} Valdation Metrics "
            val_metrics = val_metrics.to_dict()
            for k in val_metrics.keys():
                if k == "hists":
                    print(
                        val_metrics["hists"]["sofa_score"].flatten().mean(),
                        val_metrics["hists"]["sofa_score"].flatten().min(),
                        val_metrics["hists"]["sofa_score"].flatten().max(),
                    )
                    writer.add_histogram(
                        "SOFA Score", np.asarray(val_metrics["hists"]["sofa_score"].flatten()), epoch, bins=24
                    )
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
        if (epoch + 1) % save_conf.save_every == 0 and save_conf.perform:
            save_dir = writer.get_logdir() if not load_conf.from_dir else load_conf.from_dir
            save_checkpoint(
                save_dir + "/checkpoints",
                epoch,
                params_enc,
                static_enc,
                params_dec,
                static_dec,
                opt_state_enc,
                opt_state_dec,
                hyper_enc,
                hyper_dec,
            )
    writer.close()
