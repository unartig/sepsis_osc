import logging

import equinox as eqx
import jax.numpy as jnp
import jax
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import polars as pl

from sepsis_osc.model.data_loading import data
from sepsis_osc.model.train import ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE, binary_logits, ordinal_logits
from sepsis_osc.model.vae import (
    Decoder,
    Encoder,
)
from sepsis_osc.model.model_utils import load_checkpoint, LossesConfig, ConceptLossConfig, LocalityLossConfig, as_3d_indices
from sepsis_osc.simulation.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.visualisations.viz_three_dee import three_dee

LOAD_FROM_CHECKPOINT = "runs/goody"
LOAD_EPOCH = 469


def viz_latent(params, encoder, p_x, p_y, metrics, filename, figure_dir):
    fig = three_dee(
        params,
        np.asarray(metrics.f_1),
        "Parameter Space Cluster Ratio 1 + SOFA prediction",
        filename,
        "figures/model",
        show=False,
    )
    out = jax.vmap(encoder)(to_input(p_x), key=dummy_keys)
    alpha, beta, sigma, direct_sofa, direct_inf, lookup_temp, label_temp, thresholds = out

    z = jnp.concatenate(
        [
            alpha * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0],
            beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0],
            sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0],
        ],
        axis=-1,
    )

    z = SystemConfig.batch_as_index(z[0], z[1], z[2], 0.2)
    data = {"alpha": z[:, 5], "beta": z[:, 6], "sigma": z[:, 7]}
    new_fig = go.Figure(
        data=go.Scatter3d(
            x=data["sigma"],
            y=data["beta"],
            z=data["alpha"],
            opacity=1.0,
        )
    )
    if filename:
        fig.write_html(f"{figure_dir}/{filename}.html")

        fig = fig.add_traces(new_fig.data)
        new_fig.write_html(f"{figure_dir}/new_{filename}.html")

        fig.write_html(f"{figure_dir}/both_{filename}.html")


def scatter_concepts(p_y: pl.DataFrame, pred_sofa: jnp.ndarray, pred_infs: jnp.ndarray, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(range(0,24), range(0,24), color="tab:red", label="optimum")
    ax[0].scatter(p_y["sofa"], pred_sofa, alpha=0.5)
    ax[1].scatter(p_y["susp_inf_alt"], pred_infs)
    return ax


def to_input(df):
    return np.array(df.drop(["stay_id", "time"]), dtype=np.float32)


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    loss_conf = LossesConfig(
        w_recon=1.0,
        w_concept=1e3,
        w_locality=5e-1,
        w_tc=0.0,
        concept=ConceptLossConfig(w_sofa=1.0, w_inf=0.0),
        locality=LocalityLossConfig(
            sigma_input=1.0,
            sigma_sofa=2.0,
            w_input=1.0,
            w_sofa=2.0,
            z_scale=jnp.array([1.0, 1.0, 1.0]),
            temperature=2.0,
        ),
    )

    betas = np.arange(BETA_SPACE[0], BETA_SPACE[1], BETA_SPACE[2])
    sigmas = np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1], SIGMA_SPACE[2])
    alphas = np.arange(ALPHA_SPACE[0], ALPHA_SPACE[1], ALPHA_SPACE[2])

    # DATA
    size = (len(betas), len(sigmas), len(alphas))
    db_str = "Daisy"  # other/Tiny"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    indices, np_metrics = storage.get_np_lookup()
    indices, np_metrics = storage.get_np_lookup()
    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = storage.read_multiple_results(np.asarray(params))
    lookup_table = JAXLookup(
        metrics=np_metrics.to_jax(),
        indices=jnp.asarray(indices[:, 5:8]),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
    )

    beta_grid, sigma_grid, alpha_grid = np.meshgrid(betas, sigmas, alphas, indexing="ij")
    permutations = np.stack([alpha_grid, beta_grid, sigma_grid], axis=-1)
    num_permutations = len(alphas) * len(betas) * len(sigmas)
    permutations_flat = permutations.reshape(num_permutations, 3)
    a, b, s = permutations[:, :, :, 0:1], permutations[:, :, :, 1:2], permutations[:, :, :, 2:3]
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrix, _ = storage.read_multiple_results(np.asarray(params))
    storage.close()
    if not metrix:
        exit(0)

    encoder = decoder = None
    try:
        params_enc, static_enc, params_dec, static_dec, opt_state_enc, opt_state_dec = load_checkpoint(
            LOAD_FROM_CHECKPOINT + "/checkpoints", LOAD_EPOCH, None, None
        )
        logger.info(f"Successfully loaded model {LOAD_FROM_CHECKPOINT}@{LOAD_EPOCH}")

        encoder = eqx.combine(params_enc, static_enc)
        decoder = eqx.combine(params_dec, static_dec)
    except FileNotFoundError as e:
        logger.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
    assert encoder, decoder

    test_y, test_x = [
        v.drop([col for col in v.columns if col.startswith("Missing") or col in {"__index_level_0__", "los_icu"}])
        for v in data["test"].values()
    ]

    sid = np.random.choice(test_x["stay_id"], 1)

    p_x = test_x.filter(pl.col("stay_id").is_in(sid)).sort("time")
    p_y = test_y.filter(pl.col("stay_id").is_in(sid)).sort("time")

    dummy_keys = jnp.zeros((len(p_x), 2), jnp.uint32)
    out = jax.vmap(encoder)(to_input(p_x), key=dummy_keys)
    alpha, beta, sigma, direct_sofa, direct_inf, temp_lookup, temp_label, thresholds = out

    print(p_y.head(10))

    z = jnp.concatenate(
        [
            alpha * (ALPHA_SPACE[1] - ALPHA_SPACE[0]) + ALPHA_SPACE[0],
            beta * (BETA_SPACE[1] - BETA_SPACE[0]) + BETA_SPACE[0],
            sigma * (SIGMA_SPACE[1] - SIGMA_SPACE[0]) + SIGMA_SPACE[0],
        ],
        axis=-1,
    )

    print(z.shape)
    pred_concepts = lookup_table.soft_get_full(z, jnp.full((z.shape[0], 1), 0.002))

    # TODO test hard vs soft get

    # pred_sofa_logits = ordinal_logits(pred_concepts[:, 0])  # [B, 23]
    pred_sofa_logits = ordinal_logits(direct_sofa, thresholds, temp_label)  # [B, 23]
    pred_sofa = jnp.abs(jnp.argmin(jnp.abs(ordinal_logits(direct_sofa, thresholds, temp_label)), axis=1))  # [B, 23]
    pred_infs_logits = binary_logits(pred_concepts[:, 1])
    pred_infs = pred_concepts[:, 1]

    targets = jnp.arange(24 - 1)  # [0, 1, ..., 22]
    true_sofa = jnp.array(p_y["sofa"]).squeeze()
    true_sofa_logits = (true_sofa[:, None] > targets[None, :]).astype(jnp.float32)  # [B, 23]
    true_infs = jnp.array(p_y["susp_inf_alt"]).squeeze()
    true_infs_logits = binary_logits(true_infs)

    t = 12
    te = 25
    print("SOFA scores @ [0,9]")
    print("True:", true_sofa[:te])
    print("Pred:", pred_sofa[:te])
    print(f"Logits@{t}")
    print("True:", true_sofa_logits[t], true_sofa[t])
    print("Pred:", pred_sofa_logits[t], pred_sofa[t])

    print("\n" * 2)
    print("Infections @ [0,9]")
    print("True:", true_infs[:te])
    print("Pred:", pred_infs[:te])
    print(f"Logits@{t}")
    print("True:", true_infs_logits[t])
    print("Pred:", pred_infs_logits[t])

    scatter_concepts(p_y, pred_sofa, pred_infs)
    plt.show()
    # viz_latent(
    #     params=params,
    #     encoder=encoder,
    #     p_x=p_x,
    #     p_y=p_y,
    #     metrics=metrix,
    #     filename="latent_z",
    #     figure_dir="figures/model",
    # )

