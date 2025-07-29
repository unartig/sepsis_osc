import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm

from sepsis_osc.model.data_loading import get_raw_data, prepare_batches
from sepsis_osc.model.model_utils import as_3d_indices, load_checkpoint, LoadingConfig
from sepsis_osc.model.train import (
    ALPHA_SPACE,
    BETA_SPACE,
    SIGMA_SPACE,
    binary_logits,
    ordinal_logits,
    constrain_z,
    bound_z,
)
from sepsis_osc.model.ae import (
    Decoder,
    Encoder,
)
from sepsis_osc.dnm.data_classes import JAXLookup, SystemConfig
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import jax_random_seed
from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.visualisations.viz_param_space import space_plot
from sepsis_osc.visualisations.viz_three_dee import three_dee

LOAD_FROM_CHECKPOINT = "runs/Jul22_13-41-57_tinkpad"
LOAD_EPOCH = 69

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def viz_latent(df, params, p_y, metrics, indices, key, filename, figure_dir):
    print(metrics.shape)
    print(indices.shape)
    mask = (
        (indices[..., 0] >= df["alpha"].min() * 0.5)
        & (indices[..., 0] <= df["alpha"].max() * 1.5) &
         (indices[..., 1] >= df["beta"].min() * 0.5)
        & (indices[..., 1] <= df["beta"].max() * 1.5)
        & (indices[..., 2] >= df["sigma"].min() * 0.5)
        & (indices[..., 2] <= df["sigma"].max() * 1.5)
    )

    # metrics = metrics[indices[indices[]]]
    fig = three_dee(
        params[mask],
        # params,
        # np.asarray(metrics.s_1),
        np.asarray(metrics.s_1[mask]),
        "Parameter Space Cluster Ratio 1 + SOFA prediction",
        filename,
        "figures/model",
        show=False,
    )
    print(df)
    new_fig = go.Figure(
        data=go.Scatter3d(
            x=df["sigma"],
            y=df["beta"],
            z=df["alpha"],
            mode="markers",
            marker=dict(
                size=3,  # smaller marker size
                color=p_y[:, 0],  # color by this array
                colorscale="Cividis",  # choose your colormap
                colorbar=dict(title="Actual Sofa-score", x=1.1),  # optional colorbar
                opacity=0.8,
            ),
        )
    )
    if filename:
        fig.write_html(f"{figure_dir}/{filename}.html")
        fig = fig.add_traces(new_fig.data)
        new_fig.write_html(f"{figure_dir}/new_{filename}.html")
        fig.write_html(f"{figure_dir}/both_{filename}.html")


def viz_plane(params, model, p_x, p_y, lookup, key, filename, figure_dir):
    filename = filename + "_plane"
    df = do_inference(p_x, p_y, model.encoder, key)
    betas = np.arange(BETA_SPACE[0], BETA_SPACE[1], BETA_SPACE[2])
    sigmas = np.arange(SIGMA_SPACE[0], SIGMA_SPACE[1], SIGMA_SPACE[2])
    alphas = np.array(df["alpha"].mean())
    alpha_grid, beta_grid, sigma_grid = np.meshgrid(alphas, betas, sigmas, indexing="ij")
    param_grid = np.stack([alpha_grid.ravel(), beta_grid.ravel(), sigma_grid.ravel()], axis=1)

    metrics = (
        lookup.hard_get_local(param_grid, temperatures=np.ones_like(param_grid) * 1e4)
        .reshape(len(betas), len(sigmas), 2)
        .swapaxes(0, 1)[::-1, :, :]
    )
    std_sym = r"$s^{\mu}$"
    ax = space_plot(
        metrics[..., 0],
        betas,
        sigmas,
        rf"Disease Progression over {std_sym} space @$\alpha=${df['alpha'].mean():.2f}",
        filename,
        figure_dir,
    )
    print(df)
    cm = plt.cm.get_cmap("hot")
    beta_scale = len(betas) * (df["beta"] - BETA_SPACE[0]) / (BETA_SPACE[1] - BETA_SPACE[0])
    sigma_scale = len(sigmas) * (1 - (df["sigma"] - SIGMA_SPACE[0]) / (SIGMA_SPACE[1] - SIGMA_SPACE[0]))
    ax.scatter(beta_scale, sigma_scale, c=p_y[:, 0], cmap=cm)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar2.set_label("Actual SOFA-score")
    if filename:
        plt.tight_layout()  # Adjust layout
        plt.savefig(f"{figure_dir}/{filename}.svg", format="svg")


def scatter_concepts(
    true_sofa,
    true_infs,
    pred_sofa: jnp.ndarray,
    pred_infs: jnp.ndarray,
    ax=None,
    figure_dir="figures",
    file_name="sofa.png",
):
    # if ax is None:
    #     fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    # ax[0].plot(range(0, 24), range(0, 24), color="tab:red", label="optimum")
    # ax[0].scatter(true_sofa, pred_sofa, alpha=0.01)
    # ax[1].scatter(true_infs, pred_infs)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(range(0, 24), range(0, 24), color="tab:red", label="optimum")
    # ax.scatter(true_sofa, pred_sofa, alpha=1)
    ax.scatter(true_sofa, pred_sofa, alpha=0.05, color="tab:orange")
    ax.set_ylabel("Predicted SOFA-score")
    ax.set_xlabel("Actual SOFA-score")

    plt.savefig(f"{figure_dir}/{file_name}", transparent=True, dpi=800)
    return ax


def pad_to_length(arr, target_len, pad_value=0.0):
    pad_len = target_len - arr.shape[0]
    return jnp.pad(arr, ((0, pad_len), (0, 0)), constant_values=pad_value)


def unpad(arr, len):
    return arr[:len]


def to_input(df):
    return np.array(df.drop(["stay_id", "time"]), dtype=np.float32)


@eqx.filter_jit
def process_batch(x_batch, y_batch, model, enc_keys, direct=False):
    temp_lookup, temp_label, thresholds = model.get_parameters()
    alpha0, beta0, sigma0, _ = jax.vmap(model.encoder)(x_batch, dropout_keys=enc_keys)

    z0 = constrain_z(jnp.concatenate([alpha0, beta0, sigma0], axis=-1))

    pred_concepts = lookup_table.hard_get_local(z0, jnp.full((z0.shape[0], 1), temp_lookup))

    pred_sofa_logits = ordinal_logits(pred_concepts[:, 0], thresholds, temp_label)
    pred_sofa = jnp.abs(jnp.argmin(jnp.abs(pred_sofa_logits), axis=1))
    pred_infs = pred_concepts[:, 1]

    true_sofa = y_batch[:, 0]
    true_infs = y_batch[:, 1]

    return {
        "pred_sofa": pred_sofa,
        "pred_infs": pred_infs,
        "alpha": z0[..., 0].squeeze(),
        "beta": z0[..., 1],
        "sigma": z0[..., 2],
        "true_sofa": true_sofa,
        "true_infs": true_infs,
    }


def do_inference(x, y, model, key):
    x_len = x.shape[0]
    if x_len < BATCH_SIZE:
        x = pad_to_length(x, BATCH_SIZE)
        y = pad_to_length(y, BATCH_SIZE)

    key, batch_key = jr.split(key)
    x_batches, y_batches, nval_batches = prepare_batches(x, y, BATCH_SIZE, key, shuffle=False)

    num_batches = len(x_batches)
    total_rows = num_batches * BATCH_SIZE

    # Run one batch to figure out what outputs we need to allocate
    x_sample = x_batches[0]
    y_sample = y_batches[0]
    sample_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(BATCH_SIZE))

    def make_keys(base_key):
        return jr.split(base_key, 4)

    keys = jax.vmap(make_keys)(sample_keys)

    sample_result = process_batch(x_sample, y_sample, model, keys)

    # Preallocate storage for each key (only 1D arrays)
    output_buffers = {
        k: np.empty(total_rows, dtype=np.array(v).dtype) for k, v in sample_result.items() if np.array(v).ndim == 1
    }

    # Fill in the buffer batch-by-batch
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(zip(x_batches, y_batches), total=num_batches)):
        keys = jax.vmap(make_keys)(sample_keys)

        batch_result = process_batch(x_batch, y_batch, model, keys)
        offset = batch_idx * BATCH_SIZE

        for k, v in batch_result.items():
            v_np = np.array(v)
            if v_np.ndim == 1:
                output_buffers[k][offset : offset + BATCH_SIZE] = v_np

    # Truncate to original input length if padded
    if x_len < total_rows:
        for k in output_buffers:
            output_buffers[k] = output_buffers[k][:x_len]

    df = pl.DataFrame(output_buffers)
    print(df)
    return df


def process_sequence(x, y, model, key):
    preds = []
    temp_lookup, temp_label, thresholds, = model.get_parameters()

    alpha0, beta0, sigma0, h_next = model.encoder(x[0], dropout_keys=jnp.array(jr.split(key, 4)))
    z0 = constrain_z(jnp.concatenate([alpha0, beta0, sigma0], axis=-1))

    pred_concepts = lookup_table.hard_get_local(z0[None, :], jnp.asarray(temp_lookup))
    pred_sofa_logits = (pred_concepts[..., 0, None] - thresholds) / temp_lookup
    pred_sofa = jnp.abs(jnp.argmin(jnp.abs(pred_sofa_logits), axis=1))
    pred_infs = pred_concepts[:, 1]

    preds.append({
        "pred_sofa": np.asarray(pred_sofa),
        "pred_infs": np.asarray(pred_infs),
        "alpha": alpha0.squeeze(),
        "beta": beta0.squeeze(),
        "sigma": sigma0.squeeze(),
        "true_sofa": y[0, 0],
        "true_infs": y[0, 1],
    })

    # h_next = jnp.zeros((model.predictor.hidden_dim))
    z_next = z0
    for t in range(1, x.shape[0]):
        z_prev = z_next
        h_prev = h_next
        z_next, h_next = model.predictor(z_prev, h_prev)
        z_next = bound_z(z_next)

        pred_concepts = lookup_table.hard_get_local(z_next[None, :], jnp.asarray(temp_lookup))
        pred_sofa_logits = (pred_concepts[..., 0, None]- thresholds) / temp_lookup
        pred_sofa = jnp.abs(jnp.argmin(jnp.abs(pred_sofa_logits), axis=1))
        pred_infs = pred_concepts[:, 1]
        preds.append({
            "pred_sofa": np.asarray(pred_sofa),
            "pred_infs": np.asarray(pred_infs),
            "alpha": z_next[..., 0].squeeze(),
            "beta": z_next[..., 1].squeeze(),
            "sigma": z_next[..., 2].squeeze(),
            "true_sofa": y[t, 0],
            "true_infs": y[t, 1],
        })
    df = pl.DataFrame(preds)
    return df


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    key = jr.PRNGKey(jax_random_seed)
    BATCH_SIZE = 256
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
    a, b, s = as_3d_indices(ALPHA_SPACE, BETA_SPACE, SIGMA_SPACE)
    indices_3d = jnp.concatenate([a, b, s], axis=-1)
    spacing_3d = jnp.array([ALPHA_SPACE[2], BETA_SPACE[2], SIGMA_SPACE[2]])
    params = SystemConfig.batch_as_index(a, b, s, 0.2)
    metrics_3d, _ = storage.read_multiple_results(np.asarray(params))

    print(indices_3d.shape, metrics_3d.shape)
    lookup_table = JAXLookup(
        metrics=metrics_3d.copy().reshape((-1, 1)),
        indices=indices_3d.copy().reshape((-1, 3)),  # since param space only alpha beta sigma
        metrics_3d=metrics_3d,
        indices_3d=indices_3d,
        grid_spacing=spacing_3d,
    )

    if not metrics_3d:
        exit(0)

    load_conf = LoadingConfig(from_dir=LOAD_FROM_CHECKPOINT, epoch=LOAD_EPOCH)
    model, encoder, decoder = None, None, None
    if load_conf.from_dir:
        try:
            model, _ = load_checkpoint(load_conf.from_dir + "/checkpoints", load_conf.epoch, None)

            logger.info(f"Resuming training from epoch {load_conf.epoch + 1}")
        except FileNotFoundError as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting training from scratch.")
            load_conf.epoch = 0
            load_conf.from_dir = ""

    model = eqx.nn.inference_mode(model)
    assert model
    encoder, decoder = model.encoder, model.decoder
    assert encoder, decoder

    _, _, _, _, test_y, test_x = [
        v.drop([
            col for col in v.columns if col.startswith("Missing") or col in {"sep3_alt", "__index_level_0__", "los_icu"}
        ])
        for inner in get_raw_data().values()
        for v in inner.values()
    ]

    # results = do_inference(to_input(test_x), to_input(test_y), model, key)
    # scatter_concepts(results["true_sofa"], results["true_infs"], results["pred_sofa"], results["pred_infs"], figure_dir="figures/poster")
    # print(results.min())
    # print(results.max())

    stay_counts = test_x.group_by("stay_id").len()
    sofa_std = test_y.group_by("stay_id").agg([pl.col("sofa").std().alias("sofa_std")])
    valid_stays = stay_counts.filter(pl.col("len") <= 15).get_column("stay_id")
    valid_sofas = sofa_std.filter(pl.col("sofa_std") > 1.5).get_column("stay_id")
    valids = np.intersect1d(valid_stays.to_numpy(), valid_sofas.to_numpy())
    for sid in valids:
        print(sid)
        p_x = to_input(test_x.filter(pl.col("stay_id").is_in(sid)).sort("time"))
        p_y = to_input(test_y.filter(pl.col("stay_id").is_in(sid)).sort("time"))
        df = process_sequence(p_x, p_y, model, key)
        print(df)
    
    sid = np.random.choice(valids, 1)
    # sid = [244038]
    # sid = [260152]
    sid = [205200]
    # sid = [294100]
    # sid = [209826]
    # sid = [226863]
    sid =  [227512]
    p_x = to_input(test_x.filter(pl.col("stay_id").is_in(sid)).sort("time"))
    p_y = to_input(test_y.filter(pl.col("stay_id").is_in(sid)).sort("time"))
    print(p_x.shape)

    print(test_y.filter(pl.col("stay_id").is_in(sid)).sort("time"))
    # df = do_inference(p_x, p_y, model, key)
    # p_x = p_x[1:]
    # p_y =  p_y[1:]
    df = process_sequence(p_x, p_y, model, key)
    p_y = p_y[3:]
    df = df[3:, :]
    viz_latent(
        params=params,
        df=df,
        p_y=p_y,
        metrics=metrics_3d,
        indices=indices_3d,
        key=key,
        filename="latent_z",
        figure_dir="figures/model",
    )

    # valids = np.intersect1d(valid_stays.to_numpy(), valid_sofas.to_numpy())
    # for sid in valids:
    #     p_x = to_input(test_x.filter(pl.col("stay_id").is_in(sid)).sort("time"))
    #     p_y = to_input(test_y.filter(pl.col("stay_id").is_in(sid)).sort("time"))
    #     viz_plane(
    #         params=params,
    #         model=encoder,
    #         p_x=p_x,
    #         p_y=p_y,
    #         lookup=lookup_table,
    #         key=key,
    #         filename="latent_z",
    #         figure_dir="figures/model",
    #     )
    storage.close()
    plt.show()
