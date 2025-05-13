import logging

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import polars as pl

from sepsis_osc.simulation.data_classes import SystemConfig, SystemMetrics
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

num_ticks = 20


def three_dee(
    param_inds: np.ndarray,
    metric: np.ndarray,
    title: str,
    filename: str,
    figure_dir: str,
    show: bool = False,
):
    indices_flat = param_inds.reshape(-1, 9)
    parenchymal = metric.reshape(-1, 1)
    data_parenchymal = []
    for i in range(indices_flat.shape[0]):
        data_parenchymal.append({
            "alpha": indices_flat[i, 5],
            "beta": indices_flat[i, 6],
            "sigma": indices_flat[i, 7],
            "value": parenchymal[i],
            "size": parenchymal[i] + 1,
        })
    df_parenchymal = pl.DataFrame(data_parenchymal)

    # Sort and get unique grid values
    x_vals = np.sort(df_parenchymal.select("sigma").unique().to_numpy().flatten())
    y_vals = np.sort(df_parenchymal.select("beta").unique().to_numpy().flatten())
    z_vals = np.sort(df_parenchymal.select("alpha").unique().to_numpy().flatten())
    df_sorted = df_parenchymal.sort(["sigma", "beta", "alpha"])
    values = df_sorted.select("value").to_numpy().flatten()

    # Reshape into a 3D array (assuming regular grid)
    values_3d = values.reshape(len(x_vals), len(y_vals), len(z_vals))

    # Create meshgrid for plotting
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values_3d.flatten(),
            isomin=np.min(values_3d),
            isomax=np.max(values_3d),
            surface_count=20,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale="Viridis",
            opacity=0.3,
        )
    )

    fig.update_layout(scene=dict(xaxis_title="Sigma", yaxis_title="Beta", zaxis_title="Alpha"), title=title)
    if show:
        fig.show()
    if filename:
        fig.write_html(f"{figure_dir}/{filename}.html")


betas = np.arange(0.2, 1.5, 0.02)
sigmas = np.arange(0.0, 1.5, 0.04)
alphas = np.arange(0.0, 1.0, 0.04)


size = (len(betas), len(sigmas), len(alphas))
db_str = "Daisy"  # other/Tiny"
storage = Storage(
    key_dim=9,
    metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
    parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
    use_mem_cache=False,
)
params = np.ndarray((*size, 9))

for x, beta in enumerate(betas):
    for y, sigma in enumerate(sigmas):
        for z, alpha in enumerate(alphas):
            N = 100
            run_conf = SystemConfig(
                N=100,
                C=int(0.2 * N),
                omega_1=0.0,
                omega_2=0.0,
                a_1=1.0,
                epsilon_1=0.03,
                epsilon_2=0.3,
                alpha=float(alpha),
                beta=float(beta),
                sigma=float(sigma),
            )
            params[x, y, z] = np.array(run_conf.as_index)

metrix = storage.read_multiple_results(params)
storage.close()
if not metrix:
    exit(0)


print(metrix.shape)
log = False
show = False
figure_dir = "figures"
fs = (8, 8)
three_dee(params, np.asarray(metrix.r_1), "Parameter Space Kuramoto 1", "kuramoto1", "figures/3d")
three_dee(params, np.asarray(metrix.r_2), "Parameter Space Kuramoto 2", "kuramoto2", "figures/3d")
three_dee(params, np.asarray(metrix.s_1), "Parameter Space STD 1", "std1", "figures/3d")
three_dee(params, np.asarray(metrix.s_2), "Parameter Space STD 2", "std2", "figures/3d")
three_dee(params, np.asarray(metrix.m_1), "Parameter Space Mean 1", "mean1", "figures/3d")
three_dee(params, np.asarray(metrix.m_2), "Parameter Space Mean 2", "mean2", "figures/3d")
three_dee(params, np.asarray(metrix.q_1), "Parameter Space Entropy 1", "entropy1", "figures/3d")
three_dee(params, np.asarray(metrix.q_2), "Parameter Space Entropy 2", "entropy2", "figures/3d")
three_dee(params, np.asarray(metrix.f_1), "Parameter Space Cluster Ratio 1", "cluster1", "figures/3d")
three_dee(params, np.asarray(metrix.f_2), "Parameter Space Cluster Ratio 2", "cluster2", "figures/3d")
three_dee(params, np.asarray(metrix.sr_1), "Parameter Space Splay Ratio 1", "splay1", "figures/3d")
three_dee(params, np.asarray(metrix.sr_2), "Parameter Space Splay Ratio 2", "splay2", "figures/3d")
three_dee(params, np.asarray(metrix.tt), "Parameter Space Transient Time", "tt1", "figures/3d")
