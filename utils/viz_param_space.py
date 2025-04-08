import logging

import numpy as np
import matplotlib.pyplot as plt

from storage.storage_interface import Storage
from utils.logger import setup_logging
from simulation.data_classes import SystemConfig

setup_logging()
logger = logging.getLogger(__name__)

size = (100, 100)
mat1 = np.zeros(size)
mat2 = np.zeros(size)
xs = np.linspace(0.4, 0.7, size[0])
ys = np.linspace(0, 1.5, size[1])
storage = Storage(
    key_dim=9,
    metrics_kv_name="storage/100x100N100_02CSepsisMetrics.db/",
    parameter_k_name="storage/100x100N100_02CSepsisParameters_index.bin",
)
params = np.ndarray((*size, 9))
for x, beta in enumerate(xs):
    for y, sigma in enumerate(ys):
        N = 100
        run_conf = SystemConfig(
            N=100,
            C=int(0.2 * N),
            omega_1=0.0,
            omega_2=0.0,
            a_1=1.0,
            epsilon_1=0.03,
            epsilon_2=0.3,
            alpha=0.66,
            beta=beta,
            sigma=sigma,
        )
        params[x, y] = np.array(run_conf.as_index)
        metrics = storage.read_result(run_conf.as_index, threshold=0.0)
        if metrics:
            mat1[x, y] = np.clip(np.mean(np.asarray(metrics.s_1)), -np.inf, 0.15)
            mat2[x, y] = np.clip(np.mean(np.asarray(metrics.s_2)), -np.inf, 0.15)

# results = storage.read_multiple_results(params, threshold=0.0)
# if not results:
#     exit(0)
# mat1 = np.asarray(results.s_1).mean(axis=-1)
# mat2 = np.asarray(results.s_2).mean(axis=-1)
mat1 = mat1[:, ::-1].T  # np.mean(np.asarray(results.s_1), axis=-1)
mat2 = mat2[:, ::-1].T  # np.mean(np.asarray(results.s_2), axis=-1)
vmin = min(mat1.min(), mat2.min())
vmax = max(mat1.max(), mat2.max())
fig = plt.figure()
ax = fig.subplots(1, 2)
cax1 = ax[0].matshow(mat1, vmin=vmin, vmax=vmax, interpolation="none")
cax2 = ax[1].matshow(mat2, vmin=vmin, vmax=vmax, interpolation="none")
fig.colorbar(cax1, ax=ax, location="right", shrink=0.7)

num_ticks = 5  # Change this for more/less ticks
xtick_positions = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
ytick_positions = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
for a in ax:
    a.set_xticks(xtick_positions)
    a.set_xticklabels([f"{val:.2f}" for val in xs[xtick_positions]], rotation=45)
    a.set_yticks(ytick_positions)
    a.set_yticklabels([f"{val:.2f}" for val in ys[ytick_positions]][::-1])
plt.show()
storage.close()
