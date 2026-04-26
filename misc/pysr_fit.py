from scipy.ndimage import uniform_filter1d, gaussian_filter, gaussian_gradient_magnitude
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from pysr import PySRRegressor, TensorBoardLoggerSpec

from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics, DNMState, DynamicNetworkModel
from sepsis_osc.ldm.commons import build_lookup_table
from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
from sepsis_osc.storage.storage_interface import Storage
from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed, plt_params
from sepsis_osc.visualisations.viz_param_space import pretty_plot, space_plot

db_str = "DaisyFinal"
sim_storage = Storage(
    key_dim=9,
    metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
    parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
    use_mem_cache=True,
)
sim_storage.close()
lookup_table = build_lookup_table(sim_storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)

betas, sigmas = np.arange(*BETA_SPACE), np.arange(*SIGMA_SPACE)

beta_2d, sigma_2d = as_2d_indices(BETA_SPACE, SIGMA_SPACE)
a = np.ones_like(beta_2d) * ALPHA
params = DNMConfig.batch_as_index(a, beta_2d, sigma_2d, 0.2)
metrics_3d, _ = sim_storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
metrics_2d = metrics_3d.reshape([1, *metrics_3d.shape["r_1"]]).squeeze()

s1 = metrics_2d.s_1

# ###########################
# s1_smooth = gaussian_filter(s1, sigma=2.0)

# # Gradient magnitude in 2D
# grad_mag = gaussian_gradient_magnitude(s1_smooth, sigma=2.0)

# # Threshold to get the edge region
# threshold = 0.38 * grad_mag.max()
# edge_mask = grad_mag > threshold

# # Extract coordinates
# edge_i, edge_j = np.where(edge_mask)           # row=beta, col=sigma
# edge_betas  = betas[edge_i]
# edge_sigmas = sigmas[edge_j]

# Plot
# fig, ax = plt.subplots()
# ax.imshow(s1, origin="lower", aspect="auto",
#           extent=(sigmas[0], sigmas[-1], betas[0], betas[-1]),
#           cmap="viridis")
# ax.scatter(edge_sigmas, edge_betas, s=1, color="red", label="boundary")
# plt.show()

boundary_model = PySRRegressor(
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "square", "exp", "log"],
    maxsize=12,
    niterations=10000,
)

def c(s):
    return -(s-0.6)**(1/32) + s * -np.exp(-s)/100 + 1.35 + np.sin(s - 0.11)/3.9

sigma_app = np.linspace(0.6, 1.5, 100).astype(np.float64)
cs = c(sigma_app.reshape(-1, 1))
boundary_model.fit(
    sigma_app.reshape(-1, 1),
    c(sigma_app.reshape(-1, 1)),
)

y_pred = boundary_model.predict(sigma_app.reshape(-1, 1))

fig, ax = plt.subplots()
ax.imshow(s1.T, origin="lower", aspect="auto",
          extent=(betas[0], betas[-1], sigmas[0], sigmas[-1]),
          cmap="viridis")
ax.plot(np.clip(c(sigma_app), .4, .7), sigma_app)
ax.plot(np.clip(y_pred, .4, .7), sigma_app)
ax.set_title("fitted boundary")
plt.show()
#
# ###########################

# logger_spec = TensorBoardLoggerSpec(
#     log_dir="runs/pysr/run",
#     log_interval=10,  # Log every 10 iterations
# )


# model = PySRRegressor(
#     populations=50,
#     population_size=35,
#     maxsize=30,
#     niterations=100000,
#     ncycles_per_iteration=200,
#     binary_operators=[
#         "+",
#         "*",
#         "-",
#         "/",
#         "^",
#     ],
#     unary_operators=[
#         "exp",
#         "cos",
#         # "acos",
#         # "cosh",
#         "sin",
#         # "asin",
#         # "sinh",
#         "inv",
#         "log",
#         # "pow",
#         # "neg",
#         # "tan",
#         # "tanh",
#         "square",
#         "sqrt",
#         # "cube",
#         # "cbrt"
#         "softstep(x) = 1 / (1 + exp(-10 * x))",
#         "bump(x) = x * exp(-x)",
#     ],
#     extra_sympy_mappings={
#         "softstep": lambda x: 1 / (1 + sp.exp(-10 * x)),
#         "bump": lambda x: x * sp.exp(-x),
#     },
#     constraints={"^": (-1, 1)},
#     nested_constraints={
#         "softstep": {"softstep": 0, "bump": 0},
#         "bump": {"bump": 0, "softstep": 0},
#         "exp": {"exp": 0, "bump": 0},
#     },
#     turbo=True,
#     logger_spec=logger_spec,
#     parsimony=0.01,
#     adaptive_parsimony_scaling=1000,
#     warmup_maxsize_by=0.2,
# )


# y = s1.ravel()

# # Stack features into X
# X = np.column_stack([beta_2d.ravel(), sigma_2d.ravel()])
# ymax = y.max()
# model.fit(X, y / ymax)


# print(model.latex)
# y_pred_flat = model.predict(X)
# y_pred = y_pred_flat.reshape(beta_2d.shape) * y.max()


# space_plot(y_pred, betas, sigmas, "Predicted Space")


# space_plot((y_pred - s1), betas, sigmas, "Error")

# plt.show()
