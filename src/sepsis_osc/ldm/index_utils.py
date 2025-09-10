import numpy as np

def as_3d_indices(
    alpha_space: tuple[float, float, float],
    beta_space: tuple[float, float, float],
    sigma_space: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.arange(*alpha_space)
    betas = np.arange(*beta_space)
    sigmas = np.arange(*sigma_space)
    alpha_grid, beta_grid, sigma_grid = np.meshgrid(alphas, betas, sigmas, indexing="ij")
    permutations = np.stack([alpha_grid, beta_grid, sigma_grid], axis=-1)
    a, b, s = permutations[:, :, :, 0:1], permutations[:, :, :, 1:2], permutations[:, :, :, 2:3]
    return a, b, s


def as_2d_indices(
    x_space: tuple[float, float, float],
    y_space: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(*x_space)
    ys = np.arange(*y_space)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    return x_grid, y_grid


def flatten_dict(d, parent_key="", sep="_"):
    return (
        {
            f"{k}" if parent_key else k: v
            for pk, pv in d.items()
            for k, v in flatten_dict(pv, f"{parent_key}{sep}{pk}" if parent_key else pk, sep).items()
        }
        if isinstance(d, dict)
        else {parent_key: d}
    )
