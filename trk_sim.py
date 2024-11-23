import numpy as np
import torch
# from scipy.integrate import RK45
import json
from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(69)
N = 200
N_kappa = (N) ** 2
alpha = -0.28 * np.pi  # phase lag
beta = 0.66 * np.pi  # age parameter
a_1 = 1
epsilon_1 = 0.03  # adaption rate
epsilon_2 = 0.3  # adaption rate
sigma = 1
omega_1 = omega_2 = 0
T_init, T_dyn, T_max = 0, 0, 2000
T_step = 0.05

omega_1_i = (torch.ones((N,)) * omega_1).to(device)
omega_2_i = (torch.ones((N,)) * omega_2).to(device)

a_1_ij = (torch.ones((N, N)) * a_1)
a_1_ij = a_1_ij.fill_diagonal_(0).type(torch.float64).to(device)  # NOTE no self coupling


def deriv(t: float, y: torch.Tensor) -> torch.Tensor:
    # expect 1d stacked array (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN))
    # recover states from 1d array and enforce bounds
    phi_1_i = y[0 * N : 1 * N] % (torch.pi * 2)
    phi_2_i = y[1 * N : 2 * N] % (torch.pi * 2)

    kappa_1_ij = y[2 * N + 0 * N_kappa : 2 * N + 1 * N_kappa].reshape(N, N).fill_diagonal(0).clip(-1, 1)
    kappa_2_ij = y[2 * N + 1 * N_kappa : 2 * N + 2 * N_kappa].reshape(N, N).fill_diagonal(0).clip(-1, 1)

    # track bounded variables
    y[0 * N : 1 * N] = phi_1_i
    y[1 * N : 2 * N] = phi_2_i
    y[2 * N + 0 * N_kappa : 2 * N + 1 * N_kappa] = kappa_1_ij.flatten()
    y[2 * N + 1 * N_kappa : 2 * N + 2 * N_kappa] = kappa_2_ij.flatten()

    # assert symmetry of coupling matrices
    assert torch.allclose(kappa_1_ij, kappa_1_ij.T), torch.max(torch.abs(kappa_1_ij - kappa_1_ij.T))
    assert torch.allclose(kappa_2_ij, kappa_2_ij.T), torch.max(torch.abs(kappa_2_ij - kappa_2_ij.T))
    
    # sin in radians
    # phi_i coupled with itself since we do the whole N?
    phi_1_diff = phi_1_i[:, torch.newaxis] - phi_1_i
    dphi_1_i = (
        omega_1_i
        - (1/(N-1)) * torch.sum((a_1_ij + kappa_1_ij) * torch.sin(phi_1_diff + alpha), axis=1)
        - sigma * torch.sin(phi_1_i - phi_2_i)
    )
    dkappa_1_ij = -epsilon_1 * (kappa_1_ij + torch.sin(phi_1_diff - beta))
    dkappa_1_ij = (dkappa_1_ij + dkappa_1_ij.T) / 2  # keep symmetry
    dkappa_1_ij = dkappa_1_ij.fill_diagonal_(0)

    phi_2_diff = phi_2_i[:, torch.newaxis] - phi_2_i
    dphi_2_i = (
        omega_2_i 
        - (1/(N-1)) * torch.sum(kappa_2_ij * torch.sin(phi_2_diff + alpha), axis=1) 
        - sigma * torch.sin(phi_2_i - phi_1_i)
    )
    dkappa_2_ij = -epsilon_2 * (kappa_2_ij + torch.sin(phi_2_diff - beta))
    dkappa_2_ij = (dkappa_2_ij + dkappa_2_ij.T) / 2  # keep symmetry
    dkappa_2_ij = dkappa_2_ij.fill_diagonal_(0)

    # return 1d stacked array (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN))
    dy = torch.concatenate([dphi_1_i, dphi_2_i, dkappa_1_ij.flatten(), dkappa_2_ij.flatten()])
    return dy


phi_1_init = torch.from_numpy(np.random.rand(N) * (2 * torch.pi))
phi_2_init = torch.from_numpy(np.random.rand(N) * (2 * torch.pi))

kappa_1_init = torch.tensor(np.random.rand(N, N) * 2 - 1)
kappa_1_init = (kappa_1_init + kappa_1_init.T) / 2  # make symmetric
kappa_1_init = kappa_1_init.fill_diagonal_(0)

kappa_2_init = torch.ones((N,N))
kappa_2_init = (kappa_2_init + kappa_2_init.T) / 2
kappa_2_init = kappa_2_init.fill_diagonal_(0)
kappa_2_init[40:, :40] = 0
kappa_2_init[:40, 40:] = 0
init_condition = torch.concatenate(
    [phi_1_init, phi_2_init, kappa_1_init.flatten(), kappa_2_init.flatten()]
).to(device)

t = torch.arange(0, T_max).type(torch.float64).to(device)
res = odeint(deriv, init_condition, t, method="rk4", options=dict(step_size=0.05))
print(res)

t_vals = np.concat([np.array([-1]), t.cpu().numpy()]) + 1
y_vals = np.concat([[init_condition.cpu().numpy()], res.cpu().numpy()]).astype(np.float16)

print(y_vals.shape)

dir_name = "test"
np.save(f"{dir_name}/t_vals", t_vals)
np.save(f"{dir_name}/y_vals", y_vals)

info = {
    "N": N,
    "N_kappa": N_kappa,
    "alpha": alpha,
    "beta": beta,
    "a_1": a_1,
    "epsilon_1": epsilon_1,
    "epsilon_2": epsilon_2,
    "sigma": sigma,
    "omega_1": omega_1,
    "omega_2": omega_2,
    "T_init": T_init,
    "T_dyn": T_dyn,
    "T_max": T_max,
    "T_step": T_step,
    "T_tot": len(t_vals),
}
with open(f"{dir_name}/info.json", 'w') as json_file: 
    json.dump(info, json_file)

