import numpy as np
import matplotlib.pyplot as plt
import json

dir_name = "test"
ts, y_vals = np.load(f"{dir_name}/t_vals.npy"), np.load(f"{dir_name}/y_vals.npy")

print(ts.shape, y_vals.shape)
with open(f"{dir_name}/info.json", 'r') as json_file:
    info = json.load(json_file)


N = info["N"]
phis = y_vals[:, :N*2]
dphis = np.gradient(phis, axis=1)

kappas1 = y_vals[:, N*2:N*2+N*N].reshape(ts.shape[0], N, N)
kappas2 = y_vals[:, N*2+N*N:].reshape(ts.shape[0], N, N)

# sort
# sort_ind1 = np.argsort(phis[-1,:N])
# sort_ind2 = np.argsort(phis[-1, N:2*N])
sort_ind1 = np.lexsort((dphis[:, :N].mean(axis=0), phis[-1,:N]))
sort_ind2 = np.lexsort((dphis[:, N:2*N].mean(axis=0), phis[-1, N:2*N]))

phis = phis[:, np.concatenate([sort_ind1, N+sort_ind2])]
dphis = dphis[:, np.concatenate([sort_ind1, N+sort_ind2])]
# kappas1 = kappas1[:, np.ix_(sort_ind1, sort_ind1)[0], np.ix_(sort_ind1, sort_ind1)[1]]
# kappas2 = kappas2[:, np.ix_(sort_ind2, sort_ind2)[0], np.ix_(sort_ind2, sort_ind2)[1]]
print(N)
print(phis.shape)
print(dphis.shape)
print(kappas1.shape)


def mean_phase_velos(ts: np.ndarray, dphis: np.ndarray, t_1=0, t_2=-1):
    i_1 = np.argmin(np.abs(ts - t_1))
    i_2 = np.argmin(np.abs(ts - t_2)) if t_2 != -1 else -1
    dphi_1, dphi_2 = dphis[i_1], dphis[i_2]
    return (dphi_2 - dphi_1)/(ts[i_2]-ts[i_1])


def plot_phase_velos(ts: np.ndarray, dphis: np.ndarray, t_1=1100, t_2=1300, ax=None):
    if not ax:  # we want ax to have 2 elements
        fig, ax = plt.subplots(1, 2)
    ax[0].scatter(np.arange(N*2), np.abs(dphis.mean(axis=0) - mean_phase_velos(ts, dphis, t_1, t_2)))
    ax[1].scatter(np.arange(N*2), dphis.mean(axis=0))


def plot_phase_snapshot(phis: np.ndarray, t_ind=-1, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ind = np.argmin(np.abs(ts - t_ind)) if t_ind != -1 else -1
    phis = phis[ind]
    ax.scatter(np.arange(N*2), phis)


def plot_space_time_phase(phis: np.ndarray, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.matshow(phis)


def plot_kappa(ts: np.ndarray, kappas: np.ndarray, t_ind=-1, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ind = np.argmin(np.abs(ts - t_ind)) if t_ind != -1 else -1
    ax.matshow(kappas[ind, ::-1])

# plot_phase_velos(ts, dphis)
plot_space_time_phase(phis)
plot_phase_snapshot(phis, 1000)
# plot_kappa(ts, kappas1)

# print(ts)
# plot_phase_snapshot(phis, 0)
plot_kappa(ts, kappas1, t_ind=0)
plot_kappa(ts, kappas2, t_ind=0)
plt.show()
