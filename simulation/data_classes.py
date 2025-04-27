from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float


@dataclass
class SystemConfig:
    N: int  # number of oscillators per layer
    C: int  # number of infected cells
    omega_1: float  # natural frequency parenchymal layer
    omega_2: float  # natural frequency immune layer
    a_1: float  # intralayer connectivity weights
    epsilon_1: float  # adaption rate
    epsilon_2: float  # adaption rate
    alpha: float  # phase lag
    beta: float  # plasticity (age parameter)
    sigma: float  # interlayer coupling
    T_init: Optional[int] = None
    T_trans: Optional[int] = None
    T_max: Optional[int] = None
    T_step: Optional[float] = None

    @property
    def as_args(self) -> tuple[jnp.ndarray, ...]:
        ja_1_ij = jnp.ones((self.N, self.N), jnp.float64) * self.a_1
        ja_1_ij = ja_1_ij.at[jnp.diag_indices(self.N)].set(0)  # NOTE no self coupling
        jsin_alpha = jnp.sin(self.alpha * jnp.pi)
        jcos_alpha = jnp.cos(self.alpha * jnp.pi)
        jsin_beta = jnp.sin(self.beta * jnp.pi)
        jcos_beta = jnp.cos(self.beta * jnp.pi)
        jpi2 = jnp.array(jnp.pi * 2, jnp.float64)
        jadj = jnp.array(1 / (self.N - 1), jnp.float64)
        diag = (jnp.ones((self.N, self.N), jnp.float64) - jnp.eye(self.N))[None, :]  # again no self coupling
        jepsilon_1 = jnp.array(self.epsilon_1, jnp.float64) * diag
        jepsilon_2 = jnp.array(self.epsilon_2, jnp.float64) * diag
        jomega_1_i = jnp.ones((self.N,), jnp.float64) * self.omega_1
        jomega_2_i = jnp.ones((self.N,), jnp.float64) * self.omega_2
        jsigma = jnp.array(self.sigma, jnp.float64)
        return (
            ja_1_ij,
            jsin_alpha,
            jcos_alpha,
            jsin_beta,
            jcos_beta,
            jpi2,
            jadj,
            jepsilon_1,
            jepsilon_2,
            jsigma,
            jomega_1_i,
            jomega_2_i,
        )

    @property
    def as_index(self) -> tuple[float, ...]:
        return (
            float(self.omega_1),
            float(self.omega_2),
            float(self.a_1),
            float(self.epsilon_1),
            float(self.epsilon_2),
            float(self.alpha / jnp.pi),
            float(self.beta / jnp.pi),
            float(self.sigma / 2),
            float(self.C / self.N),
        )


@dataclass
class SystemState:
    phi_1: Float[Array, "t ensemble N"] | np.ndarray
    phi_2: Float[Array, "t ensemble N"] | np.ndarray
    kappa_1: Float[Array, "t ensemble N N"] | np.ndarray
    kappa_2: Float[Array, "t ensemble N N"] | np.ndarray

    # Required for JAX to recognize it as a PyTree
    def tree_flatten(self):
        return (self.phi_1, self.phi_2, self.kappa_1, self.kappa_2), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def enforce_bounds(self) -> "SystemState":
        self.phi_1 = self.phi_1 % (2 * jnp.pi)
        self.phi_2 = self.phi_2 % (2 * jnp.pi)
        # its written but not actually done
        # self.kappa_1 = jnp.clip(self.kappa_1, -1, 1)
        # self.kappa_2 = jnp.clip(self.kappa_2, -1, 1)
        return self

    def last(self) -> "SystemState":
        self.phi_1 = self.phi_1[-1]
        self.phi_2 = self.phi_2[-1]
        self.kappa_1 = self.kappa_1[-1]
        self.kappa_2 = self.kappa_2[-1]
        return self

    def squeeze(self) -> "SystemState":
        # when running with ensemble size 1, we want to get rid of the ensemble dimension
        self.phi_1 = self.phi_1.squeeze()
        self.phi_2 = self.phi_2.squeeze()
        self.kappa_1 = self.kappa_1.squeeze(axis=1)
        self.kappa_2 = self.kappa_2.squeeze(axis=1)
        return self

    def astype(self, dtype: jnp.dtype = jnp.float64) -> "SystemState":
        return SystemState(
            phi_1=self.phi_1.astype(dtype),
            phi_2=self.phi_2.astype(dtype),
            kappa_1=self.kappa_1.astype(dtype),
            kappa_2=self.kappa_2.astype(dtype),
        )

    def remove_infs(self) -> "SystemState":
        self.phi_1 = self.phi_1[~jnp.isinf(self.phi_1).any(axis=1)]
        self.phi_2 = self.phi_2[~jnp.isinf(self.phi_2).any(axis=1)]
        self.kappa_1 = self.kappa_1[~jnp.isinf(self.kappa_1).any(axis=(1, 2))]
        self.kappa_2 = self.kappa_2[~jnp.isinf(self.kappa_2).any(axis=(1, 2))]
        return self

    def copy(self) -> "SystemState":
        return SystemState(
            jnp.asarray(self.phi_1).copy(),
            jnp.asarray(self.phi_2).copy(),
            jnp.asarray(self.kappa_1).copy(),
            jnp.asarray(self.kappa_2).copy(),
        )


# Register SystemState as a JAX PyTree
jtu.register_pytree_node(SystemState, SystemState.tree_flatten, SystemState.tree_unflatten)


@dataclass
class SystemMetrics:
    # Kuramoto Order Parameter
    r_1: Float[Array, "t ensemble 1"] | np.ndarray
    r_2: Float[Array, "t ensemble 1"] | np.ndarray
    # Ensemble average velocity
    m_1: Float[Array, "t 1"] | np.ndarray
    m_2: Float[Array, "t 1"] | np.ndarray
    # Ensemble average std
    s_1: Float[Array, "t 1"] | np.ndarray
    s_2: Float[Array, "t 1"] | np.ndarray
    # Ensemble phase entropy
    q_1: Float[Array, "t 1"] | np.ndarray
    q_2: Float[Array, "t 1"] | np.ndarray
    # Frequency cluster ratio
    f_1: Float[Array, "t 1"] | np.ndarray
    f_2: Float[Array, "t 1"] | np.ndarray
    # Splay State Ratio
    sr_1: Optional[Float[Array, "t 1"] | np.ndarray] = None
    sr_2: Optional[Float[Array, "t 1"] | np.ndarray] = None

    def tree_flatten(self):
        return (
            self.r_1,
            self.r_2,
            self.m_1,
            self.m_2,
            self.s_1,
            self.s_2,
            self.q_1,
            self.q_2,
            self.f_1,
            self.f_2,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def copy(self) -> "SystemMetrics":
        return SystemMetrics(
            jnp.asarray(self.r_1).copy(),
            jnp.asarray(self.r_2).copy(),
            jnp.asarray(self.m_1).copy(),
            jnp.asarray(self.m_2).copy(),
            jnp.asarray(self.s_1).copy(),
            jnp.asarray(self.s_2).copy(),
            jnp.asarray(self.q_1).copy(),
            jnp.asarray(self.q_2).copy(),
            jnp.asarray(self.f_1).copy(),
            jnp.asarray(self.f_2).copy(),
            jnp.asarray(self.sr_1).copy() if self.sr_1 is not None else None,
            jnp.asarray(self.sr_2).copy() if self.sr_1 is not None else None,
        )

    def add_follow_ups(self) -> "SystemMetrics":
        self.sr_1 = jnp.sum(self.r_1 < 0.2, axis=-1) / self.r_1.shape[-1]
        self.sr_2 = jnp.sum(self.r_2 < 0.2, axis=-1) / self.r_2.shape[-1]
        return self

    def as_single(self) -> "SystemMetrics":
        self.add_follow_ups()
        return SystemMetrics(
            jnp.mean(jnp.asarray(self.r_1)[..., -1, :], axis=(-1,)),
            jnp.mean(jnp.asarray(self.r_2)[..., -1, :], axis=(-1,)),
            jnp.mean(jnp.asarray(self.s_1)[..., -1], axis=-1),
            jnp.mean(jnp.asarray(self.s_2)[..., -1], axis=-1),
            jnp.mean(jnp.asarray(self.m_1), axis=-1),
            jnp.mean(jnp.asarray(self.m_2), axis=-1),
            jnp.mean(jnp.asarray(self.q_1), axis=-1),
            jnp.mean(jnp.asarray(self.q_2), axis=-1),
            jnp.mean(jnp.asarray(self.f_1), axis=-1),
            jnp.mean(jnp.asarray(self.f_2), axis=-1),
            self.sr_1[..., -1] if self.sr_1 is not None else None,
            self.sr_2[..., -1] if self.sr_2 is not None else None,
        )


jtu.register_pytree_node(SystemMetrics, SystemMetrics.tree_flatten, SystemMetrics.tree_unflatten)
