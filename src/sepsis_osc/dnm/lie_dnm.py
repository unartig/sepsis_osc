from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jtree
from jax.debug import print as jprint
import numpy as np
from beartype import beartype as typechecker
from jax import vmap
from jaxlie import SO2
from jaxtyping import Array, Float, Int, ScalarLike, jaxtyped
from diffrax import AbstractAdaptiveSolver, ODETerm, LocalLinearInterpolation

from sepsis_osc.dnm.abstract_ode import StateBase
from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMConfigArgs, DNMMetrics, DNMState, DynamicNetworkModel


@jaxtyped(typechecker=typechecker)
class LieDNMState(StateBase):
    R_1: SO2  # shape (N,)
    R_2: SO2  # shape (N,)
    # NOTE shapes: Ensemble Simulation | Single Simulation | Visualisations
    kappa_1: Float[Array, "*t ensemble N N"] | Float[Array, "N N"] | np.ndarray
    kappa_2: Float[Array, "*t ensemble N N"] | Float[Array, "N N"] | np.ndarray

    # following running moments are used for steady state check
    m_1: Float[Array, "*t ensemble"] | Float[Array, ""]
    m_2: Float[Array, "*t ensemble"] | Float[Array, ""]
    v_1: Float[Array, "*t ensemble"] | Float[Array, ""]
    v_2: Float[Array, "*t ensemble"] | Float[Array, ""]

    v_1p: Float[Array, "*t ensemble"] | Float[Array, ""]
    v_2p: Float[Array, "*t ensemble"] | Float[Array, ""]

    def remove_infs(self) -> "LieDNMState":
        R_1_inf = jnp.isinf(self.R_1).any(axis=(1, 2))
        R_2_inf = jnp.isinf(self.R_2).any(axis=(1, 2))
        kappa_1_inf = jnp.isinf(self.kappa_1).any(axis=(1, 2, 3))
        kappa_2_inf = jnp.isinf(self.kappa_2).any(axis=(1, 2, 3))
        combined_mask = ~(R_1_inf | R_2_inf | kappa_1_inf | kappa_2_inf)
        return jtree.map(lambda x: x[combined_mask], self)

    @staticmethod
    def from_classical(other: DNMState) -> "LieDNMState":
        return LieDNMState(
            R_1=SO2.from_radians(other.phi_1),
            R_2=SO2.from_radians(other.phi_2),
            kappa_1=other.kappa_1,
            kappa_2=other.kappa_2,
            m_1=other.m_1,
            m_2=other.m_2,
            v_1=other.v_1,
            v_2=other.v_2,
            v_1p=other.v_1p,
            v_2p=other.v_2p,
        )

    def to_classical(self) -> DNMState:
        return DNMState(
            phi_1=self.R_1.as_radians(),
            phi_2=self.R_2.as_radians(),
            kappa_1=self.kappa_1,
            kappa_2=self.kappa_2,
            m_1=self.m_1,
            m_2=self.m_2,
            v_1=self.v_1,
            v_2=self.v_2,
            v_1p=self.v_1p,
            v_2p=self.v_2p,
        )

    def enforce_bounds(self) -> DNMState:
        # NOTE in Lie algebras we do not need to enforece any bounds
        # so we transform to classical to make use of existing
        # saving methods of DNM
        return self.to_classical()


class SO2LieAdaptive(AbstractAdaptiveSolver):
    terms = ODETerm
    interpolation_cls = LocalLinearInterpolation

    def __init__(self, base_solver: AbstractAdaptiveSolver):
        self.base = base_solver

    def order(self, terms):
        return self.base.order(terms)

    def error_order(self, terms):
        return self.base.error_order(terms)

    def init(self, terms, t0, t1, y0: LieDNMState, args):
        return self.base.init(terms, t0, t1, y0.phi, args)

    def func(self, terms, t0, y0: LieDNMState, args):
        # vf expects (t, R, args) and returns omega (scalar)
        omega = terms.vf(t0, y0.R, args)
        return omega  # scalar algebra velocity

    def step(self, terms, t0, t1, y0: LieDNMState, args, solver_state, made_jump):
        def alg_vf(t, phi_dummy, args_):
            dphi = phi_dummy - y0.phi
            R_cur = y0.R @ jaxlie.SO2.exp(jnp.array([dphi]))
            return terms.vf(t, R_cur, args_)  # returns omega(t, R)

        alg_term = dfx.ODETerm(lambda t, phi, a: alg_vf(t, phi, a))

        phi1, phi_err, dense_info_alg, solver_state, result = self.base.step(
            alg_term, t0, t1, y0.phi, args, solver_state, made_jump
        )

        dphi_main = jnp.array([phi1 - y0.phi])
        R1 = y0.R @ jaxlie.SO2.exp(dphi_main)

        y1 = LieDNMState(R=R1, phi=phi1)

        # Error estimate: keep it in the algebra; the step-size controller’s `norm`
        # will look at `.phi` only (see usage below).
        y_error = LieDNMState(R=jnp.zeros_like(y0.R), phi=phi_err if phi_err is not None else None)

        # Minimal dense info for interpolation on (R, phi). Linear in phi; for R we linearly
        # interpolate on the group via log/exp in the interpolation class; for the sketch we
        # just hand phi endpoints and let LocalLinearInterpolation work on the dataclass fields.
        dense_info = dict(y0=y0, y1=y1)

        return y1, y_error, dense_info, solver_state, result


class LieDynamicNetworkModel(DynamicNetworkModel):
    def __init__(self, full_save: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.save_method = (
            self.generate_full_save(super().system_deriv)
            if full_save
            else self.generate_metric_save(super().system_deriv)
        )

    def generate_init_conditions(self, config, M, key) -> LieDNMState:
        classical = super().generate_init_conditions(config, M, key)
        lie = LieDNMState.from_classical(classical)
        return lie

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def system_deriv(
        t: ScalarLike,
        y: LieDNMState,
        args: DNMConfigArgs,
    ) -> LieDNMState:
        def single_system_deriv(y) -> LieDNMState:
            R1_diff = SO2(y.R_1.unit_complex[None, :, :]).inverse() @ SO2(y.R_1.unit_complex[:, None, :])
            R2_diff = SO2(y.R_2.unit_complex[None, :, :]).inverse() @ SO2(y.R_2.unit_complex[:, None:])

            # TODO
            R_alpha = SO2.from_radians(jnp.asin(args.sin_alpha))
            R_beta = SO2.from_radians(jnp.asin(args.sin_beta))

            R1_diff_alpha = R1_diff @ R_alpha
            R2_diff_alpha = R2_diff @ R_alpha

            R1_diff_beta = R1_diff @ R_beta.inverse()
            R2_diff_beta = R2_diff @ R_beta.inverse()

            phi_12_diff = y.R_1 @ y.R_2.inverse()
            phi_21_diff = y.R_2 @ y.R_1.inverse()

            # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
            dphi_1 = (
                SO2.from_radians(
                    args.omega_1
                    - args.adj * jnp.einsum("ij,ij->i", (args.a_1 + y.kappa_1), R1_diff_alpha.unit_complex[..., 1])
                    - args.sigma * phi_12_diff.unit_complex[..., 1]
                )
            )

            dphi_2 = (
                SO2.from_radians(
                    args.omega_2
                    - args.adj * jnp.einsum("ij,ij->i", y.kappa_2, R2_diff_alpha.unit_complex[..., 1])
                    - args.sigma * phi_21_diff.unit_complex[..., 1]
                )
            )

            dkappa_1 = -args.epsilon_1 * (y.kappa_1 + R1_diff_beta.unit_complex[..., 1])
            dkappa_2 = -args.epsilon_2 * (y.kappa_2 + R2_diff_beta.unit_complex[..., 1])

            dkmean1 = jnp.mean(dkappa_1)
            dkmean2 = jnp.mean(dkappa_2)

            return LieDNMState(
                R_1=dphi_1,
                R_2=dphi_2,
                kappa_1=dkappa_1.squeeze(),
                kappa_2=dkappa_2.squeeze(),
                m_1=(dkmean1 - y.m_1) / args.tau,
                m_2=(dkmean2 - y.m_2) / args.tau,
                v_1=((dkmean1 - y.m_1) ** 2 - y.v_1) / args.tau,
                v_2=((dkmean2 - y.m_2) ** 2 - y.v_2) / args.tau,
                v_1p=(y.v_1 - y.v_1p) / (args.tau),
                v_2p=(y.v_2 - y.v_2p) / (args.tau),
            )

        batched_results = vmap(single_system_deriv)(y)

        return batched_results

    def generate_full_save(self, deriv) -> Callable:
        parent_full_save = super().generate_full_save(deriv)

        def full_save(t: ScalarLike, y: LieDNMState, args: DNMConfigArgs) -> DNMMetrics:
            return parent_full_save(t, y.to_classical(), args)

        return full_save

    def generate_metric_save(self, deriv) -> Callable:
        parent_metric_save = super().generate_metric_save(deriv)

        def metric_save(t: ScalarLike, y: LieDNMState, args: DNMConfigArgs) -> DNMMetrics:
            return parent_metric_save(t, y.to_classical(), args)

        return metric_save


if __name__ == "__main__":
    import logging

    import jax.random as jr
    from diffrax import Tsit5

    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.utils.config import jax_random_seed
    from sepsis_osc.utils.logger import setup_logging

    setup_logging("info", console_log=True)
    logger = logging.getLogger(__name__)

    rand_key = jr.key(jax_random_seed)
    num_parallel_runs = 100

    solver = Tsit5()
    beta_step = 0.08
    beta_step = 0.1
    betas = np.arange(0.5, 1.0, beta_step)
    sigma_step = 0.08
    sigma_step = 0.5
    sigmas = np.arange(0.0, 1.0, sigma_step)
    alpha_step = 0.1
    alphas = jnp.array([-0.28])  # jnp.array([-1.0, -0.76, -0.52, -0.28, 0.0, 0.28, 0.52, 0.76, 1.0])
    T_max_base = 1000
    T_step_base = 1
    total = len(betas) * len(sigmas) * len(alphas)
    logger.info(
        f"Starting to map parameter space of {len(betas)} beta, {len(sigmas)} sigma, {len(alphas)} alpha, total {total}"
    )

    ldnm = LieDynamicNetworkModel(full_save=False)
    solver = Tsit5()

    db_str = "Lie"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    overwrite = False

    i = 0
    for alpha in alphas:
        for beta in betas:
            for sigma in sigmas:
                N = 25
                run_conf = DNMConfig(
                    N=N,
                    alpha=float(alpha),  # phase lage
                    beta=float(beta),  # age parameter
                    sigma=float(sigma),
                )
                logger.info(f"New config {run_conf.as_index}")
                logger.info("Starting solve")
                sol = ldnm.integrate(
                    config=run_conf,
                    M=num_parallel_runs,
                    key=rand_key,
                    T_init=0,
                    T_max=T_max_base,
                    T_step=T_step_base,
                )
                logger.info(f"Solved in {sol.stats['num_steps']} steps")
                if sol.ys:
                    logger.info("Saving Result")
                    storage.add_result(run_conf.as_index, sol.ys.as_single().serialise(), overwrite=overwrite)
                    storage.write()

    storage.close()
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
