from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jtree
from jax.debug import print as jprint
import numpy as np
from beartype import beartype as typechecker
from jax import vmap
from jaxlie import SO2
from jaxtyping import Array, Float, Int, ScalarLike, jaxtyped

from sepsis_osc.dnm.abstract_ode import ConfigBase, ConfigArgBase
from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMConfigArgs, DNMMetrics, DNMState, DynamicNetworkModel

@jaxtyped(typechecker=typechecker)
class LieDNMConfigArgs(ConfigArgBase):
    # Everything precomputed for ODE
    N: Int[ScalarLike, ""]
    omega_1: Float[Array, " N"]
    omega_2: Float[Array, " N"]
    a_1: Float[Array, "N N"]
    epsilon_1: Float[Array, "broadcast N N"]
    epsilon_2: Float[Array, "broadcast N N"]
    adj: Float[Array, ""]
    R_alpha: SO2
    R_beta: SO2
    sigma: Float[Array, ""]
    tau: Float[Array, ""]

@jaxtyped(typechecker=typechecker)
class LieDNMConfig(ConfigBase):
    N: int  # number of oscillators per layer
    alpha: float  # phase lag
    beta: float  # plasticity (age parameter)
    sigma: float  # interlayer coupling
    C: float = 0.2  # number of infected cells
    omega_1: float = 0.0  # natural frequency parenchymal layer
    omega_2: float = 0.0  # natural frequency immune layer
    a_1: float = 1.0  # intralayer connectivity weights
    epsilon_1: float = 0.03  # adaption rate
    epsilon_2: float = 0.3  # adaption rate
    tau: float = 0.5
    T_init: Optional[int] = None
    T_max: Optional[int] = None
    T_step: Optional[int] = None

    @property
    def as_args(self):
        diag = (jnp.ones((self.N, self.N)) - jnp.eye(self.N))[None, :]
        return LieDNMConfigArgs(
            N=self.N,
            omega_1=jnp.ones((self.N,)) * self.omega_1,
            omega_2=jnp.ones((self.N,)) * self.omega_2,
            a_1=(jnp.ones((self.N, self.N)) * self.a_1).at[jnp.diag_indices(self.N)].set(0),
            epsilon_1=self.epsilon_1 * diag,
            epsilon_2=self.epsilon_2 * diag,
            adj=jnp.array(1 / (self.N - 1)),
            R_alpha=SO2.from_radians(self.alpha * jnp.pi),
            R_beta=SO2.from_radians(self.beta * jnp.pi),
            sigma=jnp.asarray(self.sigma),
            tau=jnp.asarray(self.tau),
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
            float(self.C),
        )


    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def batch_as_index(
        alpha: Float[Array, "... 1"] | np.ndarray,
        beta: Float[Array, "... 1"] | np.ndarray,
        sigma: Float[Array, "... 1"] | np.ndarray,
        C: float,
    ) -> jnp.ndarray:
        batch_shape = alpha.shape

        alpha, beta, sigma = alpha / jnp.pi, beta / jnp.pi, sigma / 2
        _C = jnp.full(batch_shape, C)

        consts = {
            "omega_1": LieDNMConfig.omega_1,
            "omega_2": LieDNMConfig.omega_2,
            "a_1": LieDNMConfig.a_1,
            "epsilon_1": LieDNMConfig.epsilon_1,
            "epsilon_2": LieDNMConfig.epsilon_2,
        }
        const_batches = [jnp.full(batch_shape, v) for v in consts.values()]

        batch_indices = jnp.stack([*const_batches, alpha, beta, sigma, _C], axis=-1, dtype=jnp.float32)

        return batch_indices.squeeze()

class LieDynamicNetworkModel(DynamicNetworkModel):
    def __init__(self, full_save: bool = False, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def system_deriv(
        t: ScalarLike,
        y: DNMState,
        args: LieDNMConfigArgs,
    ) -> DNMState:
        def single_system_deriv(y) -> DNMState:
            R_1 = SO2.from_radians(y.phi_1)
            R_2 = SO2.from_radians(y.phi_2)
            R1_diff = SO2(R_1.unit_complex[None, :, :]).inverse() @ SO2(R_1.unit_complex[:, None, :])
            R2_diff = SO2(R_2.unit_complex[None, :, :]).inverse() @ SO2(R_2.unit_complex[:, None, :])


            R1_diff_alpha = R1_diff @ args.R_alpha
            R2_diff_alpha = R2_diff @ args.R_alpha

            R1_diff_beta = R1_diff @ args.R_beta.inverse()
            R2_diff_beta = R2_diff @ args.R_beta.inverse()

            phi_12_diff = R_1 @ R_2.inverse()
            phi_21_diff = R_2 @ R_1.inverse()

            # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
            dphi_1 = (
                args.omega_1
                - args.adj * jnp.einsum("ij,ij->i", (args.a_1 + y.kappa_1), R1_diff_alpha.unit_complex[..., 1])
                - args.sigma * phi_12_diff.unit_complex[..., 1]
            )

            dphi_2 = (
                args.omega_2
                - args.adj * jnp.einsum("ij,ij->i", y.kappa_2, R2_diff_alpha.unit_complex[..., 1])
                - args.sigma * phi_21_diff.unit_complex[..., 1]
            )

            dkappa_1 = -args.epsilon_1 * (y.kappa_1 + R1_diff_beta.unit_complex[..., 1].T)
            dkappa_2 = -args.epsilon_2 * (y.kappa_2 + R2_diff_beta.unit_complex[..., 1].T)

            dkmean1 = jnp.mean(dkappa_1)
            dkmean2 = jnp.mean(dkappa_2)

            return DNMState(
                phi_1=dphi_1,
                phi_2=dphi_2,
                kappa_1=dkappa_1.squeeze(),
                kappa_2=dkappa_2.squeeze(),
                m_1=(dkmean1 - y.m_1) / args.tau,
                m_2=(dkmean2 - y.m_2) / args.tau,
                v_1=((dkmean1 - y.m_1) ** 2 - y.v_1) / args.tau,
                v_2=((dkmean2 - y.m_2) ** 2 - y.v_2) / args.tau,
            )

        batched_results = vmap(single_system_deriv)(y)

        return batched_results


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

    beta_step = 0.01
    beta_step = 0.1
    betas = np.arange(0.0, 1.0, beta_step)
    sigma_step = 0.01
    beta_step = 0.1
    sigmas = np.arange(0.0, 1.0, sigma_step)
    alpha_step = 0.1
    alphas = jnp.array([-0.28])  # jnp.array([-1.0, -0.76, -0.52, -0.28, 0.0, 0.28, 0.52, 0.76, 1.0])
    T_max_base = 2000
    T_step_base = 100
    total = len(betas) * len(sigmas) * len(alphas)
    logger.info(
        f"Starting to map parameter space of {len(betas)} beta, {len(sigmas)} sigma, {len(alphas)} alpha, total {total}"
    )

    ldnm = LieDynamicNetworkModel(full_save=False, steady_state_check=True, progress_bar=True)
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
                N = 100
                run_conf = LieDNMConfig(
                    N=N,
                    alpha=float(alpha),  # phase lage
                    beta=float(beta),  # age parameter
                    sigma=float(sigma),
                )
                logger.info(f"New config {run_conf.as_index}")
                logger.info(f" ~~~~~~~~~~~ {i}/{total} - {i / total * 100:.4f}% ~~~~~~~~~~~ ")
                logger.info("Starting solve")
                if storage.read_result(run_conf.as_index, DNMMetrics, 0.0) is None or overwrite:
                    sol = ldnm.integrate(
                        config=run_conf,
                        M=num_parallel_runs,
                        key=rand_key,
                        solver=solver,
                        T_init=0,
                        T_max=T_max_base,
                        T_step=T_step_base,
                    )
                    logger.info(f"Solved in {sol.stats['num_steps']} steps")
                    if sol.ys:
                        logger.info("Saving Result")
                        storage.add_result(
                            run_conf.as_index, sol.ys.remove_infs().as_single().serialise(), overwrite=overwrite
                        )
                i += 1
        storage.write()

    storage.close()
