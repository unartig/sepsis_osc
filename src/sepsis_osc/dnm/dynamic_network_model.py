from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jtree
import numpy as np
from jax import vmap
from jax.debug import print as jprint
from jaxtyping import Array, Bool, Float, Int, ScalarLike, jaxtyped, DTypeLike

from sepsis_osc.dnm.abstract_ode import ConfigArgBase, ConfigBase, MetricBase, ODEBase, StateBase
from sepsis_osc.dnm.commons import diff_angle, entropy, mean_angle, phase_entropy, std_angle
from sepsis_osc.utils.jax_config import typechecker


@jaxtyped(typechecker=typechecker)
class DNMConfigArgs(ConfigArgBase):
    # Everything precomputed for ODE
    N: Int[ScalarLike, ""]
    omega_1: Float[Array, " N"]
    omega_2: Float[Array, " N"]
    a_1: Float[Array, "N N"]
    epsilon_1: Float[Array, "broadcast N N"]
    epsilon_2: Float[Array, "broadcast N N"]
    adj: Float[Array, ""]
    sin_alpha: Float[Array, ""]
    cos_alpha: Float[Array, ""]
    sin_beta: Float[Array, ""]
    cos_beta: Float[Array, ""]
    sigma: Float[Array, ""]
    tau: Float[Array, ""]


@jaxtyped(typechecker=typechecker)
class DNMConfig(ConfigBase):
    N: int  # number of oscillators per layer
    beta: float  # plasticity (age parameter)
    sigma: float  # interlayer coupling
    alpha: float = -0.28  # phase lag
    C: float = 0.2  # number of infected cells
    omega_1: float = 0.0  # natural frequency parenchymal layer
    omega_2: float = 0.0  # natural frequency immune layer
    a_1: float = 1.0  # intralayer connectivity weights
    epsilon_1: float = 0.03  # adaption rate
    epsilon_2: float = 0.3  # adaption rate

    # Used for steady state detection
    tau: float = 0.5
    T_init: int | None = None
    T_max: int | None = None
    T_step: int | None = None

    @property
    def as_args(self) -> DNMConfigArgs:
        diag = (jnp.ones((self.N, self.N)) - jnp.eye(self.N))[None, :]
        return DNMConfigArgs(
            N=self.N,
            omega_1=jnp.ones((self.N,)) * self.omega_1,
            omega_2=jnp.ones((self.N,)) * self.omega_2,
            a_1=(jnp.ones((self.N, self.N)) * self.a_1) * ~jnp.eye(self.N).astype(jnp.bool), #.at[jnp.diag_indices(self.N)].set(0),
            epsilon_1=self.epsilon_1 * diag,
            epsilon_2=self.epsilon_2 * diag,
            adj=jnp.array(1 / (self.N - 1)),
            sin_alpha=jnp.sin(self.alpha * jnp.pi),
            cos_alpha=jnp.cos(self.alpha * jnp.pi),
            sin_beta=jnp.sin(self.beta * jnp.pi),
            cos_beta=jnp.cos(self.beta * jnp.pi),
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
            "omega_1": DNMConfig.omega_1,
            "omega_2": DNMConfig.omega_2,
            "a_1": DNMConfig.a_1,
            "epsilon_1": DNMConfig.epsilon_1,
            "epsilon_2": DNMConfig.epsilon_2,
        }
        const_batches = [jnp.full(batch_shape, v) for v in consts.values()]

        batch_indices = jnp.stack([*const_batches, alpha, beta, sigma, _C], axis=-1, dtype=jnp.float32)

        return batch_indices.squeeze()


@jaxtyped(typechecker=typechecker)
class DNMState(StateBase):
    # NOTE shapes: Ensemble Simulation | Single Simulation | Visualisations
    phi_1: Float[Array, "*t ensemble N"] | Float[Array, " N"] | np.ndarray
    phi_2: Float[Array, "*t ensemble N"] | Float[Array, " N"] | np.ndarray
    kappa_1: Float[Array, "*t ensemble N N"] | Float[Array, "N N"] | np.ndarray
    kappa_2: Float[Array, "*t ensemble N N"] | Float[Array, "N N"] | np.ndarray

    # following running moments are used for steady state check
    m_1: Float[Array, "*t ensemble"] | Float[Array, ""]
    m_2: Float[Array, "*t ensemble"] | Float[Array, ""]
    v_1: Float[Array, "*t ensemble"] | Float[Array, ""]
    v_2: Float[Array, "*t ensemble"] | Float[Array, ""]

    def enforce_bounds(self) -> "DNMState":
        return DNMState(
            phi_1=self.phi_1 % (2 * jnp.pi),
            phi_2=self.phi_2 % (2 * jnp.pi),
            kappa_1=jnp.clip(self.kappa_1),  #  in the paper they say they clip, but they dont
            kappa_2=jnp.clip(self.kappa_2),
            m_1=self.m_1,
            m_2=self.m_2,
            v_1=self.v_1,
            v_2=self.v_2,
        )

    def remove_infs(self) -> "DNMState":
        phi_1_has_inf = jnp.isinf(self.phi_1).any(axis=(1, 2))
        phi_2_has_inf = jnp.isinf(self.phi_2).any(axis=(1, 2))
        kappa_1_has_inf = jnp.isinf(self.kappa_1).any(axis=(1, 2, 3))
        kappa_2_has_inf = jnp.isinf(self.kappa_2).any(axis=(1, 2, 3))
        combined_mask = ~(phi_1_has_inf | phi_2_has_inf | kappa_1_has_inf | kappa_2_has_inf)
        return jtree.map(lambda x: x[combined_mask], self)


class DNMMetrics(MetricBase):
    # NOTE shapes: Simulation | DB/Lookup Query | Visualisations

    # Kuramoto Order Parameter
    r_1: Float[Array, "*t ensemble"] | Float[Array, "... 1"] | np.ndarray
    r_2: Float[Array, "*t ensemble"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble average velocity
    m_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    m_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble average std
    s_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    s_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble phase entropy
    q_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    q_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Frequency cluster ratio
    f_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    f_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble average Coupling Entropy
    cq_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    cq_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Ensemble average Coupling std
    cs_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    cs_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray
    # Splay State Ratio
    sr_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray | None = None
    sr_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray | None = None
    # Splay State Ratio
    cm_1: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray | None = None
    cm_2: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray | None = None
    # Measured mean transient time
    tt: Float[Array, "*t"] | Float[Array, "... 1"] | np.ndarray | None = None

    def add_follow_ups(self) -> "DNMMetrics":
        if self.sr_1 is None and self.r_1.size > 1:
            new = DNMMetrics(
                r_1=self.r_1,
                r_2=self.r_2,
                s_1=self.s_1,
                s_2=self.s_2,
                m_1=self.m_1,
                m_2=self.m_2,
                q_1=self.q_1,
                q_2=self.q_2,
                cq_1=self.cq_1,
                cq_2=self.cq_2,
                cs_1=self.cs_1,
                cs_2=self.cs_2,
                f_1=self.f_1,
                f_2=self.f_2,
                cm_1=self.cm_1,
                cm_2=self.cm_2,
                sr_1=jnp.sum(self.r_1 < 0.2, axis=-1) / self.r_1.shape[-1],
                sr_2=jnp.sum(self.r_2 < 0.2, axis=-1) / self.r_2.shape[-1],
                tt=jnp.array([self.r_1.shape[0]]) - 1,
            )
        return new

    def as_single(self) -> "DNMMetrics":
        if self.r_1.size <= 1:  # already single
            return self

        new = self.add_follow_ups()
        return DNMMetrics(
            r_1=jnp.mean(jnp.asarray(self.r_1)[..., -1, :], axis=(-1,)),
            r_2=jnp.mean(jnp.asarray(self.r_2)[..., -1, :], axis=(-1,)),
            s_1=jnp.mean(jnp.asarray(self.s_1)),
            s_2=jnp.mean(jnp.asarray(self.s_2)),
            m_1=jnp.mean(jnp.asarray(self.m_1), axis=-1),
            m_2=jnp.mean(jnp.asarray(self.m_2), axis=-1),
            q_1=jnp.mean(jnp.asarray(self.q_1), axis=-1),
            q_2=jnp.mean(jnp.asarray(self.q_2), axis=-1),
            cq_1=jnp.mean(jnp.asarray(self.cq_1)[..., -1, :], axis=(-1,)),
            cq_2=jnp.mean(jnp.asarray(self.cq_2)[..., -1, :], axis=(-1,)),
            cs_1=jnp.mean(jnp.asarray(self.cs_1)[..., -1, :], axis=(-1,)),
            cs_2=jnp.mean(jnp.asarray(self.cs_2)[..., -1, :], axis=(-1,)),
            f_1=jnp.mean(jnp.asarray(self.f_1), axis=-1),
            f_2=jnp.mean(jnp.asarray(self.f_2), axis=-1),
            cm_1=jnp.mean(jnp.asarray(self.cm_1)[..., -1, :], axis=(-1,)) if new.cm_1 is not None else None,
            cm_2=jnp.mean(jnp.asarray(self.cm_2)[..., -1, :], axis=(-1,)) if new.cm_2 is not None else None,
            sr_1=new.sr_1[..., -1] if new.sr_1 is not None else None,
            sr_2=new.sr_2[..., -1] if new.sr_2 is not None else None,
            tt=new.tt.max() if new.tt is not None else None,
        )


class DynamicNetworkModel(ODEBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def generate_init_sampler(self) -> Callable:
        def init_sampler(config: DNMConfig, M: int, key: jnp.ndarray) -> DNMState:
            C = int(config.C * config.N)
            m = np.sign(C)

            def sample(key: jnp.ndarray) -> DNMState:
                keys = jr.split(key, 3)
                phi_1_init = jr.uniform(keys[0], (config.N,)) * (2 * jnp.pi)
                phi_2_init = jr.uniform(keys[1], (config.N,)) * (2 * jnp.pi)

                kappa_1_init = jr.uniform(keys[2], (config.N, config.N)) * 2 - 1
                kappa_1_init = kappa_1_init.at[jnp.diag_indices(config.N)].set(0.0)

                kappa_2_init = jnp.ones((config.N, config.N))
                kappa_2_init = kappa_2_init.at[C:, :C].set(0.0)
                kappa_2_init = kappa_2_init.at[:C, C:].set(0.0)
                kappa_2_init = kappa_2_init.at[jnp.diag_indices(config.N)].set(0.0)
                kappa_2_init = kappa_2_init * m

                return DNMState(
                    phi_1=phi_1_init,
                    phi_2=phi_2_init,
                    kappa_1=kappa_1_init,
                    kappa_2=kappa_2_init,
                    # following are only used for steady state check
                    m_1=kappa_1_init.mean(),
                    m_2=kappa_2_init.mean(),
                    v_1=kappa_1_init.var(),
                    v_2=kappa_2_init.var(),
                )

            rand_keys = jr.split(key, M)
            return vmap(sample)(rand_keys)

        return init_sampler

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def system_deriv(
        _t: ScalarLike,
        y: DNMState,
        args: DNMConfigArgs,
    ) -> DNMState:
        def single_system_deriv(y: DNMState) -> DNMState:
            # y = y.enforce_bounds()
            # sin/cos in radians
            # https://mediatum.ub.tum.de/doc/1638503/1638503.pdf
            sin_phi_1, cos_phi_1 = jnp.sin(y.phi_1), jnp.cos(y.phi_1)
            sin_phi_2, cos_phi_2 = jnp.sin(y.phi_2), jnp.cos(y.phi_2)

            # expand dims to broadcast outer product [i:]*[:j]->[ij]
            sin_diff_phi_1 = sin_phi_1[:, None] * cos_phi_1[None, :] - cos_phi_1[:, None] * sin_phi_1[None, :]
            cos_diff_phi_1 = cos_phi_1[:, None] * cos_phi_1[None, :] + sin_phi_1[:, None] * sin_phi_1[None, :]

            sin_diff_phi_2 = sin_phi_2[:, None] * cos_phi_2[None, :] - cos_phi_2[:, None] * sin_phi_2[None, :]
            cos_diff_phi_2 = cos_phi_2[:, None] * cos_phi_2[None, :] + sin_phi_2[:, None] * sin_phi_2[None, :]

            sin_phi_1_diff_alpha = sin_diff_phi_1 * args.cos_alpha + cos_diff_phi_1 * args.sin_alpha
            sin_phi_2_diff_alpha = sin_diff_phi_2 * args.cos_alpha + cos_diff_phi_2 * args.sin_alpha

            # (phi1 (N), phi2 (N), k1 (NxN), k2 (NxN)))
            dphi_1 = (
                args.omega_1
                - args.adj * jnp.einsum("ij,ij->i", (args.a_1 + y.kappa_1), sin_phi_1_diff_alpha)
                - args.sigma * (sin_phi_1 * cos_phi_2 - cos_phi_1 * sin_phi_2)
            )
            dphi_2 = (
                args.omega_2
                - args.adj * jnp.einsum("ij,ij->i", y.kappa_2, sin_phi_2_diff_alpha)
                - args.sigma * (sin_phi_2 * cos_phi_1 - cos_phi_2 * sin_phi_1)
            )
            dkappa_1 = -args.epsilon_1 * (y.kappa_1 + (sin_diff_phi_1 * args.cos_beta - cos_diff_phi_1 * args.sin_beta))
            dkappa_2 = -args.epsilon_2 * (y.kappa_2 + (sin_diff_phi_2 * args.cos_beta - cos_diff_phi_2 * args.sin_beta))

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

        return vmap(single_system_deriv)(y)

    def generate_metric_save(self, deriv: Callable) -> Callable:
        @eqx.filter_jit
        def metric_save(_t: ScalarLike, y: DNMState, args: DNMConfigArgs) -> DNMMetrics:
            y = y.enforce_bounds()

            ###### Kuramoto Order Parameter
            r_1 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_1), axis=-1))
            r_2 = jnp.abs(jnp.mean(jnp.exp(1j * y.phi_2), axis=-1))

            ###### Entropy of Phase
            q_1 = phase_entropy(y.phi_1, 360)
            q_2 = phase_entropy(y.phi_2, 360)

            ###### Entropy of Coupling
            cq_1 = vmap(entropy, in_axes=(0, None))(y.kappa_1, 100)
            cq_2 = vmap(entropy, in_axes=(0, None))(y.kappa_2, 100)
            cs_1 = vmap(jnp.std)(y.kappa_1)
            cs_2 = vmap(jnp.std)(y.kappa_2)
            cm_1 = vmap(jnp.mean)(y.kappa_1)
            cm_2 = vmap(jnp.mean)(y.kappa_2)

            ######phase_ Ensemble average velocites and average of the standard deviations
            # For the derivatives we need to evaluate again ...
            dy = deriv(0, y, args)
            mean_1 = mean_angle(dy.phi_1, axis=-1)
            mean_2 = mean_angle(dy.phi_2, axis=-1)

            s_1 = jnp.mean(std_angle(dy.phi_1, axis=-1))
            s_2 = jnp.mean(std_angle(dy.phi_2, axis=-1))

            # redce ensemble dim
            m_1 = jnp.mean(mean_1, axis=-1)
            m_2 = jnp.mean(mean_2, axis=-1)

            ###### Frequency cluster ratio
            # check for desynchronized nodes
            eps = 10 / 360  # TODO what is the value here?
            desync_1 = jnp.any(diff_angle(y.phi_1, mean_1[:, None]) > eps, axis=-1)
            desync_2 = jnp.any(diff_angle(y.phi_2, mean_2[:, None]) > eps, axis=-1)

            # Count number of frequency clusters
            # Number of ensembles where at least one node deviates
            n_f_1 = jnp.sum(desync_1)
            n_f_2 = jnp.sum(desync_2)

            N_E = y.phi_1.shape[0]
            f_1 = n_f_1 / N_E  # N_f / N_E
            f_2 = n_f_2 / N_E

            return DNMMetrics(
                r_1=r_1,
                r_2=r_2,
                m_1=m_1,
                m_2=m_2,
                s_1=s_1,
                s_2=s_2,
                q_1=q_1,
                q_2=q_2,
                cq_1=cq_1,
                cq_2=cq_2,
                cs_1=cs_1,
                cs_2=cs_2,
                cm_1=cm_1,
                cm_2=cm_2,
                f_1=f_1,
                f_2=f_2,
            )

        return metric_save

    def generate_full_save(
        self, deriv: Callable, dtype: DTypeLike = jnp.float16, *, save_y: bool = True, save_dy: bool = True
    ) -> Callable:
        def full_compressed_save(
            _t: ScalarLike, y: DNMState, args: DNMConfigArgs | None
        ) -> tuple[DNMState, DNMState] | DNMState:
            y.enforce_bounds()
            if save_y and not save_dy:
                return y.astype(dtype)
            dy = deriv(0, y, args)
            if save_dy and not save_y:
                return dy.astype(dtype)
            return y.astype(dtype), dy.astype(dtype)

        return full_compressed_save

    def generate_steady_state_check(
        self, eps_m: float = 1e-10, eps_v: float = 1e-3, t_min: float = 1.0
    ) -> Callable[..., Bool[Array, "1"]]:
        def check(t: float, y: DNMState, _args: tuple, **kwargs) -> Bool[Array, "1"]:
            is_late = t > t_min

            # Compute errors per batch
            is_const = (jnp.abs(y.m_1).max() < eps_m) & (jnp.abs(y.m_2).max() < eps_m)
            is_hom = (y.v_1.max() < eps_v) & (y.v_2.max() < eps_v)

            return (is_hom & is_const) & is_late

        return check


if __name__ == "__main__":
    import logging

    from diffrax import Tsit5

    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.utils.config import jax_random_seed
    from sepsis_osc.utils.jax_config import setup_jax
    from sepsis_osc.utils.logger import setup_logging

    setup_logging("info", console_log=True)
    setup_jax(simulation=True)
    logger = logging.getLogger(__name__)

    rand_key = jr.key(jax_random_seed)
    num_parallel_runs = 50

    beta_step = 0.003
    betas = np.arange(0.4, 0.7, beta_step)
    sigma_step = 0.015
    sigmas = np.arange(0.0, 1.5, sigma_step)
    T_max_base = 2000
    T_step_base = 0.1
    total = len(betas) * len(sigmas)
    logger.info(
        f"Starting to map parameter space of {len(betas)} beta, {len(sigmas)} sigma, total {total}"
    )

    dnm = DynamicNetworkModel(full_save=False, steady_state_check=False, progress_bar=False)
    solver = Tsit5(scan_kind="bounded")

    db_str = "Final"
    storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    overwrite = False

    i = 0
    for beta in betas:
        for sigma in sigmas:
            N = 200
            run_conf = DNMConfig(
                N=N,
                C=0.2,  # phase lage
                alpha=-0.28,
                beta=beta,  # age parameter
                sigma=sigma,
            )
            logger.info(f"New config {run_conf.as_index}")
            logger.info(f" ~~~~~~~~~~~ {i}/{total} - {i / total * 100:.4f}% ~~~~~~~~~~~ ")
            if storage.read_result(run_conf.as_index, DNMMetrics, 1e-15) is None or overwrite:
                logger.info("Starting solve")
                sol = dnm.integrate(
                    config=run_conf,
                    M=num_parallel_runs,
                    solver=solver,
                    key=rand_key,
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
            else:
                logger.error("Keeping existing result")
            i += 1
        storage.write()
    storage.close()
