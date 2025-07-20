import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from time import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from numpy.typing import DTypeLike
from jax.tree_util import register_dataclass
import numpy as np
from jaxtyping import Array, Float, PyTree
from optax import GradientTransformation, OptState

from sepsis_osc.model.ae import make_decoder, make_encoder
from sepsis_osc.model.gru import make_predictor
from sepsis_osc.model.latent_dynamics import LatentDynamicsModel

logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("func:%r took: %2.6f sec" % (f.__name__, te - ts))
        return result

    return wrap


@register_dataclass
@dataclass
class AuxLosses:
    alpha: Array
    beta: Array
    sigma: Array

    infection_loss: Array
    sofa_loss: Array
    lookup_temperature: Array
    label_temperature: Array

    total_loss: Array
    recon_loss: Array
    concept_loss: Array
    tc_loss: Array
    kl_loss: Array

    hists_sofa_score: Array
    hists_sofa_metric: Array

    infection_t: Array
    sofa_t: Array

    @staticmethod
    def empty() -> "AuxLosses":
        empty_losses = AuxLosses(
            alpha=jnp.zeros(()),
            beta=jnp.zeros(()),
            sigma=jnp.zeros(()),
            infection_loss=jnp.zeros(()),
            sofa_loss=jnp.zeros(()),
            infection_t=jnp.zeros(()),
            sofa_t=jnp.zeros(()),
            lookup_temperature=jnp.zeros(()),
            label_temperature=jnp.zeros(()),
            total_loss=jnp.zeros(()),
            recon_loss=jnp.zeros(()),
            concept_loss=jnp.zeros(()),
            tc_loss=jnp.zeros(()),
            kl_loss=jnp.zeros(()),
            hists_sofa_score=jnp.ones(()),
            hists_sofa_metric=jnp.ones(()),
        )
        return empty_losses

    def to_dict(self) -> dict[str, dict[str, jnp.ndarray]]:
        return {
            "latents": {
                "alpha": self.alpha,
                "beta": self.beta,
                "sigma": self.sigma,
            },
            "concepts": {
                "infection": self.infection_loss,
                "sofa": self.sofa_loss,
                "lookup_temperature": self.lookup_temperature,
                "label_temperature": self.label_temperature,
            },
            "losses": {
                "total_loss": self.total_loss,
                "recon_loss": self.recon_loss,
                "concept_loss": self.concept_loss,
                "tc_loss": self.tc_loss,
                "kl_loss": self.kl_loss,
            },
            "hists": {"sofa_score": self.hists_sofa_score, "sofa_metric": self.hists_sofa_metric},
            "mult": {
                "infection_t": self.infection_t,
                "sofa_t": self.sofa_t,
            },
        }


@dataclass
class ModelConfig:
    latent_dim: int
    input_dim: int
    enc_hidden: int
    dec_hidden: int
    predictor_hidden: int


@dataclass
class TrainingConfig:
    batch_size: int
    window_len: int
    epochs: int
    perc_train_set: float = 1.0
    validate_every: float = 1.0


@dataclass
class LoadingConfig:
    from_dir: str
    epoch: int


@dataclass
class SaveConfig:
    save_every: int
    perform: bool


@dataclass
class LRConfig:
    init: float
    peak: float
    peak_decay: float
    end: float
    warmup_epochs: int
    enc_wd: float
    grad_norm: float


@register_dataclass
@dataclass
class ConceptLossConfig:
    w_sofa: float
    w_inf: float


@register_dataclass
@dataclass
class LossesConfig:
    w_concept: float
    w_recon: float
    w_tc: float
    w_kl: float
    concept: ConceptLossConfig
    anneal_concept_iter: float
    anneal_kl_iter: float
    steps_per_epoch: int = 0


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model,
    opt_state: OptState,
    hyper_enc: dict[str, float | int],
    hyper_dec: dict[str, int],
    hyper_pred: dict[str, int],
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04d}.eqx")

    # = {"opt_state": opt_state}
    hyper = {"encoder": hyper_enc, "decoder": hyper_dec, "predictor": hyper_pred}

    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyper)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, (model, opt_state))
    logger.info(f"Model checkpoint saved for epoch {epoch} to {filename}")


def load_checkpoint(
    load_dir: str,
    epoch: int,
    opt_template: GradientTransformation | None = None,
) -> tuple[PyTree, OptState]:
    filename = os.path.join(load_dir, f"checkpoint_epoch_{epoch:04d}.eqx")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found at {filename}")

    with open(filename, "rb") as f:
        hyper = json.loads(f.readline().decode())
        hyper_enc = hyper["encoder"]
        hyper_dec = hyper["decoder"]
        hyper_predictor = hyper["predictor"]

        # Build skeleton full model
        key_dummy = jr.PRNGKey(0)
        encoder = make_encoder(key_dummy, **hyper_enc)
        decoder = make_decoder(key_dummy, **hyper_dec)
        predictor = make_predictor(key_dummy, **hyper_predictor)

        model = eqx.nn.inference_mode(
            LatentDynamicsModel(encoder=encoder, predictor=predictor, decoder=decoder), value=False
        )
        params_model, static_model = eqx.partition(model, eqx.is_array)

        opt_state = opt_template.init(params_model) if opt_template else None

        loaded_model, loaded_opt_state = eqx.tree_deserialise_leaves(f, (model, opt_state))
    logger.info(f"Model checkpoint loaded for epoch {epoch} from {filename}")
    return loaded_model, loaded_opt_state


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


def log_train_metrics(metrics, epoch, writer):
    log_msg = f"Epoch {epoch} Training Metrics "
    metrics = metrics.to_dict()
    for group_key, metrics_group in metrics.items():
        if group_key in ("hists", "mult"):
            continue
        for metric_name, metric_values in metrics_group.items():
            metric_values = np.asarray(metric_values)
            if group_key == "losses":
                log_msg += f"{metric_name} = {float(metric_values.mean()):.4f} ({float(metric_values.std()):.4f}), "
            writer.add_scalar(f"train_{group_key}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
    return log_msg


def log_val_metrics(metrics, epoch, writer):
    log_msg = f"Epoch {epoch} Valdation Metrics "
    metrics = metrics.to_dict()
    for k in metrics.keys():
        if k == "hists":
            writer.add_histogram("SOFA Score", np.asarray(metrics["hists"]["sofa_score"].flatten()), epoch, bins=25)
            writer.add_histogram("SOFA metric", np.asarray(metrics["hists"]["sofa_metric"].flatten()), epoch, bins=25)
        elif k == "mult":
            for t, v in enumerate(np.asarray(metrics["mult"]["infection_t"]).mean(axis=0)):
                writer.add_scalar(f"infection_per_timestep/t{t}", np.asarray(v), epoch)
            for t, v in enumerate(np.asarray(metrics["mult"]["sofa_t"]).mean(axis=0)):
                writer.add_scalar(f"sofa_per_timestep/t{t}", np.asarray(v), epoch)
        else:
            for metric_name, metric_values in metrics[k].items():
                log_msg += (
                    f"{metric_name} = {float(metric_values.mean()):.4f} ({float(metric_values.std()):.4f}), "
                    if k == "losses"
                    else ""
                )
                writer.add_scalar(f"val_{k}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
                writer.add_scalar(f"val_{k}/{metric_name}_std", np.asarray(metric_values.std()), epoch)
    return log_msg
