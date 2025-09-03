import json
import logging
import os
from dataclasses import dataclass, fields

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.tree_util import register_dataclass
from jaxtyping import Array, PyTree
from optax import GradientTransformation, OptState
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis_osc.ldm.ae import make_decoder, make_encoder
from sepsis_osc.ldm.gru import make_predictor
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass
class AuxLosses:
    alpha: Array
    beta: Array
    sigma: Array

    lookup_temperature: Array
    label_temperature: Array

    total_loss: Array
    recon_loss: Array
    concept_loss: Array
    directional_loss: Array
    tc_loss: Array
    accel_loss: Array
    diff_loss: Array
    thresh_loss: Array

    hists_sofa_score: Array
    hists_sofa_metric: Array
    hists_inf_error: Array

    infection_loss_t: Array
    sofa_loss_t: Array
    sep3_loss_t: Array

    anneal_concept: Array
    anneal_recon: Array
    anneal_threshs: Array

    sofa_d2_p: Array
    susp_inf_p: Array
    sep3_p: Array

    @staticmethod
    def empty() -> "AuxLosses":
        initialized_fields = {f.name: np.zeros(()) for f in fields(AuxLosses)}
        return AuxLosses(**initialized_fields)

    def to_dict(self) -> dict[str, dict[str, jnp.ndarray]]:
        return {
            "latents": {
                "alpha": self.alpha,
                "beta": self.beta,
                "sigma": self.sigma,
            },
            "concepts": {
                "sofa": self.sofa_loss_t,
                "infection": self.infection_loss_t,
                "sepsis-3": self.sep3_loss_t,
            },
            "losses": {
                "total_loss": self.total_loss,
                "recon_loss": self.recon_loss,
                "concept_loss": self.concept_loss,
                "directional_loss": self.directional_loss,
                "tc_loss": self.tc_loss,
                "accel_loss": self.accel_loss,
                "diff_loss": self.diff_loss,
                "thresh_loss": self.thresh_loss,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
                "sofa_metric": self.hists_sofa_metric,
                "inf_error": self.hists_inf_error,
            },
            "mult": {
                "infection_t": self.infection_loss_t,
                "sofa_t": self.sofa_loss_t,
                "sep3_t": self.sep3_loss_t,
            },
            "cosine_annealings": {
                "concept": self.anneal_concept,
                "recon": self.anneal_recon,
                "thresholds": self.anneal_threshs,
            },
            "sepsis_metrics": {
                "sofa_d2_p": self.sofa_d2_p,
                "susp_inf_p": self.susp_inf_p,
                "sep3_p": self.sep3_p,
            },
        }


@dataclass
class ModelConfig:
    latent_dim: int
    input_dim: int
    enc_hidden: int
    dec_hidden: int
    predictor_hidden: int
    dropout_rate: float


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
    w_sep3: float


@register_dataclass
@dataclass
class LossesConfig:
    w_concept: float
    w_recon: float
    w_tc: float
    w_accel: float
    w_diff: float
    w_direction: float
    w_thresh: float
    concept: ConceptLossConfig
    anneal_concept_iter: float
    anneal_recon_iter: float
    anneal_threshs_iter: float
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


def log_train_metrics(metrics, model_params, epoch, writer):
    log_msg = f"Epoch {epoch} Training "
    metrics = metrics.to_dict()
    metrics = {**metrics, "model_params": {**model_params}}
    for group_key, metrics_group in metrics.items():
        if group_key in ("hists", "mult", "sepsis_metrics"):
            continue
        for metric_name, metric_values in metrics_group.items():
            metric_values = np.asarray(metric_values)
            if metric_name in ("total_loss", "sofa", "infection", "sepsis-3"):
                log_msg += f"{metric_name} = {float(metric_values.mean()):.4f} ({float(metric_values.std()):.4f}), "
            writer.add_scalar(f"train_{group_key}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
    return log_msg


def log_val_metrics(metrics, y, epoch, writer):
    log_msg = f"Epoch {epoch} Valdation "
    metrics = metrics.to_dict()
    for k in metrics.keys():
        if k == "hists":
            writer.add_histogram("SOFA Score", np.asarray(metrics["hists"]["sofa_score"].flatten()), epoch, bins=25)
            writer.add_histogram("SOFA metric", np.asarray(metrics["hists"]["sofa_metric"].flatten()), epoch, bins=25)
            writer.add_histogram("Inf Error", np.asarray(metrics["hists"]["inf_error"].flatten()), epoch)
        elif k == "mult":
            for t, v in enumerate(np.asarray(metrics["mult"]["infection_t"]).mean(axis=0)):
                writer.add_scalar(f"infection_per_timestep/t{t}", np.asarray(v), epoch)
            for t, v in enumerate(np.asarray(metrics["mult"]["sofa_t"]).mean(axis=0)):
                writer.add_scalar(f"sofa_per_timestep/t{t}", np.asarray(v), epoch)
        elif k == "sepsis_metrics":
            sep3 = np.asarray(y[..., 2].any(axis=-1)).flatten()
            pred_sep3_p = np.asarray(metrics[k]["sep3_p"]).flatten()
            pred_sofa_d2_p = np.asarray(metrics[k]["sofa_d2_p"]).flatten()
            pred_susp_inf_p = np.asarray(metrics[k]["susp_inf_p"]).flatten()
            writer.add_scalar(k + "/AUROC_pred_sep", roc_auc_score(sep3, pred_sep3_p), epoch)
            writer.add_scalar(k + "/AUPRC_pred_sep", average_precision_score(sep3, pred_sep3_p), epoch)
            writer.add_scalar(k + "/AUROC_pred_sofa_d2", roc_auc_score(sep3, pred_sofa_d2_p), epoch)
            writer.add_scalar(k + "/AUPRC_pred_sofa_d2", average_precision_score(sep3, pred_sofa_d2_p), epoch)
            writer.add_scalar(k + "/AUROC_pred_susp_inf", roc_auc_score(sep3, pred_susp_inf_p), epoch)
            writer.add_scalar(k + "/AUPRC_pred_susp_inf", average_precision_score(sep3, pred_susp_inf_p), epoch)
            log_msg += (f"AUROC = {float(roc_auc_score(sep3, pred_sep3_p)):.4f}, ")
            log_msg += (f"AUPRC = {float(average_precision_score(sep3, pred_sep3_p)):.4f}, ")
        elif k in ("cosine_annealings"):
            continue
        else:
            for metric_name, metric_values in metrics[k].items():
                log_msg += (
                    f"{metric_name} = {float(metric_values.mean()):.4f} ({float(metric_values.std()):.4f}), "
                    if metric_name in ("total_loss", "sofa", "infection", "sepsis-3")
                    else ""
                )
                writer.add_scalar(f"val_{k}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
                writer.add_scalar(f"val_{k}/{metric_name}_std", np.asarray(metric_values.std()), epoch)
    return log_msg
