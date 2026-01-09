from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.tensorboard.writer import SummaryWriter

from sepsis_osc.ldm.lookup import LatentLookup
from sepsis_osc.ldm.model_structs import AuxLosses
from sepsis_osc.visualisations.viz_model_results import (
    viz_curves,
    viz_heatmap_concepts,
    viz_plane,
    viz_progression,
    viz_starter,
)


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    return (
        {
            f"{k}" if parent_key else k: v
            for pk, pv in d.items()
            for k, v in flatten_dict(pv, f"{parent_key}{sep}{pk}" if parent_key else pk, sep).items()
        }
        if isinstance(d, dict)
        else {parent_key: d}
    )


def log_train_metrics(
    aux_losses: AuxLosses,
    model_params: dict[str, Float[Array, "1"]],
    epoch: int,
    writer: SummaryWriter,
    *,
    mask: np.ndarray | None = None,
) -> str:
    if mask is None:
        mask = np.ones_like(aux_losses.sep3_p.astype(np.bool))
    log_msg = f"Epoch {epoch} Training "
    metrics = aux_losses.to_dict()
    metrics = {**metrics, "model_params": {**model_params}}
    for group_key, metrics_group in metrics.items():
        if group_key in ("hists", "mult", "sepsis_metrics"):
            continue
        for metric_name, metric_values in metrics_group.items():
            masked_values = metric_values[mask] if metric_values.shape == mask.shape else metric_values
            if metric_name in ("total_loss", "sofa", "infection", "sepsis-3"):
                log_msg += f"{metric_name}={float(masked_values.mean()):.4f}({float(masked_values.std()):.2f}),"
            writer.add_scalar(f"train_{group_key}/{metric_name}_mean", np.asarray(masked_values.mean()), epoch)
    return log_msg


def log_val_metrics(
    aux_losses: AuxLosses,
    y: np.ndarray,
    lookup: LatentLookup,
    epoch: int,
    writer: SummaryWriter,
    *,
    mask: np.ndarray | None = None,
) -> str:
    metrics = aux_losses.to_dict()
    if mask is None:  # if prediction
        mask = np.ones_like(aux_losses.sep3_p.astype(np.bool))
        true_sofa = np.asarray(y[..., 0]).flatten()
        true_sofa_d2 = np.asarray(jnp.diff(y[..., 0], axis=-1) > 0).any(axis=-1).flatten()
        true_inf = np.asarray((y[..., 1]).max(axis=-1)).flatten()
        true_sep3 = np.asarray((y[..., 2] == 1.0).any(axis=-1)).flatten()
        pred_sep3_p = np.asarray(metrics["sepsis_metrics"]["sep3_p"]).flatten()
        pred_sofa_d2_p = np.asarray(metrics["sepsis_metrics"]["sofa_d2_p"]).flatten()
        pred_susp_inf_p = np.asarray(metrics["sepsis_metrics"]["susp_inf_p"]).flatten()
        pred_sofa_score = np.asarray(metrics["hists"]["sofa_score"]).flatten()
    else:  # if detection
        true_sofa = np.asarray(y[..., 0])[mask]
        true_sofa_d2 = np.concat(
            [np.zeros(y.shape[:-2])[..., None], np.asarray(jnp.diff(y[..., 0], axis=-1) > 0)], axis=-1
        )[mask]
        true_inf = np.asarray(y[..., 1])[mask]
        true_sep3 = np.asarray(y[..., 2] == 1.0)[mask]
        pred_sep3_p = np.asarray(metrics["sepsis_metrics"]["sep3_p"])[mask]
        pred_sofa_d2_p = np.asarray(metrics["sepsis_metrics"]["sofa_d2_p"])[mask]
        pred_susp_inf_p = np.asarray(metrics["sepsis_metrics"]["susp_inf_p"])[mask].squeeze()
        pred_sofa_score = np.asarray(metrics["hists"]["sofa_score"])[mask]

    print("lims sofa", pred_sofa_d2_p.min(), pred_sofa_d2_p.max())
    print("lims inf ", pred_susp_inf_p.min(), pred_susp_inf_p.max())
    print("lims sep ", pred_sep3_p.min(), pred_sep3_p.max())

    fig, ax = plt.subplots(1, 1)
    ax = viz_starter(
        metrics["latents"]["alpha"][0],
        metrics["latents"]["beta"][0],
        metrics["latents"]["sigma"][0],
        lookup=lookup,
        mask=mask[0],
        filename="",
        figax=(fig, ax),
    )  # only first batch
    writer.add_figure("Latents@0", fig, epoch, close=True)

    fig, ax = plt.subplots(1, 1)
    ax = viz_progression(
        y[..., 0],
        y[..., 1],
        metrics["hists"]["sofa_score"],
        metrics["sepsis_metrics"]["susp_inf_p"],
        mask=mask,
        filename="",
        ax=ax,
    )
    writer.add_figure("Progression", fig, epoch, close=True)

    fig, ax = plt.subplots(1, 1)
    seq_idx = np.argmax(y[0, :, :, 0].var(axis=-1))  # highest sofa std in first batch

    ax = viz_plane(
        true_sofa=y[0, seq_idx, :, 0],
        alphas=metrics["latents"]["alpha"][0, seq_idx],
        betas=metrics["latents"]["beta"][0, seq_idx],
        sigmas=metrics["latents"]["sigma"][0, seq_idx],
        mask=mask[0, seq_idx],
        lookup=lookup,
        cmaps=False,
        filename="",
        figax=(fig, ax),
    )
    writer.add_figure("In Param Space", fig, epoch, close=True)

    fig, ax = plt.subplots(1, 2)
    ax = viz_heatmap_concepts(
        true_sofa,
        true_inf,
        pred_sofa_score,
        pred_susp_inf_p,
        cmap=False,
        filename="",
        figax=(fig, ax),
    )
    writer.add_figure("Performance", fig, epoch, close=True)

    fig, ax = plt.subplots(1, 2)
    ax = viz_curves(
        true_sofa_d2,
        true_inf > 0,
        true_sep3,
        pred_sofa_d2_p,
        pred_susp_inf_p,
        pred_sep3_p,
        filename="",
        figax=(fig, ax),
    )
    writer.add_figure("Confusion", fig, epoch, close=True)

    log_msg = f"Epoch {epoch} Valdation "
    for k in metrics:
        if k == "hists":
            writer.add_histogram("SOFA Score", np.asarray(pred_sofa_score), epoch, bins=25)
        elif k == "mult":
            for t, v in enumerate(np.asarray(metrics["mult"]["sofa_t"]).mean(axis=(0, 1))):
                writer.add_scalar(f"sofa_per_timestep/t{t}", np.asarray(v), epoch)
        elif k == "sepsis_metrics":
            writer.add_scalar(k + "/AUROC_pred_sep", roc_auc_score(true_sep3, pred_sep3_p), epoch)
            writer.add_scalar(k + "/AUPRC_pred_sep", average_precision_score(true_sep3, pred_sep3_p), epoch)
            writer.add_scalar(k + "/AUROC_pred_sofa_d2", roc_auc_score(true_sofa_d2, pred_sofa_d2_p), epoch)
            writer.add_scalar(k + "/AUPRC_pred_sofa_d2", average_precision_score(true_sofa_d2, pred_sofa_d2_p), epoch)
            writer.add_scalar(k + "/AUROC_pred_susp_inf", roc_auc_score(true_inf > 0, pred_susp_inf_p), epoch)
            writer.add_scalar(k + "/AUPRC_pred_susp_inf", average_precision_score(true_inf > 0, pred_susp_inf_p), epoch)
            log_msg += f"AUROC = {float(roc_auc_score(true_sep3, pred_sep3_p)):.4f}, "
            log_msg += f"AUPRC = {float(average_precision_score(true_sep3, pred_sep3_p)):.4f}, "
        elif k in ("cosine_annealings"):
            continue
        else:
            for metric_name, metric_values in metrics[k].items():
                masked_values = metric_values[mask] if metric_values.shape == mask.shape else metric_values
                log_msg += (
                    f"{metric_name}={float(masked_values.mean()):.4f}({float(masked_values.std()):.2f}), "
                    if metric_name in ("total_loss", "sofa", "infection", "sepsis-3")
                    else ""
                )
                writer.add_scalar(f"val_{k}/{metric_name}_mean", np.asarray(masked_values.mean()), epoch)
                writer.add_scalar(f"val_{k}/{metric_name}_std", np.asarray(masked_values.std()), epoch)
    return log_msg
