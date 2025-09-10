import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
                log_msg += f"{metric_name}={float(metric_values.mean()):.4f}({float(metric_values.std()):.2f}), "
            writer.add_scalar(f"train_{group_key}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
    return log_msg


def log_val_metrics(metrics, y, epoch, writer):
    log_msg = f"Epoch {epoch} Valdation "
    metrics = metrics.to_dict()
    for k in metrics.keys():
        if k == "hists":
            writer.add_histogram(
                "SOFA Score", np.asarray(metrics["hists"]["sofa_score"][:, 0].flatten()), epoch, bins=25
            )
            writer.add_histogram("SOFA metric", np.asarray(metrics["hists"]["sofa_metric"].flatten()), epoch, bins=25)
            writer.add_histogram(
                "Inf Error", np.asarray(metrics["hists"]["inf_prob"].flatten() - (y[..., 1]).flatten()), epoch
            )
        elif k == "mult":
            for t, v in enumerate(np.asarray(metrics["mult"]["infection_t"]).mean(axis=(0, 1))):
                writer.add_scalar(f"infection_per_timestep/t{t}", np.asarray(v), epoch)
            for t, v in enumerate(np.asarray(metrics["mult"]["sofa_t"]).mean(axis=(0, 1))):
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
            log_msg += f"AUROC = {float(roc_auc_score(sep3, pred_sep3_p)):.4f}, "
            log_msg += f"AUPRC = {float(average_precision_score(sep3, pred_sep3_p)):.4f}, "
        elif k in ("cosine_annealings"):
            continue
        else:
            for metric_name, metric_values in metrics[k].items():
                log_msg += (
                    f"{metric_name}={float(metric_values.mean()):.4f}({float(metric_values.std()):.2f}), "
                    if metric_name in ("total_loss", "sofa", "infection", "sepsis-3")
                    else ""
                )
                writer.add_scalar(f"val_{k}/{metric_name}_mean", np.asarray(metric_values.mean()), epoch)
                writer.add_scalar(f"val_{k}/{metric_name}_std", np.asarray(metric_values.std()), epoch)
    return log_msg
