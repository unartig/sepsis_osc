from dataclasses import dataclass, fields

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass
from jaxtyping import Array


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
    tc_loss: Array
    accel_loss: Array
    diff_loss: Array
    thresh_loss: Array
    matching_loss: Array

    hists_sofa_score: Array
    hists_sofa_metric: Array
    hists_inf_prob: Array

    sofa_loss_t: Array
    sofa_d2_p_loss: Array
    infection_p_loss_t: Array
    sep3_p_loss: Array
    directional_loss: Array
    trend_loss: Array

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
                "directional_loss": self.directional_loss,
                "trend_loss": self.trend_loss,
                "sofa": self.sofa_loss_t,
                "sofa_d2": self.sofa_d2_p_loss,
                "infection": self.infection_p_loss_t,
                "sepsis-3": self.sep3_p_loss,
            },
            "losses": {
                "total_loss": self.total_loss,
                "recon_loss": self.recon_loss,
                "tc_loss": self.tc_loss,
                "accel_loss": self.accel_loss,
                "diff_loss": self.diff_loss,
                "thresh_loss": self.thresh_loss,
                "matching_loss": self.matching_loss,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
                "sofa_metric": self.hists_sofa_metric,
                "inf_prob": self.hists_inf_prob,
            },
            "mult": {
                "infection_t": self.infection_p_loss_t,
                "sofa_t": self.sofa_loss_t,
            },
            "cosine_annealings": {
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
    z_latent_dim: int
    v_latent_dim: int
    input_dim: int
    enc_hidden: int
    dec_hidden: int
    predictor_z_hidden: int
    predictor_v_hidden: int
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
class LossesConfig:
    w_recon: float
    w_tc: float
    w_accel: float
    w_diff: float
    w_sofa_direction: float
    w_sofa_trend: float
    w_sofa_classification: float
    w_sofa_d2: float
    w_inf: float
    w_inf_alpha: float
    w_inf_gamma: float
    w_sep3: float
    w_matching: float
    w_thresh: float
    anneal_threshs_iter: float
    steps_per_epoch: int = 0
