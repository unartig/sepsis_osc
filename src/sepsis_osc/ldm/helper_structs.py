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

    total_loss: Array
    recon_loss: Array
    sequence_loss: Array
    tc_loss: Array
    acceleration_loss: Array
    velocity_loss: Array
    diff_loss: Array
    thresh_loss: Array
    spreading_loss: Array
    distr_loss: Array

    hists_sofa_score: Array
    hists_sofa_metric: Array
    hists_inf_prob: Array

    sofa_loss_t: Array
    sofa_d2_p_loss: Array
    infection_p_loss_t: Array
    sep3_p_loss: Array
    sofa_directional_loss: Array

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
                "sofa_directional_loss": self.sofa_directional_loss,
                "sofa": self.sofa_loss_t,
                "sofa_d2": self.sofa_d2_p_loss,
                "infection": self.infection_p_loss_t,
                "sepsis-3": self.sep3_p_loss,
            },
            "losses": {
                "total_loss": self.total_loss,
                "recon_loss": self.recon_loss,
                "tc_loss": self.tc_loss,
                "sequence_loss": self.sequence_loss,
                "accel_loss": self.acceleration_loss,
                "velocity_loss": self.velocity_loss,
                "diff_loss": self.diff_loss,
                "thresh_loss": self.thresh_loss,
                "spreading_loss": self.spreading_loss,
                "distr_loss": self.distr_loss,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
                "sofa_metric": self.hists_sofa_metric,
                "inf_prob": self.hists_inf_prob,
            },
            "mult": {
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
    input_dim: int
    enc_hidden: int
    inf_pred_hidden: int
    dec_hidden: int
    predictor_z_hidden: int
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
    w_sequence: float
    w_tc: float
    w_acceleration: float
    w_velocity: float
    w_diff: float
    w_sofa_direction: float
    w_sofa_classification: float
    w_sofa_d2: float
    w_inf: float
    w_sep3: float
    w_matching: float
    w_spreading: float
    w_distr: float
    w_thresh: float
    anneal_threshs_iter: float
    steps_per_epoch: int = 0
