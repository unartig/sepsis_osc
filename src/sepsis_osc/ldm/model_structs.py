from dataclasses import dataclass, fields

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array


@register_dataclass
@dataclass
class AuxLosses:
    """
    Container for all loss components and auxiliary metrics generated during a model step.
    Organizes reconstruction, conceptual, and clinical losses (SOFA, Sepsis-3).
    Used for logging and gradient computation.
    """

    beta: Array
    sigma: Array

    total_loss: Array
    recon_loss: Array

    hists_sofa_score: Array
    hists_sofa_metric: Array
    hists_inf_prob: Array

    total_loss: Array
    recon_loss: Array
    spreading_loss: Array
    boundary_loss: Array
    sofa_loss_t: Array
    sofa_d2_p_loss: Array
    infection_p_loss_t: Array
    sep3_p_loss: Array
    sofa_directional_loss: Array

    sofa_d2_p: Array
    susp_inf_p: Array
    sep3_p: Array

    @staticmethod
    def empty() -> "AuxLosses":
        initialized_fields = {f.name: jnp.zeros(()) for f in fields(AuxLosses)}
        return AuxLosses(**initialized_fields)

    def to_dict(self) -> dict[str, dict[str, jnp.ndarray]]:
        return {
            "latents": {
                "beta": self.beta,
                "sigma": self.sigma,
            },
            "losses": {
                "total_loss": self.total_loss,
                "spreading_loss": self.spreading_loss,
                "boundary_loss": self.boundary_loss,
                "recon_loss": self.recon_loss,
                "sofa_directional_loss": self.sofa_directional_loss,
                "sofa": self.sofa_loss_t,
                "sofa_d2": self.sofa_d2_p_loss,
                "infection": self.infection_p_loss_t,
                "sepsis-3": self.sep3_p_loss,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
            },
            "mult": {
                "sofa_t": self.sofa_loss_t,
            },
            "sepsis_metrics": {
                "sofa_d2_p": self.sofa_d2_p,
                "susp_inf_p": self.susp_inf_p,
                "sep3_p": self.sep3_p,
            },
        }


@dataclass
class TrainingConfig:
    """
    Configuration dataclass for the training loop execution.
    """

    batch_size: int
    window_len: int
    epochs: int
    perc_train_set: float = 1.0
    validate_every: float = 1.0
    calibrate: bool = False
    early_stop: bool = False


@dataclass
class LoadingConfig:
    """
    Configuration dataclass for the loading of pretrained models.
    """

    from_dir: str
    epoch: int


@dataclass
class SaveConfig:
    """
    Configuration dataclass for the saving of models in training.
    """

    save_every: int
    perform: bool


@dataclass
class LRConfig:
    """
    Configuration dataclass for the learning rate schedule.
    """

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
    """
    Configuration dataclass for the losses involved in training.
    """

    w_spreading: float
    w_boundary: float
    w_recon: float
    w_sofa_direction: float
    w_sofa_classification: float
    w_sofa_d2: float
    w_inf: float
    w_sep3: float
    steps_per_epoch: int = 0

