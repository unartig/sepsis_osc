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

    hists_sofa_score: Array
    hists_sofa_metric: Array
    hists_inf_prob: Array

    total_loss: Array
    sep3_loss_t: Array
    sofa_loss_t: Array
    infection_p_loss_t: Array

    recon_loss: Array
    spreading_loss: Array
    boundary_loss: Array

    sofa_d2_risk: Array
    susp_inf_p: Array
    sep3_risk: Array

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
                "sofa": self.sofa_loss_t,
                "infection": self.infection_p_loss_t,
                "sepsis-3": self.sep3_loss_t,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
            },
            "mult": {
                "sofa_t": self.sofa_loss_t,
            },
            "sepsis_metrics": {
                "sofa_d2_risk": self.sofa_d2_risk,
                "susp_inf_p": self.susp_inf_p,
                "sep3_risk": self.sep3_risk,
            },
        }


@dataclass
class TrainingConfig:
    """
    Configuration dataclass for the training loop execution.
    """

    batch_size: int
    epochs: int
    mini_epochs: int = 1
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
    warmup_epochs: int
    enc_wd: float


@register_dataclass
@dataclass
class LossesConfig:
    """
    Configuration dataclass for the losses involved in training.
    """

    lambda_sep3: float
    lambda_sofa_classification: float
    lambda_inf: float

    lambda_spreading: float
    lambda_boundary: float

    lambda_recon: float

    steps_per_epoch: int = 0

