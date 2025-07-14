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

from sepsis_osc.model.vae import make_decoder, make_encoder

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

    infection_direct: Array
    sofa_direct: Array
    infection_lookup: Array
    sofa_lookup: Array
    lookup_temperature: Array
    label_temperature: Array

    sigma_recon: Array
    sigma_locality: Array
    sigma_concept: Array
    sigma_tc: Array

    total_loss: Array
    recon_loss: Array
    concept_loss: Array
    tc_loss: Array

    hists_sofa_score: Array

    @staticmethod
    def empty() -> "AuxLosses":
        empty_losses = AuxLosses(
            alpha=jnp.zeros(()),
            beta=jnp.zeros(()),
            sigma=jnp.zeros(()),
            infection_direct=jnp.zeros(()),
            sofa_direct=jnp.zeros(()),
            infection_lookup=jnp.zeros(()),
            sofa_lookup=jnp.zeros(()),
            lookup_temperature=jnp.zeros(()),
            label_temperature=jnp.zeros(()),
            sigma_recon=jnp.zeros(()),
            sigma_locality=jnp.zeros(()),
            sigma_concept=jnp.zeros(()),
            sigma_tc=jnp.zeros(()),
            total_loss=jnp.zeros(()),
            recon_loss=jnp.zeros(()),
            concept_loss=jnp.zeros(()),
            tc_loss=jnp.zeros(()),
            hists_sofa_score=jnp.ones(()),
        )
        return empty_losses

    def to_dict(self) -> dict[str, dict[str, jnp.ndarray]]:
        return {
            "latents": {
                "alpha": self.alpha,
                "beta": self.beta,
                "sigma": self.sigma,
            },
            "sigmas": {
                "s_recon": self.sigma_recon,
                "s_locality": self.sigma_locality,
                "s_concept": self.sigma_concept,
                "s_tc": self.sigma_tc,
            },
            "concepts": {
                "infection_direct": self.infection_direct,
                "sofa_direct": self.sofa_direct,
                "infection_lookup": self.infection_lookup,
                "sofa_lookup": self.sofa_lookup,
                "lookup_temperature": self.lookup_temperature,
                "label_temperature": self.label_temperature,
            },
            "losses": {
                "total_loss": self.total_loss,
                "recon_loss": self.recon_loss,
                "concept_loss": self.concept_loss,
                "tc_loss": self.tc_loss,
            },
            "hists": {
                "sofa_score": self.hists_sofa_score,
            },
        }


@dataclass
class ModelConfig:
    latent_dim: int
    input_dim: int
    enc_hidden: int
    dec_hidden: int


@dataclass
class TrainingConfig:
    batch_size: int
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
    w_tc:float
    lookup_vs_direct: float
    concept: ConceptLossConfig


def save_checkpoint(
    save_dir: str,
    epoch: int,
    params_enc: PyTree,
    static_enc: PyTree,
    params_dec: PyTree,
    static_dec: PyTree,
    opt_state_enc: OptState,
    opt_state_dec: OptState,
    hyper_enc: dict[str, float | int],
    hyper_dec: dict[str, int],
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04d}.eqx")

    full_model_state = {
        "encoder_params": params_enc,
        "encoder_static": static_enc,
        "decoder_params": params_dec,
        "decoder_static": static_dec,
        "opt_state_enc": opt_state_enc,
        "opt_state_dec": opt_state_dec,
    }
    hyper = {"encoder": hyper_enc, "decoder": hyper_dec}

    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyper)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, full_model_state)
    logger.info(f"Model checkpoint saved for epoch {epoch} to {filename}")


def load_checkpoint(
    load_dir: str,
    epoch: int,
    opt_enc_template: GradientTransformation | None = None,
    opt_dec_template: GradientTransformation | None = None,
) -> tuple[PyTree, PyTree, PyTree, PyTree, OptState, OptState]:
    filename = os.path.join(load_dir, f"checkpoint_epoch_{epoch:04d}.eqx")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found at {filename}")

    with open(filename, "rb") as f:
        hyper = json.loads(f.readline().decode())
        hyper_enc = hyper["encoder"]
        hyper_dec = hyper["decoder"]

        key_dummy = jr.PRNGKey(0)
        skeleton_encoder = make_encoder(key_dummy, **hyper_enc)
        skeleton_decoder = make_decoder(key_dummy, **hyper_dec)

        skeleton_params_enc, skeleton_static_enc = eqx.partition(skeleton_encoder, eqx.is_array)
        skeleton_params_dec, skeleton_static_dec = eqx.partition(skeleton_decoder, eqx.is_array)

        if opt_enc_template is None or opt_dec_template is None:
            logger.warning(
                "Reading empty optimizer templates, if you want to resume training, provide suitable templates."
            )
        skeleton_opt_state_enc = opt_enc_template.init(skeleton_params_enc) if opt_enc_template else None
        skeleton_opt_state_dec = opt_dec_template.init(skeleton_params_dec) if opt_dec_template else None

        skeleton_full_model_state = {
            "encoder_params": skeleton_params_enc,
            "encoder_static": skeleton_static_enc,
            "decoder_params": skeleton_params_dec,
            "decoder_static": skeleton_static_dec,
            "opt_state_enc": skeleton_opt_state_enc,
            "opt_state_dec": skeleton_opt_state_dec,
        }

        full_model_state = eqx.tree_deserialise_leaves(f, skeleton_full_model_state)

    logger.info(f"Model checkpoint loaded for epoch {epoch} from {filename}")

    return (
        full_model_state["encoder_params"],
        full_model_state["encoder_static"],
        full_model_state["decoder_params"],
        full_model_state["decoder_static"],
        full_model_state["opt_state_enc"],
        full_model_state["opt_state_dec"],
    )


def prepare_batches(
    x_data: Float[Array, "nsamples dim"],
    y_data: Float[Array, "nsamples dim"],
    batch_size: int,
    key: jnp.ndarray,
    perc: float = 1.0,
    shuffle=True,
) -> tuple[Float[Array, "nbatches batch dim"], Float[Array, "nbatches batch dim"], int]:
    # TODO balance classes for training?

    num_samples = int(perc * x_data.shape[0])
    num_features = x_data.shape[1]
    num_targets = y_data.shape[1]
    x_data = x_data[:num_samples]
    y_data = y_data[:num_samples]

    # Shuffle data
    if shuffle:
        perm = jr.permutation(key, num_samples)
        x_shuffled = x_data[perm]
        y_shuffled = y_data[perm]
    else:
        x_shuffled = x_data
        y_shuffled = y_data
        

    # Ensure full batches only
    num_full_batches = num_samples // batch_size
    x_truncated = x_shuffled[: num_full_batches * batch_size]
    y_truncated = y_shuffled[: num_full_batches * batch_size]

    # Reshape into batches
    x_batched = x_truncated.reshape(num_full_batches, batch_size, num_features)
    y_batched = y_truncated.reshape(num_full_batches, batch_size, num_targets)

    return x_batched, y_batched, num_full_batches


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
