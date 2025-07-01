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
from sepsis_osc.utils.logger import setup_logging

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
    latents_alpha: jnp.ndarray
    latents_beta: jnp.ndarray
    latents_sigma: jnp.ndarray

    concepts_infection_direct: jnp.ndarray
    concepts_sofa_direct: jnp.ndarray
    concepts_infection_lookup: jnp.ndarray
    concepts_sofa_lookup: jnp.ndarray
    concepts_lookup_temperature: jnp.ndarray
    concepts_label_temperature: jnp.ndarray

    losses_recon_loss: jnp.ndarray
    losses_locality_loss: jnp.ndarray
    losses_total_loss: jnp.ndarray
    losses_concept_loss: jnp.ndarray

    hists_sofa_score: jnp.ndarray

    @staticmethod
    def empty() -> "AuxLosses":
        empty_losses = AuxLosses(
            latents_alpha=jnp.zeros(()),
            latents_beta=jnp.zeros(()),
            latents_sigma=jnp.zeros(()),
            concepts_infection_direct=jnp.zeros(()),
            concepts_sofa_direct=jnp.zeros(()),
            concepts_infection_lookup=jnp.zeros(()),
            concepts_sofa_lookup=jnp.zeros(()),
            concepts_lookup_temperature=jnp.zeros(()),
            concepts_label_temperature=jnp.zeros(()),
            losses_recon_loss=jnp.zeros(()),
            losses_locality_loss=jnp.zeros(()),
            losses_total_loss=jnp.zeros(()),
            losses_concept_loss=jnp.zeros(()),
            hists_sofa_score=jnp.ones(()),
        )
        return empty_losses

    def to_dict(self) -> dict[str, dict[str, jnp.ndarray]]:
        return {
            "latents": {
                "alpha": self.latents_alpha,
                "beta": self.latents_beta,
                "sigma": self.latents_sigma,
            },
            "concepts": {
                "infection_direct": self.concepts_infection_direct,
                "sofa_direct": self.concepts_sofa_direct,
                "infection_lookup": self.concepts_infection_lookup,
                "sofa_lookup": self.concepts_sofa_lookup,
                "lookup_temperature": self.concepts_lookup_temperature,
                "label_temperature": self.concepts_label_temperature,
            },
            "losses": {
                "total_loss": self.losses_total_loss,
                "recon_loss": self.losses_recon_loss,
                "locality_loss": self.losses_locality_loss,
                "concept_loss": self.losses_concept_loss,
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
class LocalityLossConfig:
    sigma_input: float
    sigma_sofa: float
    w_input: float
    w_sofa: float
    z_scale: jnp.ndarray
    temperature: float


@register_dataclass
@dataclass
class LossesConfig:
    w_recon: float
    w_concept: float
    w_locality: float
    w_tc: float
    concept: ConceptLossConfig
    locality: LocalityLossConfig


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
) -> tuple[Float[Array, "nbatches batch dim"], Float[Array, "nbatches batch dim"], int]:
    # TODO balance classes for training?

    num_samples = int(perc * x_data.shape[0])
    num_features = x_data.shape[1]
    num_targets = y_data.shape[1]
    x_data = x_data[:num_samples]
    y_data = y_data[:num_samples]

    # Shuffle data
    perm = jr.permutation(key, num_samples)
    x_shuffled = x_data[perm]
    y_shuffled = y_data[perm]

    # Ensure full batches only
    num_full_batches = num_samples // batch_size
    x_truncated = x_shuffled[: num_full_batches * batch_size]
    y_truncated = y_shuffled[: num_full_batches * batch_size]

    # Reshape into batches
    x_batched = x_truncated.reshape(num_full_batches, batch_size, num_features)
    y_batched = y_truncated.reshape(num_full_batches, batch_size, num_targets)

    return x_batched, y_batched, num_full_batches


def infer_grid_params(coords: np.ndarray):
    origin = np.min(coords, axis=0)

    # Safe spacing: avoid zero-spacing errors
    spacing = []
    shape = []

    for i in range(3):
        unique_vals = np.unique(coords[:, i])
        if len(unique_vals) > 1:
            d = np.min(np.diff(unique_vals))
            spacing.append(d)
            dim_size = int(np.round((coords[:, i].max() - origin[i]) / d)) + 1
            shape.append(dim_size)
        else:
            # Grid is flat along this axis
            spacing.append(1.0)  # or a small epsilon
            shape.append(1)

    return np.array(origin), np.array(spacing), np.array(shape)


def as_3d_indices(
    alpha_space: tuple[float, float, float],
    beta_space: tuple[float, float, float],
    sigma_space: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    betas = np.arange(*beta_space)
    sigmas = np.arange(*sigma_space)
    alphas = np.arange(*alpha_space)
    beta_grid, sigma_grid, alpha_grid = np.meshgrid(betas, sigmas, alphas, indexing="ij")
    permutations = np.stack([alpha_grid, beta_grid, sigma_grid], axis=-1)
    a, b, s = permutations[:, :, :, 0:1], permutations[:, :, :, 1:2], permutations[:, :, :, 2:3]
    return a, b, s
