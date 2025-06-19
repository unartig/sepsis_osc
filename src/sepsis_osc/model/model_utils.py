import json
import logging
import os
from functools import wraps
from time import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PyTree
from optax import GradientTransformation, OptState

from sepsis_osc.model.vae import make_decoder, make_encoder
from sepsis_osc.utils.logger import setup_logging

setup_logging("info")
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
) -> tuple[Float[Array, "nbatches batch dim"], Float[Array, "nbatches batch dim"], int]:
    num_samples = x_data.shape[0]
    num_features = x_data.shape[1]
    num_targets = y_data.shape[1]

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
    origin = coords.min(axis=0)

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
