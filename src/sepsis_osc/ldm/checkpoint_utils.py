import json
import logging
from pathlib import Path

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PyTree
from optax import GradientTransformation, OptState

from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel, make_ldm

logger = logging.getLogger(__name__)


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: LatentDynamicsModel,
    opt_state: OptState,
    hyper_ldm: dict[str, int | float | Array],
) -> None:
    """
    Serializes the model, optimizer state, and hyperparameters to a file.

    This function creates a hybrid checkpoint file. The first line is a JSON-encoded
    string of hyperparameters, followed by the binary serialization of the model
    and optimizer state leaves.
    """
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    filename = Path(f"{save_dir}/checkpoint_epoch_{epoch:04d}.eqx")

    with Path.open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyper_ldm)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, (model, opt_state))
    logger.info(f"Model checkpoint saved for epoch {epoch} to {filename}")


def load_checkpoint(
    load_dir: str,
    epoch: int,
    opt_template: GradientTransformation | None = None,
) -> tuple[PyTree, OptState]:
    """
    Reconstructs a model and optimizer state from a saved checkpoint.

    The function first reads the JSON header to retrieve hyperparameters,
    instantiates a 'skeleton' model using `make_ldm`, and then populates that
    skeleton with the saved binary weights.
    """
    filename = Path(f"{load_dir}/checkpoint_epoch_{epoch:04d}.eqx")

    if not Path(filename).exists():
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found at {filename}")

    with Path.open(filename, "rb") as f:
        hyper = json.loads(f.readline().decode())

        # Build skeleton full model
        key_dummy = jr.PRNGKey(0)
        model = make_ldm(key_dummy, **hyper)

        params_model, _static_model = eqx.partition(model, eqx.is_array)

        opt_state = opt_template.init(params_model) if opt_template else None

        loaded_model, loaded_opt_state = eqx.tree_deserialise_leaves(f, (model, opt_state))
    logger.info(f"Model checkpoint loaded for epoch {epoch} from {filename}")
    return loaded_model, loaded_opt_state
