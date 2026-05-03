import json
import logging
import pickle
import struct
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
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(f"{save_dir}/checkpoint_epoch_{epoch:04d}.eqx")

    with filename.open("wb") as f:
        f.write((json.dumps(hyper_ldm) + "\n").encode())

        lookup_bytes = pickle.dumps(model.lookup)
        f.write(struct.pack(">Q", len(lookup_bytes)))  # 8-byte big-endian length
        f.write(lookup_bytes)

        eqx.tree_serialise_leaves(f, (model, opt_state))

    logger.info(f"Checkpoint saved for epoch {epoch} to {filename}")


def load_checkpoint(
    load_dir: str,
    epoch: int,
    opt_template: GradientTransformation | None = None,
) -> tuple[PyTree, OptState]:
    filename = Path(f"{load_dir}/checkpoint_epoch_{epoch:04d}.eqx")
    if not filename.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    with filename.open("rb") as f:
        hyper = json.loads(f.readline().decode())

        (lookup_len,) = struct.unpack(">Q", f.read(8))
        lookup = pickle.loads(f.read(lookup_len))

        key_dummy = jr.PRNGKey(0)
        model = make_ldm(key_dummy, lookup=lookup, **hyper)
        params_model, _ = eqx.partition(model, eqx.is_array)
        opt_state = opt_template.init(params_model) if opt_template else None
        loaded_model, loaded_opt_state = eqx.tree_deserialise_leaves(f, (model, opt_state))

    logger.info(f"Checkpoint loaded for epoch {epoch} from {filename}")
    return loaded_model, loaded_opt_state
