import json
import logging
import os

import equinox as eqx
import jax.random as jr
from jaxtyping import PyTree
from optax import GradientTransformation, OptState

from sepsis_osc.ldm.ae import make_decoder, make_encoder
from sepsis_osc.ldm.latent_dynamics import LatentDynamicsModel
from sepsis_osc.ldm.gru import make_predictor

logger = logging.getLogger(__name__)



def save_checkpoint(
    save_dir: str,
    epoch: int,
    model,
    opt_state: OptState,
    hyper_enc: dict[str, float | int],
    hyper_dec: dict[str, int],
    hyper_pred: dict[str, int],
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04d}.eqx")

    # = {"opt_state": opt_state}
    hyper = {"encoder": hyper_enc, "decoder": hyper_dec, "predictor": hyper_pred}

    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyper)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, (model, opt_state))
    logger.info(f"Model checkpoint saved for epoch {epoch} to {filename}")


def load_checkpoint(
    load_dir: str,
    epoch: int,
    opt_template: GradientTransformation | None = None,
) -> tuple[PyTree, OptState]:
    filename = os.path.join(load_dir, f"checkpoint_epoch_{epoch:04d}.eqx")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found at {filename}")

    with open(filename, "rb") as f:
        hyper = json.loads(f.readline().decode())
        hyper_enc = hyper["encoder"]
        hyper_dec = hyper["decoder"]
        hyper_predictor = hyper["predictor"]

        # Build skeleton full model
        key_dummy = jr.PRNGKey(0)
        encoder = make_encoder(key_dummy, **hyper_enc)
        decoder = make_decoder(key_dummy, **hyper_dec)
        predictor = make_predictor(key_dummy, **hyper_predictor)

        model = eqx.nn.inference_mode(
            LatentDynamicsModel(encoder=encoder, predictor=predictor, decoder=decoder), value=False
        )
        params_model, static_model = eqx.partition(model, eqx.is_array)

        opt_state = opt_template.init(params_model) if opt_template else None

        loaded_model, loaded_opt_state = eqx.tree_deserialise_leaves(f, (model, opt_state))
    logger.info(f"Model checkpoint loaded for epoch {epoch} from {filename}")
    return loaded_model, loaded_opt_state
