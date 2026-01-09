import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.ldm.ae import Decoder, LatentEncoder
from sepsis_osc.ldm.commons import qr_init
from sepsis_osc.utils.jax_config import typechecker

ones_24 = jnp.ones(24)


class LatentDynamicsModel(eqx.Module):
    latent_pre_encoder: LatentEncoder
    latent_encoder: eqx.nn.GRUCell
    latent_rollout: eqx.nn.GRUCell
    latent_proj_out: eqx.nn.Linear
    latent_enc_hidden_dim: int = eqx.field(static=True)
    latent_hidden_dim: int = eqx.field(static=True)

    inf_encoder: eqx.nn.GRUCell
    inf_rollout: eqx.nn.GRUCell
    inf_proj_out: eqx.nn.Linear
    inf_hidden_dim: int = eqx.field(static=True)

    latent_h0: Array
    inf_h0: Array

    dec_hidden_dim: int = eqx.field(static=True)
    decoder: Decoder

    input_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    inf_dim: int = eqx.field(static=True)

    sofa_dist: Float[Array, "24"] = eqx.field(static=True)
    lookup_kernel_size: int

    _lookup_temperature: Float[Array, "1"]
    _d_diff: Array
    _d_scale: Array

    _sofa_label_smooth: Array
    _sofa_d2_pred_smooth: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        input_dim: int,
        latent_dim: int,
        latent_enc_hidden_dim: int,
        latent_hidden_dim: int,
        inf_dim: int,
        inf_hidden_dim: int,
        dec_hidden_dim: int,
        lookup_kernel_size: int = 3,
        sofa_dist: Float[Array, "24"] = ones_24,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        key_dec, key_enc, keyz, keyinf = jr.split(key, 4)

        self.lookup_kernel_size = lookup_kernel_size
        self.sofa_dist = sofa_dist
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inf_dim = inf_dim
        self.latent_enc_hidden_dim = latent_enc_hidden_dim
        self.latent_hidden_dim = latent_hidden_dim
        self.inf_hidden_dim = inf_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        self.latent_pre_encoder = LatentEncoder(
            key_enc,
            input_dim=input_dim * 2,
            latent_enc_hidden=latent_enc_hidden_dim,
            latent_pred_hidden=latent_hidden_dim,
            dtype=dtype,
        )
        self.latent_encoder = eqx.nn.GRUCell((input_dim * 2) + latent_dim, latent_hidden_dim, key=keyz, use_bias=False)
        self.latent_h0 = jnp.zeros(latent_hidden_dim)
        self.latent_rollout = eqx.nn.GRUCell(latent_dim, latent_hidden_dim, key=keyz, dtype=dtype)
        self.latent_proj_out = eqx.nn.Linear(latent_hidden_dim, latent_dim, key=keyz, dtype=dtype, use_bias=False)

        self.inf_encoder = eqx.nn.GRUCell(input_dim * 2, inf_hidden_dim, key=keyinf, use_bias=False)
        self.inf_h0 = jnp.zeros(inf_hidden_dim)
        self.inf_rollout = eqx.nn.GRUCell(inf_dim, inf_hidden_dim, key=keyinf, dtype=dtype)
        self.inf_proj_out = eqx.nn.Linear(inf_hidden_dim, inf_dim, key=keyinf, dtype=dtype)

        self.decoder = Decoder(
            key_dec,
            input_dim=input_dim,
            z_latent_dim=latent_dim,
            dec_hidden=dec_hidden_dim,
            dtype=dtype,
        )

        self._d_diff = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 1 / 25)
        self._d_scale = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 50)

        self._sofa_label_smooth = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.7)
        self._sofa_d2_pred_smooth = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.7)
        self._lookup_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.05)

    @property
    def n_params(self) -> int:
        return sum(x.size if isinstance(x, jnp.ndarray) else 1 for x in jax.tree_util.tree_leaves(self))

    @property
    def lookup_temperature(self) -> Float[Array, "1"]:
        return jnp.exp(self._lookup_temperature)

    @property
    def d_thresh(self) -> Float[Array, "1"]:
        return jnp.exp(self._d_diff)

    @property
    def d_scale(self) -> Float[Array, "1"]:
        return jnp.exp(self._d_scale)

    @property
    def sofa_d2_label_smooth(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_label_smooth)

    @property
    def sofa_d2_pred_smooth(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_d2_pred_smooth)

    def hypers_dict(self) -> dict[str, int | float | Array]:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "latent_enc_hidden_dim": self.latent_enc_hidden_dim,
            "latent_hidden_dim": self.latent_hidden_dim,
            "inf_dim": self.inf_dim,
            "inf_hidden_dim": self.inf_hidden_dim,
            "dec_hidden_dim": self.dec_hidden_dim,
            "lookup_kernel_size": self.lookup_kernel_size,
            "sofa_dist": self.sofa_dist,
        }

    def params_to_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "lookup_temperature": self.lookup_temperature,
            "d_diff": self.d_thresh,
            "d_scale": self.d_scale,
            "sofa_d2_pred_smooth": self.sofa_d2_pred_smooth,
            "sofa_d2_label_smooth": self.sofa_d2_label_smooth,
        }

    @jaxtyped(typechecker=typechecker)
    def online_sequence(
        self, xs: Float[Array, "time input_dim"]
    ) -> tuple[Float[Array, "time latent_dim"], Float[Array, "time 1"]]:
        @jaxtyped(typechecker=typechecker)
        def z_step(
            carry: tuple[Float[Array, " latent_hidden_dim"], Float[Array, " latent_dim"]],
            x_t: Float[Array, " input_dim"],
        ) -> tuple[
            tuple[Float[Array, " latent_hidden_dim"], Float[Array, " latent_dim"]],
            Float[Array, " latent_dim"],
        ]:
            h_prev, z_prev = carry
            h_t = self.latent_encoder(jnp.concat([x_t, z_prev]), h_prev)
            dz_t = self.latent_proj_out(h_t)
            z_t = z_prev + dz_t
            return (h_t, z_t), z_t

        h0, z0 = self.latent_pre_encoder(xs[0])

        (_, _), z_preds = jax.lax.scan(z_step, (h0, z0), xs[1:])
        z_seq = jnp.concat([z0[None], z_preds], axis=0)

        @jaxtyped(typechecker=typechecker)
        def inf_step(
            h_prev: Float[Array, " inf_hidden_dim"], x_t: Float[Array, " input_dim"]
        ) -> tuple[Float[Array, " inf_hidden_dim"], Float[Array, "* inf_hidden_dim"]]:
            h_next = self.inf_encoder(x_t, h_prev)
            return h_next, h_next

        _, ihs = jax.lax.scan(inf_step, self.inf_h0, xs)
        inf_seq = jax.vmap(self.inf_proj_out)(ihs)

        return jax.nn.sigmoid(z_seq), inf_seq

    @jaxtyped(typechecker=typechecker)
    def offline_sequence(
        self, x: Float[Array, "1 input_dim"], key: jnp.ndarray
    ) -> tuple[Float[Array, "time latent_dim"], Float[Array, "time 1"]]:
        @jaxtyped(typechecker=typechecker)
        def z_step(
            carry: tuple[
                Float[Array, " latent_hidden_dim"],  # h_t
                Float[Array, " latent_dim"],  # z_t
            ],
            _: tuple[Float[Array, " input_dim"], jnp.ndarray],
        ) -> tuple[tuple[Float[Array, " latent_hidden_dim"], Float[Array, " latent_dim"]], Float[Array, " latent_dim"]]:
            h_prev, z_t = carry
            h_next = self.latent_rollout(z_t, h_prev)
            z_next = self.latent_proj_out(h_next)
            return (h_next, z_next), z_next

        pre_x = self.latent_pre_encoder(x)
        zh0 = self.latent_encoder(pre_x, self.latent_h0)
        _, zhs = jax.lax.scan(z_step, zh0, x[1:])
        z_pred = jnp.concat([zh0, zhs], axis=-1)
        z_seq = jax.vmap(self.latent_proj_out)(z_pred)

        @jaxtyped(typechecker=typechecker)
        def inf_step(
            carry: tuple[
                Float[Array, " inf_hidden_dim"],  # h_t
                Float[Array, " inf_dim"],  # i_t
            ],
            _: tuple[Float[Array, " input_dim"], jnp.ndarray],
        ) -> tuple[tuple[Float[Array, " inf_hidden_dim"], Float[Array, " inf_dim"]], Float[Array, " inf_dim"]]:
            h_prev, i_t = carry
            h_next = self.inf_rollout(i_t, h_prev)
            i_next = self.inf_proj_out(h_next)
            return (h_next, i_next), i_next

        ih0 = self.inf_encoder(x, self.inf_h0)
        _, ihs = jax.lax.scan(inf_step, ih0, x[1:])
        inf_pred = jnp.concat([ih0, ihs], axis=-1)
        inf_seq = jax.vmap(self.latent_proj_out)(inf_pred)

        return jax.nn.sigmoid(z_seq), inf_seq


def make_ldm(
    key: PRNGKeyArray,
    input_dim: int,
    latent_dim: int,
    latent_enc_hidden_dim: int,
    latent_hidden_dim: int,
    inf_dim: int,
    inf_hidden_dim: int,
    dec_hidden_dim: int,
    lookup_kernel_size: int,
    sofa_dist: Float[Array, "24"],
):
    return LatentDynamicsModel(
        key=key,
        input_dim=input_dim,
        latent_dim=latent_dim,
        latent_enc_hidden_dim=latent_enc_hidden_dim,
        latent_hidden_dim=latent_hidden_dim,
        inf_dim=inf_dim,
        inf_hidden_dim=inf_hidden_dim,
        dec_hidden_dim=dec_hidden_dim,
        lookup_kernel_size=lookup_kernel_size,
        sofa_dist=sofa_dist,
    )


def init_ldm_weights(ldm: LatentDynamicsModel, key: jnp.ndarray, scale: float = 1e-1) -> LatentDynamicsModel:
    enc_key, gru_key = jr.split(key)
    # ldm = eqx.tree_at(
    #     lambda e: e.latent_pre_encoder, ldm, apply_initialization(ldm.latent_pre_encoder, qr_init, zero_bias_init, enc_key)
    # )
    ldm = eqx.tree_at(lambda e: e.latent_proj_out.weight, ldm, qr_init(ldm.latent_proj_out.weight, gru_key) * scale)
    return ldm
