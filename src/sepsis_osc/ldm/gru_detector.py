import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, jaxtyped
from numpy.typing import DTypeLike

from sepsis_osc.ldm.ae import CoordinateEncoder, Decoder, make_decoder
from sepsis_osc.utils.config import ALPHA
from sepsis_osc.utils.jax_config import typechecker

ones_24 = jnp.ones(24)


class GRUDetector(eqx.Module):
    decoder: Decoder
    encoder: CoordinateEncoder
    z_gru_cell: eqx.nn.GRUCell
    z_proj_out: eqx.nn.Linear
    z_hidden_dim: int = eqx.field(static=True)

    inf_gru_cell: eqx.nn.GRUCell
    inf_proj_out: eqx.nn.Linear
    inf_h0: Float[Array, " inf_hidden_dim"]
    inf_hidden_dim: int = eqx.field(static=True)

    input_dim: int = eqx.field(static=True)
    z_dim: int = eqx.field(static=True)
    inf_dim: int = eqx.field(static=True)

    sofa_dist: Float[Array, "24"] = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    kernel_size: int

    _lookup_temperature: Float[Array, "1"]
    _d_diff: Array
    _d_scale: Array

    _sofa_dir_lsigma: Array
    _sofa_d2_lsigma: Array
    _sofa_label_smooth: Array
    _sofa_d2_pred_smooth: Array
    _sofa_class_lsigma: Array
    _inf_lsigma: Array
    _sep3_lsigma: Array
    _recon_lsigma: Array
    _spread_lsigma: Array

    _c: Array

    def __init__(
        self,
        key: jnp.ndarray,
        input_dim: int,
        z_dim: int,
        z_hidden_dim: int,
        inf_dim: int,
        inf_hidden_dim: int,
        decoder: Decoder,
        encoder: CoordinateEncoder,
        alpha: float = ALPHA,
        kernel_size: int = 3,
        sofa_dist: Float[Array, "24"] = ones_24,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        keyz, keyinf = jr.split(key, 2)

        self.decoder = decoder
        self.encoder = encoder

        self.alpha = alpha
        self.kernel_size = kernel_size
        self.sofa_dist = sofa_dist
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.inf_dim = inf_dim
        self.z_hidden_dim = z_hidden_dim
        self.inf_hidden_dim = inf_hidden_dim

        # self.z_gru_cell = eqx.nn.GRUCell(input_dim + z_dim, z_hidden_dim, key=keyz, dtype=dtype)
        self.z_gru_cell = eqx.nn.GRUCell(z_dim, z_hidden_dim, key=keyz, dtype=dtype, use_bias=False)
        self.z_proj_out = eqx.nn.Linear(z_hidden_dim, z_dim, key=keyz, dtype=dtype, use_bias=False)

        self.inf_gru_cell = eqx.nn.GRUCell(input_dim, inf_hidden_dim, key=keyinf, dtype=dtype, use_bias=False)
        self.inf_h0 = jnp.zeros(self.inf_gru_cell.hidden_size)
        self.inf_proj_out = eqx.nn.Linear(inf_hidden_dim, inf_dim, key=keyinf, dtype=dtype, use_bias=False)

        self._d_diff = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 1 / 25)
        self._d_scale = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 50)

        self._sofa_dir_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sofa_d2_lsigma = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.7)
        self._sofa_label_smooth = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.7)
        self._sofa_d2_pred_smooth = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.7)
        self._sofa_class_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._inf_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._sep3_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._recon_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._spread_lsigma = jnp.zeros((1,), dtype=jnp.float32)
        self._lookup_temperature = jnp.log(jnp.ones((1,), dtype=jnp.float32) * 0.05)

        self._c = jnp.ones((1,)) * 0.7

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
    def sofa_dir_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_dir_lsigma)

    @property
    def sofa_class_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_class_lsigma)

    @property
    def sofa_d2_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_d2_lsigma)

    @property
    def sofa_d2_label_smooth(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_label_smooth)

    @property
    def sofa_d2_pred_smooth(self) -> Float[Array, "1"]:
        return jnp.exp(self._sofa_d2_pred_smooth)

    @property
    def inf_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._inf_lsigma)

    @property
    def sep3_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._sep3_lsigma)

    @property
    def spread_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._spread_lsigma)

    @property
    def recon_lsigma(self) -> Float[Array, "1"]:
        return jnp.exp(self._recon_lsigma)

    @property
    def c(self) -> Float[Array, "1"]:
        return jax.nn.sigmoid(self._c)

    def hypers_dict(self) -> tuple[dict[str, int | float], dict[str, int], dict[str, int | float | Array]]:
        hyper_enc = {
            "input_dim": self.encoder.input_dim,
            "enc_hidden": self.encoder.enc_hidden,
            "pred_hidden": self.encoder.pred_hidden,
            "dropout_rate": self.encoder.dropout_rate,
        }
        hyper_dec = {
            "input_dim": self.decoder.input_dim,
            "z_latent_dim": self.decoder.z_latent_dim,
            "dec_hidden": self.decoder.dec_hidden,
        }
        hyper_gru = {
            "alpha": self.alpha,
            "kernel_size": self.kernel_size,
            "sofa_dist": self.sofa_dist,
            "input_dim": self.input_dim,
            "z_hidden_dim": self.z_hidden_dim,
            "inf_hidden_dim": self.inf_hidden_dim,
        }
        return hyper_enc, hyper_dec, hyper_gru

    def params_to_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "lookup_temperature": self.lookup_temperature,
            "d_diff": self.d_thresh,
            "d_scale": self.d_scale,
            "sofa_d2_pred_smooth": self.sofa_d2_pred_smooth,
            "sofa_d2_label_smooth": self.sofa_d2_label_smooth,
            "c": self.c
        }

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, xs: Float[Array, "time input_dim"], key: jnp.ndarray
    ) -> tuple[Float[Array, "time latent_dim"], Float[Array, "time 1"]]:
        keys = jax.random.split(key, xs.shape[0] + 1)
        key0, scan_keys = keys[0], keys[1:]

        beta, sigma, zh0 = self.encoder(xs[0], dropout_keys=self.encoder.make_keys(key0))
        z0 = jnp.concatenate([beta, sigma], axis=-1)

        @jaxtyped(typechecker=typechecker)
        def z_step(
            carry: tuple[
                Float[Array, " z_hidden_dim"],  # h_t
                Float[Array, " latent_dim"],  # z_t
            ],
            inputs_t: tuple[Float[Array, " input_dim"], jnp.ndarray],
        ) -> tuple[tuple[Float[Array, " z_hidden_dim"], Float[Array, " latent_dim"]], Float[Array, " latent_dim"]]:
            h_prev, z_t = carry
            x_t, key_t = inputs_t
            _, _, h_enc = self.encoder(x_t, dropout_keys=self.encoder.make_keys(key_t))
            h_comb = h_enc * self.c + h_prev * (1.0 - self.c)
            h_next = self.z_gru_cell(z_t, h_comb)
            dz_t = self.z_proj_out(h_next)
            z_next = z_t + dz_t
            return (h_next, z_next), z_next

        (_, _), z_pred = jax.lax.scan(z_step, (zh0, z0), (xs, scan_keys))

        @jaxtyped(typechecker=typechecker)
        def inf_step(
            h_prev: Float[Array, " inf_hidden_dim"], x_t: Float[Array, " input_dim"]
        ) -> tuple[Float[Array, " inf_hidden_dim"], Float[Array, "* inf_hidden_dim"]]:
            h_next = self.inf_gru_cell(x_t, h_prev)
            return h_next, h_next

        _, hs_inf = jax.lax.scan(inf_step, self.inf_h0, xs)

        inf_pred = jax.vmap(self.inf_proj_out)(hs_inf)
        return jax.nn.sigmoid(z_pred), inf_pred


def init_gru_weights(gru: GRUDetector, key: jnp.ndarray, scale: float =1e-1) -> GRUDetector:
    rows, cols = gru.z_gru_cell.weight_ih.shape
    A = jax.random.normal(key, (rows, cols), dtype=jnp.float32)

    if rows < cols:
        Q, _ = jnp.linalg.qr(A.T)
        Q = Q.T
    else:
        Q, _ = jnp.linalg.qr(A)
    Q = Q[:rows, :cols].astype(jnp.float32)

    gru = eqx.tree_at(lambda e: e.z_gru_cell.weight_ih, gru, Q)
    gru = eqx.tree_at(lambda e: e.z_proj_out.weight, gru, gru.z_proj_out.weight * scale)
    return gru
