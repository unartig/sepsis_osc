import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PyTree


class TransformerBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    mha: eqx.nn.MultiheadAttention
    
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout_rate: float, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(dim, use_weight=True)
        self.ln2 = eqx.nn.LayerNorm(dim, use_weight=True)
        self.mha = eqx.nn.MultiheadAttention(dim, num_heads, key=key1, dropout_p=0)
        
        self.linear1 = eqx.nn.Linear(dim, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, dim, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: Float[Array, "batch_size t latent_dim"], key: jnp.ndarray, mask=None):
        x = jnp.where(mask[..., None], x, 0.0)

        x_norm = jax.vmap(self.ln1)(x)

        x = self.mha(x_norm, x_norm, x_norm, mask[:, None] * mask[None, :])      # [T, B, D]

        x_norm= jax.vmap(self.ln2)(x)
        input_x= jax.vmap(self.linear1)(x_norm)
        input_x= jax.nn.gelu(input_x)

        key1, key2 = jr.split(key, num=2)

        input_x = self.dropout1(input_x, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, key=key2)

        x = x + input_x


        return x

class TransformerForecaster(eqx.Module):
    blocks: list[TransformerBlock]
    proj_out: eqx.nn.Linear

    # Hyperparams
    dim: int
    depth: int
    num_heads: int
    hidden_dim: int
    dropout_rate: float
    
    def __init__(self, key, dim: int, depth: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.3):
        keys = jax.random.split(key, depth + 1)

        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.blocks = [TransformerBlock(dim, num_heads, hidden_dim, dropout_rate, keys[i]) for i in range(depth)]
        self.proj_out = eqx.nn.Linear(dim, dim, key=keys[-1])

    def __call__(self, z_t, key: jnp.ndarray, mask: None):
        x = z_t
        for block in self.blocks:
            x = block(x, key, mask)
        return self.proj_out(x[-1])  # [B, z_dim]


def make_forecaster(key, dim: int, depth: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.3):
    return TransformerForecaster(key, dim, depth, num_heads, hidden_dim, dropout_rate)
