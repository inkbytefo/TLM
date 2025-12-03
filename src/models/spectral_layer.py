import jax.numpy as jnp
from flax import linen as nn
from src.models.spectral_block import SpectralBlock

class SpectralLayer(nn.Module):
    """
    A single layer consisting of a Spectral Block and a FeedForward network,
    with residual connections and normalization.
    """
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Spectral Block Branch
        residual = x
        x = nn.LayerNorm()(x) # Pre-Norm
        x = SpectralBlock(hidden_dim=self.hidden_dim)(x, train=train)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + residual
        
        # FeedForward Branch
        residual = x
        x = nn.LayerNorm()(x) # Pre-Norm
        x = nn.Dense(self.hidden_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + residual
        
        return x
