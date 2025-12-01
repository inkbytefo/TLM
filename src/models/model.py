## Developer: inkbytefo
## Modified: 2025-12-01
import jax.numpy as jnp
from flax import linen as nn
from src.models.spectral_layer import SpectralLayer

class SinusoidalPositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 16384
    def setup(self):
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[jnp.newaxis, :, :]
    def __call__(self, x):
        return x + self.pe[:, :x.shape[1], :]

class SpectralModel(nn.Module):
    """
    Full Spectral-JAX Model.
    """
    vocab_size: int
    hidden_dim: int
    num_layers: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x: (batch, seq_len) indices
        
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(x)
        x = SinusoidalPositionalEncoding(d_model=self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        for _ in range(self.num_layers):
            x = SpectralLayer(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)(x, train=train)
            
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        
        return logits
