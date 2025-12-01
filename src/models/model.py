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
    Full Spectral-JAX Model with Masked Pooling.
    """
    vocab_size: int
    hidden_dim: int
    num_layers: int
    dropout_rate: float = 0.1
    num_classes: int = None 

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x: (batch, seq_len) indices
        
        # 1. Maskeyi oluştur (0 olmayanlar 1, padding 0)
        mask = (x != 0).astype(jnp.float32)
        mask = jnp.expand_dims(mask, axis=-1) # (B, L, 1)
        
        x_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(x)
        x_emb = SinusoidalPositionalEncoding(d_model=self.hidden_dim)(x_emb)
        x_emb = nn.Dropout(rate=self.dropout_rate)(x_emb, deterministic=not train)
        
        # Layer döngüsü
        curr_x = x_emb
        for _ in range(self.num_layers):
            curr_x = SpectralLayer(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)(curr_x, train=train)
            
        curr_x = nn.LayerNorm()(curr_x)
        
        if self.num_classes is not None:
            # --- KRİTİK DÜZELTME: MASKED MEAN POOLING ---
            # Sadece gerçek tokenların ortalamasını al
            # x * mask -> Paddingler 0 olur
            # sum(x) / sum(mask) -> Sadece dolu token sayısına böl
            
            sum_embeddings = jnp.sum(curr_x * mask, axis=1)
            sum_mask = jnp.sum(mask, axis=1) + 1e-9 # Sıfıra bölünmeyi önle
            
            pooled_x = sum_embeddings / sum_mask
            logits = nn.Dense(self.num_classes)(pooled_x)
        else:
            logits = nn.Dense(self.vocab_size)(curr_x)
        
        return logits
