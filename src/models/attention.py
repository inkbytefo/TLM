import jax
import jax.numpy as jnp
from typing import Any
from flax import linen as nn

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention Layer.
    Standard Multi-Head Attention with a causal sliding window mask.
    """
    hidden_dim: int
    num_heads: int = 8
    window_size: int = 512
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, Seq, Hidden)
        b, seq_len, hidden = x.shape
        
        # Create sliding window mask
        # 1 means attend, 0 means mask out (Flax attention expects boolean mask or bias)
        # We'll create a bias: 0 for attend, -inf for mask
        
        # Indices
        i = jnp.arange(seq_len)[:, None]
        j = jnp.arange(seq_len)[None, :]
        
        # Causal mask: i >= j
        causal_mask = i >= j
        
        # Window mask: i - j < window_size
        window_mask = (i - j) < self.window_size
        
        # Combined mask
        mask = causal_mask & window_mask
        
        # Convert to bias: 0 where True, -1e9 where False
        # Shape: (1, 1, Seq, Seq) for broadcasting over Batch and Heads
        mask_bias = jnp.where(mask, 0.0, -1e9).astype(jnp.float32)
        mask_bias = mask_bias[None, None, :, :]
        
        # Layer Norm before Attention (Pre-Norm)
        y = nn.LayerNorm(dtype=self.dtype)(x)
        
        # Multi-Head Attention
        # Flax's MultiHeadAttention expects inputs_q, inputs_kv
        # For self-attention, both are y
        y = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            dtype=self.dtype
        )(inputs_q=y, inputs_kv=y, mask=mask_bias, deterministic=not train)
        
        # Dropout
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        
        return y
