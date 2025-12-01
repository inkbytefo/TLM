import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.causal_utils import causal_fft_conv

class HyenaBlock(nn.Module):
    """
    Hyena Block: A causal spectral block for autoregressive modeling.
    Replaces Attention with Causal FFT Convolution.
    
    y = h * (x * v)
    """
    hidden_dim: int
    filter_order: int = 64 # Length of the learnable filter (can be shorter than seq_len)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, Length, Hidden)
        seq_len = x.shape[1]
        
        # 1. Projections (Q, K, V equivalent)
        # In Hyena/H3, we typically have 3 projections.
        # Let's call them x_proj (input), v_proj (value), and gate_proj.
        # For simplicity, we'll follow a standard structure:
        # u = x_proj(x)
        # v = v_proj(x)
        # y = h * (u * v)  <-- This is a simplification of the full Hyena hierarchy
        
        # Let's implement a standard "Gated Convolution" block
        u = nn.Dense(self.hidden_dim)(x)
        v = nn.Dense(self.hidden_dim)(x)
        
        # 2. Learnable Filter 'h'
        # We need a filter of length 'seq_len'. 
        # Ideally, this is generated from a positional embedding or a small MLP.
        # For this MVP, we will learn a direct parameter and interpolate/pad if needed.
        # Or better: Learn a filter of size 'seq_len' directly (simple but fixed length).
        
        # To support variable length, we usually use a small MLP on positional embeddings.
        # Let's use a simple learnable parameter for now, assuming fixed max_len or padding.
        # h_param: (Length, Hidden)
        h_param = self.param('h_filter', nn.initializers.normal(stddev=0.02), (seq_len, self.hidden_dim))
        
        # Apply exponential decay window to enforce locality/stability (optional but good)
        t = jnp.arange(seq_len)
        decay = jnp.exp(-0.01 * t)[:, None]
        h = h_param * decay
        
        # 3. Causal Convolution
        # y = h * v (Convolution)
        v_conv = causal_fft_conv(v, h)
        
        # 4. Gating
        # y = u * v_conv (Element-wise gating)
        y = u * v_conv
        
        # 5. Output Projection
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        
        return y
