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
    max_len: int = 2048 # Maximum sequence length for filter
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, Length, Hidden)
        seq_len = x.shape[1]
        
        # 1. Projections
        u = nn.Dense(self.hidden_dim)(x)
        v = nn.Dense(self.hidden_dim)(x)
        
        # 2. Learnable Filter 'h'
        # We initialize with max_len to handle variable lengths
        h_param = self.param('h_filter', nn.initializers.normal(stddev=0.02), (self.max_len, self.hidden_dim))
        
        # Slice to current sequence length
        h = h_param[:seq_len, :]
        
        # Apply exponential decay window to enforce locality/stability
        t = jnp.arange(seq_len)
        decay = jnp.exp(-0.01 * t)[:, None]
        h = h * decay
        
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
