## Developer: inkbytefo
## Modified: 2025-12-01
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable

from src.utils.fft_utils import fft_transform, ifft_transform

class SpectralBlock(nn.Module):
    """
    Spectral-JAX Block: FFT -> MLP -> iFFT -> Gate
    """
    hidden_dim: int
    mlp_expansion: int = 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        seq_len = x.shape[1]  # Store original length for irfft
        residual = x
        x_norm = nn.LayerNorm()(x)
        
        # 1. Real-to-Complex FFT
        x_hat = fft_transform(x_norm) # Shape: (B, N//2 + 1, D)
        
        # 2. Spectral Filter Generation
        # Global context from time domain
        context = jnp.mean(x_norm, axis=1, keepdims=True)
        
        filter_signal = nn.Dense(self.hidden_dim * self.mlp_expansion, dtype=self.dtype)(context)
        filter_signal = nn.gelu(filter_signal)
        filter_signal = nn.Dense(self.hidden_dim, dtype=self.dtype)(filter_signal)
        filter_signal = jnp.tanh(filter_signal)
        
        # Broadcast to time domain then FFT -> This learns a frequency filter implicitly
        h_time = jnp.tile(filter_signal, (1, seq_len, 1))
        h_hat = fft_transform(h_time) # Shape: (B, N//2 + 1, D)
        
        # 3. Spectral Mixing
        y_hat = x_hat * h_hat
        
        # 4. Complex-to-Real iFFT
        y = ifft_transform(y_hat, n=seq_len)
        
        # 5. Energy Gate
        gate = nn.Dense(self.hidden_dim, dtype=self.dtype)(x_norm)
        gate = nn.sigmoid(gate)
        y = y * gate
        
        return residual + y
