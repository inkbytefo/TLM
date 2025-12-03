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
        
        # 2. Spectral Filter (Learnable in Frequency Domain)
        # We learn complex weights directly: H(f)
        # Shape: (N//2 + 1, D, D) to allow mixing between hidden dimensions
        freq_len = x_hat.shape[1]
        
        w_real = self.param('w_real', nn.initializers.normal(stddev=0.02), 
                           (freq_len, self.hidden_dim, self.hidden_dim))
        w_imag = self.param('w_imag', nn.initializers.normal(stddev=0.02), 
                           (freq_len, self.hidden_dim, self.hidden_dim))
        
        # 3. Spectral Mixing: Y(f) = X(f) * H(f)
        # Complex multiplication with matrix mixing:
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        # x_hat: (B, S, I)
        # w: (S, I, O)
        
        # Real part: x.real * w.real - x.imag * w.imag
        real_part = jnp.einsum('bsi,sio->bso', x_hat.real, w_real) - \
                    jnp.einsum('bsi,sio->bso', x_hat.imag, w_imag)
                    
        # Imag part: x.real * w.imag + x.imag * w.real
        imag_part = jnp.einsum('bsi,sio->bso', x_hat.real, w_imag) + \
                    jnp.einsum('bsi,sio->bso', x_hat.imag, w_real)
                    
        y_hat = real_part + 1j * imag_part
        
        # 4. Complex-to-Real iFFT
        y = ifft_transform(y_hat, n=seq_len)
        
        # 5. Energy Gate (Optional but kept for stability)
        gate = nn.Dense(self.hidden_dim, dtype=self.dtype)(x_norm)
        gate = nn.sigmoid(gate)
        y = y * gate
        
        return residual + y
