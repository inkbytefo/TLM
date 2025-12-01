import jax
import jax.numpy as jnp

def causal_fft_conv(u, k):
    """
    Performs causal convolution using FFT.
    
    Args:
        u: Input signal (Batch, Length, Dim)
        k: Filter kernel (Length, Dim) - assumed to be causal (starts at t=0)
        
    Returns:
        y: Convolved signal (Batch, Length, Dim)
    """
    seq_len = u.shape[1]
    
    # 1. Pad to at least 2*L to avoid circular convolution wrapping
    fft_len = 2 * seq_len
    
    # 2. FFT
    u_fft = jnp.fft.rfft(u, n=fft_len, axis=1) # (B, L+1, D)
    k_fft = jnp.fft.rfft(k, n=fft_len, axis=0) # (L+1, D)
    
    # 3. Element-wise Multiplication (Convolution in Time)
    # Broadcast k_fft across batch dimension
    y_fft = u_fft * k_fft[None, :, :]
    
    # 4. Inverse FFT
    y = jnp.fft.irfft(y_fft, n=fft_len, axis=1)
    
    # 5. Crop to original length (Causal masking)
    # The result of standard convolution is length L+L-1.
    # We want the first L elements (0 to L-1).
    y = y[:, :seq_len, :]
    
    return y
