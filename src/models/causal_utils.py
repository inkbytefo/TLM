import jax
import jax.numpy as jnp

def next_power_of_2(n):
    """Returns the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()

def causal_fft_conv(u, k):
    """
    Performs causal convolution using FFT with optimized padding.

    Args:
        u: Input signal (Batch, Length, Dim)
        k: Filter kernel (Length, Dim) - assumed to be causal (starts at t=0)

    Returns:
        y: Convolved signal (Batch, Length, Dim)
    """
    seq_len = u.shape[1]

    # 1. Pad to next power of 2 for optimal FFT performance
    # Must be at least 2*L to avoid circular convolution wrapping
    min_fft_len = 2 * seq_len
    fft_len = next_power_of_2(min_fft_len)
    
    # Cast to float32 for FFT (JAX requires float32 for FFT)
    u_32 = u.astype(jnp.float32)
    k_32 = k.astype(jnp.float32)
    
    # 2. FFT
    u_fft = jnp.fft.rfft(u_32, n=fft_len, axis=1) # (B, L+1, D)
    k_fft = jnp.fft.rfft(k_32, n=fft_len, axis=0) # (L+1, D)
    
    # 3. Element-wise Multiplication (Convolution in Time)
    # Broadcast k_fft across batch dimension
    y_fft = u_fft * k_fft[None, :, :]
    
    # 4. Inverse FFT
    y = jnp.fft.irfft(y_fft, n=fft_len, axis=1)
    
    # 5. Crop to original length (Causal masking)
    # The result of standard convolution is length L+L-1.
    # We want the first L elements (0 to L-1).
    y = y[:, :seq_len, :]
    
    # Cast back to original dtype
    y = y.astype(u.dtype)
    
    return y
