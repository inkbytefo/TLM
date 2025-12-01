import jax.numpy as jnp

def fft_transform(x: jnp.ndarray) -> jnp.ndarray:
    """
    Applies Real Fast Fourier Transform to the input.
    Input: (..., seq_len, hidden_dim)
    Output: (..., seq_len // 2 + 1, hidden_dim) - Complex64/128
    """
    # Real-to-Complex FFT is 2x faster and uses 50% memory for real inputs
    return jnp.fft.rfft(x, axis=-2)

def ifft_transform(x_hat: jnp.ndarray, n: int = None) -> jnp.ndarray:
    """
    Applies Inverse Real Fast Fourier Transform.
    Input: (..., seq_len // 2 + 1, hidden_dim)
    Output: (..., seq_len, hidden_dim) - Real
    
    Args:
        n: Output sequence length (required for odd/even ambiguity in irfft)
    """
    # Complex-to-Real iFFT
    # If n is not provided, it assumes even length (2 * (input_len - 1))
    return jnp.fft.irfft(x_hat, n=n, axis=-2)
