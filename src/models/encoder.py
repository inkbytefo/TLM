import jax
import jax.numpy as jnp
from flax import linen as nn

class ByteLatentEncoder(nn.Module):
    """
    Byte-Level Latent Encoder with Patching.
    
    Converts raw bytes (0-255) into compressed latent representations.
    Uses a 1D Convolution with stride to reduce sequence length by 4x.
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x: (Batch, Length) -> uint8 or int32
        # Input validation (optional but good for debugging)
        # assert x.dtype == jnp.uint8 or x.dtype == jnp.int32
        
        # 1. Byte Embedding
        # Vocab size is fixed to 256 (0-255 byte values)
        x = nn.Embed(num_embeddings=256, features=self.hidden_dim)(x)
        # x shape: (Batch, Length, HiddenDim)
        
        # 2. Causal Padding for Convolution
        # Kernel Size = 6, Stride = 4
        # We want to maintain causality, so we pad ONLY on the left (past).
        # To ensure the convolution window covers the start properly and aligns with the stride:
        # We add 2 padding tokens to the left.
        # This allows the first kernel window to see [Pad, Pad, Byte0, Byte1, Byte2, Byte3]
        # And produce the first patch.
        
        # Padding configuration: ((Batch_Left, Batch_Right), (Len_Left, Len_Right), (Dim_Left, Dim_Right))
        # We only pad the Length dimension (axis 1)
        padding_config = ((0, 0), (2, 0), (0, 0))
        x_padded = jnp.pad(x, padding_config, mode='constant', constant_values=0)
        
        # 3. Patching via Strided Convolution
        # Kernel=6, Stride=4 reduces length by factor of 4.
        # padding='VALID' because we handled padding manually.
        x_patched = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(6,),
            strides=(4,),
            padding='VALID'
        )(x_padded)
        
        # 4. Normalization
        x_out = nn.LayerNorm()(x_patched)
        
        return x_out
