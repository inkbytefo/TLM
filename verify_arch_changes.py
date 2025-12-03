
import jax
import jax.numpy as jnp
from flax import linen as nn
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.models.encoder import ByteLatentEncoder
from src.models.spectral_layer import SpectralLayer
from config import Config

def verify_encoder():
    print("Verifying ByteLatentEncoder...")
    config = Config()
    model = ByteLatentEncoder(
        hidden_dim=config.model.hidden_dim,
        encoder_dense_units=config.model.encoder_dense_units
    )
    
    # Dummy input: Batch=2, Length=1024 (must be divisible by stride 4 after padding logic)
    # The encoder adds 2 padding tokens. 
    # If input is 1024, padded is 1026. 
    # Conv stride 4. 
    # (1026 - 6) / 4 + 1 = 1020 / 4 + 1 = 255 + 1 = 256.
    
    x = jnp.zeros((2, 1024), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    
    variables = model.init(rng, x)
    out = model.apply(variables, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    expected_len = 256
    assert out.shape == (2, expected_len, config.model.hidden_dim)
    print("ByteLatentEncoder verification passed!")

def verify_spectral_layer():
    print("\nVerifying SpectralLayer...")
    config = Config()
    model = SpectralLayer(
        hidden_dim=config.model.hidden_dim,
        dropout_rate=config.model.dropout_rate
    )
    
    x = jnp.zeros((2, 64, config.model.hidden_dim), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    rng_dropout = jax.random.PRNGKey(1)
    
    variables = model.init(rng, x, train=False)
    
    # Check for new parameters (Nested in SpectralBlock_0)
    params = variables['params']
    print("Param keys:", params.keys())
    
    # SpectralLayer contains SpectralBlock. 
    # Depending on Flax version/naming, it might be 'SpectralBlock_0'
    block_params = params.get('SpectralBlock_0')
    if block_params is None:
         # Fallback check if naming is different
         for k in params.keys():
             if 'SpectralBlock' in k:
                 block_params = params[k]
                 break
    
    assert block_params is not None, "SpectralBlock params not found"
    assert 'w_real' in block_params
    assert 'w_imag' in block_params
    print("SpectralBlock parameters (w_real, w_imag) found!")

    # Test forward pass (train=False)
    out = model.apply(variables, x, train=False)
    assert out.shape == x.shape
    print("SpectralLayer forward (eval) passed!")
    
    # Test forward pass (train=True)
    out_train = model.apply(variables, x, train=True, rngs={'dropout': rng_dropout})
    assert out_train.shape == x.shape
    print("SpectralLayer forward (train) passed!")

if __name__ == "__main__":
    try:
        verify_encoder()
        verify_spectral_layer()
        print("\nAll verifications passed successfully.")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
