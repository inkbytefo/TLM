import pytest
import jax
import jax.numpy as jnp
from src.models.spectral_block import SpectralBlock

def test_spectral_block_shape():
    batch_size = 2
    seq_len = 16
    hidden_dim = 32
    
    model = SpectralBlock(hidden_dim=hidden_dim)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))
    
    params = model.init(key, x)
    y = model.apply(params, x)
    
    assert y.shape == x.shape
    assert not jnp.isnan(y).any()
