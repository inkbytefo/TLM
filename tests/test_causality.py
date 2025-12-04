import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gpt import SpectralGPT
from config import ModelConfig

def test_causality():
    print("Running Causality/Leakage Test...")
    
    # Use a small config for testing
    class TestConfig:
        vocab_size = 264
        hidden_dim = 64
        num_layers = 2
        num_heads = 2
        dropout_rate = 0.0
        encoder_dense_units = 64
        use_memory = True
        memory_dim = 16
        memory_interval = 1
        decay_min = 0.9
        decay_max = 0.99
        use_attention = True
        attention_interval = 1
        attention_window = 16
    
    config = TestConfig()
    
    rng = jax.random.PRNGKey(0)
    model = SpectralGPT(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        use_memory=True
    )
    
    # Girdi oluştur: (1, 10)
    seq_len = 10
    x = jax.random.randint(rng, (1, seq_len), 0, 256)
    
    # Parametreleri başlat
    variables = model.init(rng, x, train=False)
    params = variables['params']
    
    # 1. Tam geçiş
    logits_1, _ = model.apply({'params': params}, x, train=False)
    
    # 2. Son tokeni değiştir
    x_modified = x.at[:, -1].set(0) # Sadece son token değişti
    logits_2, _ = model.apply({'params': params}, x_modified, train=False)
    
    # KONTROL: t=0'dan t=8'e kadar olan çıktılar AYNI kalmalı.
    # Sadece t=9 (son token) değişmeli.
    # Eğer t=8 değişirse, model geleceği (t=9) görüyordur -> HATA.
    
    # Logits shape: (Batch, Seq, Vocab)
    # We check diff up to the last token (exclusive)
    diff = jnp.abs(logits_1[:, :-1, :] - logits_2[:, :-1, :])
    max_diff = jnp.max(diff)
    
    print(f"Max difference in past tokens: {max_diff}")
    
    if max_diff > 1e-5:
        print(f"FAILED: Causality leakage detected! Max diff: {max_diff}")
        print("Model gelecekteki token'dan etkileniyor.")
        sys.exit(1)
    else:
        print("PASSED: No leakage detected. Model is strictly causal.")
        sys.exit(0)

if __name__ == "__main__":
    test_causality()
