import jax
import jax.numpy as jnp
from flax import linen as nn
import unittest
from src.models.gpt import SpectralGPT

class TestInfiniteContext(unittest.TestCase):
    def test_extrapolation(self):
        print("\n[Testing Infinite Context Extrapolation - Full Model]")
        
        # 1. Initialize Model
        # We don't need max_len anymore!
        hidden_dim = 64
        model = SpectralGPT(
            vocab_size=100,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_memory=False # Keep it simple for this test
        )
        
        key = jax.random.PRNGKey(0)
        
        # 2. Init with SHORT sequence (e.g., Training length)
        train_seq_len = 128
        # Input is integer tokens for GPT
        dummy_input = jnp.ones((1, train_seq_len), dtype=jnp.int32)
        variables = model.init(key, dummy_input)
        
        print(f"Model initialized with sequence length: {train_seq_len}")
        
        # 3. Apply with LONG sequence (Extrapolation)
        # e.g., 4x longer than "training"
        test_seq_len = 512 
        print(f"Testing extrapolation to length: {test_seq_len}")
        
        long_input = jnp.ones((1, test_seq_len), dtype=jnp.int32)
        
        # This would crash with the old implementation due to shape mismatch
        output, _ = model.apply(variables, long_input, train=False)
        
        print(f"Output shape: {output.shape}")
        
        # Check shapes
        self.assertEqual(output.shape, (1, test_seq_len, 100))
        
        # Check for NaNs
        self.assertFalse(jnp.isnan(output).any(), "Output contains NaNs")
        
        print("[SUCCESS] Full Model successfully extrapolated to longer sequence!")

if __name__ == '__main__':
    unittest.main()
