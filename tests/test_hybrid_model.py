import jax
import jax.numpy as jnp
from src.models.gpt import SpectralGPT

def test_hybrid_architecture():
    print("Initializing SpectralGPT with Hybrid Architecture...")
    model = SpectralGPT(
        vocab_size=100,
        hidden_dim=128,
        num_layers=12, # Should have 2 Attention layers (at 6 and 12)
        dropout_rate=0.0,
        use_memory=False
    )
    
    rng = jax.random.PRNGKey(0)
    seq_len = 1024
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    
    print("Initializing parameters...")
    variables = model.init(rng, dummy_input, train=False)
    
    print("Running forward pass...")
    logits, _ = model.apply(variables, dummy_input, train=False)
    
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (1, seq_len, 100)
    print("Test Passed: Output shape is correct.")

if __name__ == "__main__":
    test_hybrid_architecture()
