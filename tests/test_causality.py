import jax
import jax.numpy as jnp
import numpy as np
from src.models.gpt import SpectralGPT

def test_causality():
    print("Running Causality Test for SpectralGPT...")
    
    # 1. Initialize Model
    rng = jax.random.PRNGKey(0)
    model = SpectralGPT(vocab_size=256, hidden_dim=64, num_layers=2)
    
    seq_len = 10
    dummy_input = jnp.zeros((1, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)['params']
    
    # 2. Create two sequences that differ ONLY at the last position
    # Seq A: [1, 2, 3, ..., 10]
    # Seq B: [1, 2, 3, ..., 99]
    seq_a = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    seq_b = seq_a.at[:, -1].set(99)
    
    # 3. Run Model
    # We use train=False to disable dropout (deterministic)
    out_a = model.apply({'params': params}, seq_a, train=False)
    out_b = model.apply({'params': params}, seq_b, train=False)
    
    # 4. Check Outputs
    # For a causal model, the output at position t should ONLY depend on inputs 0...t.
    # Therefore, changing input at t=9 (last pos) should NOT affect outputs at t=0...8.
    
    diff = jnp.abs(out_a - out_b)
    
    # Check differences at the last position (should be different)
    last_diff = jnp.max(diff[:, -1, :])
    print(f"Difference at last position (expected > 0): {last_diff}")
    
    # Check differences at previous positions (should be EXACTLY 0)
    prev_diff = jnp.max(diff[:, :-1, :])
    print(f"Difference at previous positions (expected 0.0): {prev_diff}")
    
    if prev_diff < 1e-5:
        print("SUCCESS: Model is Causal! Past outputs are not affected by future inputs.")
    else:
        print("FAILURE: Model is NOT Causal. Future inputs leaked into past outputs.")

if __name__ == "__main__":
    test_causality()
