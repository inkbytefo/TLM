import numpy as np
import jax.numpy as jnp

def get_dummy_dataloader(batch_size, seq_len, vocab_size):
    """Generates dummy data for testing."""
    while True:
        inputs = np.random.randint(0, vocab_size, (batch_size, seq_len))
        labels = np.random.randint(0, vocab_size, (batch_size, seq_len)) # Simple sequence-to-sequence
        yield {'input': inputs, 'label': labels}
