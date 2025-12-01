## Developer: inkbytefo
## Modified: 2025-12-01
import numpy as np
import jax.numpy as jnp

def get_adding_problem_data(batch_size: int, seq_len: int):
    while True:
        values = np.random.uniform(0, 1, (batch_size, seq_len)).astype(np.float32)
        masks = np.zeros((batch_size, seq_len), dtype=np.float32)
        for i in range(batch_size):
            idx1 = np.random.randint(0, seq_len // 2)
            idx2 = np.random.randint(seq_len // 2, seq_len)
            masks[i, idx1] = 1.0
            masks[i, idx2] = 1.0
        inputs = np.stack([values, masks], axis=-1)
        labels = np.sum(values * masks, axis=1, keepdims=True).astype(np.float32)
        yield {'input': jnp.array(inputs), 'label': jnp.array(labels)}
