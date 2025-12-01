## Developer: inkbytefo
## Modified: 2025-12-01
import jax
import jax.numpy as jnp
import optax

def eval_step(state, batch):
    """Evaluate for a single step."""
    logits = state.apply_fn({'params': state.params}, batch['input'], train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    }
    return metrics
