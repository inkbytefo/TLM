import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from src.models.model import SpectralModel

def create_train_state(rng, config):
    """Creates `TrainState` with Scheduler and Gradient Clipping."""
    
    model = SpectralModel(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout_rate=config.model.dropout_rate,
        num_classes=10 # ListOps has 10 classes (0-9)
    )
    
    dummy_input = jnp.ones((1, config.data.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)['params']
    
    # --- SCHEDULER & OPTIMIZER (GÜNCELLENDİ) ---
    # Adding Problem'den öğrenilen stabilizasyon ayarları
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.num_steps
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient Clipping
        optax.adamw(learning_rate=schedule, weight_decay=config.training.weight_decay)
    )
    # -------------------------------------------
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch, rng):
    """Train for a single step."""
    dropout_rng, rng = jax.random.split(rng)
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input'], train=True, rngs={'dropout': dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    # Metrics
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    
    return state, loss, accuracy, rng