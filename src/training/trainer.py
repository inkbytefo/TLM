import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from src.models.model import SpectralModel

def create_train_state(rng, config):
    model = SpectralModel(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout_rate=config.model.dropout_rate,
        num_classes=10
    )
    
    # Dummy input shape: (1, SeqLen)
    dummy_input = jnp.ones((1, config.data.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)['params']
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.num_steps
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.training.weight_decay)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch, rng):
    """
    Gradient Accumulation destekli train step.
    batch['input'] shape: (AccumSteps, MicroBatch, SeqLen)
    """
    # Dropout key'ini accum_steps kadar çoğalt
    accum_steps = batch['input'].shape[0]
    dropout_rngs = jax.random.split(rng, accum_steps)
    
    def compute_loss(params, minibatch, dropout_rng):
        logits = state.apply_fn(
            {'params': params}, 
            minibatch['input'], 
            train=True, 
            rngs={'dropout': dropout_rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=minibatch['label']
        ).mean()
        
        # Accuracy hesabı (sadece loglama için)
        acc = jnp.mean(jnp.argmax(logits, -1) == minibatch['label'])
        return loss, (logits, acc)

    def scan_step(carry, x):
        # x: (minibatch, dropout_rng)
        minibatch, dropout_rng = x
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, (logits, acc)), grads = grad_fn(state.params, minibatch, dropout_rng)
        return carry, (loss, acc, grads)

    # Scan over accumulation steps
    # batch'i (Accum, ...) formatında olduğu için doğrudan scan edebiliriz
    # Ancak batch bir dict, onu scan uyumlu hale getirmeliyiz
    scan_inputs = (batch, dropout_rngs) 
    # Not: jax.lax.scan dict inputları otomatik parçalar (tree_map mantığı)
    
    _, (losses, accuracies, grads) = jax.lax.scan(
        scan_step, 
        None, # carry yok
        scan_inputs
    )
    
    # Gradyanların ve metriklerin ortalamasını al
    avg_loss = jnp.mean(losses)
    avg_acc = jnp.mean(accuracies)
    
    # Grads bir PyTree, yaprakların ortalamasını almalıyız
    avg_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    
    state = state.apply_gradients(grads=avg_grads)
    
    # Yeni RNG döndür (sonuncusu değil, split edilmişlerden türetilen yeni bir tane)
    new_rng = jax.random.fold_in(rng, state.step)
    
    return state, avg_loss, avg_acc, new_rng