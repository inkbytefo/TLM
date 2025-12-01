## Developer: inkbytefo
## Modified: 2025-12-01
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state
from src.models.model import SpectralModel

VOCAB_SIZE = 100
HIDDEN_DIM = 32
SEQ_LEN = 64
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
STEPS = 500

def calculate_grad_norm(grads):
    leaves, _ = jax.tree_util.tree_flatten(grads)
    return jnp.sqrt(sum([jnp.sum(leaf ** 2) for leaf in leaves]))

def run_sanity_check():
    print(f"--- BAŞLANGIÇ: Sanity Check & Overfitting Test ---")
    print(f"Hedef: Loss -> 0, Accuracy -> 1.0, Grad Norm < Sonsuz")

    key = jax.random.PRNGKey(0)
    dummy_input = jax.random.randint(key, (1, SEQ_LEN), 0, VOCAB_SIZE)
    dummy_label = jax.random.randint(key, (1, SEQ_LEN), 0, VOCAB_SIZE)

    model = SpectralModel(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=0.0
    )

    params = model.init(key, dummy_input, train=False)['params']
    tx = optax.adam(learning_rate=LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, x, train=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)

        grad_norm = calculate_grad_norm(grads)
        state = state.apply_gradients(grads=grads)

        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
        return state, loss, accuracy, grad_norm

    for i in range(1, STEPS + 1):
        state, loss, acc, g_norm = train_step(state, dummy_input, dummy_label)

        if i % 50 == 0:
            print(f"Step {i:03d} | Loss: {loss:.6f} | Acc: {acc:.2f} | Grad Norm: {g_norm:.4f}")

        if acc == 1.0 and loss < 0.01:
            print(f"\n[BAŞARILI] Model {i}. adımda veriyi ezberledi.")
            break

        if jnp.isnan(g_norm) or g_norm > 1000.0:
            print(f"\n[BAŞARISIZ] Gradyan patlaması tespit edildi! Norm: {g_norm}")
            return False

    if acc < 1.0:
        print("\n[BAŞARISIZ] Model veriyi ezberleyemedi (Underfitting/Architecture Bug).")
    else:
        print(f"[SONUÇ] Hipotez doğrulandı: Model öğrenebiliyor ve gradyanlar stabil.")

if __name__ == "__main__":
    run_sanity_check()
