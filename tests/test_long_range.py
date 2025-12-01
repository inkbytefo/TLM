## Developer: inkbytefo
## Modified: 2025-12-01
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from src.models.spectral_layer import SpectralLayer
from src.data.synthetic import get_adding_problem_data

SEQ_LEN = 1024
BATCH_SIZE = 32
HIDDEN_DIM = 128
NUM_LAYERS = 4
STEPS = 3000
LEARNING_RATE = 1e-3

class SinusoidalPositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[jnp.newaxis, :, :]

    def __call__(self, x):
        return x + self.pe[:, :x.shape[1], :]

class SpectralRegressor(nn.Module):
    hidden_dim: int
    num_layers: int
    @nn.compact
    def __call__(self, x, train: bool = True):
        h = nn.Dense(self.hidden_dim)(x)
        h = SinusoidalPositionalEncoding(d_model=self.hidden_dim, max_len=SEQ_LEN)(h)
        for _ in range(self.num_layers):
            h = SpectralLayer(hidden_dim=self.hidden_dim, dropout_rate=0.0)(h, train=train)
        h = jnp.mean(h, axis=1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        out = nn.Dense(1)(h)
        return out

def run_long_range_test():
    print(f"--- Adding Problem Testi (Uzun Menzil: {SEQ_LEN}) ---")
    print("Başarı Kriteri: MSE Loss < 0.005")
    data_gen = get_adding_problem_data(BATCH_SIZE, SEQ_LEN)
    model = SpectralRegressor(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, SEQ_LEN, 2), dtype=jnp.float32)
    params = model.init(key, dummy_input, train=False)['params']
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=LEARNING_RATE,
        warmup_steps=500,
        decay_steps=STEPS,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            preds = state.apply_fn({'params': params}, batch['input'], train=True)
            loss = jnp.mean((preds - batch['label']) ** 2)
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    for step in range(1, STEPS + 1):
        batch = next(data_gen)
        state, loss = train_step(state, batch)
        if step % 100 == 0:
            print(f"Step {step:04d} | MSE Loss: {loss:.6f}")
        if loss < 0.002:
            print(f"\n[BAŞARILI] Model {step}. adımda problemi çözdü (MSE < 0.002).")
            print("YORUM: Spectral-JAX uzun menzilli bağımlılıkları yakalayabiliyor.")
            return
    print(f"\n[SONUÇ] Final Loss: {loss:.6f}")
    if loss > 0.16:
        print("[BAŞARISIZ] Model öğrenemedi (Baseline seviyesinde).")
    else:
        print("[KISMEN BAŞARILI] Model öğreniyor ancak tam yakınsamadı.")

if __name__ == "__main__":
    run_long_range_test()
