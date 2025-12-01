## Developer: inkbytefo
## Modified: 2025-12-01
import jax
import jax.numpy as jnp
import time
from src.models.model import SpectralModel

SEQLENS = [1024, 2048, 4096, 8192, 16384]
BATCH_SIZE = 1
HIDDEN_DIM = 64
VOCAB_SIZE = 1000
NUM_LAYERS = 2

def profile_run():
    print(f"--- Hız ve Ölçeklenebilirlik Testi (Teorem 1.1) ---")
    print(f"Cihaz: {jax.devices()[0]}")
    print("-" * 65)
    print(f"{'Seq Len':<10} | {'Time (ms)':<10} | {'Tokens/sec':<12} | {'Katlanma (x)':<12}")
    print("-" * 65)

    prev_time = None

    for seq_len in SEQLENS:
        model = SpectralModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout_rate=0.0)
        key = jax.random.PRNGKey(0)

        x = jax.random.randint(key, (BATCH_SIZE, seq_len), 0, VOCAB_SIZE)

        try:
            params = model.init(key, x, train=False)['params']
        except Exception as e:
            print(f"{seq_len:<10} | OOM/Error: {str(e)[:30]}")
            break

        @jax.jit
        def forward(p, x):
            return model.apply({'params': p}, x, train=False)

        try:
            _ = forward(params, x).block_until_ready()
        except Exception as e:
            print(f"{seq_len:<10} | OOM (Warmup): {str(e)[:20]}")
            break

        start = time.time()
        loops = 10
        for _ in range(loops):
            _ = forward(params, x).block_until_ready()
        end = time.time()

        avg_time = (end - start) / loops
        tokens_per_sec = (BATCH_SIZE * seq_len) / avg_time
        scaling_factor = avg_time / prev_time if prev_time else 1.0

        print(f"{seq_len:<10} | {avg_time*1000:<10.2f} | {tokens_per_sec:<12.0f} | {scaling_factor:<12.2f}")

        prev_time = avg_time

    print("-" * 65)
    print("YORUM: 'Katlanma (x)' değeri 2.0 - 2.5 arasındaysa O(N log N) doğrulanmıştır.")
    print("       Eğer 3.5 - 4.0 üzerindeyse O(N^2) davranışı vardır (HATA).")

if __name__ == "__main__":
    profile_run()
