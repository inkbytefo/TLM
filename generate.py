
import jax
import jax.numpy as jnp
import numpy as np
import os
import functools
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

@functools.partial(jax.jit, static_argnames=['apply_fn'])
def generate_step(params, apply_fn, curr_seq, t, rng, temperature=1.0, top_k=0):
    """
    Single step generation with Sampling (Temperature + Random).
    """
    # Forward pass
    logits = apply_fn({'params': params}, curr_seq, train=False)
    
    # t-1 anındaki logitleri al (bir sonraki token tahmini)
    next_token_logits = logits[:, t-1, :] / temperature
    
    # Sampling (Categorical)
    # Argmax yerine dağılımdan örnek alıyoruz
    next_token = jax.random.categorical(rng, next_token_logits, axis=-1)
    
    # Update sequence
    next_seq = curr_seq.at[:, t].set(next_token)
    
    return next_seq

def generate(state, prompt, rng, max_new_tokens=100, temperature=0.8, max_len=2048):
    """
    Autoregressive generation loop.
    """
    B, P = prompt.shape
    curr_seq = jnp.zeros((B, max_len), dtype=jnp.int32)
    curr_seq = curr_seq.at[:, :P].set(prompt)
    
    print(f"Starting generation ({max_new_tokens} tokens, Temp: {temperature})...")
    
    # Loop
    for i in range(max_new_tokens):
        t = P + i
        if t >= max_len:
            break
        
        # Her adımda yeni bir RNG key üret
        rng, step_rng = jax.random.split(rng)
        
        curr_seq = generate_step(state.params, state.apply_fn, curr_seq, t, step_rng, temperature)
        
        if i % 10 == 0:
            print(f"Step {i}/{max_new_tokens}", end='\r')
            
    print(f"Generation complete.          ")
    return curr_seq[:, :P + max_new_tokens]

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)
    
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_listops", "best")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_listops")

    logger.info(f"Loading checkpoint from: {ckpt_dir}")
    
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)
    
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    logger.info(f"Checkpoint restored from step: {state.step}")
    
    # Prompt: Biraz daha yönlendirici bir prompt kullanalım
    prompt_text = "(MAX 2 "
    prompt_bytes = [ord(c) for c in prompt_text]
    prompt_tensor = jnp.array([prompt_bytes], dtype=jnp.int32)
    
    logger.info(f"Prompt: {prompt_text}")
    
    # Generate
    rng, gen_rng = jax.random.split(rng)
    generated_ids = generate(state, prompt_tensor, gen_rng, max_new_tokens=100, temperature=0.8)
    
    generated_list = generated_ids[0].tolist()
    generated_text = "".join([chr(c) for c in generated_list])
    
    logger.info(f"Generated Output:\n{generated_text}")

if __name__ == "__main__":
    main()