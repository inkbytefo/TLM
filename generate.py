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

@functools.partial(jax.jit, static_argnames=['apply_fn', 'temperature'])
def generate_step(params, apply_fn, curr_seq, t, temperature=1.0):
    """
    Single step generation (JIT compiled).
    curr_seq: (Batch, MaxLen) - Padded
    t: int - Index of the next token to generate (we need logits at t-1)
    """
    # Forward pass on the full sequence (masked by causality)
    logits = apply_fn({'params': params}, curr_seq, train=False)
    
    # logits: (Batch, MaxLen, Vocab)
    # We want the prediction for position t, which comes from logits at t-1
    next_token_logits = logits[:, t-1, :] / temperature
    
    # Greedy sampling
    next_token = jnp.argmax(next_token_logits, axis=-1) # (Batch,)
    
    # Update sequence
    # curr_seq is immutable, return new one
    next_seq = curr_seq.at[:, t].set(next_token)
    
    return next_seq

def generate(state, prompt, max_new_tokens=100, temperature=1.0, max_len=2048):
    """
    Autoregressive generation with fixed shape to avoid recompilation.
    """
    # prompt: (Batch, PromptLen)
    B, P = prompt.shape
    
    # Pad to max_len
    # We assume 0 is a safe pad token (it's a byte value, but causal masking handles it)
    curr_seq = jnp.zeros((B, max_len), dtype=jnp.int32)
    curr_seq = curr_seq.at[:, :P].set(prompt)
    
    print(f"Starting generation ({max_new_tokens} tokens)...")
    
    for i in range(max_new_tokens):
        t = P + i
        if t >= max_len:
            break
            
        curr_seq = generate_step(state.params, state.apply_fn, curr_seq, t, temperature)
        
        # Optional: Print progress
        if i % 10 == 0:
            print(f"Step {i}/{max_new_tokens}", end='\r')
            
    print(f"Generation complete.          ")
    
    # Return only the valid part
    final_len = P + max_new_tokens
    return curr_seq[:, :final_len]

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)
    
    # Checkpoint directory
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_listops", "best")
    if not os.path.exists(ckpt_dir):
        logger.warning(f"Best checkpoint not found at {ckpt_dir}, trying last checkpoint.")
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_listops")

    logger.info(f"Loading checkpoint from: {ckpt_dir}")
    
    # Initialize Model
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)
    
    # Restore Checkpoint
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    logger.info(f"Checkpoint restored from step: {state.step}")
    
    # Create a prompt
    # ListOps usually uses digits 0-9 and operators. 
    # Since we use Byte-Level (0-255), we can just encode a string.
    # Let's try a simple start of a ListOps expression.
    prompt_text = "(MAX"
    prompt_bytes = [ord(c) for c in prompt_text]
    prompt_tensor = jnp.array([prompt_bytes], dtype=jnp.int32) # (1, Len)
    
    logger.info(f"Prompt: {prompt_text}")
    
    # Generate
    generated_ids = generate(state, prompt_tensor, max_new_tokens=200)
    
    # Decode
    generated_list = generated_ids[0].tolist()
    generated_text = "".join([chr(c) for c in generated_list])
    
    logger.info(f"Generated Output:\n{generated_text}")

if __name__ == "__main__":
    main()
