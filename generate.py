import jax
import jax.numpy as jnp
import numpy as np
import os
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def generate(state, prompt, max_new_tokens=100, temperature=1.0):
    """
    Autoregressive generation.
    """
    # prompt: (Batch, Current_Len)
    curr_seq = prompt
    
    for _ in range(max_new_tokens):
        # Forward pass
        # We only care about the last token's logits for the next prediction
        logits = state.apply_fn({'params': state.params}, curr_seq, train=False)
        
        # logits: (Batch, SeqLen, Vocab)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Greedy sampling (argmax)
        next_token = jnp.argmax(next_token_logits, axis=-1)
        
        # Reshape for concatenation: (Batch, 1)
        next_token = next_token[:, None]
        
        # Append to sequence
        curr_seq = jnp.concatenate([curr_seq, next_token], axis=1)
        
        # Stop if we hit a specific token? (Optional, skipping for now)
        
    return curr_seq

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
