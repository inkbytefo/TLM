import jax
import jax.numpy as jnp
import optax
import os
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state, train_step_generative
from src.data.autonomous_data import get_autonomous_dataloader

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def main():
    config = Config()
    logger = setup_logger()
    set_seed(42)
    
    # Config for Final Mix Training
    config.data.seq_len = 256 # Increased context
    config.data.batch_size = 32
    config.training.learning_rate = 1e-4 
    config.training.num_steps = 3000 # Enough steps to learn diverse tasks
    
    # Load from Stage 3 (Skills) - The most capable model before autonomy
    load_dir = os.path.join(os.getcwd(), "checkpoints", "curriculum", "Stage_3_Skills", "best")
    
    # Save to Final Autonomy
    save_dir = os.path.join(os.getcwd(), "checkpoints", "curriculum", "Stage_4_Autonomy")
    best_dir = os.path.join(save_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    
    logger.info("Starting Final Brain Training (Multi-Task)...")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)
    
    if os.path.exists(load_dir):
        state = checkpoints.restore_checkpoint(load_dir, target=state)
        logger.info(f"Restored Brain from: {load_dir}")
    else:
        logger.error("Stage 3 checkpoint not found! Please run train_curriculum.py first.")
        # Fallback to whatever is available or init from scratch (not recommended)
        return

    train_loader = get_autonomous_dataloader(config.data.batch_size, config.data.seq_len)
    
    best_loss = float('inf')

    for step in range(1, config.training.num_steps + 1):
        batch_np = next(train_loader)
        
        # Reshape to (Batch, 1, SeqLen) for scan over batch dimension
        input_data = jnp.array(batch_np['input'])
        input_data = input_data[:, None, :] 
        batch = {'input': input_data}
        
        state, loss, acc, rng = train_step_generative(state, batch, rng)
        
        if step % 50 == 0:
            logger.info(f"Step {step} | Loss: {loss:.4f} | Acc: {acc:.4f}")
            
        if step % 200 == 0:
            checkpoints.save_checkpoint(save_dir, state, step, keep=1, overwrite=True)
            if loss < best_loss:
                best_loss = loss
                checkpoints.save_checkpoint(best_dir, state, step, keep=1, overwrite=True)
                logger.info(f"Best model saved (Loss: {best_loss:.4f})")

    logger.info("Final Brain Training Complete.")

if __name__ == "__main__":
    main()
