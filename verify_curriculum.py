
"""
Verification Script for Curriculum Learning.
Runs 1 step per stage to ensure pipeline integrity.
"""

import os
import jax
import jax.numpy as jnp
import logging
from config import Config
from src.utils.common import setup_logger, set_seed
from flax.training import checkpoints

# Import Data Loaders
from src.data.lra import get_lra_dataloader
from src.data.text_generation import get_text_dataloaders
from src.data.autonomous_data import get_autonomous_dataloader

# Import Trainer
from src.training.trainer import create_generative_train_state, train_step_generative

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def run_stage_test(stage_name, config, train_loader):
    """Runs a single step of the stage."""
    print(f"Testing Stage: {stage_name}")
    
    # Initialize Model
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)
    
    # Get Batch
    try:
        if stage_name == "Stage_4_Autonomy":
            batch_np = next(train_loader)
            # Reshape: (B, L) -> (B, 1, L) for gradient accumulation scan
            batch = {'input': jnp.array(batch_np['input'])[:, None, :]}
        elif stage_name == "Stage_1_Logic":
            batch_np = next(train_loader)
            batch = {'input': jnp.array(batch_np['input'])[:, None, :]}
        else:
            inputs, targets = next(train_loader)
            batch = {'input': inputs[:, None, :]}
    except Exception as e:
        print(f"FAILED to get batch for {stage_name}: {e}")
        return False

    # Train Step
    try:
        state, loss, acc, rng = train_step_generative(state, batch, rng)
        print(f"Stage {stage_name} Step 1 | Loss: {loss:.4f} | Acc: {acc:.4f}")
        return True
    except Exception as e:
        print(f"FAILED training step for {stage_name}: {e}")
        return False

def main():
    config = Config()
    # Reduce size for test
    config.model.num_layers = 2
    config.model.hidden_dim = 64
    config.model.vocab_size = 260
    config.data.vocab_size = 260
    config.data.batch_size = 2
    config.data.seq_len = 32
    
    set_seed(42)
    
    print("=== STARTING CURRICULUM VERIFICATION ===")
    
    # --- STAGE 1: LOGIC ---
    try:
        # Mock LRA loader if real one fails or takes too long
        # For verification, we can just use random data if LRA download is heavy
        # But let's try to use the real one if it's there
        if os.path.exists('data/lra_listops/basic_train.tsv'):
            train_loader_1 = get_lra_dataloader('train', 32, 2, task_name='lra_listops', repeat=True)
            if not run_stage_test("Stage_1_Logic", config, train_loader_1):
                print("Stage 1 Failed")
        else:
            print("Skipping Stage 1 (Data not found)")
    except Exception as e:
        print(f"Stage 1 Error: {e}")
    
    # --- STAGE 2: LANGUAGE ---
    try:
        train_loader_2, _, _, _, _ = get_text_dataloaders('data/sonnet.txt', 32, 2)
        train_iter_2 = iter(train_loader_2)
        if not run_stage_test("Stage_2_Language", config, train_iter_2):
            print("Stage 2 Failed")
    except Exception as e:
        print(f"Stage 2 Error: {e}")
        
    # --- STAGE 3: SKILLS ---
    try:
        # Use sonnet as dummy if agent data missing
        agent_file = 'data/agent_dataset.txt'
        if not os.path.exists(agent_file):
            agent_file = 'data/sonnet.txt'
            
        train_loader_3, _, _, _, _ = get_text_dataloaders(agent_file, 32, 2)
        train_iter_3 = iter(train_loader_3)
        if not run_stage_test("Stage_3_Skills", config, train_iter_3):
            print("Stage 3 Failed")
    except Exception as e:
        print(f"Stage 3 Error: {e}")
        
    # --- STAGE 4: AUTONOMY ---
    try:
        train_loader_4 = get_autonomous_dataloader(2, 32)
        if not run_stage_test("Stage_4_Autonomy", config, train_loader_4):
            print("Stage 4 Failed")
    except Exception as e:
        print(f"Stage 4 Error: {e}")

    print("=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()
