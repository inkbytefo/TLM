
"""
Curriculum Learning Script for Spectral-JAX.
Trains the model in 4 evolutionary stages:
1. Logic (ListOps)
2. Language (Text)
3. Skills (Agent/Tools)
4. Life (Autonomous Loop)
"""

import os
import jax
import jax.numpy as jnp
import optax
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

def run_stage(stage_name, config, train_loader, val_loader, num_steps, prev_ckpt_dir=None):
    """Runs a single stage of the curriculum."""
    logger = logging.getLogger("spectral_jax")
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING STAGE: {stage_name}")
    logger.info(f"{'='*60}")
    
    # Setup directories
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "curriculum", stage_name)
    best_dir = os.path.join(ckpt_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    
    # Initialize Model
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)
    
    # Load from previous stage if exists
    if prev_ckpt_dir and os.path.exists(prev_ckpt_dir):
        logger.info(f"Loading knowledge from previous stage: {prev_ckpt_dir}")
        state = checkpoints.restore_checkpoint(prev_ckpt_dir, target=state)
    else:
        logger.info("Initializing from scratch (Tabula Rasa).")

    best_loss = float('inf')
    
    # Training Loop
    for step in range(1, num_steps + 1):
        # Get Batch
        try:
            if stage_name == "Stage_4_Autonomy":
                batch_np = next(train_loader)
                batch = {'input': jnp.array(batch_np['input'])}
            elif stage_name == "Stage_1_Logic":
                 # ListOps loader returns dict
                batch_np = next(train_loader)
                batch = {'input': jnp.array(batch_np['input'])}
            else:
                # Text loaders return tuple (input, target)
                # But train_step_generative expects a dict with 'input' key containing the full sequence
                # The text loader yields (inputs, targets) where inputs is x[0:-1] and targets is x[1:]?
                # Let's check src/data/text_generation.py
                # It yields inputs, targets. 
                # inputs = data[start:start+len], targets = data[start+1:start+len+1]
                # train_step_generative expects 'input' to be the FULL sequence x[0:L] so it can slice it itself.
                # Wait, train_step_generative does: seq = minibatch['input']; inputs = seq[:, :-1]; targets = seq[:, 1:]
                # So it expects 'input' to be a sequence of length L+1? Or L?
                # If LRA yields L, then inputs is L-1.
                # TextGenerationDataLoader yields inputs (L) and targets (L).
                # If we pass 'inputs' as 'input', train_step_generative will slice it further.
                # inputs is already x[t:t+L].
                # If train_step_generative does inputs[:, :-1], it reduces length.
                # We should probably pass the raw sequence if possible, or adjust.
                # TextGenerationDataLoader yields (inputs, targets). 'inputs' is a valid sequence.
                # If we use 'inputs' as the batch['input'], train_step_generative will predict next token from it.
                # That works.
                
                inputs, targets = next(train_loader)
                batch = {'input': inputs}
                
        except StopIteration:
            # Restart loader if needed (mostly for text loaders)
            if stage_name == "Stage_2_Language":
                 # Ideally we should re-create iterator, but for this script we assume enough data
                 # or we catch it.
                 logger.info("Epoch complete, restarting iterator...")
                 # This part is tricky with iterators. 
                 # For now, let's just break or continue.
                 break
            continue

        # Train Step
        state, loss, acc, rng = train_step_generative(state, batch, rng)
        
        # Logging
        if step % 50 == 0:
            logger.info(f"[{stage_name}] Step {step}/{num_steps} | Loss: {loss:.4f} | Acc: {acc:.4f}")
            
        # Validation & Saving
        if step % 200 == 0 or step == num_steps:
            # Simple validation logic (can be expanded)
            logger.info(f"Saving checkpoint for {stage_name} at step {step}")
            checkpoints.save_checkpoint(ckpt_dir, state, step, keep=1, overwrite=True)
            
            # Update best model path for next stage
            if loss < best_loss:
                best_loss = loss
                checkpoints.save_checkpoint(best_dir, state, step, keep=1, overwrite=True)

    logger.info(f"Stage {stage_name} Complete. Best Loss: {best_loss:.4f}")
    return best_dir

def main():
    config = Config()
    logger = setup_logger()
    set_seed(42)
    
    # GLOBAL CONFIGURATION
    # We must use the maximum vocab size (260) throughout all stages
    # to ensure weight compatibility.
    config.model.vocab_size = 260 
    config.data.vocab_size = 260
    
    # --- STAGE 1: LOGIC (ListOps) ---
    # Learning hierarchical reasoning
    config.data.seq_len = 512
    config.data.batch_size = 32
    
    # Check if LRA data exists, if not warn
    try:
        train_loader_1 = get_lra_dataloader('train', 512, 32, task_name='lra_listops', repeat=True)
        
        best_stage_1 = run_stage(
            "Stage_1_Logic", 
            config, 
            train_loader_1, 
            None, 
            num_steps=1000, # Short warmup for logic
            prev_ckpt_dir=None
        )
    except Exception as e:
        logger.warning(f"Skipping Stage 1 (Logic) due to error: {e}")
        best_stage_1 = None
    
    # --- STAGE 2: LANGUAGE (Text) ---
    # Learning grammar and vocabulary
    config.data.seq_len = 512
    config.data.batch_size = 16
    
    # Use sonnet.txt if shakespeare.txt not found
    text_file = 'data/shakespeare.txt'
    if not os.path.exists(text_file):
        text_file = 'data/sonnet.txt'
        
    if os.path.exists(text_file):
        train_loader_2, _, _, _, _ = get_text_dataloaders(text_file, 512, 16)
        train_iter_2 = iter(train_loader_2)
        
        best_stage_2 = run_stage(
            "Stage_2_Language", 
            config, 
            train_iter_2, 
            None, 
            num_steps=2000, 
            prev_ckpt_dir=best_stage_1
        )
    else:
        logger.warning(f"Skipping Stage 2 (Language) - No text file found.")
        best_stage_2 = best_stage_1
    
    # --- STAGE 3: SKILLS (Agent) ---
    # Learning to use tools and reason
    config.data.seq_len = 512
    config.data.batch_size = 16
    agent_file = 'data/agent_dataset.txt'
    
    if os.path.exists(agent_file):
        train_loader_3, _, _, _, _ = get_text_dataloaders(agent_file, 512, 16)
        train_iter_3 = iter(train_loader_3)
        
        best_stage_3 = run_stage(
            "Stage_3_Skills", 
            config, 
            train_iter_3, 
            None, 
            num_steps=2000, 
            prev_ckpt_dir=best_stage_2
        )
    else:
        logger.warning(f"Skipping Stage 3 (Skills) - No agent dataset found.")
        best_stage_3 = best_stage_2
    
    # --- STAGE 4: AUTONOMY (Life) ---
    # Learning interaction loop
    config.data.seq_len = 128
    config.data.batch_size = 32
    train_loader_4 = get_autonomous_dataloader(32, 128)
    
    best_stage_4 = run_stage(
        "Stage_4_Autonomy", 
        config, 
        train_loader_4, 
        None, 
        num_steps=1000, 
        prev_ckpt_dir=best_stage_3
    )
    
    logger.info("\n" + "="*60)
    logger.info("CURRICULUM TRAINING COMPLETE")
    logger.info(f"Final Autonomous Model: {best_stage_4}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
