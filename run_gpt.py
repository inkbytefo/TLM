import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging
import time
import os
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.data.lra import get_lra_dataloader
from src.training.trainer import create_generative_train_state, train_step_generative
from src.training.evaluator import eval_step_generative

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    # Checkpoint directory
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_listops")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Starting Generative Training (SpectralGPT) on: {config.data.task_name}")
    logger.info(f"Checkpoint Dir: {ckpt_dir}")

    accum_steps = getattr(config.training, 'accum_steps', 1)
    
    logger.info(f"Gradient Accumulation Steps: {accum_steps}")
    logger.info(f"Micro Batch Size: {config.data.batch_size}")
    logger.info(f"Effective Batch Size: {config.data.batch_size * accum_steps}")

    # Load Data (We use ListOps data but treat it as text for autoregression)
    train_loader = get_lra_dataloader(
        task_name=config.data.task_name,
        batch_size=config.data.batch_size,
        seq_len=config.data.seq_len,
        split='train',
        repeat=True,
    )
    val_loader = get_lra_dataloader(
        task_name=config.data.task_name,
        batch_size=config.data.batch_size,
        seq_len=config.data.seq_len,
        split='validation',
        repeat=True,
    )

    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize Generative Model
    state = create_generative_train_state(init_rng, config)
    
    # Restore if exists
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    start_step = int(state.step)
    logger.info(f"Starting from step: {start_step}")

    best_acc = 0.0
    
    for step in range(start_step + 1, config.training.num_steps + 1):
        # --- BATCH PREPARATION (Accumulation) ---
        inputs = []
        
        for _ in range(accum_steps):
            try:
                batch_np = next(train_loader)
            except StopIteration:
                train_loader = get_lra_dataloader(
                    task_name=config.data.task_name,
                    batch_size=config.data.batch_size,
                    seq_len=config.data.seq_len,
                    split='train',
                    repeat=True,
                )
                batch_np = next(train_loader)
                
            inputs.append(batch_np['input'])
            
        # Stack: (Accum, MicroBatch, SeqLen)
        batch = {
            'input': jnp.array(np.stack(inputs)),
            # We don't need 'label' for generative training
        }
        # -------------------------------------------

        state, loss, acc, rng = train_step_generative(
            state, 
            batch, 
            rng
        )

        if step % config.training.eval_every == 0:
            # --- VALIDATION LOOP ---
            val_metrics = []
            eval_steps = 50 
            
            for _ in range(eval_steps):
                try:
                    val_np = next(val_loader)
                except StopIteration:
                    pass
                    
                val_batch = {
                    'input': jnp.array(val_np['input']),
                    # No label needed
                }
                metrics = eval_step_generative(state, val_batch)
                val_metrics.append(metrics)
            
            mean_loss = jnp.mean(jnp.array([m['loss'] for m in val_metrics]))
            mean_acc = jnp.mean(jnp.array([m['accuracy'] for m in val_metrics]))
            
            val_acc = float(mean_acc)
            
            logger.info(
                f"Step: {step}/{config.training.num_steps} | "
                f"Train Loss: {float(loss):.4f} | Train Acc: {float(acc):.4f} | "
                f"Val Loss: {float(mean_loss):.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=step, keep=1, overwrite=True)
                logger.info(f"New Best Accuracy: {best_acc:.4f} - Checkpoint Saved")

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
