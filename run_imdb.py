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
from src.data.imdb import get_imdb_dataloader
from src.training.trainer import create_train_state, train_step
from src.training.evaluator import eval_step

# Prevent TensorFlow from grabbing GPU memory (JAX needs it)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    # Checkpoint directory
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "imdb")
    best_dir = os.path.join(ckpt_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    logger.info(f"Starting IMDB Sentiment Analysis")
    logger.info(f"Checkpoint Dir: {ckpt_dir}")

    accum_steps = getattr(config.training, 'accum_steps', 1)
    
    logger.info(f"Gradient Accumulation Steps: {accum_steps}")
    logger.info(f"Micro Batch Size: {config.data.batch_size}")
    
    # Use IMDB specific config if available, else fallback to default data config
    # We will assume config.data has been updated or we override here
    seq_len = getattr(config.data, 'imdb_seq_len', 1024)
    batch_size = config.data.batch_size
    
    logger.info(f"Sequence Length: {seq_len}")

    # Load Data
    train_loader = get_imdb_dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        split='train',
        repeat=True,
    )
    val_loader = get_imdb_dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        split='test', # IMDB uses 'test' as validation/test
        repeat=True,
    )

    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    # Force num_classes = 2 for IMDB (Pos/Neg)
    # We need to temporarily modify config or pass it explicitly if create_train_state allows
    # create_train_state uses config.model.num_classes. Let's patch it.
    config.model.num_classes = 2
    
    state = create_train_state(init_rng, config)
    
    # Restore if exists
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    start_step = int(state.step)
    logger.info(f"Starting training from step {start_step}")

    best_acc = 0.0
    
    for step in range(start_step, config.training.num_steps):
        # --- TRAINING STEP ---
        try:
            batch = next(train_loader)
        except StopIteration:
            # Should not happen with repeat=True
            logger.warning("Train loader exhausted, restarting...")
            train_loader = get_imdb_dataloader(batch_size=batch_size, seq_len=seq_len, split='train', repeat=True)
            batch = next(train_loader)

        # Gradient Accumulation Logic
        # We need to split the batch into micro-batches if we were doing real accumulation,
        # but here we assume the loader gives us the micro-batch size.
        # If we want accumulation, we need to fetch `accum_steps` batches.
        
        # Simplified Accumulation:
        # Fetch `accum_steps` batches and stack them? 
        # Or just rely on the trainer to handle a larger batch if provided?
        # The current trainer implementation expects a single batch and splits it internally if needed
        # OR it expects the loop to call it multiple times.
        # Let's look at trainer.py... it uses jax.lax.scan over the batch dimension if we pass a large batch.
        # So we should collect `accum_steps` batches and stack them.
        
        inputs = []
        labels = []
        inputs.append(batch['input'])
        labels.append(batch['label'])
        
        for _ in range(accum_steps - 1):
            b = next(train_loader)
            inputs.append(b['input'])
            labels.append(b['label'])
            
        large_batch = {
            'input': jnp.stack(inputs, axis=0),
            'label': jnp.stack(labels, axis=0)
        }

        state, loss, acc, rng = train_step(state, large_batch, rng)

        if step % 10 == 0:
            logger.info(f"Step {step} | Loss: {loss:.4f} | Acc: {acc:.4f}")

        if step % config.training.eval_every == 0:
            # --- VALIDATION LOOP ---
            val_metrics = []
            eval_steps = 50 
            
            for _ in range(eval_steps):
                try:
                    val_batch_np = next(val_loader)
                except StopIteration:
                    pass
                    
                val_batch = {
                    'input': jnp.array(val_batch_np['input']),
                    'label': jnp.array(val_batch_np['label'])
                }
                metrics = eval_step(state, val_batch)
                val_metrics.append(metrics)
            
            mean_loss = jnp.mean(jnp.array([m['loss'] for m in val_metrics]))
            mean_acc = jnp.mean(jnp.array([m['accuracy'] for m in val_metrics]))
            
            val_acc = float(mean_acc)
            
            logger.info(
                f"Step: {step}/{config.training.num_steps} | "
                f"Train Loss: {float(loss):.4f} | Train Acc: {float(acc):.4f} | "
                f"Val Loss: {float(mean_loss):.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Save Latest
            checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=step, keep=1, overwrite=True)
            
            # Save Best
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoints.save_checkpoint(ckpt_dir=best_dir, target=state, step=step, keep=1, overwrite=True)
                logger.info(f"New Best Accuracy: {best_acc:.4f} - Saved")

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
