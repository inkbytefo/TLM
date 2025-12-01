## Developer: inkbytefo
## Modified: 2025-12-01
import jax
import jax.numpy as jnp
import os
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.data.lra import get_lra_dataloader
from src.training.trainer import create_train_state, train_step
from src.training.evaluator import eval_step

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    # Checkpoint directory
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", config.data.task_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Starting LRA Task: {config.data.task_name}")
    logger.info(f"Checkpoint Dir: {ckpt_dir}")

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
    state = create_train_state(init_rng, config)
    
    # Restore if exists
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    start_step = int(state.step)
    logger.info(f"Starting from step: {start_step}")

    best_acc = 0.0
    
    for step in range(start_step + 1, config.training.num_steps + 1):
        batch_np = next(train_loader)
        batch = {
            'input': jnp.array(batch_np['input']),
            'label': jnp.array(batch_np['label']),
        }
        state, loss, acc, rng = train_step(state, batch, rng)

        if step % config.training.eval_every == 0:
            val_np = next(val_loader)
            val_batch = {
                'input': jnp.array(val_np['input']),
                'label': jnp.array(val_np['label']),
            }
            metrics = eval_step(state, val_batch)
            val_acc = float(metrics['accuracy'])
            
            logger.info(
                f"Step: {step}/{config.training.num_steps} | "
                f"Train Loss: {float(loss):.4f} | Train Acc: {float(acc):.4f} | "
                f"Val Loss: {float(metrics['loss']):.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=step, keep=1, overwrite=True)
                logger.info(f"New Best Accuracy: {best_acc:.4f} - Checkpoint Saved")

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
