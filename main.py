## Developer: inkbytefo
## Modified: 2025-12-01
import jax
import jax.numpy as jnp
import time
from config import Config
from src.data.dataloader import get_dummy_dataloader
from src.training.trainer import create_train_state, train_step
from src.training.evaluator import eval_step
from src.utils.common import setup_logger, set_seed

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    logger.info("Initializing Spectral-JAX Model...")
    logger.info(f"Device: {jax.devices()[0]}")

    train_loader = get_dummy_dataloader(
        batch_size=config.data.batch_size,
        seq_len=config.data.seq_len,
        vocab_size=config.model.vocab_size,
    )

    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    logger.info("Starting training...")

    for step in range(1, config.training.num_steps + 1):
        batch = next(train_loader)
        batch = {k: jnp.array(v) for k, v in batch.items()}

        start_time = time.time()
        state, loss, rng = train_step(state, batch, rng)

        if step % config.training.eval_every == 0:
            eval_batch = next(train_loader)
            eval_batch = {k: jnp.array(v) for k, v in eval_batch.items()}
            metrics = eval_step(state, eval_batch)

            elapsed = time.time() - start_time
            logger.info(
                f"Step: {step}/{config.training.num_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"Time: {elapsed:.4f}s"
            )

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
