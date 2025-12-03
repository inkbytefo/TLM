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

# Prevent TensorFlow from grabbing GPU memory (JAX needs it)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.data.text_generation import get_text_dataloaders, decode_text
from src.training.trainer import create_generative_train_state, train_step_generative
from src.training.evaluator import eval_step_generative
from generate import generate

def sample_text(state, prompt_text, char_to_idx, idx_to_char, rng, max_new_tokens=200, temperature=0.8):
    """
    Generate sample text from the model.

    Args:
        state: Training state with model parameters
        prompt_text: String to use as prompt
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        rng: Random key
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature

    Returns:
        generated_text: Full generated text including prompt
    """
    # Encode prompt
    prompt_indices = [char_to_idx.get(ch, 0) for ch in prompt_text]
    prompt = jnp.array([prompt_indices], dtype=jnp.int32)  # (1, prompt_len)

    # Generate
    generated_seq = generate(
        state=state,
        prompt=prompt,
        rng=rng,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_len=len(prompt_indices) + max_new_tokens
    )

    # Decode
    generated_indices = np.array(generated_seq[0])
    generated_text = decode_text(generated_indices, idx_to_char)

    return generated_text

def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    # Get text generation config
    text_config = getattr(config, 'text_gen', None)
    if text_config is None:
        logger.warning("TextGenConfig not found in config. Using defaults.")
        text_file = 'data/shakespeare.txt'
        seq_len = 1024
        batch_size = 32
        num_steps = 5000
    else:
        text_file = text_config.dataset_path
        seq_len = text_config.seq_len
        batch_size = text_config.batch_size
        num_steps = getattr(text_config, 'num_steps', 5000)

    # Override config
    config.data.task_name = 'text_generation'
    config.data.seq_len = seq_len
    config.data.batch_size = batch_size
    config.training.num_steps = num_steps

    # Checkpoint directory
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_text_gen")
    best_dir = os.path.join(ckpt_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    logger.info(f"Starting Generative Training (SpectralGPT) on: {text_file}")
    logger.info(f"Checkpoint Dir: {ckpt_dir}")
    logger.info(f"Sequence Length: {seq_len}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Training Steps: {num_steps}")

    # Load Text Data with new data loader
    train_loader, val_loader, vocab_size, char_to_idx, idx_to_char = get_text_dataloaders(
        filepath=text_file,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.9
    )

    logger.info(f"Vocabulary Size: {vocab_size}")
    logger.info(f"Training Batches: {len(train_loader)}")
    logger.info(f"Validation Batches: {len(val_loader)}")

    # Update config with vocab size
    config.data.vocab_size = vocab_size

    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)

    # Initialize Generative Model
    state = create_generative_train_state(init_rng, config)

    # Restore if exists
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    start_step = int(state.step)
    logger.info(f"Starting from step: {start_step}")

    best_loss = float('inf')

    # Sample prompts for generation
    sample_prompts = [
        "The ",
        "To be or not to be",
        "ROMEO:"
    ]

    # Training loop
    global_step = start_step
    for epoch in range(1000):  # Large number of epochs
        logger.info(f"\n=== Epoch {epoch + 1} ===")

        # Training
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            global_step += 1

            if global_step > config.training.num_steps:
                break

            # Prepare batch for train_step_generative
            # The function expects batch['input'] to contain the input tokens
            batch = {'input': inputs}

            state, loss, acc, rng = train_step_generative(
                state,
                batch,
                rng
            )

            if global_step % 50 == 0:
                logger.info(
                    f"Step: {global_step}/{config.training.num_steps} | "
                    f"Train Loss: {float(loss):.4f} | Train Acc: {float(acc):.4f}"
                )

            # Validation and text generation
            if global_step % 200 == 0:
                # --- VALIDATION LOOP ---
                val_losses = []
                val_accs = []

                for val_inputs, val_targets in val_loader:
                    val_batch = {'input': val_inputs}
                    metrics = eval_step_generative(state, val_batch)
                    val_losses.append(metrics['loss'])
                    val_accs.append(metrics['accuracy'])

                mean_val_loss = float(jnp.mean(jnp.array(val_losses)))
                mean_val_acc = float(jnp.mean(jnp.array(val_accs)))

                logger.info(
                    f"Validation - Step: {global_step} | "
                    f"Val Loss: {mean_val_loss:.4f} | Val Acc: {mean_val_acc:.4f}"
                )

                # Save Latest Checkpoint
                checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=global_step, keep=2, overwrite=False)

                if mean_val_loss < best_loss:
                    best_loss = mean_val_loss
                    checkpoints.save_checkpoint(ckpt_dir=best_dir, target=state, step=global_step, keep=1, overwrite=True)
                    logger.info(f"New best model saved! Loss: {best_loss:.4f}")

                # --- TEXT GENERATION SAMPLING ---
                logger.info("\n=== Generated Samples ===")
                for prompt in sample_prompts:
                    rng, gen_rng = jax.random.split(rng)
                    try:
                        generated = sample_text(
                            state=state,
                            prompt_text=prompt,
                            char_to_idx=char_to_idx,
                            idx_to_char=idx_to_char,
                            rng=gen_rng,
                            max_new_tokens=100,
                            temperature=0.8
                        )
                        logger.info(f"\nPrompt: '{prompt}'")
                        logger.info(f"Generated: {generated[:200]}...")  # Show first 200 chars
                    except Exception as e:
                        logger.error(f"Generation failed for prompt '{prompt}': {e}")

                logger.info("=" * 50 + "\n")

            if global_step >= config.training.num_steps:
                break

        if global_step >= config.training.num_steps:
            break

    logger.info(f"\nTraining complete. Best Validation Loss: {best_loss:.4f}")

    # Final text generation
    logger.info("\n=== Final Generated Samples (Temperature=0.7) ===")
    for prompt in sample_prompts:
        rng, gen_rng = jax.random.split(rng)
        try:
            generated = sample_text(
                state=state,
                prompt_text=prompt,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                rng=gen_rng,
                max_new_tokens=200,
                temperature=0.7
            )
            logger.info(f"\nPrompt: '{prompt}'")
            logger.info(f"Generated:\n{generated}\n")
        except Exception as e:
            logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    main()
