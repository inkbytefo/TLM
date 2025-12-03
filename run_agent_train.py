"""
Agent training script.
Train the model on synthetic agent dataset to learn tool use.
"""

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

    # Get agent config
    agent_config = config.agent
    dataset_path = agent_config.dataset_path
    temperature = agent_config.temperature
    seq_len = agent_config.seq_len
    batch_size = agent_config.batch_size
    num_steps = agent_config.num_steps
    accum_steps = agent_config.accum_steps
    learning_rate = agent_config.learning_rate

    # Override config
    config.data.task_name = 'agent_training'
    config.data.seq_len = seq_len
    config.data.batch_size = batch_size
    config.training.num_steps = num_steps
    config.training.learning_rate = learning_rate  # Use agent-specific learning rate

    # Checkpoint directory
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "agent_model")
    best_dir = os.path.join(ckpt_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    logger.info(f"Starting Agent Training on: {dataset_path}")
    logger.info(f"Checkpoint Dir: {ckpt_dir}")
    logger.info(f"Sequence Length: {seq_len}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Training Steps: {num_steps}")
    logger.info(f"Gradient Accumulation Steps: {accum_steps}")
    logger.info(f"Effective Batch Size: {batch_size * accum_steps}")

    # Load Agent Data with text_generation loader
    train_loader, val_loader, vocab_size, char_to_idx, idx_to_char = get_text_dataloaders(
        filepath=dataset_path,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.9
    )

    logger.info(f"Vocabulary Size: {vocab_size}")
    logger.info(f"Training Batches: {len(train_loader)}")
    logger.info(f"Validation Batches: {len(val_loader)}")

    # Check if <EXEC> and </EXEC> are in vocabulary
    if agent_config.tool_start_token[0] in char_to_idx:
        logger.info(f"✓ Tool tokens are in vocabulary")
    else:
        logger.warning(f"⚠ Tool tokens might not be fully in vocabulary")

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

    # Sample prompts for testing agent behavior
    sample_prompts = [
        "SORU: 345 * 982 nedir?\nDÜŞÜNCE: Bu bir çarpma işlemi. Python kullanarak hesaplayacağım.\nEYLEM: ",
        "SORU: 25 faktöriyel kaçtır?\nDÜŞÜNCE: Faktöriyel hesaplamak için math modülünü kullanacağım.\nEYLEM: ",
        "SORU: [10, 20, 30] listesinin toplamı nedir?\nDÜŞÜNCE: ",
    ]

    # Training loop with gradient accumulation
    global_step = start_step
    train_iter = iter(train_loader)

    for epoch in range(1000):  # Large number of epochs
        logger.info(f"\n=== Epoch {epoch + 1} ===")

        # Training
        while global_step < config.training.num_steps:
            global_step += 1

            if global_step > config.training.num_steps:
                break

            # Collect multiple batches for gradient accumulation
            accumulated_inputs = []
            for _ in range(accum_steps):
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    # Reset iterator when epoch ends
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)

                accumulated_inputs.append(inputs)

            # Stack to create 3D tensor: (Accum, Batch, SeqLen)
            batch = {'input': jnp.stack(accumulated_inputs, axis=0)}

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

                # --- AGENT BEHAVIOR SAMPLING ---
                logger.info("\n=== Agent Behavior Samples ===")
                for prompt in sample_prompts:
                    rng, gen_rng = jax.random.split(rng)
                    try:
                        generated = sample_text(
                            state=state,
                            prompt_text=prompt,
                            char_to_idx=char_to_idx,
                            idx_to_char=idx_to_char,
                            rng=gen_rng,
                            max_new_tokens=150,
                            temperature=temperature
                        )
                        logger.info(f"\nPrompt: '{prompt[:60]}...'")
                        logger.info(f"Generated:\n{generated[len(prompt):200]}...")

                        # Check if model is generating tool tags
                        if agent_config.tool_start_token in generated and agent_config.tool_end_token in generated:
                            logger.info("✓ Model is using tool tags correctly!")
                        else:
                            logger.info("⚠ Model not yet using tool tags")

                    except Exception as e:
                        logger.error(f"Generation failed for prompt: {e}")

                logger.info("=" * 50 + "\n")

            if global_step >= config.training.num_steps:
                break

        if global_step >= config.training.num_steps:
            break

    logger.info(f"\nTraining complete. Best Validation Loss: {best_loss:.4f}")

    # Final agent behavior test
    logger.info("\n" + "="*60)
    logger.info("FINAL AGENT BEHAVIOR TEST")
    logger.info("="*60)

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
                temperature=0.6  # Lower temperature for final test
            )
            logger.info(f"\n{'='*60}")
            logger.info(f"Prompt:\n{prompt}")
            logger.info(f"\nGenerated:\n{generated}")
            logger.info(f"{'='*60}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")


if __name__ == "__main__":
    main()
