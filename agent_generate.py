"""
Agent generation with stop-and-go tool execution.
Model generates text, executes code when it produces <EXEC> tags, and continues.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import functools
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state
from src.tools.executor import execute_code, extract_code_from_exec_tags

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def generate_step(params, apply_fn, curr_seq, t, rng, temperature=1.0):
    """
    Single step generation with temperature sampling.

    Args:
        params: Model parameters
        apply_fn: Model forward function
        curr_seq: Current sequence (Batch, MaxLen)
        t: Current time step
        rng: Random key
        temperature: Sampling temperature

    Returns:
        Updated sequence with new token at position t
    """
    # Forward pass
    logits = apply_fn({'params': params}, curr_seq, train=False)

    # Get logits for next token prediction
    next_token_logits = logits[:, t-1, :] / temperature

    # Sample from distribution
    next_token = jax.random.categorical(rng, next_token_logits, axis=-1)

    # Update sequence
    next_seq = curr_seq.at[:, t].set(next_token)

    return next_seq, next_token


def decode_indices(indices, idx_to_char):
    """Decode integer indices to text."""
    return ''.join([idx_to_char.get(int(i), '') for i in indices])


def encode_text(text, char_to_idx):
    """Encode text to integer indices."""
    return [char_to_idx.get(ch, 0) for ch in text]


def agent_generate(
    state,
    prompt,
    char_to_idx,
    idx_to_char,
    rng,
    max_iterations=5,
    max_tokens_per_iteration=200,
    temperature=0.8,
    max_len=2048,
    tool_start_token="<EXEC>",
    tool_end_token="</EXEC>",
    verbose=True
):
    """
    Agent generation with stop-and-go tool execution.

    Algorithm:
        1. Generate text until </EXEC> token or max_tokens
        2. If </EXEC> found, extract and execute code
        3. Append result to prompt and continue generation
        4. Repeat until max_iterations or no more </EXEC> tokens

    Args:
        state: Training state with model
        prompt: Initial prompt text
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        rng: Random key
        max_iterations: Maximum number of tool execution loops
        max_tokens_per_iteration: Max tokens to generate per iteration
        temperature: Sampling temperature
        max_len: Maximum sequence length
        tool_start_token: Start token for code execution
        tool_end_token: End token for code execution
        verbose: Whether to print progress

    Returns:
        Final generated text
    """
    # Encode prompt
    current_text = prompt
    current_indices = encode_text(current_text, char_to_idx)

    if verbose:
        print(f"\n{'='*60}")
        print(f"AGENT GENERATION STARTING")
        print(f"{'='*60}")
        print(f"Initial Prompt: {prompt}")
        print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Prepare current sequence
        B = 1  # Batch size 1
        P = len(current_indices)

        if P >= max_len:
            if verbose:
                print("Maximum sequence length reached.")
            break

        curr_seq = jnp.zeros((B, max_len), dtype=jnp.int32)
        curr_seq = curr_seq.at[:, :P].set(jnp.array([current_indices], dtype=jnp.int32))

        # Generate tokens until </EXEC> or max_tokens
        generated_tokens = []
        found_end_tag = False

        for i in range(max_tokens_per_iteration):
            t = P + i
            if t >= max_len:
                break

            # Generate next token
            rng, step_rng = jax.random.split(rng)
            curr_seq, next_token = generate_step(
                state.params,
                state.apply_fn,
                curr_seq,
                t,
                step_rng,
                temperature
            )

            token_id = int(next_token[0])
            generated_tokens.append(token_id)

            # Decode to check for end tag
            current_generated = decode_indices(generated_tokens, idx_to_char)
            if tool_end_token in current_generated:
                found_end_tag = True
                if verbose:
                    print(f"Found {tool_end_token} at token {i}")
                break

        # Decode generated text
        generated_text = decode_indices(generated_tokens, idx_to_char)
        current_text += generated_text

        if verbose:
            print(f"\nGenerated ({len(generated_tokens)} tokens):")
            print(f"{generated_text}")

        # Check if we found execution tags
        if found_end_tag and tool_start_token in current_text:
            # Extract code
            code = extract_code_from_exec_tags(current_text, tool_start_token, tool_end_token)

            if code:
                if verbose:
                    print(f"\n>>> EXECUTING CODE:")
                    print(f"{code}")
                    print(f">>> RUNNING...")

                # Execute code
                try:
                    result = execute_code(code, timeout=5)
                    if verbose:
                        print(f">>> RESULT:")
                        print(f"{result}")
                except Exception as e:
                    result = f"ERROR: {str(e)}"
                    if verbose:
                        print(f">>> ERROR:")
                        print(f"{result}")

                # Append result to current text
                result_text = f"\nSONUÇ: {result.strip()}\nCEVAP: "
                current_text += result_text
                current_indices = encode_text(current_text, char_to_idx)

                if verbose:
                    print(f"\nContinuing generation after execution...")
            else:
                if verbose:
                    print(f"Warning: Found {tool_end_token} but could not extract code")
                break
        else:
            # No execution tag found, we're done
            if verbose:
                print(f"\nNo {tool_end_token} found. Generation complete.")
            break

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL OUTPUT:")
        print(f"{'='*60}")
        print(current_text)
        print(f"{'='*60}\n")

    return current_text


def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    # Load agent config
    agent_config = getattr(config, 'agent', None)
    if agent_config:
        tool_start_token = agent_config.tool_start_token
        tool_end_token = agent_config.tool_end_token
        max_exec_len = agent_config.max_exec_len
    else:
        tool_start_token = "<EXEC>"
        tool_end_token = "</EXEC>"
        max_exec_len = 256

    # Load checkpoint
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "agent_model", "best")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_text_gen", "best")
        if not os.path.exists(ckpt_dir):
            ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "gpt_text_gen")

    logger.info(f"Loading checkpoint from: {ckpt_dir}")

    # Note: In a real scenario, we would need the char_to_idx and idx_to_char mappings
    # from the training data. For now, we'll create a simple character mapping.

    # Simple ASCII character mapping for demonstration
    chars = [chr(i) for i in range(32, 127)]  # Printable ASCII
    chars.extend(['\n', '\t'])
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    logger.info(f"Vocabulary size: {vocab_size}")

    # Update config
    config.data.vocab_size = vocab_size

    # Initialize model
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)

    # Try to restore checkpoint
    try:
        state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
        logger.info(f"Checkpoint restored from step: {state.step}")
    except Exception as e:
        logger.warning(f"Could not restore checkpoint: {e}")
        logger.warning("Using randomly initialized model for demonstration")

    # Test prompts
    test_prompts = [
        "SORU: 345 * 982 nedir?\nDÜŞÜNCE: Bu bir çarpma işlemi. Python kullanarak hesaplayacağım.\nEYLEM: ",
        "SORU: 25 faktöriyel kaçtır?\nDÜŞÜNCE: Faktöriyel hesaplamak için math modülünü kullanacağım.\nEYLEM: ",
    ]

    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"Test Prompt {i+1}")
        logger.info(f"{'#'*60}")

        rng, gen_rng = jax.random.split(rng)

        result = agent_generate(
            state=state,
            prompt=prompt,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            rng=gen_rng,
            max_iterations=5,
            max_tokens_per_iteration=200,
            temperature=0.7,
            tool_start_token=tool_start_token,
            tool_end_token=tool_end_token,
            verbose=True
        )


if __name__ == "__main__":
    main()
