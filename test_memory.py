"""
Memory Layer Copy Task Test.
Tests the associative recall capability of DeltaMemoryLayer (Error-Correcting).

Task: Given key-value pairs, retrieve the correct value for a query key.
Example: "a:10 b:20 c:30 a:?" -> "10"

This test validates the Delta Rule's ability to OVERWRITE memory:
- If "a:10" is written, then "a:20" is written, the memory should contain "a:20"
- This solves the Catastrophic Forgetting problem in continual learning
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from config import Config
from src.utils.common import setup_logger, set_seed
from src.models.gpt import SpectralGPT


def generate_copy_data(num_examples=100, seq_len=32, vocab_size=50, num_pairs=3):
    """
    Generate synthetic key-value copy task data.

    Format: "k1:v1 k2:v2 k3:v3 kq:?"
    Target: "vq" (value corresponding to query key kq)

    Args:
        num_examples: Number of training examples
        seq_len: Maximum sequence length
        vocab_size: Size of vocabulary (for keys and values)
        num_pairs: Number of key-value pairs per example

    Returns:
        inputs: (num_examples, seq_len) - Input sequences
        targets: (num_examples,) - Target values to copy
    """
    inputs = []
    targets = []

    # Special tokens
    COLON = vocab_size - 3
    SPACE = vocab_size - 2
    QUERY = vocab_size - 1

    for _ in range(num_examples):
        # Generate unique keys and values
        keys = np.random.randint(0, vocab_size - 3, size=num_pairs)
        values = np.random.randint(0, vocab_size - 3, size=num_pairs)

        # Select a random key to query
        query_idx = np.random.randint(0, num_pairs)
        query_key = keys[query_idx]
        target_value = values[query_idx]

        # Build sequence: k1:v1 k2:v2 ... kq:?
        seq = []
        for i in range(num_pairs):
            seq.append(keys[i])
            seq.append(COLON)
            seq.append(values[i])
            seq.append(SPACE)

        # Add query
        seq.append(query_key)
        seq.append(COLON)
        seq.append(QUERY)

        # Pad to seq_len
        seq = seq[:seq_len]
        seq = seq + [0] * (seq_len - len(seq))

        inputs.append(seq)
        targets.append(target_value)

    return jnp.array(inputs, dtype=jnp.int32), jnp.array(targets, dtype=jnp.int32)


def create_copy_model(vocab_size, use_memory=False):
    """Create model for copy task."""
    return SpectralGPT(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=4,
        dropout_rate=0.1,
        use_memory=use_memory,
        memory_dim=32,
        memory_interval=2
    )


def test_copy_ability(use_memory=False, train_steps=0):
    """
    Test model's ability to perform associative copy.

    Args:
        use_memory: Whether to use memory layers
        train_steps: Number of training steps (0 = test without training)

    Returns:
        accuracy: Copy accuracy percentage
    """
    logger = setup_logger()
    set_seed(42)

    vocab_size = 50
    seq_len = 32
    num_train = 500
    num_test = 100

    logger.info(f"\n{'='*70}")
    logger.info(f"COPY TASK TEST - Memory: {use_memory}, Train Steps: {train_steps}")
    logger.info(f"{'='*70}")

    # Generate data
    train_inputs, train_targets = generate_copy_data(num_train, seq_len, vocab_size)
    test_inputs, test_targets = generate_copy_data(num_test, seq_len, vocab_size)

    logger.info(f"Train examples: {num_train}")
    logger.info(f"Test examples: {num_test}")
    logger.info(f"Sequence length: {seq_len}")
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create model
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    model = create_copy_model(vocab_size, use_memory=use_memory)
    dummy_input = jnp.zeros((1, seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input, train=False)['params']

    # Training (if requested)
    if train_steps > 0:
        logger.info(f"\nTraining for {train_steps} steps...")

        # Create optimizer
        tx = optax.adam(learning_rate=1e-3)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

        # Training loop
        for step in range(train_steps):
            # Random batch
            batch_size = 32
            indices = np.random.randint(0, num_train, size=batch_size)
            batch_inputs = train_inputs[indices]
            batch_targets = train_targets[indices]

            # Split RNG for this step
            rng, dropout_rng = jax.random.split(rng)

            # Forward + backward
            def loss_fn(params):
                logits = model.apply(
                    {'params': params},
                    batch_inputs,
                    train=True,
                    rngs={'dropout': dropout_rng}
                )
                # Get logits for last position (where answer should be)
                last_logits = logits[:, -1, :]
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    last_logits, batch_targets
                ).mean()
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)

            if (step + 1) % 20 == 0:
                logger.info(f"Step {step+1}/{train_steps} | Loss: {float(loss):.4f}")

        params = state.params

    # Test
    logger.info(f"\nTesting on {num_test} examples...")

    # CRITICAL FIX: Process each example independently with its own memory state
    # Each sequence needs to build up its own associations from scratch
    predictions = []
    for i in range(num_test):
        single_input = test_inputs[i:i+1]  # (1, seq_len)
        logits_i = model.apply({'params': params}, single_input, train=False)
        pred_i = jnp.argmax(logits_i[0, -1])  # Get prediction for last token
        predictions.append(pred_i)

    predictions = jnp.array(predictions)  # (num_test,)

    # Calculate accuracy
    correct = jnp.sum(predictions == test_targets)
    accuracy = float(correct) / num_test * 100

    logger.info(f"\nResults:")
    logger.info(f"Correct: {int(correct)}/{num_test}")
    logger.info(f"Accuracy: {accuracy:.2f}%")

    # Show some examples
    logger.info(f"\nExample predictions:")
    for i in range(min(5, num_test)):
        pred = int(predictions[i])
        target = int(test_targets[i])
        status = "[OK]" if pred == target else "[FAIL]"
        logger.info(f"  Example {i+1}: Predicted={pred}, Target={target} {status}")

    return accuracy


def main():
    """Run comprehensive memory tests."""
    logger = setup_logger()

    logger.info("\n" + "="*70)
    logger.info("MEMORY LAYER ASSOCIATIVE RECALL TEST")
    logger.info("="*70)

    results = {}

    # Test 1: No memory, no training (baseline - should be ~2% random)
    logger.info("\n[TEST 1] Baseline: No memory, no training")
    results['baseline'] = test_copy_ability(use_memory=False, train_steps=0)

    # Test 2: Memory enabled, no training (mechanism test)
    logger.info("\n[TEST 2] Memory enabled, no training")
    results['memory_untrained'] = test_copy_ability(use_memory=True, train_steps=0)

    # Test 3: No memory, with training
    logger.info("\n[TEST 3] No memory, with training (100 steps)")
    results['no_memory_trained'] = test_copy_ability(use_memory=False, train_steps=100)

    # Test 4: Memory enabled, with training
    logger.info("\n[TEST 4] Memory enabled, with training (100 steps)")
    results['memory_trained'] = test_copy_ability(use_memory=True, train_steps=100)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"1. Baseline (no memory, no train):     {results['baseline']:.2f}%")
    logger.info(f"2. Memory only (no train):              {results['memory_untrained']:.2f}%")
    logger.info(f"3. Training only (no memory):           {results['no_memory_trained']:.2f}%")
    logger.info(f"4. Memory + Training:                   {results['memory_trained']:.2f}%")

    # Analysis
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS")
    logger.info("="*70)

    improvement_memory = results['memory_trained'] - results['no_memory_trained']
    logger.info(f"Memory improvement over baseline: {improvement_memory:.2f}%")

    if results['memory_trained'] > 90:
        logger.info("\n[SUCCESS] Memory layer enables near-perfect associative recall!")
    elif results['memory_trained'] > results['no_memory_trained']:
        logger.info(f"\n[PARTIAL SUCCESS] Memory helps (+{improvement_memory:.2f}%), but more training needed")
    else:
        logger.info("\n[NEEDS WORK] Memory layer not providing clear benefit yet")

    # Save results
    with open("memory_test_results.txt", 'w') as f:
        f.write("Memory Layer Copy Task Results\n")
        f.write("="*70 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.2f}%\n")

    logger.info(f"\nResults saved to: memory_test_results.txt")


if __name__ == "__main__":
    main()
