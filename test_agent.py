"""
End-to-end agent testing script.
Tests the trained agent's ability to solve problems using code execution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state
from src.data.text_generation import get_text_dataloaders
from agent_generate import agent_generate

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


def test_agent_problem_solving(state, char_to_idx, idx_to_char, rng, agent_config, logger):
    """
    Test the agent on various problem-solving tasks.

    Args:
        state: Trained model state
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        rng: Random key
        agent_config: Agent configuration
        logger: Logger instance

    Returns:
        List of test results
    """
    # Test problems
    test_problems = [
        {
            "name": "Simple Multiplication",
            "prompt": "SORU: 12345 * 67890 nedir?\nDÜŞÜNCE: Bu bir çarpma işlemi. Python kullanarak hesaplayacağım.\nEYLEM: ",
            "expected_result": "838102050",
        },
        {
            "name": "Factorial",
            "prompt": "SORU: 10 faktöriyel kaçtır?\nDÜŞÜNCE: Faktöriyel hesaplamak için Python math modülünü kullanacağım.\nEYLEM: ",
            "expected_result": "3628800",
        },
        {
            "name": "List Sum",
            "prompt": "SORU: [10, 20, 30, 40, 50] listesinin toplamı nedir?\nDÜŞÜNCE: Liste işlemleri için Python'un built-in fonksiyonlarını kullanacağım.\nEYLEM: ",
            "expected_result": "150",
        },
        {
            "name": "Power Operation",
            "prompt": "SORU: 2^10 kaçtır?\nDÜŞÜNCE: Üs alma işlemi için Python kullanacağım.\nEYLEM: ",
            "expected_result": "1024",
        },
        {
            "name": "Division",
            "prompt": "SORU: 1000 // 7 kaçtır?\nDÜŞÜNCE: Tam bölme işlemi için Python kullanacağım.\nEYLEM: ",
            "expected_result": "142",
        },
    ]

    results = []

    for i, problem in enumerate(test_problems):
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST {i+1}: {problem['name']}")
        logger.info(f"{'='*70}")

        rng, test_rng = jax.random.split(rng)

        try:
            # Run agent generation
            result_text = agent_generate(
                state=state,
                prompt=problem['prompt'],
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                rng=test_rng,
                max_iterations=agent_config.max_iterations,
                max_tokens_per_iteration=agent_config.max_tokens_per_iteration,
                temperature=agent_config.temperature,
                tool_start_token=agent_config.tool_start_token,
                tool_end_token=agent_config.tool_end_token,
                verbose=True
            )

            # Check if expected result is in output
            success = problem['expected_result'] in result_text

            result = {
                "name": problem['name'],
                "success": success,
                "expected": problem['expected_result'],
                "output": result_text,
            }

            results.append(result)

            if success:
                logger.info(f"\n✓ TEST PASSED - Found expected result: {problem['expected_result']}")
            else:
                logger.info(f"\n✗ TEST FAILED - Expected result not found: {problem['expected_result']}")

        except Exception as e:
            logger.error(f"\n✗ TEST ERROR: {e}")
            result = {
                "name": problem['name'],
                "success": False,
                "expected": problem['expected_result'],
                "error": str(e),
            }
            results.append(result)

    return results


def main():
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    agent_config = config.agent

    logger.info("="*70)
    logger.info("AGENT END-TO-END TEST")
    logger.info("="*70)

    # Load checkpoint
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "agent_model", "best")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "agent_model")

    logger.info(f"Loading checkpoint from: {ckpt_dir}")

    # Load vocabulary from agent dataset
    dataset_path = agent_config.dataset_path
    seq_len = 512  # Match training config
    batch_size = 16

    # Get vocabulary mappings
    from src.data.text_generation import get_text_dataloaders
    _, _, vocab_size, char_to_idx, idx_to_char = get_text_dataloaders(
        filepath=dataset_path,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.9
    )

    logger.info(f"Vocabulary size: {vocab_size}")

    # Update config
    config.data.vocab_size = vocab_size
    config.data.seq_len = seq_len

    # Initialize model
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)

    # Restore checkpoint
    try:
        state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
        logger.info(f"✓ Checkpoint restored from step: {state.step}")
    except Exception as e:
        logger.error(f"✗ Could not restore checkpoint: {e}")
        logger.error("Please train the agent model first using: python run_agent_train.py")
        return

    # Run tests
    logger.info("\nStarting agent problem-solving tests...\n")

    rng, test_rng = jax.random.split(rng)
    results = test_agent_problem_solving(
        state=state,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        rng=test_rng,
        agent_config=agent_config,
        logger=logger
    )

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    passed = sum(1 for r in results if r['success'])
    total = len(results)

    logger.info(f"\nTotal Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")

    logger.info("\nDetailed Results:")
    for i, result in enumerate(results):
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        logger.info(f"{i+1}. {result['name']}: {status}")

    # Save results
    results_file = "agent_test_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("AGENT TEST RESULTS\n")
        f.write("="*70 + "\n\n")

        for i, result in enumerate(results):
            f.write(f"Test {i+1}: {result['name']}\n")
            f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n")
            f.write(f"Expected: {result['expected']}\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Output:\n{result['output']}\n")
            f.write("\n" + "-"*70 + "\n\n")

        f.write(f"Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)\n")

    logger.info(f"\nResults saved to: {results_file}")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! The agent is working correctly! 🎉")
    elif passed > 0:
        logger.info(f"\n⚠ PARTIAL SUCCESS: {passed}/{total} tests passed")
    else:
        logger.info("\n✗ ALL TESTS FAILED: The agent needs more training")


if __name__ == "__main__":
    main()
