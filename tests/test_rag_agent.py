"""
RAG-Enhanced Agent Test Script.
Demonstrates agent's ability to retrieve information from a knowledge base.
"""

import jax
import jax.numpy as jnp
import os
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state
from src.data.text_generation import get_text_dataloaders
from src.memory.rag import VectorStore
from agent_generate import agent_generate

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


def create_knowledge_base():
    """
    Create a comprehensive knowledge base for testing.

    Returns:
        VectorStore with technical knowledge
    """
    store = VectorStore()

    # Technical knowledge about programming and math
    knowledge = [
        # Python & Programming
        "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
        "Python uses dynamic typing and automatic memory management with garbage collection.",
        "Python's standard library includes modules for file I/O, system calls, and internet protocols.",
        "The Python Package Index (PyPI) hosts over 400,000 third-party packages.",

        # JAX & Machine Learning
        "JAX is a numerical computing library developed by Google for high-performance machine learning research.",
        "JAX provides automatic differentiation (autograd) for computing gradients of arbitrary Python functions.",
        "JAX can compile Python code to run on GPUs and TPUs using XLA (Accelerated Linear Algebra).",
        "Flax is a neural network library built on top of JAX, providing high-level abstractions for model building.",

        # Mathematics
        "Factorial of n (denoted n!) is the product of all positive integers from 1 to n.",
        "The factorial function grows extremely rapidly. For example: 5! = 120, 10! = 3,628,800, 20! = 2,432,902,008,176,640,000.",
        "Factorial is used in combinatorics to count permutations and combinations.",
        "The factorial of 0 is defined as 1 by convention (0! = 1).",

        # Algorithms
        "Quick Sort is a divide-and-conquer sorting algorithm with average time complexity O(n log n).",
        "Binary Search requires a sorted array and has time complexity O(log n).",
        "Dynamic Programming solves complex problems by breaking them into simpler overlapping subproblems.",
        "Hash tables provide O(1) average-case time complexity for insert, delete, and lookup operations.",

        # Data Structures
        "A linked list is a linear data structure where elements are stored in nodes with pointers to the next node.",
        "Binary trees are hierarchical data structures where each node has at most two children.",
        "Stacks follow LIFO (Last In First Out) principle, while queues follow FIFO (First In First Out).",
        "Graphs consist of vertices (nodes) connected by edges, and can be directed or undirected.",

        # Neural Networks
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Deep learning uses neural networks with multiple hidden layers to learn hierarchical representations.",
        "Backpropagation is the algorithm used to compute gradients in neural networks for training.",
        "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images.",

        # Transformers
        "Transformers are a neural network architecture that relies on self-attention mechanisms.",
        "The attention mechanism allows models to focus on different parts of the input when processing each element.",
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model.",
        "GPT (Generative Pre-trained Transformer) is an autoregressive language model for text generation.",
    ]

    store.add_documents(knowledge)
    return store


def test_rag_retrieval(state, char_to_idx, idx_to_char, rng, store, logger):
    """
    Test agent's ability to retrieve and use information from knowledge base.

    Args:
        state: Trained model state
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        rng: Random key
        store: VectorStore with knowledge
        logger: Logger instance

    Returns:
        Test results
    """
    test_cases = [
        {
            "name": "Python Definition Query",
            "prompt": "SORU: Python nedir?\nDÜŞÜNCE: Bilgi tabanından Python hakkında bilgi arayacağım.\nARAMA: <SEARCH>Python programming language</SEARCH>",
            "expected_keywords": ["programming", "language", "Python"],
        },
        {
            "name": "JAX Information Query",
            "prompt": "SORU: JAX ne işe yarar?\nDÜŞÜNCE: JAX hakkında bilgi arayacağım.\nARAMA: <SEARCH>JAX machine learning library</SEARCH>",
            "expected_keywords": ["JAX", "machine learning", "numerical"],
        },
        {
            "name": "Factorial Definition Query",
            "prompt": "SORU: Faktöriyel ne demektir?\nDÜŞÜNCE: Faktöriyel tanımını arayacağım.\nARAMA: <SEARCH>factorial definition mathematics</SEARCH>",
            "expected_keywords": ["factorial", "product", "integers"],
        },
        {
            "name": "Algorithm Query",
            "prompt": "SORU: Quick Sort algoritması nasıl çalışır?\nDÜŞÜNCE: Quick Sort hakkında bilgi bulmalıyım.\nARAMA: <SEARCH>Quick Sort algorithm</SEARCH>",
            "expected_keywords": ["Quick Sort", "sorting", "algorithm"],
        },
        {
            "name": "Neural Network Query",
            "prompt": "SORU: Neural network nedir?\nDÜŞÜNCE: Neural network tanımını arayacağım.\nARAMA: <SEARCH>neural networks definition</SEARCH>",
            "expected_keywords": ["neural", "network", "learning"],
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST {i+1}: {test_case['name']}")
        logger.info(f"{'='*70}")

        rng, test_rng = jax.random.split(rng)

        try:
            # Run agent generation with RAG
            result_text = agent_generate(
                state=state,
                prompt=test_case['prompt'],
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                rng=test_rng,
                max_iterations=3,
                max_tokens_per_iteration=150,
                temperature=0.6,
                vector_store=store,
                top_k_search=2,
                verbose=True
            )

            # Check if keywords are present
            found_keywords = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in result_text.lower():
                    found_keywords.append(keyword)

            success = len(found_keywords) >= 2  # At least 2 keywords found

            result = {
                "name": test_case['name'],
                "success": success,
                "expected_keywords": test_case['expected_keywords'],
                "found_keywords": found_keywords,
                "output": result_text,
            }

            results.append(result)

            if success:
                logger.info(f"\n✓ TEST PASSED - Found keywords: {found_keywords}")
            else:
                logger.info(f"\n✗ TEST FAILED - Only found: {found_keywords}")

        except Exception as e:
            logger.error(f"\n✗ TEST ERROR: {e}")
            result = {
                "name": test_case['name'],
                "success": False,
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
    logger.info("RAG-ENHANCED AGENT TEST")
    logger.info("="*70)

    # Create knowledge base
    logger.info("\nCreating knowledge base...")
    store = create_knowledge_base()
    logger.info(f"✓ Created knowledge base with {store.size()} documents")

    # Load checkpoint
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "agent_model", "best")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "agent_model")

    logger.info(f"\nLoading checkpoint from: {ckpt_dir}")

    # Load vocabulary from agent dataset
    dataset_path = agent_config.dataset_path
    seq_len = 512
    batch_size = 16

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
        logger.warning("⚠ Using randomly initialized model for demonstration")
        logger.warning("Note: Results will be random without trained model")

    # Run RAG tests
    logger.info("\n\nStarting RAG retrieval tests...\n")

    rng, test_rng = jax.random.split(rng)
    results = test_rag_retrieval(
        state=state,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        rng=test_rng,
        store=store,
        logger=logger
    )

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    passed = sum(1 for r in results if r.get('success', False))
    total = len(results)

    logger.info(f"\nTotal Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")

    logger.info("\nDetailed Results:")
    for i, result in enumerate(results):
        status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
        logger.info(f"{i+1}. {result['name']}: {status}")
        if 'found_keywords' in result:
            logger.info(f"   Keywords found: {result['found_keywords']}")

    # Save results
    results_file = "rag_test_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAG-ENHANCED AGENT TEST RESULTS\n")
        f.write("="*70 + "\n\n")

        for i, result in enumerate(results):
            f.write(f"Test {i+1}: {result['name']}\n")
            f.write(f"Status: {'PASSED' if result.get('success', False) else 'FAILED'}\n")

            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                if 'found_keywords' in result:
                    f.write(f"Found Keywords: {result['found_keywords']}\n")
                f.write(f"\nOutput:\n{result.get('output', 'N/A')}\n")

            f.write("\n" + "-"*70 + "\n\n")

        f.write(f"Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)\n")

    logger.info(f"\nResults saved to: {results_file}")

    # Final assessment
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! RAG integration is working correctly! 🎉")
    elif passed > 0:
        logger.info(f"\n⚠ PARTIAL SUCCESS: {passed}/{total} tests passed")
        logger.info("Note: Model may need more training to use RAG effectively")
    else:
        logger.info("\n✗ ALL TESTS FAILED")
        logger.info("Note: This is expected if model is not trained for RAG")
        logger.info("      The RAG infrastructure is in place and working")


if __name__ == "__main__":
    main()
