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
from src.memory.rag import VectorStore

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


def extract_search_query(text, start_token="<SEARCH>", end_token="</SEARCH>"):
    """
    Extract search query from text between <SEARCH> tags.

    Args:
        text: Full text containing search tags
        start_token: Start tag for search
        end_token: End tag for search

    Returns:
        Search query string or None if not found
    """
    try:
        start_idx = text.rfind(start_token)  # Last occurrence
        if start_idx == -1:
            return None

        start_idx += len(start_token)
        end_idx = text.find(end_token, start_idx)

        if end_idx == -1:
            return None

        query = text[start_idx:end_idx].strip()
        return query if query else None
    except Exception:
        return None


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
    search_start_token="<SEARCH>",
    search_end_token="</SEARCH>",
    vector_store=None,
    top_k_search=3,
    verbose=True
):
    """
    Agent generation with stop-and-go tool execution and RAG search.

    Algorithm:
        1. Generate text until </EXEC>, </SEARCH>, or max_tokens
        2. If </EXEC> found, extract and execute code
        3. If </SEARCH> found, retrieve from vector store
        4. Append result to prompt and continue generation
        5. Repeat until max_iterations or no more tool tokens

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
        search_start_token: Start token for RAG search
        search_end_token: End token for RAG search
        vector_store: Optional VectorStore for RAG retrieval
        top_k_search: Number of documents to retrieve
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

            # Decode to check for end tags (tool or search)
            current_generated = decode_indices(generated_tokens, idx_to_char)
            if tool_end_token in current_generated or search_end_token in current_generated:
                found_end_tag = True
                if verbose:
                    tag = tool_end_token if tool_end_token in current_generated else search_end_token
                    print(f"Found {tag} at token {i}")
                break

        # Decode generated text
        generated_text = decode_indices(generated_tokens, idx_to_char)
        current_text += generated_text

        if verbose:
            print(f"\nGenerated ({len(generated_tokens)} tokens):")
            print(f"{generated_text}")

        # Check if we found tool execution tags
        if found_end_tag and tool_start_token in current_text and tool_end_token in current_text:
            # Extract code
            code = extract_code_from_exec_tags(current_text, tool_start_token, tool_end_token)

            if code:
                if verbose:
                    print(f"\n>>> EXECUTING CODE:")
                    print(f"{code}")
                    print(f">>> RUNNING...")

                # Execute code
                result = execute_code(code, timeout=5)

                # Check if execution resulted in an error
                if result.startswith("ERROR:"):
                    if verbose:
                        print(f">>> EXECUTION ERROR:")
                        print(f"{result}")
                        print(f"\n⚠ Feeding error back to model for correction...")

                    # Feed error back to model for self-correction
                    # Format: SONUÇ: HATA: {error_message}
                    result_text = f"\nSONUÇ: {result}\nDÜŞÜNCE: "
                    current_text += result_text
                    current_indices = encode_text(current_text, char_to_idx)

                    if verbose:
                        print(f"Continuing with error feedback for model to self-correct...")
                else:
                    # Success case
                    if verbose:
                        print(f">>> RESULT:")
                        print(f"{result}")

                    # Append successful result
                    result_text = f"\nSONUÇ: {result.strip()}\nCEVAP: "
                    current_text += result_text
                    current_indices = encode_text(current_text, char_to_idx)

                    if verbose:
                        print(f"\nContinuing generation after successful execution...")
            else:
                if verbose:
                    print(f"⚠ Warning: Found {tool_end_token} but could not extract code")
                    print(f"Continuing generation...")
                # Don't break - let model continue generating

        # Check if we found search tags
        elif found_end_tag and search_start_token in current_text and search_end_token in current_text:
            # Extract search query
            query = extract_search_query(current_text, search_start_token, search_end_token)

            if query and vector_store is not None:
                if verbose:
                    print(f"\n>>> SEARCHING KNOWLEDGE BASE:")
                    print(f"Query: {query}")
                    print(f">>> RETRIEVING...")

                # Search vector store
                try:
                    results = vector_store.search_with_scores(query, top_k=top_k_search)

                    if results:
                        if verbose:
                            print(f">>> RETRIEVED {len(results)} DOCUMENTS:")
                            for i, (doc, score) in enumerate(results):
                                print(f"\n{i+1}. [Similarity: {score:.3f}]")
                                print(f"   {doc.text[:100]}...")

                        # Format results as context
                        context_parts = []
                        for i, (doc, score) in enumerate(results):
                            context_parts.append(f"[Kaynak {i+1}] {doc.text}")

                        context = "\n".join(context_parts)

                        # Append context to current text
                        result_text = f"\nBULUNAN BİLGİ:\n{context}\n\nCEVAP: "
                        current_text += result_text
                        current_indices = encode_text(current_text, char_to_idx)

                        if verbose:
                            print(f"\nContinuing generation with retrieved context...")
                    else:
                        if verbose:
                            print(f">>> No relevant documents found")
                        # Continue without context
                        result_text = f"\nBULUNAN BİLGİ: İlgili bilgi bulunamadı.\n\nCEVAP: "
                        current_text += result_text
                        current_indices = encode_text(current_text, char_to_idx)

                except Exception as e:
                    if verbose:
                        print(f">>> SEARCH ERROR: {e}")
                    # Continue without context
                    result_text = f"\nBULUNAN BİLGİ: Arama hatası.\n\nCEVAP: "
                    current_text += result_text
                    current_indices = encode_text(current_text, char_to_idx)
            else:
                if verbose:
                    if not query:
                        print(f"⚠ Warning: Found {search_end_token} but could not extract query")
                    elif vector_store is None:
                        print(f"⚠ Warning: Search requested but no vector store provided")
                    print(f"Continuing generation...")
                # Don't break - let model continue generating

        else:
            # No tool tags found, we're done
            if verbose:
                print(f"\nNo tool tags found. Generation complete.")
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

    # Create a sample knowledge base for RAG testing
    knowledge_store = VectorStore()
    knowledge_store.add_documents([
        "Python is a high-level programming language created by Guido van Rossum.",
        "JAX is a numerical computing library for high-performance machine learning research.",
        "Factorial of n (n!) is the product of all positive integers less than or equal to n.",
        "The factorial function grows very rapidly. For example, 10! = 3,628,800.",
        "Matrix multiplication is a binary operation that produces a matrix from two matrices.",
        "Neural networks are computing systems inspired by biological neural networks.",
    ])
    logger.info(f"✓ Created knowledge base with {knowledge_store.size()} documents")

    # Test prompts
    test_prompts = [
        "SORU: 345 * 982 nedir?\nDÜŞÜNCE: Bu bir çarpma işlemi. Python kullanarak hesaplayacağım.\nEYLEM: ",
        "SORU: Faktöriyel nedir?\nDÜŞÜNCE: Bilgi tabanından faktöriyel hakkında bilgi arayacağım.\nARAMA: <SEARCH>factorial definition</SEARCH>",
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
            vector_store=knowledge_store,
            top_k_search=2,
            verbose=True
        )


if __name__ == "__main__":
    main()
