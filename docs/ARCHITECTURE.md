# Architecture Overview

This document details the internal architecture of the Spectral-JAX framework, focusing on the custom model components and the autonomous agent system.

## SpectralGPT

**SpectralGPT** is the core generative model. Unlike standard Transformers, it leverages **Hyena Operators** to replace the traditional attention mechanism, offering better scaling properties for long sequences.

### Key Components

1.  **Hyena Blocks**:
    *   Replaces Multi-Head Attention for the majority of layers.
    *   Uses implicit convolutions and gating mechanisms to mix information across the sequence.
    *   Provides sub-quadratic complexity with respect to sequence length.

2.  **Sliding Window Attention (Hybrid)**:
    *   Interleaved every **6 Hyena layers**.
    *   Standard Multi-Head Attention with a causal sliding window mask.
    *   Provides "needle-in-a-haystack" retrieval capabilities that pure signal processing models sometimes lack.
    *   Ensures the model can focus intensely on local details while Hyena handles the global context.

2.  **Residual Memory Blocks**:
    *   Interleaved with Hyena blocks.
    *   Allows the model to read and write to a persistent memory vector.
    *   Crucial for maintaining context across long interaction sessions in the autonomous loop.

3.  **Positional Embedding**:
    *   Uses dynamic sinusoidal positional embeddings to handle variable sequence lengths and enable extrapolation.

### Model Definition (`src/models/gpt.py`)

```python
class SpectralGPT(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    use_memory: bool
    # ...
```

The model takes a sequence of token IDs and outputs logits for the next token, optionally updating its internal memory state.

## Autonomous Agent System

The agent operates in a continuous loop defined in `autonomous_agent.py`.

### The Loop

1.  **Observation**: The agent receives input (user messages, tool outputs, or system prompts).
2.  **Thinking**: The agent generates internal thought traces (`THINK` token) to plan its actions.
3.  **Action**:
    *   **Speak**: Output text to the user (`SPEAK` token).
    *   **Tool Use**: Execute Python code (`<EXEC>` tags).
    *   **Wait**: Pause and wait for external input (`WAIT` token).
4.  **Memory Update**: The agent's short-term memory (context window) and long-term memory (RAG) are updated.

### RAG (Retrieval-Augmented Generation)

Located in `src/memory/rag.py`.
*   Uses **FAISS** for efficient vector similarity search.
*   Uses **SentenceTransformers** for embedding text.
*   Stores past interactions and knowledge snippets to augment the agent's context.

### Tool Execution

Located in `src/tools/executor.py`.
*   Sandboxed Python execution environment.
*   Captures `stdout` and `stderr`.
*   Time-limited to prevent infinite loops.

## Training Pipeline

The training system (`train.py`) supports:
*   **Curriculum Learning**: Progressing through stages of difficulty.
*   **Generative Training**: Standard next-token prediction.
*   **Memory Training**: Optimizing the memory usage (if enabled).

Checkpoints are managed using `flax.training.checkpoints`.
