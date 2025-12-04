# Autonomous Agent System

The **Spectral-JAX** agent is designed to be a fully autonomous entity capable of long-term reasoning, tool use, and interaction.

## Core Concepts

### 1. The Loop (`autonomous_agent.py`)
The agent operates in a continuous `Observation -> Thought -> Action` loop. Unlike traditional chatbots that stop after generating a response, this agent:
1.  **Observes** the environment (user input, tool output).
2.  **Thinks** about what to do (`THINK` token).
3.  **Acts** (Speaks, executes code, or waits).
4.  **Repeats** until it decides to stop or wait.

### 2. Special Tokens
We have reserved special tokens in the vocabulary (IDs 256-259) to control this flow:

| Token | ID | Description |
| :--- | :--- | :--- |
| `SILENCE` | 256 | Agent decides to stay silent and observe. |
| `WAIT` | 257 | Agent pauses execution to wait for user input. |
| `THINK` | 258 | Triggers internal monologue (Chain of Thought). |
| `SPEAK` | 259 | Triggers output to the user. |

### 3. Hybrid Memory
The agent leverages the **Hybrid Hyena-Attention** architecture:
- **Hyena**: Maintains an infinite context of the entire interaction history.
- **Sliding Window Attention**: Focuses on the most recent tokens (e.g., last 512) for precise instruction following.

## Future: Phase 3 (Agency)

Currently, in **Phase 1**, the model is learning the *structure* of language and code. It does not yet know *how* to use these tokens autonomously.

In **Phase 3**, we will fine-tune the model on interaction datasets where these tokens are used explicitly, teaching the model:
- "When I see a complex question, I should output `THINK` first."
- "When I need to calculate something, I should use `<EXEC>`."
- "When I am done, I should output `WAIT`."
