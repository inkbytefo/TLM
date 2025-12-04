# API Reference

## Models

### `src.models.gpt.SpectralGPT`

The main model class.

```python
class SpectralGPT(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    dropout_rate: float
    use_memory: bool
    memory_dim: int
    memory_interval: int
```

*   `__call__(x, init_memory_state=None, train=True)`: Forward pass.

### `src.models.hyena_block.HyenaBlock`

Implements the Hyena operator for long-context mixing.

```python
class HyenaBlock(nn.Module):
    hidden_dim: int
    dropout_rate: float
```

## Memory

### `src.memory.rag.RAGSystem`

Handles retrieval-augmented generation.

```python
class RAGSystem:
    def __init__(self, index_path="memory_index.faiss"): ...
    def add(self, text: str): ...
    def query(self, query_text: str, k=3): ...
```

## Tools

### `src.tools.executor.execute_code`

Executes Python code strings.

```python
def execute_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Returns:
        dict: {'success': bool, 'output': str, 'error': str}
    """
```

## Training

### `src.training.trainer.create_train_state`

Initializes the Flax `TrainState`.

```python
def create_train_state(rng, config): ...
```

### `src.training.trainer.train_step`

Performs a single training step (forward + backward).

```python
def train_step(state, batch, rng): ...
```
