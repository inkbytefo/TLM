"""
Gated Linear Memory Layer for Spectral-Associative Hybrid Architecture.

This layer implements a dynamic associative memory that can:
1. WRITE: Store Key-Value pairs in a memory matrix
2. READ: Query the memory to retrieve information
3. FORGET: Gradually decay old information

Based on:
- Fast Weights (Schmidhuber, 1992)
- Linear Transformers (Katharopoulos et al., 2020)
- Gated Linear Attention (Yang et al., 2023)

Key advantage: O(N) complexity with infinite theoretical context window.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple


class GatedLinearMemory(nn.Module):
    """
    Gated Linear Memory layer with dynamic state updates.

    This layer maintains a hidden state (memory matrix) that gets updated
    with each token, allowing the model to "remember" information beyond
    the training context window.

    Attributes:
        hidden_dim: Dimension of hidden representations
        memory_dim: Dimension of memory keys/queries (typically smaller for efficiency)
        dropout_rate: Dropout probability
        decay_min: Minimum decay factor (0 = full forgetting, 1 = perfect memory)
        decay_max: Maximum decay factor
    """
    hidden_dim: int
    memory_dim: int = 64  # Compressed memory dimension
    dropout_rate: float = 0.1
    decay_min: float = 0.9
    decay_max: float = 0.999

    def setup(self):
        """Initialize learnable parameters."""
        # Query, Key, Value projections (like in attention)
        self.W_q = nn.Dense(self.memory_dim, use_bias=False, name='query_proj')
        self.W_k = nn.Dense(self.memory_dim, use_bias=False, name='key_proj')
        self.W_v = nn.Dense(self.memory_dim, use_bias=False, name='value_proj')

        # Output projection
        self.W_o = nn.Dense(self.hidden_dim, use_bias=True, name='output_proj')

        # Learnable gating parameters
        self.gate_proj = nn.Dense(1, use_bias=True, name='gate_proj')

        # Learnable decay factor (per-head)
        self.decay_param = self.param(
            'decay',
            nn.initializers.constant(0.5),  # Initialize to middle value
            (1,)
        )

        # Dropout
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def compute_decay(self) -> jnp.ndarray:
        """
        Compute decay factor from learnable parameter.
        Maps unbounded parameter to [decay_min, decay_max] range.
        """
        # Sigmoid maps to [0, 1], then scale to [decay_min, decay_max]
        alpha = jax.nn.sigmoid(self.decay_param)
        decay = self.decay_min + alpha * (self.decay_max - self.decay_min)
        return decay

    def init_memory_state(self, batch_size: int) -> jnp.ndarray:
        """
        Initialize memory state matrix.

        Args:
            batch_size: Batch size

        Returns:
            Memory state matrix of shape (batch_size, memory_dim, memory_dim)
        """
        return jnp.zeros((batch_size, self.memory_dim, self.memory_dim))

    def update_memory_step(
        self,
        memory_state: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        decay: float
    ) -> jnp.ndarray:
        """
        Update memory state with new Key-Value pair.

        Memory update equation (Fast Weights):
            S_t = decay * S_{t-1} + K_t^T @ V_t

        Args:
            memory_state: Current memory state (B, D_mem, D_mem)
            key: Key vector (B, D_mem)
            value: Value vector (B, D_mem)
            decay: Decay factor for old memories

        Returns:
            Updated memory state (B, D_mem, D_mem)
        """
        # Outer product of key and value: K^T @ V
        # (B, D_mem, 1) @ (B, 1, D_mem) -> (B, D_mem, D_mem)
        update = jnp.einsum('bd,be->bde', key, value)

        # Apply decay to old memory and add new information
        new_state = decay * memory_state + update

        return new_state

    def read_memory(
        self,
        memory_state: jnp.ndarray,
        query: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Read from memory using query.

        Read equation:
            output = Q @ S_t

        Args:
            memory_state: Current memory state (B, D_mem, D_mem)
            query: Query vector (B, D_mem)

        Returns:
            Retrieved value (B, D_mem)
        """
        # Q @ S: (B, D_mem) @ (B, D_mem, D_mem) -> (B, D_mem)
        output = jnp.einsum('bd,bde->be', query, memory_state)
        return output

    def __call__(
        self,
        x: jnp.ndarray,
        memory_state: Optional[jnp.ndarray] = None,
        train: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass with memory update and retrieval.

        Args:
            x: Input tensor (Batch, SeqLen, Hidden)
            memory_state: Optional previous memory state (Batch, D_mem, D_mem)
            train: Whether in training mode

        Returns:
            output: Output tensor (Batch, SeqLen, Hidden)
            final_memory_state: Updated memory state (Batch, D_mem, D_mem)
        """
        B, L, D = x.shape

        # Initialize memory if not provided
        if memory_state is None:
            memory_state = self.init_memory_state(B)

        # Project to Query, Key, Value
        # (B, L, D) -> (B, L, D_mem)
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # Compute decay factor
        decay = self.compute_decay()

        # Compute gating scores (which tokens to remember strongly)
        # (B, L, D) -> (B, L, 1)
        gate_scores = self.gate_proj(x)
        gates = jax.nn.sigmoid(gate_scores).squeeze(-1)  # (B, L)

        def scan_fn(carry, inputs):
            """Scan function to process sequence step by step."""
            mem_state = carry
            query_t, key_t, value_t, gate_t = inputs

            # Apply gating to key and value
            gated_key = key_t * gate_t[..., None]
            gated_value = value_t * gate_t[..., None]

            # Update memory with new Key-Value pair
            mem_state = self.update_memory_step(mem_state, gated_key, gated_value, decay)

            # Read from memory using query
            output_t = self.read_memory(mem_state, query_t)

            return mem_state, output_t

        # Process sequence step by step using scan
        # inputs: queries (B,L,D_mem), keys (B,L,D_mem), values (B,L,D_mem), gates (B,L)
        final_memory_state, outputs = jax.lax.scan(
            scan_fn,
            memory_state,
            (
                jnp.transpose(queries, (1, 0, 2)),  # (L, B, D_mem)
                jnp.transpose(keys, (1, 0, 2)),
                jnp.transpose(values, (1, 0, 2)),
                jnp.transpose(gates, (1, 0))  # (L, B)
            )
        )

        # outputs: (L, B, D_mem) -> (B, L, D_mem)
        outputs = jnp.transpose(outputs, (1, 0, 2))

        # Project back to hidden dimension
        outputs = self.W_o(outputs)

        # Apply dropout
        outputs = self.dropout(outputs, deterministic=not train)

        return outputs, final_memory_state


class ResidualMemoryBlock(nn.Module):
    """
    Memory block with residual connection and layer norm.
    Combines memory layer with standard feedforward processing.
    """
    hidden_dim: int
    memory_dim: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        memory_state: Optional[jnp.ndarray] = None,
        train: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass with memory and residual connection.

        Args:
            x: Input tensor (Batch, SeqLen, Hidden)
            memory_state: Optional previous memory state
            train: Whether in training mode

        Returns:
            output: Output tensor (Batch, SeqLen, Hidden)
            final_memory_state: Updated memory state
        """
        # Layer norm before memory
        normed_x = nn.LayerNorm()(x)

        # Memory layer
        memory_layer = GatedLinearMemory(
            hidden_dim=self.hidden_dim,
            memory_dim=self.memory_dim,
            dropout_rate=self.dropout_rate
        )
        memory_output, final_memory_state = memory_layer(normed_x, memory_state, train)

        # Residual connection
        output = x + memory_output

        return output, final_memory_state


if __name__ == "__main__":
    # Test the memory layer
    print("Testing Gated Linear Memory Layer...")

    # Create a simple test
    batch_size = 2
    seq_len = 10
    hidden_dim = 128
    memory_dim = 32

    # Initialize layer
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))

    # Create module
    memory_layer = GatedLinearMemory(
        hidden_dim=hidden_dim,
        memory_dim=memory_dim,
        dropout_rate=0.0  # No dropout for testing
    )

    # Initialize parameters
    variables = memory_layer.init(key, x, train=False)
    params = variables['params']

    # Forward pass
    output, memory_state = memory_layer.apply(
        {'params': params},
        x,
        train=False
    )

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Memory state shape: {memory_state.shape}")

    # Test decay computation
    decay = jax.nn.sigmoid(params['decay'][0]) * (0.999 - 0.9) + 0.9
    print(f"Decay factor: {float(decay):.4f}")

    print("\n[OK] Memory layer test passed!")
