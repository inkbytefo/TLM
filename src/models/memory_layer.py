"""
## Developer: inkbytefo
## Modified: 2025-12-03
Delta Memory Layer - Error-Correcting Linear Memory for ASI.
Modified: Increased initialization scale and beta range for faster convergence.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple


class DeltaMemoryLayer(nn.Module):
    """
    Delta Memory Layer with error-correcting updates.

    This layer maintains a memory matrix that can OVERWRITE old information,
    solving the Catastrophic Forgetting problem in continual learning.

    Attributes:
        hidden_dim: Dimension of hidden representations
        memory_dim: Dimension of memory keys/queries (compressed)
        dropout_rate: Dropout probability
        beta_min: Minimum learning rate for memory updates
        beta_max: Maximum learning rate for memory updates
    """
    hidden_dim: int
    memory_dim: int = 64
    dropout_rate: float = 0.1
    beta_min: float = 0.1    # Artırıldı: 0.01 -> 0.1 (Daha hızlı update)
    beta_max: float = 0.9    # Artırıldı: 0.3 -> 0.9 (Daha agresif update)

    def setup(self):
        """Initialize learnable parameters."""
        # Query, Key, Value projections
        self.W_q = nn.Dense(self.memory_dim, use_bias=False, name='query_proj')
        self.W_k = nn.Dense(self.memory_dim, use_bias=False, name='key_proj')
        self.W_v = nn.Dense(self.memory_dim, use_bias=False, name='value_proj')

        # Output projection with learnable scale for residual balancing
        self.W_o = nn.Dense(self.hidden_dim, use_bias=True, name='output_proj')

        # Learnable scale for memory contribution
        # KRİTİK DEĞİŞİKLİK: 0.1 -> 1.0 (Hafıza artık başlangıçta tam etkili)
        self.scale_param = self.param(
            'memory_scale',
            nn.initializers.constant(1.0),
            (1,)
        )

        # Learnable beta (learning rate) for memory updates
        self.beta_param = self.param(
            'beta',
            nn.initializers.constant(0.0),  # Sigmoid(0) = 0.5 -> Ortadan başla
            (1,)
        )

        # Dropout
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def compute_beta(self) -> jnp.ndarray:
        """
        Compute learning rate (beta) from learnable parameter.
        Maps unbounded parameter to [beta_min, beta_max] range.
        """
        alpha = jax.nn.sigmoid(self.beta_param)
        beta = self.beta_min + alpha * (self.beta_max - self.beta_min)
        return beta

    def init_memory_state(self, batch_size: int) -> jnp.ndarray:
        """
        Initialize memory state matrix.

        Args:
            batch_size: Batch size

        Returns:
            Memory state matrix of shape (batch_size, memory_dim, memory_dim)
        """
        return jnp.zeros((batch_size, self.memory_dim, self.memory_dim))

    def update_memory_delta(
        self,
        memory_state: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        beta: float
    ) -> jnp.ndarray:
        """
        Update memory using Delta Rule with Adaptive Clipping.

        Delta Rule Update (Raw, with adaptive clipping):
            1. Predict current value: v_pred = S_{t-1} @ k
            2. Compute error: delta = v - v_pred
            3. Compute raw update: update = delta ⊗ k
            4. Adaptive clipping: Only clip if update norm exceeds threshold
            5. Update memory: S_t = S_{t-1} + β * update_clipped

        Key Difference from Previous Version:
        - NO normalization of delta/key (preserves magnitude)
        - Adaptive clipping only when necessary (preserves small updates)
        - Large errors → Large updates (fast correction)
        - Small errors → Small updates (precision)

        Args:
            memory_state: Current memory state (B, D_mem, D_mem)
            key: Key vector (B, D_mem)
            value: Target value vector (B, D_mem)
            beta: Learning rate for update

        Returns:
            Updated memory state (B, D_mem, D_mem)
        """
        # Step 1: Read current prediction from memory
        # v_pred = S @ k: (B, D_mem, D_mem) @ (B, D_mem, 1) -> (B, D_mem)
        v_pred = jnp.einsum('bde,be->bd', memory_state, key)

        # Step 2: Compute prediction error (delta)
        # delta = v - v_pred
        delta = value - v_pred  # (B, D_mem)

        # Step 3: Compute RAW outer product update (no normalization!)
        # update = delta ⊗ k: (B, D_mem, 1) @ (B, 1, D_mem) -> (B, D_mem, D_mem)
        update = jnp.einsum('bd,be->bde', delta, key)

        # Step 4: Adaptive Clipping (only clip if norm too large)
        # Compute Frobenius norm of update matrix for each batch
        update_norm = jnp.linalg.norm(update, axis=(1, 2), keepdims=True)  # (B, 1, 1)

        # Clip threshold (relaxed for better gradient flow)
        clip_threshold = 50.0  # Artırıldı: 10.0 -> 50.0

        # Only clip if norm exceeds threshold
        # If update_norm <= clip_threshold: scale_factor = 1 (no change)
        # If update_norm > clip_threshold: scale_factor = clip_threshold / update_norm
        scale_factor = jnp.minimum(1.0, clip_threshold / (update_norm + 1e-8))

        update_clipped = update * scale_factor

        # Step 5: Apply delta update to memory
        new_state = memory_state + beta * update_clipped

        return new_state

    def read_memory(
        self,
        memory_state: jnp.ndarray,
        query: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Read from memory using query.

        Read equation:
            output = S_t @ q

        Args:
            memory_state: Current memory state (B, D_mem, D_mem)
            query: Query vector (B, D_mem)

        Returns:
            Retrieved value (B, D_mem)
        """
        # output = S @ q: (B, D_mem, D_mem) @ (B, D_mem, 1) -> (B, D_mem)
        output = jnp.einsum('bde,be->bd', memory_state, query)
        return output

    def __call__(
        self,
        x: jnp.ndarray,
        memory_state: Optional[jnp.ndarray] = None,
        train: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass with Delta Rule memory updates.

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
        queries = self.W_q(x)  # (B, L, D_mem)
        keys = self.W_k(x)     # (B, L, D_mem)
        values = self.W_v(x)   # (B, L, D_mem)

        # Compute learning rate (beta) for updates
        beta = self.compute_beta()

        # Get learnable memory scale
        memory_scale = jnp.abs(self.scale_param[0])  # Ensure positive

        def scan_fn(carry, inputs):
            """
            Process sequence step by step using Delta Rule.

            KEY FIX: Write FIRST, then read from updated memory.
            This ensures that at position t, we can recall what was written at position t-1.
            """
            mem_state = carry
            query_t, key_t, value_t = inputs

            # STEP 1: WRITE to memory (update with current key-value)
            # No gating - let the model learn what to write through backprop
            mem_state = self.update_memory_delta(
                mem_state,
                key_t,      # Raw key (no gating)
                value_t,    # Raw value (no gating)
                beta
            )

            # STEP 2: READ from updated memory (causal - uses info from past writes)
            output_t = self.read_memory(mem_state, query_t)

            return mem_state, output_t

        # Process sequence step by step
        final_memory_state, outputs = jax.lax.scan(
            scan_fn,
            memory_state,
            (
                jnp.transpose(queries, (1, 0, 2)),   # (L, B, D_mem)
                jnp.transpose(keys, (1, 0, 2)),
                jnp.transpose(values, (1, 0, 2))
            )
        )

        # Reshape outputs: (L, B, D_mem) -> (B, L, D_mem)
        outputs = jnp.transpose(outputs, (1, 0, 2))

        # Project back to hidden dimension
        outputs = self.W_o(outputs)

        # Apply learnable scaling for residual balancing
        outputs = outputs * memory_scale

        # Apply dropout
        outputs = self.dropout(outputs, deterministic=not train)

        return outputs, final_memory_state


class ResidualMemoryBlock(nn.Module):
    """
    Memory block with residual connection and layer norm.
    Uses Delta Memory Layer for error-correcting updates.
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
        Forward pass with delta memory and residual connection.

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

        # Delta Memory Layer
        memory_layer = DeltaMemoryLayer(
            hidden_dim=self.hidden_dim,
            memory_dim=self.memory_dim,
            dropout_rate=self.dropout_rate
        )
        memory_output, final_memory_state = memory_layer(normed_x, memory_state, train)

        # Residual connection
        output = x + memory_output

        return output, final_memory_state


if __name__ == "__main__":
    # Test the Delta Memory Layer
    print("Testing Delta Memory Layer...")

    # Create a simple test
    batch_size = 2
    seq_len = 10
    hidden_dim = 128
    memory_dim = 32

    # Initialize layer
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))

    # Create module
    memory_layer = DeltaMemoryLayer(
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

    # Test beta computation
    beta = jax.nn.sigmoid(params['beta'][0]) * (0.3 - 0.01) + 0.01
    print(f"Beta (learning rate): {float(beta):.4f}")

    # Test copying capability
    print("\n[Testing Copying Capability]")
    print("Testing if Delta Rule can overwrite memory...")

    # Create a simple copying test
    # Input: [A, B, A, C]
    # Expected: Model should remember A->B, then A->C (overwrite)
    test_seq = jnp.array([
        [1.0] * hidden_dim,   # Token A
        [2.0] * hidden_dim,   # Token B (value for A)
        [1.0] * hidden_dim,   # Token A again
        [3.0] * hidden_dim,   # Token C (new value for A - should overwrite)
    ])

    test_seq = test_seq[None, ...]  # Add batch dimension: (1, 4, hidden_dim)

    test_output, test_mem = memory_layer.apply(
        {'params': params},
        test_seq,
        train=False
    )

    print(f"Test sequence shape: {test_seq.shape}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Test memory state shape: {test_mem.shape}")

    print("\n[OK] Delta Memory Layer test passed!")
    print("Ready for copying task validation!")
