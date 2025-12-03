import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.hyena_block import HyenaBlock
from src.models.memory_layer import ResidualMemoryBlock  # Uses DeltaMemoryLayer internally

class SpectralGPT(nn.Module):
    """
    SpectralGPT: A Decoder-Only Generative Model using Causal Spectral Blocks (Hyena).

    Architecture:
    1. Embedding (Byte-Level)
    2. Stack of HyenaBlocks (with optional Memory Layers)
    3. Final Norm
    4. Output Head (Next Token Prediction)
    """
    vocab_size: int = 256 # Byte-level
    hidden_dim: int = 512
    num_layers: int = 6
    dropout_rate: float = 0.1
    max_len: int = 2048 # Maximum sequence length for PE
    use_memory: bool = False  # Enable dynamic associative memory
    memory_dim: int = 64  # Memory dimension
    memory_interval: int = 2  # Insert memory layer every N hyena layers
    
    @nn.compact
    def __call__(self, x, init_memory_state=None, train: bool = True):
        # x: (Batch, Length) -> uint8
        # init_memory_state: List of memory states for each memory layer (optional)
        
        # 1. Embedding
        x_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(x)
        
        # 2. Positional Encoding
        # We initialize PE with max_len to handle variable input lengths (e.g. L vs L-1)
        seq_len = x.shape[1]
        pe = self.param('pe', nn.initializers.normal(0.02), (1, self.max_len, self.hidden_dim))
        
        # Slice PE to current sequence length
        # pe[:, :seq_len, :]
        pe_slice = pe[:, :seq_len, :]
        
        x_emb = x_emb + pe_slice
        x_emb = nn.Dropout(rate=self.dropout_rate)(x_emb, deterministic=not train)
        
        # 3. Hybrid Hyena-Memory Blocks
        curr_x = x_emb
        memory_state = init_memory_state  # Use passed state
        new_memory_states = [] # To collect updated states

        for layer_idx in range(self.num_layers):
            # Hyena block (causal convolution)
            residual = curr_x
            curr_x = nn.LayerNorm()(curr_x)
            curr_x = HyenaBlock(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)(curr_x, train=train)
            curr_x = residual + curr_x

            # Interleaved memory layer for precise copying
            if self.use_memory and (layer_idx + 1) % self.memory_interval == 0:
                # If we have an initial memory state passed in, use it for the first memory layer
                # Note: This simple logic assumes one memory layer or that we only care about the last one's state
                # for stateful execution. For multiple memory layers, we might need a list of states.
                # For now, we'll assume the state corresponds to the *last* memory layer or a shared state if appropriate.
                # But wait, each memory layer has its own parameters. 
                # To support full stateful execution across multiple layers, we'd need to pass a dict or list of states.
                # Let's simplify: We will pass 'memory_state' to *all* memory layers, but in a recurrent loop,
                # usually we only persist the state of the *Linear Memory* which is often shared or we just track the last one.
                # Actually, in the current architecture, each ResidualMemoryBlock has its own DeltaMemoryLayer.
                # So each layer needs its own state.
                
                # For the Autonomous Loop, we likely want to persist the state of ALL memory layers.
                # Let's assume 'memory_state' input is a LIST of states, one for each memory layer.
                
                layer_mem_state = None
                if memory_state is not None:
                    # Find which memory layer index this is
                    mem_layer_idx = (layer_idx + 1) // self.memory_interval - 1
                    if mem_layer_idx < len(memory_state):
                        layer_mem_state = memory_state[mem_layer_idx]
                
                curr_x, new_mem_state = ResidualMemoryBlock(
                    hidden_dim=self.hidden_dim,
                    memory_dim=self.memory_dim,
                    dropout_rate=self.dropout_rate
                )(curr_x, layer_mem_state, train=train)
                
                # Collect new states
                if new_memory_states is None:
                    new_memory_states = []
                new_memory_states.append(new_mem_state)

        curr_x = nn.LayerNorm()(curr_x)
        
        # 4. Output Head
        logits = nn.Dense(self.vocab_size)(curr_x)
        
        return logits, new_memory_states
