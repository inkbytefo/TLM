import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.hyena_block import HyenaBlock
from src.models.memory_layer import ResidualMemoryBlock

class SpectralGPT(nn.Module):
    """
    SpectralGPT: Decoder-Only Generative Model.
    Modified to return memory state for autonomous loop.
    """
    vocab_size: int = 260
    hidden_dim: int = 512
    num_layers: int = 6
    dropout_rate: float = 0.1
    use_memory: bool = False
    memory_dim: int = 64
    memory_interval: int = 2
    
    @nn.compact
    def __call__(self, x, init_memory_state=None, train: bool = True):
        # x: (Batch, Length)
        
        # 1. Embedding
        x_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(x)
        
        # 2. Positional Encoding (Dynamic / Infinite)
        # We use the same Sinusoidal PE as in HyenaBlock for consistency and extrapolation
        from src.models.hyena_block import PositionalEmbedding
        
        seq_len = x.shape[1]
        # Generate PE for current length
        pe = PositionalEmbedding(emb_dim=self.hidden_dim)(seq_len) # (Seq, Hidden)
        
        # Add to embeddings (Broadcast batch dim)
        x_emb = x_emb + pe[None, :, :]
        x_emb = nn.Dropout(rate=self.dropout_rate)(x_emb, deterministic=not train)
        
        # 3. Hybrid Hyena-Memory Blocks
        curr_x = x_emb
        
        # Handle memory state initialization/passing
        # init_memory_state is expected to be a list of states for each memory layer
        new_memory_states = []
        
        mem_layer_count = 0

        for layer_idx in range(self.num_layers):
            # Hyena block
            residual = curr_x
            curr_x = nn.LayerNorm()(curr_x)
            curr_x = HyenaBlock(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)(curr_x, train=train)
            curr_x = residual + curr_x

            # Interleaved memory layer
            if self.use_memory and (layer_idx + 1) % self.memory_interval == 0:
                # Extract state for this specific layer if available
                layer_state = None
                if init_memory_state is not None and mem_layer_count < len(init_memory_state):
                    layer_state = init_memory_state[mem_layer_count]
                
                curr_x, new_mem_state = ResidualMemoryBlock(
                    hidden_dim=self.hidden_dim,
                    memory_dim=self.memory_dim,
                    dropout_rate=self.dropout_rate
                )(curr_x, layer_state, train=train)
                
                new_memory_states.append(new_mem_state)
                mem_layer_count += 1

        curr_x = nn.LayerNorm()(curr_x)
        
        # 4. Output Head
        logits = nn.Dense(self.vocab_size)(curr_x)
        
        # Return logits AND memory states
        return logits, new_memory_states
