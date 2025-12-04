import jax
import jax.numpy as jnp
from typing import Any
from flax import linen as nn
from src.models.hyena_block import HyenaBlock
from src.models.memory_layer import ResidualMemoryBlock
from src.models.attention import SlidingWindowAttention

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
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, init_memory_state=None, train: bool = True):
        # x: (Batch, Length)
        
        # 1. Embedding
        # 1. Embedding
        x_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim, dtype=self.dtype)(x)
        
        # 2. Positional Encoding (Dynamic / Infinite)
        # We use the same Sinusoidal PE as in HyenaBlock for consistency and extrapolation
        from src.models.hyena_block import PositionalEmbedding
        
        seq_len = x.shape[1]
        # Generate PE for current length
        # Generate PE for current length
        pe = PositionalEmbedding(emb_dim=self.hidden_dim, dtype=self.dtype)(seq_len) # (Seq, Hidden)
        
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
            # Gradient Checkpointing (Remat) to save memory
            # We use static_argnums=(2,) because 'train' appears to be the 3rd argument (index 2)
            # likely due to (self/scope, x, train) signature in the traced function.
            # likely due to (self/scope, x, train) signature in the traced function.
            curr_x = nn.remat(HyenaBlock, static_argnums=(2,))(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate, dtype=self.dtype)(curr_x, train)
            curr_x = residual + curr_x

            # Hybrid: Add Sliding Window Attention every 6 Hyena layers
            # We add it AFTER the 6th Hyena block (1-indexed count)
            # layer_idx is 0-indexed, so (layer_idx + 1) % 6 == 0
            if (layer_idx + 1) % 6 == 0:
                residual_attn = curr_x
                # Attention layer handles its own LayerNorm (Pre-Norm) inside or we do it here.
                # My implementation of SlidingWindowAttention does Pre-Norm internally.
                # But to be consistent with Hyena block usage here (Norm -> Block -> Add),
                # let's check my implementation.
                # My implementation: y = nn.LayerNorm()(x) -> Attention -> Dropout -> Return
                # So it expects raw input and returns residual branch content.
                # Wait, usually we do x + block(norm(x)).
                # My implementation does norm(x) inside.
                # So we just do: curr_x = curr_x + Attention(curr_x)
                
                attn_out = nn.remat(SlidingWindowAttention, static_argnums=(2,))(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    window_size=512, # Can be parameterized if needed
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype
                )(curr_x, train)
                
                curr_x = curr_x + attn_out

            # Interleaved memory layer
            if self.use_memory and (layer_idx + 1) % self.memory_interval == 0:
                # Extract state for this specific layer if available
                layer_state = None
                if init_memory_state is not None and mem_layer_count < len(init_memory_state):
                    layer_state = init_memory_state[mem_layer_count]
                
                curr_x, new_mem_state = ResidualMemoryBlock(
                    hidden_dim=self.hidden_dim,
                    memory_dim=self.memory_dim,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype
                )(curr_x, layer_state, train=train)
                
                new_memory_states.append(new_mem_state)
                mem_layer_count += 1

        curr_x = nn.LayerNorm(dtype=self.dtype)(curr_x)
        
        # 4. Output Head
        logits = nn.Dense(self.vocab_size, dtype=self.dtype)(curr_x)
        
        # Return logits AND memory states
        return logits, new_memory_states
