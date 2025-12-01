import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.hyena_block import HyenaBlock

class SpectralGPT(nn.Module):
    """
    SpectralGPT: A Decoder-Only Generative Model using Causal Spectral Blocks (Hyena).
    
    Architecture:
    1. Embedding (Byte-Level)
    2. Stack of HyenaBlocks
    3. Final Norm
    4. Output Head (Next Token Prediction)
    """
    vocab_size: int = 256 # Byte-level
    hidden_dim: int = 512
    num_layers: int = 6
    dropout_rate: float = 0.1
    max_len: int = 2048 # Maximum sequence length for PE
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, Length) -> uint8
        
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
        
        # 3. Hyena Blocks
        curr_x = x_emb
        for _ in range(self.num_layers):
            residual = curr_x
            curr_x = nn.LayerNorm()(curr_x)
            curr_x = HyenaBlock(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)(curr_x, train=train)
            curr_x = residual + curr_x
            
        curr_x = nn.LayerNorm()(curr_x)
        
        # 4. Output Head
        logits = nn.Dense(self.vocab_size)(curr_x)
        
        return logits
