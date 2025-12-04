import jax
import jax.numpy as jnp
from flax import linen as nn
from src.models.causal_utils import causal_fft_conv

class PositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding for the Filter Network"""
    emb_dim: int

    @nn.compact
    def __call__(self, seq_len):
        # Create position indices: [0, 1, ..., seq_len-1]
        positions = jnp.arange(seq_len)[:, None] # (Seq, 1)
        
        # Create frequencies
        # div_term = exp(arange(0, dim, 2) * -(log(10000.0) / dim))
        div_term = jnp.exp(jnp.arange(0, self.emb_dim, 2) * -(jnp.log(10000.0) / self.emb_dim))
        
        # Calculate sine and cosine components
        pe = jnp.zeros((seq_len, self.emb_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))
        
        return pe # (Seq, Emb_Dim)

class FilterNetwork(nn.Module):
    """
    Implicit Neural Filter (MLP).
    Maps Position -> Filter Value.
    Allows for infinite context extrapolation.
    """
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, positions_emb):
        # positions_emb: (Seq, Emb_Dim)
        
        # Simple MLP: Pos -> Dense -> Act -> Dense -> Act -> Dense -> Filter
        x = nn.Dense(self.hidden_dim)(positions_emb)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x) # (Seq, Output_Dim)
        
        return x

class HyenaBlock(nn.Module):
    """
    Hyena Block: A causal spectral block for autoregressive modeling.
    Replaces Attention with Causal FFT Convolution.
    
    Now with Infinite Context Support via Implicit Neural Filters.
    """
    hidden_dim: int
    filter_order: int = 64 # Dimension of the positional embedding for the filter
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, Length, Hidden)
        seq_len = x.shape[1]
        
        # 1. Projections
        u = nn.Dense(self.hidden_dim)(x)
        v = nn.Dense(self.hidden_dim)(x)
        
        # 2. Implicit Neural Filter Generation
        # Instead of learning a fixed tensor h[max_len], we generate h(t)
        
        # A. Generate Positional Embeddings for current length
        # We use a small embedding dimension for the filter network (e.g., 64)
        pos_emb = PositionalEmbedding(emb_dim=self.filter_order)(seq_len) # (Seq, Filter_Order)
        
        # B. Pass through MLP to get the filter
        # The filter needs to be (Seq, Hidden) to convolve with v (Seq, Hidden)
        # We initialize the MLP to output small values to start
        h = FilterNetwork(hidden_dim=64, output_dim=self.hidden_dim)(pos_emb)
        
        # Apply exponential decay window to enforce locality/stability
        # This is crucial for Hyena stability
        t = jnp.arange(seq_len)
        decay = jnp.exp(-0.01 * t)[:, None]
        h = h * decay
        
        # 3. Causal Convolution
        # y = h * v (Convolution)
        v_conv = causal_fft_conv(v, h)
        
        # 4. Gating
        # y = u * v_conv (Element-wise gating)
        y = u * v_conv
        
        # 5. Output Projection
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        
        return y
