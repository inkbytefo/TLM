import jax
import jax.numpy as jnp
from flax import linen as nn
from .encoder import ByteLatentEncoder
from .memory_layer import ResidualMemoryBlock

class SpectralLayer(nn.Module):
    hidden_dim: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, Length, Hidden)
        
        # 1. Frequency Domain Transformation (FFT)
        x_fft = jnp.fft.rfft(x, axis=1)
        
        # 2. Spectral Gating (Learnable Filters)
        # Parametre sayısı: (Length // 2 + 1, Hidden, 2) -> Real & Imag
        seq_len = x_fft.shape[1]
        
        # Karmaşık sayı ağırlıkları (Complex Weights)
        # Real ve Imaginary kısımları ayrı ayrı öğreniyoruz
        w_real = self.param('w_real', nn.initializers.normal(stddev=0.02), 
                           (seq_len, self.hidden_dim, self.hidden_dim))
        w_imag = self.param('w_imag', nn.initializers.normal(stddev=0.02), 
                           (seq_len, self.hidden_dim, self.hidden_dim))
        
        # (Batch, Seq, Hidden) @ (Seq, Hidden, Hidden) -> (Batch, Seq, Hidden)
        # Einsum ile batch matrix multiplication
        # b: batch, s: seq, i: input_dim, o: output_dim
        
        # Real * Real - Imag * Imag
        real_part = jnp.einsum('bsi,sio->bso', x_fft.real, w_real) - \
                    jnp.einsum('bsi,sio->bso', x_fft.imag, w_imag)
                    
        # Real * Imag + Imag * Real
        imag_part = jnp.einsum('bsi,sio->bso', x_fft.real, w_imag) + \
                    jnp.einsum('bsi,sio->bso', x_fft.imag, w_real)
                    
        x_gated = real_part + 1j * imag_part
        
        # 3. Inverse FFT
        x_out = jnp.fft.irfft(x_gated, n=x.shape[1], axis=1)
        
        # Residual + Norm + Activation
        x = x + x_out
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        return x

class SinusoidalPositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        # x: (Batch, Length, Dim)
        seq_len = x.shape[1]
        
        pe = jnp.zeros((seq_len, self.d_model))
        position = jnp.arange(0, seq_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model))
        
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        pe = jnp.expand_dims(pe, axis=0) # (1, Length, Dim)
        return x + pe

class AttentionPooling(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x, mask=None):
        # x: (Batch, SeqLen, Hidden)
        # mask: (Batch, SeqLen, 1)
        
        # Öğrenilebilir bir "Sorgu" vektörü (Query Vector)
        attn_query = self.param('attn_query', nn.initializers.normal(0.02), (1, 1, self.hidden_dim))
        
        # Attention Skorları: (B, L, D) * (1, 1, D) -> (B, L, D) -> sum -> (B, L)
        scores = jnp.sum(x * attn_query, axis=-1, keepdims=True) 
        
        if mask is not None:
            # Maskelenmiş alanlara çok küçük değer ver (softmax'te 0 olsunlar)
            scores = scores - 1e9 * (1.0 - mask)
            
        attn_weights = nn.softmax(scores, axis=1) # (B, L, 1)
        
        # Ağırlıklı Toplam
        context = jnp.sum(x * attn_weights, axis=1) # (B, D)
        return context

class SpectralModel(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_classes: int
    dropout_rate: float
    encoder_dense_units: int = 128 # Default value
    use_memory: bool = False  # Enable memory layers
    memory_dim: int = 64  # Memory dimension
    memory_interval: int = 2  # Insert memory layer every N spectral layers

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, L_original) -> uint8

        # --- 1. Byte-Level Patching Encoder ---
        x_encoded = ByteLatentEncoder(hidden_dim=self.hidden_dim, encoder_dense_units=self.encoder_dense_units)(x)

        # --- 2. Positional Encoding ---
        x_emb = SinusoidalPositionalEncoding(d_model=self.hidden_dim)(x_encoded)
        x_emb = nn.Dropout(rate=self.dropout_rate)(x_emb, deterministic=not train)

        # --- 3. Hybrid Spectral-Memory Layers ---
        curr_x = x_emb
        memory_state = None  # Will be initialized by first memory layer

        for layer_idx in range(self.num_layers):
            # Spectral processing
            curr_x = SpectralLayer(
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate
            )(curr_x, train=train)

            # Interleaved memory layer
            if self.use_memory and (layer_idx + 1) % self.memory_interval == 0:
                curr_x, memory_state = ResidualMemoryBlock(
                    hidden_dim=self.hidden_dim,
                    memory_dim=self.memory_dim,
                    dropout_rate=self.dropout_rate
                )(curr_x, memory_state, train=train)

        curr_x = nn.LayerNorm()(curr_x)
        
        # --- 4. Pooling & Classification ---
        if self.num_classes is not None:
            # Maskeleme Mantığı
            mask_orig = (x != 0).astype(jnp.float32) # (B, L)
            mask_orig = jnp.expand_dims(mask_orig, axis=-1) # (B, L, 1)
            
            # Mask Downsampling (AvgPool ile)
            mask_padded = jnp.pad(mask_orig, ((0,0), (2,0), (0,0)))
            mask_compressed = nn.avg_pool(mask_padded, window_shape=(6,), strides=(4,), padding='VALID')
            
            # --- ATTENTION POOLING ---
            pooled_x = AttentionPooling(hidden_dim=self.hidden_dim)(curr_x, mask=mask_compressed)
            
            logits = nn.Dense(self.num_classes)(pooled_x)
        else:
            logits = nn.Dense(self.vocab_size)(curr_x)
        
        return logits
