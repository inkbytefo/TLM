import jax
import jax.numpy as jnp
from flax import linen as nn
from .encoder import ByteLatentEncoder

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

class SpectralModel(nn.Module):
    vocab_size: int # Artık kullanılmıyor ama config uyumluluğu için tutulabilir veya kaldırılabilir.
    hidden_dim: int
    num_layers: int
    num_classes: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (Batch, L_original) -> uint8
        
        # --- 1. Byte-Level Patching Encoder ---
        # Girdiyi sıkıştır: L -> L/4
        x_encoded = ByteLatentEncoder(hidden_dim=self.hidden_dim)(x)
        # x_encoded shape: (Batch, L_compressed, Hidden)
        
        # --- 2. Positional Encoding ---
        # Sıkıştırılmış uzunluk üzerinde çalışır
        x_emb = SinusoidalPositionalEncoding(d_model=self.hidden_dim)(x_encoded)
        x_emb = nn.Dropout(rate=self.dropout_rate)(x_emb, deterministic=not train)
        
        # --- 3. Spectral Layers ---
        curr_x = x_emb
        for _ in range(self.num_layers):
            curr_x = SpectralLayer(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)(curr_x, train=train)
            
        curr_x = nn.LayerNorm()(curr_x)
        
        # --- 4. Pooling & Classification ---
        if self.num_classes is not None:
            # Maskeleme Mantığı Güncellemesi:
            # Orijinal maske (B, L) boyutundaydı.
            # Patching sonrası maske (B, L/4) olmalı.
            # Basitçe: Orijinal x'te 0 olmayanlar 1'di.
            # Patching işlemi (Conv stride=4) ile boyut düştü.
            # Yeni maskeyi oluşturmak için:
            # Orijinal maskeyi (B, L, 1) alıp Average Pooling (k=4, s=4) yapabiliriz.
            # Veya daha basiti: Encoder çıkışındaki padding'i anlamak zor olabilir (Conv padding='VALID' vs).
            # Alternatif: Encoder çıkışındaki tüm tokenları kullan (Global Average Pooling).
            # Ancak padding tokenları (0) modelin kafasını karıştırabilir.
            
            # Pratik Çözüm:
            # Orijinal maskeyi oluştur
            mask_orig = (x != 0).astype(jnp.float32) # (B, L)
            mask_orig = jnp.expand_dims(mask_orig, axis=-1) # (B, L, 1)
            
            # Maskeyi de aynı Conv işlemiyle (veya AvgPool) küçült
            # Conv(kernel=6, stride=4, padding='VALID') benzeri bir işlem.
            # Maske binary olduğu için, patch içinde EN AZ BİR dolu token varsa o patch dolu sayılmalı (Max Pooling).
            # Veya patch'in ne kadarının dolu olduğu ağırlıklandırılmalı (Avg Pooling).
            
            # Avg Pooling yaklaşımı:
            # Önce sol padding ekle (Encoder ile uyumlu olması için)
            mask_padded = jnp.pad(mask_orig, ((0,0), (2,0), (0,0)))
            
            # Avg Pooling (Stride=4, Window=6 - Encoder ile tam eşleşmesi için Conv yerine AvgPool)
            # Flax AvgPool N-D destekler.
            mask_compressed = nn.avg_pool(mask_padded, window_shape=(6,), strides=(4,), padding='VALID')
            
            # Eğer mask_compressed > 0 ise o patch kısmen doludur.
            # Bunu ağırlık olarak kullanabiliriz.
            
            # Weighted Mean Pooling
            # sum(x * mask) / sum(mask)
            
            # Boyut eşleşmesi kontrolü:
            # x_encoded: (B, L_new, D)
            # mask_compressed: (B, L_new, 1)
            
            sum_embeddings = jnp.sum(curr_x * mask_compressed, axis=1)
            sum_mask = jnp.sum(mask_compressed, axis=1) + 1e-9
            
            pooled_x = sum_embeddings / sum_mask
            logits = nn.Dense(self.num_classes)(pooled_x)
        else:
            logits = nn.Dense(self.vocab_size)(curr_x) # Vocab size 256 olmalı
        
        return logits
