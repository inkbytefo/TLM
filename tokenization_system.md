# AgiFormer Tokenization System Technical Documentation

## 1. Overview
The AgiFormer model utilizes a **Byte-Level Tokenization** strategy, completely bypassing traditional subword tokenizers (like BPE, WordPiece, or SentencePiece). This approach allows the model to process raw binary data (UTF-8 bytes) directly, making it vocabulary-agnostic and capable of handling any language or data format without preprocessing.

## 2. Technical Specifications

### 2.1. Vocabulary
- **Type**: Raw Bytes (uint8)
- **Vocabulary Size**: 256 (Values 0-255)
- **Special Tokens**: None (The model operates purely on data bytes).
- **Encoding**: UTF-8 (for text data).

### 2.2. Input Processing Pipeline
1.  **Text to Bytes**: Input text is encoded into UTF-8 bytes.
    - Example: `"A"` -> `[65]`
    - Example: `"İ"` (Turkish) -> `[196, 176]`
2.  **Tensor Conversion**: Byte sequences are converted to `torch.long` tensors.
3.  **Embedding**: A standard lookup table embeds the 256 possible byte values into a high-dimensional space (`d_model`).
    - `nn.Embedding(256, d_model)`

### 2.3. Patching Mechanism (Implicit Tokenization)
While the input is byte-level, the model does not process one byte at a time in the Transformer layers. Instead, it uses a **Patching Mechanism** to group bytes into larger semantic units, effectively creating "visual tokens" from 1D byte sequences.

- **Component**: `ByteLatentEncoder` (`src/models/encoder.py`)
- **Operation**: 1D Convolution (`nn.Conv1d`)
- **Kernel Size**: 6
- **Stride**: 4
- **Patch Size**: 4 bytes
- **Overlap**: 2 bytes

#### How it works:
The encoder slides a window of 6 bytes over the sequence, moving 4 bytes at a time.
- **Input Sequence Length**: $L$ bytes
- **Output Latent Sequence Length**: $L / 4$ latents

This reduces the sequence length by a factor of 4, significantly improving computational efficiency ($O(N^2)$ attention becomes $O((N/4)^2)$).

#### Causal Padding
To ensure the model remains autoregressive (cannot see the future), the patching uses **Left-Only Padding**.
- **Padding Size**: 2 bytes (Kernel Size 6 - Stride 4).
- This ensures that the convolution for position $t$ only depends on bytes $t$ and previous bytes, maintaining causality.

### 2.4. Output Decoding
The model predicts raw bytes directly from the latent representations.

- **Component**: `LocalAutoregressiveHead` (`src/models/agiformer.py`)
- **Mechanism**: Parallel MLP Decoder
- **Input**: Latent vector (dimension $D$)
- **Output**: Logits for 4 bytes simultaneously ($4 \times 256$ logits).
- **Prediction**: The model predicts the next 4 bytes in the sequence at once.

## 3. Advantages of this Approach
1.  **Universal**: Works for any language (Turkish, English, Code) without retraining a tokenizer.
2.  **Robust**: No "unknown token" (<UNK>) issues; can represent any string.
3.  **Efficient**: The stride-4 patching makes it 4x more efficient than a naive byte-level transformer, approaching the efficiency of subword models while retaining byte-level flexibility.
4.  **Morphology-Aware**: For agglutinative languages like Turkish, byte-level processing (especially with overlap) can better capture morphological structures than rigid subword splits.
