"""
Text generation data loader for character/byte-level language modeling.
"""

import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple

def load_text_file(filepath: str) -> str:
    """Load text file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_char_mappings(text: str) -> Tuple[dict, dict, int]:
    """
    Create character to index and index to character mappings.

    Args:
        text: Input text string

    Returns:
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        vocab_size: Total number of unique characters
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, vocab_size

def encode_text(text: str, char_to_idx: dict) -> np.ndarray:
    """
    Encode text to integer indices.

    Args:
        text: Input text string
        char_to_idx: Character to index mapping

    Returns:
        Encoded text as numpy array of integers
    """
    return np.array([char_to_idx[ch] for ch in text], dtype=np.int32)

def decode_text(indices: np.ndarray, idx_to_char: dict) -> str:
    """
    Decode integer indices to text.

    Args:
        indices: Array of integer indices
        idx_to_char: Index to character mapping

    Returns:
        Decoded text string
    """
    # Handle out-of-vocabulary indices gracefully
    return ''.join([idx_to_char.get(int(i), '?') for i in indices])

class TextGenerationDataLoader:
    """
    Data loader for character-level text generation.
    Creates sliding windows of text with input (x[t]) and target (x[t+1]) pairs.
    """

    def __init__(self, filepath: str, seq_len: int, batch_size: int, split: str = 'train', train_split: float = 0.9):
        """
        Initialize text generation data loader.

        Args:
            filepath: Path to text file
            seq_len: Length of each sequence
            batch_size: Number of sequences per batch
            split: 'train' or 'val'
            train_split: Fraction of data to use for training
        """
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.split = split

        # Load and encode text
        text = load_text_file(filepath)
        self.char_to_idx, self.idx_to_char, self.vocab_size = create_char_mappings(text)
        self.encoded_text = encode_text(text, self.char_to_idx)

        # Split into train/val
        split_idx = int(len(self.encoded_text) * train_split)
        if split == 'train':
            self.data = self.encoded_text[:split_idx]
        else:
            self.data = self.encoded_text[split_idx:]

        # Calculate number of complete sequences
        self.num_sequences = (len(self.data) - 1) // seq_len
        self.num_batches = self.num_sequences // batch_size

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Iterate over batches of (input, target) pairs.

        Yields:
            inputs: (batch_size, seq_len) array of input tokens
            targets: (batch_size, seq_len) array of target tokens (shifted by 1)
        """
        # Shuffle indices for training
        if self.split == 'train':
            indices = np.random.permutation(self.num_sequences)
        else:
            indices = np.arange(self.num_sequences)

        for batch_idx in range(self.num_batches):
            batch_inputs = []
            batch_targets = []

            for i in range(self.batch_size):
                seq_idx = indices[batch_idx * self.batch_size + i]
                start_pos = seq_idx * self.seq_len

                # Extract input and target sequences
                input_seq = self.data[start_pos:start_pos + self.seq_len]
                target_seq = self.data[start_pos + 1:start_pos + self.seq_len + 1]

                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)

            inputs = jnp.array(np.stack(batch_inputs), dtype=jnp.int32)
            targets = jnp.array(np.stack(batch_targets), dtype=jnp.int32)

            yield inputs, targets

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches

def get_text_dataloaders(filepath: str, seq_len: int, batch_size: int, train_split: float = 0.9):
    """
    Create train and validation data loaders for text generation.

    Args:
        filepath: Path to text file
        seq_len: Length of each sequence
        batch_size: Number of sequences per batch
        train_split: Fraction of data to use for training

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab_size: Size of vocabulary
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    train_loader = TextGenerationDataLoader(filepath, seq_len, batch_size, split='train', train_split=train_split)
    val_loader = TextGenerationDataLoader(filepath, seq_len, batch_size, split='val', train_split=train_split)

    return train_loader, val_loader, train_loader.vocab_size, train_loader.char_to_idx, train_loader.idx_to_char
