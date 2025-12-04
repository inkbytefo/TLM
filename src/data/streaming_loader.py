import numpy as np
import os
import jax.numpy as jnp

def prepare_data(input_file_path, output_dir, vocab_size=256):
    """
    Reads a text file and saves it as a numpy memmap file (uint8 or uint16).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.basename(input_file_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}.bin")
    
    print(f"Processing {input_file_path}...")
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Simple byte-level encoding if vocab_size <= 256
    # For larger vocabs, you'd need a tokenizer here.
    # Assuming byte-level for now as per project specs.
    if vocab_size <= 256:
        # Encode as bytes directly
        ids = [ord(c) % 256 for c in data]
        dtype = np.uint8
    else:
        # Simple char-level fallback
        chars = sorted(list(set(data)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        ids = [char_to_idx[c] for c in data]
        dtype = np.uint16
        
    # Save as binary
    ids = np.array(ids, dtype=dtype)
    ids.tofile(output_path)
    
    print(f"Saved {len(ids)} tokens to {output_path}")
    return output_path, len(ids)

class MemmapDataLoader:
    """
    Efficient Data Loader using Numpy Memmap.
    Allows training on datasets larger than RAM.
    """
    def __init__(self, bin_path, batch_size, seq_len, split='train', val_split=0.1):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        
        # Load data as memmap (Zero RAM usage)
        # Determine dtype by file size or assume uint8 for now
        # Ideally metadata should be saved alongside, but we'll assume uint8 for byte-level
        self.data = np.memmap(bin_path, dtype=np.uint8, mode='r')
        
        total_tokens = len(self.data)
        val_tokens = int(total_tokens * val_split)
        train_tokens = total_tokens - val_tokens
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_tokens
        else:
            self.start_idx = train_tokens
            self.end_idx = total_tokens
            
        self.num_tokens = self.end_idx - self.start_idx
        print(f"[{split}] Loaded {self.num_tokens / 1e6:.2f}M tokens from {bin_path}")

    def __iter__(self):
        return self

    def __next__(self):
        # Random sampling for training
        if self.split == 'train':
            # Generate random start positions
            # Ensure we don't go out of bounds
            high = self.end_idx - self.seq_len - 1
            if high <= self.start_idx:
                 raise ValueError("Dataset too small for requested sequence length")
                 
            ix = np.random.randint(self.start_idx, high, size=(self.batch_size,))
            
            x_batch = []
            y_batch = []
            
            for i in ix:
                # Slicing memmap reads from disk
                chunk = self.data[i:i+self.seq_len+1]
                x_batch.append(chunk[:-1])
                y_batch.append(chunk[1:])
                
            x = np.stack(x_batch).astype(np.int32)
            y = np.stack(y_batch).astype(np.int32)
            
            return {'input': x, 'label': y}
            
        else:
            # Sequential sampling for validation (simplified)
            # Just return random chunks for now to keep it stateless
            # A proper eval would iterate sequentially
            high = self.end_idx - self.seq_len - 1
            ix = np.random.randint(self.start_idx, high, size=(self.batch_size,))
            
            x_batch = []
            y_batch = []
            
            for i in ix:
                chunk = self.data[i:i+self.seq_len+1]
                x_batch.append(chunk[:-1])
                y_batch.append(chunk[1:])
                
            x = np.stack(x_batch).astype(np.int32)
            y = np.stack(y_batch).astype(np.int32)
            
            return {'input': x, 'label': y}

class MixedDataLoader:
    """
    Data Loader that mixes multiple MemmapDataLoaders with specified weights.
    Useful for Curriculum Learning (e.g., mixing Turkish, Code, and English).
    """
    def __init__(self, datasets, weights, batch_size, seq_len):
        """
        Args:
            datasets: List of dicts {'path': str, 'split': str}
            weights: List of floats (probabilities) summing to 1.0
            batch_size: Global batch size
            seq_len: Sequence length
        """
        self.loaders = []
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= self.weights.sum() # Normalize
        self.batch_size = batch_size
        
        for ds in datasets:
            loader = MemmapDataLoader(
                bin_path=ds['path'],
                batch_size=batch_size, # We request full batch from sub-loader
                seq_len=seq_len,
                split=ds.get('split', 'train')
            )
            self.loaders.append(loader)
            
        print(f"Initialized MixedDataLoader with {len(self.loaders)} datasets. Weights: {self.weights}")

    def __iter__(self):
        return self

    def __next__(self):
        # Sample a dataset index based on weights
        # We pick ONE dataset for this batch to avoid padding/masking issues between different sources
        # Alternatively, we could mix within batch, but that's complex for now.
        dataset_idx = np.random.choice(len(self.loaders), p=self.weights)
        selected_loader = self.loaders[dataset_idx]
        
        return next(selected_loader)
