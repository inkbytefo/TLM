import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jnp

def get_imdb_dataloader(batch_size=16, seq_len=1024, split='train', repeat=True):
    """
    Loads IMDB Reviews dataset using TFDS and converts text to byte sequences.
    """
    # Load dataset
    # split='train' or 'test' (IMDB has train/test splits)
    ds = tfds.load('imdb_reviews', split=split, shuffle_files=True)
    
    def process_example(example):
        text = example['text']
        label = example['label']
        
        # Convert string to bytes
        # TF strings are tensors, we need to use tf.io.decode_raw or similar if possible,
        # but for simplicity in TF graph, we can use tf.strings.unicode_decode
        # However, a simpler way for byte-level is just casting to uint8 if it were raw bytes.
        # Since 'text' is a string tensor, we treat it as UTF-8.
        
        # 1. Decode UTF-8 to values (unicode code points) - but we want BYTES.
        # tf.io.decode_raw requires bytes input.
        # Let's use a py_function for robust text-to-byte conversion if needed,
        # or tf.strings.bytes_split (returns chars).
        
        # Efficient approach:
        # tf.strings.unicode_decode(text, 'UTF-8') gives code points.
        # We specifically want raw bytes. 
        # tf.io.decode_raw(text, tf.uint8) works if input is bytes.
        
        # In TFDS, 'text' is tf.string.
        bytes_seq = tf.io.decode_raw(text, tf.uint8)
        
        # Pad or Truncate
        # We want fixed length `seq_len`
        current_len = tf.shape(bytes_seq)[0]
        
        if current_len < seq_len:
            # Pad with 0
            paddings = [[0, seq_len - current_len]]
            bytes_seq = tf.pad(bytes_seq, paddings, constant_values=0)
        else:
            # Truncate
            bytes_seq = bytes_seq[:seq_len]
            
        # Ensure shape is set
        bytes_seq.set_shape([seq_len])
        
        return {'input': bytes_seq, 'label': label}

    # Apply processing
    ds = ds.map(process_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    if split == 'train':
        ds = ds.shuffle(buffer_size=10000)
        
    if repeat:
        ds = ds.repeat()
        
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # Convert to Numpy Iterator (compatible with JAX/Flax)
    return iter(tfds.as_numpy(ds))
