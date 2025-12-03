import os
import numpy as np
import tensorflow as tf

class TextDataset:
    def __init__(self, file_path, seq_len, batch_size, repeat=False):
        self.file_path = file_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.repeat = repeat
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
            
        with open(file_path, 'rb') as f:
            self.data = f.read()
            
        self.total_len = len(self.data)
        print(f"Loaded text file: {file_path} ({self.total_len} bytes)")

    def _generator(self):
        # Basit bir kayan pencere (sliding window) veya chunking mantığı
        # Overfitting testi için genellikle dosyanın başından sonuna kadar gidip başa dönmek istenir.
        # Next token prediction: Input=x[t], Target=x[t+1]
        
        # Veriyi uint8 array'e çevir
        data_arr = np.frombuffer(self.data, dtype=np.uint8)
        
        # Basit chunking: seq_len + 1 (input + target için)
        # Örnek: "HELLO" -> Input="HELL", Target="ELLO"
        
        idx = 0
        while True:
            if idx + self.seq_len + 1 > self.total_len:
                if self.repeat:
                    idx = 0 # Başa dön
                else:
                    break # Bitir
            
            chunk = data_arr[idx : idx + self.seq_len + 1]
            
            # Input ve Target ayır
            x = chunk[:-1]
            y = chunk[1:]
            
            # Padding gerekirse (son chunk için, ama yukarıdaki mantıkta tam chunk alıyoruz)
            # Eğer veri çok kısaysa ve seq_len'den küçükse padding lazım olabilir.
            if len(x) < self.seq_len:
                 # Padding (0)
                pad_len = self.seq_len - len(x)
                x = np.pad(x, (0, pad_len), 'constant')
                y = np.pad(y, (0, pad_len), 'constant')

            yield {'input': x, 'label': y}
            
            idx += self.seq_len # Non-overlapping chunks
            # idx += 1 # Overlapping chunks (daha fazla veri üretir ama yavaş olabilir)

    def get_dataset(self):
        output_signature = {
            'input': tf.TensorSpec(shape=(self.seq_len,), dtype=tf.uint8),
            'label': tf.TensorSpec(shape=(self.seq_len,), dtype=tf.uint8)
        }
        
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=output_signature
        )
        
        if self.repeat:
            dataset = dataset.repeat()
            
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        # Prefetch performans için
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def get_text_dataloader(file_path, seq_len, batch_size, repeat=True):
    """Helper function to get numpy iterator."""
    ds = TextDataset(file_path, seq_len, batch_size, repeat).get_dataset()
    return ds.as_numpy_iterator()
