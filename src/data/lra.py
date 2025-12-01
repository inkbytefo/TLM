import os
import requests
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm

# ListOps Veri Seti URL'leri
LISTOPS_URLS = {
    'train': 'https://storage.googleapis.com/long-range-arena/lra_release/listops-1000/basic_train.tsv',
    'validation': 'https://storage.googleapis.com/long-range-arena/lra_release/listops-1000/basic_val.tsv',
    'test': 'https://storage.googleapis.com/long-range-arena/lra_release/listops-1000/basic_test.tsv'
}

def download_file(url, dest_path):
    """Dosyayı indirir (Progress bar ile)."""
    # Eğer dosya varsa ve boyutu çok küçükse (muhtemelen hata mesajı), sil
    if os.path.exists(dest_path):
        if os.path.getsize(dest_path) < 1000:
            print(f"File {dest_path} is too small (<1KB), deleting and redownloading...")
            os.remove(dest_path)
        else:
            return
    
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status() # Hata durumunda (404, 403 vs) exception fırlat
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def tokenize(text, seq_len):
    """Metni Raw Byte (uint8) formatına çevirir ve pad'ler."""
    # 1. String -> UTF-8 Bytes
    byte_data = text.encode('utf-8')
    
    # 2. Bytes -> Numpy Array (uint8)
    # np.frombuffer, byte string üzerinde doğrudan çalışır ve çok hızlıdır.
    tokens = np.frombuffer(byte_data, dtype=np.uint8)
    
    # 3. Truncate (Eğer çok uzunsa kes)
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
        
    # 4. Padding (0 ile doldur - NULL byte)
    if len(tokens) < seq_len:
        # Pad değeri 0 (NULL byte)
        padding = np.zeros(seq_len - len(tokens), dtype=np.uint8)
        tokens = np.concatenate([tokens, padding])
        
    return tokens

def get_lra_dataloader(split, seq_len, batch_size, task_name=None, repeat=False):
    """LRA ListOps veri setini indirir ve yükler (Byte-Level)."""
    
    # Veri klasörünü oluştur
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/lra_listops')
    os.makedirs(data_dir, exist_ok=True)
    
    # Dosya yolu
    file_path = os.path.join(data_dir, f'basic_{split}.tsv')
    
    # İndir
    download_file(LISTOPS_URLS[split], file_path)
    
    # Yükle ve İşle
    texts = []
    labels = []
    
    print(f"Loading and tokenizing {split} data (Byte-Level)...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Header parsing
        header_line = f.readline().strip()
        headers = header_line.split('\t')
        
        source_idx = 0
        target_idx = 1
        
        if "Source" in headers and "Target" in headers:
            source_idx = headers.index("Source")
            target_idx = headers.index("Target")
            print(f"Detected columns: Source at {source_idx}, Target at {target_idx}")
        else:
            print(f"Header not found or unrecognized: {header_line}. Assuming Source at 0, Target at 1.")
            # Reset file pointer if no header (unlikely for LRA but good safety)
            # But readline() consumed it. If it wasn't a header, we lost data.
            # LRA files usually have headers. Let's assume standard format if header missing.
            if "Source" not in header_line:
                 # If the first line doesn't look like a header, maybe it's data?
                 # But we already consumed it. Re-opening or seeking is better.
                 f.seek(0)
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            try:
                # Parse based on detected or default indices
                text = parts[source_idx]
                label = int(parts[target_idx])
                
                tokenized_text = tokenize(text, seq_len)
                texts.append(tokenized_text)
                labels.append(label)
            except (ValueError, IndexError):
                continue
                
    X = np.array(texts, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)
    
    print(f"{split} data loaded. Shape: {X.shape}")
    
    if len(X) == 0:
        raise ValueError(f"No data loaded for {split}. Check file format.")

    # tf.data.Dataset oluştur
    dataset = tf.data.Dataset.from_tensor_slices({'input': X, 'label': y})
    
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=10000)
        
    if repeat:
        dataset = dataset.repeat()
        
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Numpy Iterator'a çevir
    import tensorflow_datasets as tfds
    return iter(tfds.as_numpy(dataset))
