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

def load_lra_dataset(split, seq_len, batch_size):
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
        # Başlık satırını atla (TSV formatında genelde başlık olur mu? LRA'da Source, Target var)
        # LRA ListOps formatı: "Label \t Source" veya "Source \t Label" olabilir.
        # Genellikle: Label \t Sequence
        
        first_line = True
        for line in f:
            if first_line:
                # Başlık kontrolü (opsiyonel, veri setine bağlı)
                if "Source" in line:
                    first_line = False
                    continue
                first_line = False
            
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            # ListOps formatı: Label <tab> Sequence
            # Örnek: 2 <tab> (MAX 2 3 )
            try:
                label = int(parts[0])
                text = parts[1]
                
                # Tokenize (Byte-Level)
                tokenized_text = tokenize(text, seq_len)
                
                texts.append(tokenized_text)
                labels.append(label)
            except ValueError:
                continue # Hatalı satırları atla
                
    # Numpy array'e çevir
    X = np.array(texts, dtype=np.uint8) # (N, L) uint8
    y = np.array(labels, dtype=np.int32) # (N,)
    
    print(f"{split} data loaded. Shape: {X.shape}")
    
    # tf.data.Dataset oluştur (Batching ve Shuffle için)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=10000)
        
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # JAX uyumluluğu için numpy iterator'a çevir (tf.data -> numpy)
    # Not: tfds.as_numpy(dataset) de kullanılabilir ama manuel döngü daha güvenli olabilir.
    
    return dataset
