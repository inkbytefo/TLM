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

# GÜNCELLENMİŞ VOCAB (Hata logundaki formata uygun)
VOCAB = {
    '<PAD>': 0,
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    '(': 11, ')': 12, 
    '[MAX': 13, '[MIN': 14, '[MED': 15, '[SM': 16, # [SM = SUM_MOD
    ']': 17 # Operatör kapatma parantezi
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
    """Metni token'lara çevirir ve pad'ler."""
    # Bilinmeyen tokenları 0 (PAD) olarak işaretle
    tokens = [VOCAB.get(t, 0) for t in text.split()]
    tokens = tokens[:seq_len]
    padding = [0] * (seq_len - len(tokens))
    return tokens + padding

def load_listops_data(split, seq_len, data_dir='./data/lra_listops'):
    """TSV dosyasını okur ve numpy array'e çevirir."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f'{split}.tsv')
    download_file(LISTOPS_URLS[split], file_path)
    
    inputs = []
    targets = []
    
    print(f"Processing {split} data...")
    with open(file_path, 'r') as f:
        # Read first few lines for debugging
        first_line = f.readline()
        print(f"First line of {split}: {first_line.strip()}")
        f.seek(0)
        
        header = next(f) # Header'ı atla (Source \t Target)
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            # DİNAMİK PARSING: Hangi sütunun label olduğunu anla
            p0, p1 = parts[0].strip(), parts[1].strip()
            
            # Eğer p1 kısa ve sayıysa, format: Sequence \t Label (LRA standardı)
            if len(p1) < 5 and p1.isdigit():
                text = p0
                label = int(p1)
            # Eğer p0 kısa ve sayıysa, format: Label \t Sequence
            elif len(p0) < 5 and p0.isdigit():
                label = int(p0)
                text = p1
            else:
                # Format bozuksa atla
                continue
            
            inputs.append(tokenize(text, seq_len))
            targets.append(label)
            
    print(f"Loaded {len(inputs)} samples for {split}")
    if len(inputs) == 0:
        raise ValueError(f"No data loaded for {split}! Check TSV format.")
            
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)

def get_lra_dataloader(task_name, batch_size, seq_len, split='train', repeat=True):
    """
    JAX uyumlu veri yükleyici.
    """
    if 'listops' not in task_name:
        raise NotImplementedError("Sadece 'lra_listops' şu an destekleniyor.")
        
    # Veriyi indir ve yükle
    inputs, targets = load_listops_data(split, seq_len)
    
    # TF Dataset pipeline (Shuffle, Batch, Prefetch)
    ds = tf.data.Dataset.from_tensor_slices({'input': inputs, 'label': targets})
    
    if split == 'train':
        ds = ds.shuffle(buffer_size=10000)
        
    if repeat:
        ds = ds.repeat()
        
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds.as_numpy_iterator()
