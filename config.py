class ModelConfig:
    vocab_size = 256    # Byte-Level (0-255)
    hidden_dim = 256    # 128 -> 256 (Kapasite artışı)
    num_layers = 6      # 4 -> 6 (Derinlik artışı)
    dropout_rate = 0.2  # 0.1 -> 0.2 (Overfitting önlemi)
    encoder_dense_units = 128 # YENİ: Encoder dense layer boyutu

class DataConfig:
    task_name = 'text_modeling' # 'lra_listops' or 'text_modeling'
    seq_len = 2048      # Byte-Level için artırıldı
    batch_size = 16     # 32 -> 16 (OOM Fix)
    text_file_path = 'data/sonnet.txt' # Text task için dosya yolu
    imdb_seq_len = 1024 # IMDB için sequence length

class TextDataConfig:
    seq_len = 1024 # Text için belki daha kısa tutabiliriz veya aynı
    batch_size = 8

class TextGenConfig:
    """Configuration for text generation training (Shakespeare, code, etc.)"""
    dataset_path = 'data/shakespeare.txt'
    seq_len = 1024
    batch_size = 8  # Smaller batch for gradient accumulation (8 * 8 = 64 effective)
    num_steps = 5000
    eval_every = 200
    sample_every = 200  # Generate sample text every N steps
    accum_steps = 4  # Gradient accumulation steps (8 * 4 = 32 effective batch size)

class TrainingConfig:
    learning_rate = 1e-4  # 2e-4 -> 1e-4 (Daha stabil öğrenme)
    weight_decay = 0.1    # 0.01 -> 0.1 (Güçlü regülarizasyon)
    gradient_clip_value = 1.0 # YENİ: Gradyan kırpma eşiği
    warmup_steps = 2000
    num_steps = 20000
    eval_every = 200
    seed = 42
    accum_steps = 8       # YENİ: 16 * 8 = 128 Efektif Batch Size
    label_smoothing = 0.1 # YENİ: Etiket yumuşatma faktörü

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.text_gen = TextGenConfig()