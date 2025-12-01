class ModelConfig:
    vocab_size = 256    # Byte-Level (0-255)
    hidden_dim = 256    # 128 -> 256 (Kapasite artışı)
    num_layers = 6      # 4 -> 6 (Derinlik artışı)
    dropout_rate = 0.1

class DataConfig:
    task_name = 'lra_listops'
    seq_len = 2048      # Byte-Level için artırıldı
    batch_size = 16     # 32 -> 16 (OOM Fix)

class TrainingConfig:
    learning_rate = 2e-4
    weight_decay = 0.01   # 0.05 -> 0.01 (Daha rahat öğrenme için düşürüldü)
    warmup_steps = 2000
    num_steps = 20000
    eval_every = 200
    seed = 42
    accum_steps = 8       # YENİ: 16 * 8 = 128 Efektif Batch Size

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()