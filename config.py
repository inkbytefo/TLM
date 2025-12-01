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
    learning_rate = 2e-4  # 4e-4 -> 2e-4 (Daha güvenli)
    weight_decay = 0.05   # 0.01 -> 0.05 (Overfitting'i engellemek için artırıldı)
    warmup_steps = 2000   # 1000 -> 2000 (Daha yavaş ısınma)
    num_steps = 20000     # 10000 -> 20000 (Daha uzun eğitim)
    eval_every = 200    
    seed = 42

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()