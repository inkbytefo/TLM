class ModelConfig:
    vocab_size = 20     # ListOps için (0-9, parantezler, operatörler)
    hidden_dim = 128    # LRA standardı
    num_layers = 4
    dropout_rate = 0.1

class DataConfig:
    task_name = 'lra_listops'
    seq_len = 2048      # ListOps max uzunluk
    batch_size = 32

class TrainingConfig:
    learning_rate = 1e-3
    weight_decay = 0.01
    warmup_steps = 1000
    num_steps = 5000    # Test için kısa, gerçek eğitim için 100k+
    eval_every = 100
    seed = 42

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()