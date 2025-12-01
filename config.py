class ModelConfig:
    vocab_size = 20     
    hidden_dim = 256    # 128 -> 256 (Kapasite artışı)
    num_layers = 6      # 4 -> 6 (Derinlik artışı)
    dropout_rate = 0.1

class DataConfig:
    task_name = 'lra_listops'
    seq_len = 2048      
    batch_size = 32     

class TrainingConfig:
    learning_rate = 4e-4 # 1e-3 -> 4e-4 (Stabilite için düşürüldü)
    weight_decay = 0.01
    warmup_steps = 1000
    num_steps = 10000   # 5000 -> 10000 (Daha uzun eğitim)
    eval_every = 200    
    seed = 42

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()