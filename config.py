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

class AgentConfig:
    """Configuration for agent tool use and code execution"""
    max_exec_len = 256  # Maximum length of code block in characters
    tool_start_token = "<EXEC>"  # Token that marks start of code execution
    tool_end_token = "</EXEC>"  # Token that marks end of code execution
    max_iterations = 5  # Maximum number of tool execution loops
    max_tokens_per_iteration = 200  # Max tokens to generate before checking for tool use
    execution_timeout = 5  # Timeout for code execution in seconds
    temperature = 0.7  # Sampling temperature for agent generation
    dataset_path = 'data/agent_dataset.txt'  # Path to agent training dataset
    num_train_examples = 1000  # Number of synthetic training examples to generate

    # Training specific parameters
    seq_len = 512  # Shorter sequences for agent training (agent examples are concise)
    batch_size = 16  # Larger batch size for synthetic data
    num_steps = 3000  # Fewer steps needed (synthetic data is easier to learn)
    learning_rate = 2e-4  # Slightly higher LR for faster learning on synthetic data
    accum_steps = 2  # Gradient accumulation (16 * 2 = 32 effective batch)

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
        self.agent = AgentConfig()