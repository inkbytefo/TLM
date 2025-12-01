# Spectral-JAX Research Roadmap
## 12-Week Implementation Plan for JAX-Based Spectral Sequence Modeling

**Project Lead**: [Your Name]  
**Timeline**: 12 Weeks (3 Phases)  
**Compute Requirements**: 1x TPU v4 Pod Slice or 8x A100 GPUs  

---

## 🎯 PROJECT OBJECTIVES

1. **Scientific**: Validate that selective spectral mixing achieves competitive performance on LRA benchmark
2. **Engineering**: Build production-ready JAX/Flax implementation with <1000 lines of core code
3. **Publication**: Submit to ICLR 2025 or NeurIPS 2025 workshop track

---

## 📅 PHASE 1: FOUNDATION (Weeks 1-4)

### Week 1: Environment Setup & Literature Deep Dive

**Tasks**:
- [ ] Set up JAX development environment (TPU access via Google Cloud or Colab Pro+)
- [ ] Install Flax, Optax, TensorFlow Datasets (TFDS)
- [ ] Clone and study reference repositories:
  - [google-research/long-range-arena](https://github.com/google-research/long-range-arena)
  - [state-spaces/mamba](https://github.com/state-spaces/mamba) (for S4/Mamba reference)
  - [HazyResearch/safari](https://github.com/HazyResearch/safari) (Hyena implementation)

**Deliverables**:
- Annotated bibliography (15-20 papers) with key takeaways
- Working JAX environment with all dependencies
- Initial project structure:

```
spectral-jax/
├── spectral_jax/
│   ├── __init__.py
│   ├── layers.py          # Core Spectral Block
│   ├── models.py          # Full model definitions
│   ├── data.py            # LRA dataloaders
│   └── train_utils.py     # Training loop helpers
├── configs/
│   ├── lra_listops.yaml
│   ├── lra_text.yaml
│   └── base_model.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── sweep.py           # Hyperparameter sweeps
├── tests/
│   ├── test_layers.py
│   └── test_gradients.py
├── notebooks/
│   └── 01_fft_visualization.ipynb
├── docs/
│   ├── whitepaper.md
│   ├── math_proofs.md
│   └── README.md
├── requirements.txt
├── setup.py
└── .pre-commit-config.yaml  # Code quality checks
```

---

### Week 2: Core Layer Implementation

**Goal**: Implement `SpectralBlock` layer in pure JAX

**Code Structure** (`spectral_jax/layers.py`):

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

class SpectralBlock(nn.Module):
    """
    Single Spectral-JAX block with selective filtering.
    
    Architecture:
        Input → FFT → Selective Filter → iFFT → Energy Gate → Output
    """
    hidden_dim: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            train: bool for dropout
        Returns:
            y: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, d_model = x.shape
        
        # Stage 1: Forward FFT
        x_freq = jnp.fft.rfft(x, axis=1)  # [B, N//2+1, D]
        
        # Stage 2: Selective Filter (Input-Dependent)
        # Use global average as context
        context = jnp.mean(x, axis=1, keepdims=True)  # [B, 1, D]
        
        # MLP to generate frequency-dependent weights
        filter_mlp = nn.Sequential([
            nn.Dense(self.hidden_dim * self.mlp_ratio),
            nn.gelu,
            nn.Dropout(self.dropout_rate, deterministic=not train),
            nn.Dense(self.hidden_dim)
        ])
        
        filter_weights = filter_mlp(context)  # [B, 1, D]
        
        # Broadcast to frequency dimension
        filter_weights = jnp.tile(filter_weights, (1, x_freq.shape[1], 1))
        
        # Apply filter in frequency domain
        x_freq_filtered = x_freq * filter_weights
        
        # Stage 3: Inverse FFT
        x_time = jnp.fft.irfft(x_freq_filtered, n=seq_len, axis=1)
        
        # Stage 4: Energy Gate
        energy = jnp.linalg.norm(x_time, axis=-1, keepdims=True)  # [B, N, 1]
        gate = nn.sigmoid(energy)
        
        output = x_time * gate
        
        return output

class SpectralJAXModel(nn.Module):
    """Full Spectral-JAX Model"""
    vocab_size: int
    num_layers: int = 6
    hidden_dim: int = 512
    num_classes: int = 2
    
    @nn.compact
    def __call__(self, tokens, train: bool = True):
        # Embedding
        x = nn.Embed(self.vocab_size, self.hidden_dim)(tokens)
        
        # Add positional encoding (sinusoidal)
        pos_enc = self._positional_encoding(x.shape[1], self.hidden_dim)
        x = x + pos_enc
        
        # Stack Spectral Blocks
        for _ in range(self.num_layers):
            x = SpectralBlock(self.hidden_dim)(x, train=train)
            x = nn.LayerNorm()(x)  # Stabilize training
        
        # Classification head
        x = jnp.mean(x, axis=1)  # Global average pooling
        x = nn.Dense(self.num_classes)(x)
        
        return x
    
    def _positional_encoding(self, seq_len, d_model):
        # Standard sinusoidal encoding
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * 
                          -(jnp.log(10000.0) / d_model))
        
        pe = jnp.zeros((seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return pe[None, :, :]  # [1, seq_len, d_model]
```

**Tests** (`tests/test_layers.py`):

```python
import jax
import jax.numpy as jnp
from spectral_jax.layers import SpectralBlock

def test_forward_pass():
    """Test that layer executes without errors"""
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 128, 256))  # [B, N, D]
    
    layer = SpectralBlock(hidden_dim=256)
    params = layer.init(rng, x, train=False)
    
    output = layer.apply(params, x, train=False)
    
    assert output.shape == x.shape
    assert jnp.isfinite(output).all()

def test_gradient_flow():
    """Ensure gradients don't explode/vanish"""
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 128, 256))
    
    layer = SpectralBlock(hidden_dim=256)
    
    def loss_fn(params):
        out = layer.apply(params, x, train=True)
        return jnp.mean(out ** 2)
    
    params = layer.init(rng, x, train=False)
    grads = jax.grad(loss_fn)(params)
    
    # Check no NaN/Inf gradients
    flat_grads = jax.tree_util.tree_leaves(grads)
    assert all(jnp.isfinite(g).all() for g in flat_grads)

if __name__ == "__main__":
    test_forward_pass()
    test_gradient_flow()
    print("✅ All tests passed!")
```

**Deliverables**:
- Working `SpectralBlock` implementation
- Passing unit tests
- Gradient stability verification

---

### Week 3: Dataloader & Training Infrastructure

**Tasks**:
1. Implement LRA dataloaders using TFDS
2. Set up training loop with Optax
3. Add logging (Weights & Biases integration)

**Key Files**:

`spectral_jax/data.py`:
```python
import tensorflow_datasets as tfds
import jax.numpy as jnp

def load_lra_dataset(task_name, batch_size=32, seq_len=1024):
    """
    Load LRA benchmark tasks.
    
    Args:
        task_name: 'listops', 'text', 'retrieval', 'pathfinder', 'pathfinder_x'
        batch_size: Batch size
        seq_len: Maximum sequence length
    
    Returns:
        train_ds, val_ds, test_ds: JAX-compatible datasets
    """
    # Load raw data from TFDS
    ds = tfds.load(f'lra_{task_name}', split='train')
    
    def preprocess(example):
        # Tokenize, pad, truncate to seq_len
        tokens = example['input_ids'][:seq_len]
        tokens = jnp.pad(tokens, (0, seq_len - len(tokens)))
        label = example['label']
        return {'tokens': tokens, 'label': label}
    
    ds = ds.map(preprocess).batch(batch_size).prefetch(2)
    
    return ds

# Similar for val/test splits
```

`scripts/train.py`:
```python
import jax
import optax
from spectral_jax.models import SpectralJAXModel
from spectral_jax.data import load_lra_dataset
import wandb

def train_model(config):
    # Initialize model
    model = SpectralJAXModel(**config['model'])
    
    # Load data
    train_ds = load_lra_dataset(config['task'], batch_size=config['batch_size'])
    
    # Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=config['learning_rate'],
        warmup_steps=2000,
        decay_steps=100000
    )
    optimizer = optax.adamw(schedule, weight_decay=0.01)
    
    # Training loop
    rng = jax.random.PRNGKey(config['seed'])
    
    # Initialize parameters
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)
    opt_state = optimizer.init(params)
    
    # JIT-compiled training step
    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(params):
            logits = model.apply(params, batch['tokens'], train=True)
            labels_one_hot = jax.nn.one_hot(batch['label'], config['num_classes'])
            loss = -jnp.sum(labels_one_hot * jax.nn.log_softmax(logits))
            return jnp.mean(loss)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    # Main loop
    for epoch in range(config['num_epochs']):
        for batch in train_ds:
            params, opt_state, loss = train_step(params, opt_state, batch)
            
            wandb.log({'train/loss': float(loss)})
    
    return params

if __name__ == "__main__":
    config = {
        'task': 'listops',
        'model': {'vocab_size': 20, 'num_layers': 4, 'hidden_dim': 256},
        'batch_size': 32,
        'seq_len': 2048,
        'learning_rate': 3e-4,
        'num_epochs': 50,
        'seed': 42
    }
    
    wandb.init(project='spectral-jax', config=config)
    trained_params = train_model(config)
```

**Deliverables**:
- Working data pipeline for at least 2 LRA tasks (ListOps, Text)
- End-to-end training script that runs
- WandB dashboard showing loss curves

---

### Week 4: Sanity Checks & Overfitting Tests

**Goal**: Verify model can learn (overfit on small dataset)

**Experiments**:
1. **Memorization Test**: Train on 100 examples of ListOps
   - Expected: 100% train accuracy within 500 steps
   - If not achieved → architecture bug

2. **Gradient Magnitude Tracking**: Log $\|\nabla_{\theta} \mathcal{L}\|$ every 10 steps
   - Should stabilize around $10^{-3}$ to $10^{-1}$
   - If exploding → add gradient clipping

3. **Spectral Analysis**: Visualize learned filters $H_{\theta}$
   - Plot frequency response at different training steps
   - Expected: Low frequencies active early, high frequencies emerge later

**Notebook** (`notebooks/01_sanity_checks.ipynb`):
```python
# Visualize frequency response evolution
import matplotlib.pyplot as plt

def plot_frequency_response(params, epoch):
    # Extract filter weights from params
    H = params['SpectralBlock_0']['filter_mlp']['Dense_1']['kernel']
    
    # Compute FFT of impulse response
    h_time = jnp.fft.ifft(H)
    
    plt.figure(figsize=(10, 4))
    plt.plot(jnp.abs(H))
    plt.title(f'Frequency Response at Epoch {epoch}')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.savefig(f'filter_epoch_{epoch}.png')

# Call after each epoch in training loop
```

**Deliverables**:
- Confirmed overfitting on toy dataset
- Stable gradient norms
- Frequency response visualizations showing learning

---

## 📅 PHASE 2: VALIDATION (Weeks 5-8)

### Week 5-6: LRA Benchmark Implementation

**Tasks**:
- Implement all 5 LRA tasks (ListOps, Text, Retrieval, Image, Pathfinder-X)
- Tune hyperparameters per task (use Optuna for automated search)
- Run baseline comparisons (Transformer, Linear Transformer)

**Hyperparameter Ranges** (from LRA paper):
```yaml
# configs/lra_sweep.yaml
search_space:
  learning_rate: [1e-4, 3e-4, 1e-3]
  hidden_dim: [256, 512]
  num_layers: [4, 6, 8]
  mlp_ratio: [2, 4]
  dropout_rate: [0.0, 0.1, 0.2]
```

**Expected Results** (Based on LRA Benchmark):

| Model | ListOps | Text | Retrieval | Image | Avg |
|-------|---------|------|-----------|-------|-----|
| Transformer | 36.4 | 64.3 | 57.5 | 42.4 | 50.2 |
| Linformer | 35.7 | 53.9 | 52.3 | 38.6 | 45.1 |
| S4 | 58.3 | 76.0 | 87.1 | 88.7 | 77.5 |
| **Spectral-JAX (Target)** | **>55** | **>70** | **>80** | **>85** | **>72** |

---

### Week 7: Performance Profiling

**Goal**: Optimize for speed and memory

**Tasks**:
1. Profile training speed (tokens/sec)
2. Memory usage analysis (peak VRAM)
3. Comparison with FlashAttention baseline

**Tools**:
```python
# Use JAX's built-in profiler
from jax.profiler import trace

with trace('/tmp/jax-trace'):
    train_step(params, batch)

# Analyze with TensorBoard
```

**Optimization Techniques**:
- Mixed precision (bfloat16)
- Gradient checkpointing for long sequences
- Fused kernels for FFT+multiply operations

---

### Week 8: Ablation Studies

**Research Questions**:
1. **Q**: Does input-dependent filtering matter?
   - **Experiment**: Compare Spectral-JAX with fixed filters
   
2. **Q**: Is energy gating necessary?
   - **Experiment**: Remove $\sigma(\|Y\|_2)$ gate
   
3. **Q**: How does performance scale with sequence length?
   - **Experiment**: Test on N = [1K, 2K, 4K, 8K, 16K, 32K, 64K]

**Expected Findings**:
- Input-dependent filtering: +5-10% accuracy on complex tasks
- Energy gating: Prevents training divergence on long sequences
- Scaling: Linear memory, sublinear time (due to FFT)

---

## 📅 PHASE 3: PUBLICATION (Weeks 9-12)

### Week 9-10: Paper Writing

**Target Conference**: ICLR 2025 (Deadline: ~October 2024) or NeurIPS Workshop

**Paper Outline** (8 pages + appendix):

1. **Abstract** (150 words)
2. **Introduction** (1.5 pages)
   - Problem: Transformer inefficiency
   - Contribution: Selective spectral mixing
   - Results preview: LRA benchmark scores
3. **Related Work** (1 page)
   - Transformers & Efficient Variants
   - State Space Models (S4, Mamba)
   - Fourier Neural Operators
4. **Method** (2.5 pages)
   - Spectral Layer Derivation
   - Architecture Details
   - Complexity Analysis
5. **Experiments** (2 pages)
   - LRA Benchmark Results
   - Ablation Studies
   - Scaling Analysis
6. **Discussion & Limitations** (0.5 pages)
7. **Conclusion** (0.5 pages)
8. **Appendix**
   - Mathematical Proofs
   - Hyperparameter Tables
   - Additional Visualizations

---

### Week 11: Code Release & Documentation

**Tasks**:
1. Clean up codebase (remove debugging code)
2. Write comprehensive README with:
   - Installation instructions
   - Quick start example
   - Reproducing paper results
3. Add pre-trained checkpoints to HuggingFace Hub
4. Create demo notebook (Colab-compatible)

**Documentation Structure**:
```markdown
# Spectral-JAX

Official implementation of "Spectral-JAX: Energy-Gated Spectral Networks for Sequence Modeling"

## Installation
```bash
pip install spectral-jax
```

## Quick Start
```python
from spectral_jax import SpectralJAXModel

model = SpectralJAXModel(vocab_size=10000, num_layers=6)
# ... training code
```

## Reproducing Results
See `scripts/reproduce_lra.sh` for full LRA benchmark.

## Citation
```bibtex
@inproceedings{spectral-jax-2025,
  title={Spectral-JAX: Energy-Gated Spectral Networks},
  author={Your Name},
  booktitle={ICLR},
  year={2025}
}
```
```

---

### Week 12: Submission & Backup Experiments

**Tasks**:
1. Submit paper to ICLR/NeurIPS
2. Prepare rebuttal document (anticipate reviewer questions)
3. Run additional experiments if needed:
   - Language modeling (WikiText-103)
   - DNA sequence classification
   - Vision Transformers with Spectral-JAX

**Contingency Plan**:
If results don't meet expectations:
- Pivot to workshop paper (lower bar)
- Focus on "negative results" (what didn't work and why)
- Submit to ICML 2025 instead (later deadline)

---

## 🛠️ TOOLING & INFRASTRUCTURE

### Required Resources

**Compute**:
- **Option A** (Recommended): 1x TPU v4-8 pod slice (~$400/month on Google Cloud)
- **Option B**: 4-8× NVIDIA A100 GPUs (university cluster or Lambda Labs)
- **Option C** (Budget): Google Colab Pro+ with T4 GPU ($50/month) + longer runtimes

**Storage**:
- 500GB for datasets (LRA, WikiText)
- 100GB for checkpoints & logs

**Software Stack**:
```txt
# requirements.txt
jax[cuda]==0.4.20  # or jax[tpu]
flax==0.7.5
optax==0.1.7
tensorflow-datasets==4.9.3
wandb==0.16.0
pytest==7.4.3
matplotlib==3.8.2
seaborn==0.13.0
```

---

## 📊 SUCCESS METRICS

### Minimum Viable Results (for publication)
- ✅ LRA Average Score ≥ 70% (matches S4)
- ✅ Training 2× faster than Transformer at seq_len=8K
- ✅ Memory usage ≤ Transformer up to seq_len=16K

### Stretch Goals
- 🎯 LRA Average Score ≥ 75% (beats S4)
- 🎯 Successful language modeling (perplexity < 20 on WikiText-103)
- 🎯 Open-source adoption (>100 GitHub stars within 3 months)

---

## ⚠️ RISK MITIGATION

### Potential Issues & Solutions

1. **Problem**: Can't match Transformer accuracy
   - **Solution**: Hybrid model (Spectral-JAX + sparse local attention)

2. **Problem**: Training instability
   - **Solution**: Add LayerNorm after each block, reduce learning rate

3. **Problem**: FFT too slow on TPU
   - **Solution**: Use JAX's `jax.lax.fft` instead of `jnp.fft` (hardware-optimized)

4. **Problem**: Reviewer skepticism ("just another efficient Transformer")
   - **Solution**: Emphasize theoretical contributions (connection to SSMs, universal approximation)

---

## 📚 ADDITIONAL RESOURCES

### Must-Read Papers (Beyond Initial List)
- "Attention is All You Need" (Vaswani et al., 2017) - baseline
- "How to Train Your HiPPO" (Gu et al., 2020) - S4 initialization
- "FlashAttention" (Dao et al., 2022) - speed benchmark
- "On the Spectral Bias of Neural Networks" (Rahaman et al., 2019) - theory

### Helpful JAX Tutorials
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Flax NNX Tutorial](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [JAX on TPU Guide](https://cloud.google.com/tpu/docs/jax-quickstart)

---

## 🎉 CONCLUSION

This roadmap provides a structured 12-week path to:
1. Implement a novel spectral sequence model in JAX
2. Validate it on standard benchmarks
3. Publish in a top-tier venue

**Key Success Factors**:
- Rigorous testing at each stage
- Daily progress tracking (use GitHub Projects)
- Willingness to pivot if experiments fail

**Good luck, and may your gradients be ever stable! 🚀**

---

**END OF ROADMAP**