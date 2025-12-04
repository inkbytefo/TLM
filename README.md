# Spectral-JAX: Hybrid Hyena-Attention Autonomous Agent

**Spectral-JAX** is a cutting-edge research framework for training autonomous agents using a **Hybrid Architecture** that combines the infinite context capabilities of **Hyena Operators** with the precise "needle-in-a-haystack" retrieval of **Sliding Window Attention**.

## 🚀 Key Features

- **Hybrid Architecture**: 
  - **Hyena Blocks**: For global context and sub-quadratic scaling.
  - **Sliding Window Attention**: Interleaved every 6 layers for high-resolution local focus.
- **Autonomous Agent Loop**: Built-in support for `THINK`, `SPEAK`, `WAIT`, and `SILENCE` tokens to enable complex reasoning and interaction.
- **JAX/Flax Implementation**: High-performance training on TPUs and GPUs.
- **Curriculum Learning**: Structured 3-phase training pipeline (Language -> Knowledge -> Agency).

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/inkbytefo/TLM.git
cd TLM

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### Phase 1: Pre-training (Hybrid Logic)

Train the model on language and code data to establish the base capabilities.

```bash
python train.py \
    --run_name phase1_hybrid \
    --data_paths data/turkish_academic.txt,data/github_code.txt \
    --data_weights 0.8,0.2 \
    --hidden_dim 256 \
    --num_layers 12 \
    --seq_len 1024
```

### Run the Agent

Interact with the trained agent in the autonomous loop.

```bash
python autonomous_agent.py
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
- [Agent System](docs/AGENT.md)

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` (coming soon) for details.

## 📄 License

Copyright (c) 2025 **Tevfik İşkın**. All Rights Reserved.
This project is proprietary and confidential. See the [LICENSE](LICENSE) file for details.
