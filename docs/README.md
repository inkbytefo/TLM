# Spectral-JAX: Autonomous Agent Framework

## Overview

Spectral-JAX is a cutting-edge autonomous agent framework built on JAX and Flax. It features a custom **SpectralGPT** architecture that integrates **Hyena Operators** for long-context processing and **Residual Memory Blocks** for persistent state management. The framework is designed to train and deploy agents capable of autonomous reasoning, tool use, and continuous learning.

## Key Features

- **SpectralGPT Architecture**: A decoder-only transformer alternative using Hyena blocks for efficient long-sequence modeling.
- **Residual Memory**: Integrated memory layers that allow the model to maintain and update internal state over time.
- **Autonomous Loop**: A robust agent loop that supports `THINK`, `SPEAK`, `WAIT`, and `SILENCE` modes.
- **Tool Use**: Built-in Python code execution capabilities via `<EXEC>` tags.
- **RAG Integration**: Retrieval-Augmented Generation using FAISS for long-term memory and knowledge retrieval.
- **Curriculum Learning**: Structured training stages from basic language modeling to full autonomy.

## Project Structure

```
TLM/
├── autonomous_agent.py    # Main autonomous agent loop
├── config.py              # Configuration management
├── train.py               # Training script
├── server.py              # API server for the agent
├── src/
│   ├── models/            # SpectralGPT, Hyena, Memory layers
│   ├── memory/            # RAG and memory management
│   ├── tools/             # Tool execution (Python executor)
│   ├── training/          # Trainer state and logic
│   └── utils/             # Utilities (logging, seeding)
└── docs/                  # Documentation
```

## Quick Start

### Prerequisites

- Python 3.8+
- JAX (with CUDA support recommended for training)
- Flax, Optax
- FAISS, SentenceTransformers

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Agent

To start the autonomous agent loop:

```bash
python autonomous_agent.py
```

### Training

To train the model from scratch or resume training:

```bash
python train.py --config config.py
```

## Documentation

- [User Guide](user_guide.md): Detailed instructions on training, configuration, and running the agent.
- [Architecture](architecture.md): Deep dive into SpectralGPT, Hyena blocks, and Memory layers.
- [API Reference](api_reference.md): Technical documentation for classes and functions.

## License

MIT License
