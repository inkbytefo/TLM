# Spectral-JAX R&D

## Overview
This project implements a Spectral-JAX architecture for sequence modeling, leveraging Fast Fourier Transforms (FFT) for efficient long-range dependency modeling. It is built using JAX and Flax.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd TLM
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To start training with default configuration:
```bash
python -m src.training.trainer
```

### Testing
To run unit tests:
```bash
pytest tests/
```

## Project Structure
- `src/`: Source code
    - `models/`: Spectral-JAX model components
    - `training/`: Training and evaluation loops
    - `data/`: Data loading utilities
    - `utils/`: Helper functions (FFT, logging)
- `configs/`: Hydra configuration files
- `tests/`: Unit tests
