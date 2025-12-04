# User Guide

This guide provides step-by-step instructions for configuring, training, and running the Spectral-JAX autonomous agent.

## Configuration

Configuration is managed in `config.py`. You can modify the `Config` class or pass arguments via command line.

### Key Configuration Options

*   **Model Config**:
    *   `hidden_dim`: Size of the hidden layers (default: 512).
    *   `num_layers`: Number of Hyena/Memory blocks (default: 6).
    *   `seq_len`: Maximum sequence length (default: 1024).
*   **Training Config**:
    *   `batch_size`: Training batch size.
    *   `lr`: Learning rate.
    *   `total_steps`: Total training steps.
*   **Agent Config**:
    *   `SILENCE_TOKEN`, `WAIT_TOKEN`, etc.: Special control tokens.

## Training the Model

To train the model, use `train.py`.

### 1. Prepare Data
Ensure your data is in the `data/` directory. The script handles tokenization and binary conversion.

### 2. Run Training
```bash
python train.py --batch_size 8 --lr 3e-4 --steps 50000
```

### 3. Monitoring
Training metrics are logged to **WandB** (Weights & Biases). Ensure you are logged in:
```bash
wandb login
```

## Running the Autonomous Agent

Once trained, you can run the agent in autonomous mode.

### 1. Start the Agent
```bash
python autonomous_agent.py
```

### 2. Interaction
The agent will start the loop. It may:
*   Print thoughts to the console.
*   Execute code if it decides to.
*   Wait for user input.

### 3. Server Mode
To expose the agent via an API:
```bash
python server.py
```
This starts a FastAPI server (default port 8000).

## Tool Usage

The agent can execute Python code.
*   **Syntax**: `<EXEC>print("Hello")</EXEC>`
*   **Safety**: The executor runs in a restricted scope. Avoid running untrusted models with full system access.

## Troubleshooting

*   **OOM Errors**: Reduce `batch_size` or `seq_len` in `config.py`.
*   **Checkpoint Not Found**: Ensure `checkpoints/` directory exists and contains valid Flax checkpoints.
