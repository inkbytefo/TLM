import jax
import jax.numpy as jnp
import optax
import wandb
import os
import time
import argparse
import numpy as np
from flax.training import checkpoints, train_state
from src.models.gpt import SpectralGPT
from src.data.streaming_loader import MemmapDataLoader, MixedDataLoader, prepare_data
from src.utils.common import setup_logger, set_seed

# --- CONFIGURATION ---
class TrainConfig:
    def __init__(self, args):
        self.project_name = "spectral-jax-scaling"
        self.run_name = args.run_name
        
        # Model
        self.vocab_size = 260 # Byte-level
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = 0.1
        
        # Training
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.accum_steps = args.accum_steps
        self.learning_rate = args.lr
        self.total_steps = args.steps
        self.warmup_steps = int(args.steps * 0.05)
        
        # Data
        self.data_paths = args.data_paths.split(',')
        self.data_weights = [float(w) for w in args.data_weights.split(',')]
        
        # Checkpointing
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)

# --- TRAIN STATE ---
class TrainState(train_state.TrainState):
    # Custom train state if needed, currently standard is fine
    pass

def create_train_state(rng, config):
    model = SpectralGPT(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_rate=config.dropout,
        use_memory=True
    )
    
    dummy_input = jnp.ones((1, config.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)['params']
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1)
    )
    
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# --- TRAINING STEP (With Gradient Accumulation) ---
@jax.jit
def train_step(state, batch, rng):
    # batch: [Batch, Seq]
    # We assume batch is already shaped for accumulation if needed, 
    # but for simplicity here we do standard step. 
    # To do real accumulation, we'd scan over microbatches.
    
    def loss_fn(params, x, y, dropout_rng):
        logits, _ = state.apply_fn(
            {'params': params}, 
            x, 
            train=True, 
            rngs={'dropout': dropout_rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, logits

    dropout_rng = jax.random.fold_in(rng, state.step)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, batch['input'], batch['label'], dropout_rng)
    
    state = state.apply_gradients(grads=grads)
    return state, loss

# --- EXTRAPOLATION EVAL ---
def eval_extrapolation(state, config, logger):
    """Test model on sequences 2x and 4x longer than training."""
    logger.info("Running Extrapolation Test...")
    
    lengths = [config.seq_len * 2, config.seq_len * 4]
    
    for length in lengths:
        # Create dummy input
        dummy_input = jnp.ones((1, length), dtype=jnp.int32)
        
        try:
            start = time.time()
            # Run forward pass (inference mode)
            out, _ = state.apply_fn({'params': state.params}, dummy_input, train=False)
            dt = time.time() - start
            
            logger.info(f"Length {length}: SUCCESS ({dt:.4f}s)")
            wandb.log({f"extrap_success_{length}": 1})
        except Exception as e:
            logger.error(f"Length {length}: FAILED - {e}")
            wandb.log({f"extrap_success_{length}": 0})

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run_001")
    parser.add_argument("--data_paths", type=str, default="data/tinystories.txt", help="Comma separated paths")
    parser.add_argument("--data_weights", type=str, default="1.0", help="Comma separated weights")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()
    
    config = TrainConfig(args)
    logger = setup_logger()
    
    # WandB
    wandb.init(project=config.project_name, name=config.run_name, config=args)
    
    # Data Preparation & Loading
    datasets = []
    for path in config.data_paths:
        bin_path = path.replace('.txt', '.bin')
        if not os.path.exists(bin_path):
            logger.info(f"Preparing binary data from {path}...")
            prepare_data(path, os.path.dirname(path))
        datasets.append({'path': bin_path, 'split': 'train'})
    
    if len(datasets) == 1:
        loader = MemmapDataLoader(datasets[0]['path'], config.batch_size, config.seq_len, split='train')
    else:
        loader = MixedDataLoader(datasets, config.data_weights, config.batch_size, config.seq_len)
    
    # Init Model
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config)
    
    logger.info(f"Starting training for {config.total_steps} steps...")
    
    # Loop
    for step in range(1, config.total_steps + 1):
        batch = next(loader)
        
        # JAX Step
        state, loss = train_step(state, batch, rng)
        
        if step % 100 == 0:
            wandb.log({"train_loss": float(loss), "step": step})
            logger.info(f"Step {step} | Loss: {float(loss):.4f}")
            
        if step % 1000 == 0:
            # Save Checkpoint
            checkpoints.save_checkpoint(config.ckpt_dir, state, step, keep=2, overwrite=True)
            
            # Extrapolation Test
            eval_extrapolation(state, config, logger)

if __name__ == "__main__":
    main()
