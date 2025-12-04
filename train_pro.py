import jax
import jax.numpy as jnp
import optax
import os
import time
import wandb
from tqdm import tqdm
from flax.training import checkpoints
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state, train_step_generative
from src.training.evaluator import eval_step_generative
from src.data.text_generation import get_text_dataloaders, decode_text
from generate import generate

# Prevent TensorFlow from grabbing GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def main():
    # 1. CONFIGURATION
    config = Config()
    logger = setup_logger()
    set_seed(config.training.seed)

    # PRO SETTINGS (Override defaults for serious training)
    config.data.seq_len = 1024          # Byte-level için en az 1024 olmalı
    config.data.batch_size = 8          # GPU belleğine göre ayarla
    accum_steps = 8                     # 8 * 8 = 64 Efektif Batch Size
    
    # Training Duration
    # Byte-level modeller için en az 100M byte görülmeli.
    # 100,000 adım * 64 batch * 1024 seq = ~6.5 Milyar byte (İdeal)
    config.training.num_steps = 50000   
    config.training.warmup_steps = 2000
    config.training.learning_rate = 3e-4 # Biraz daha agresif başlangıç
    
    # WANDB INIT
    wandb.init(
        project="spectral-jax-pro",
        config={
            "model": "SpectralGPT",
            "layers": config.model.num_layers,
            "hidden": config.model.hidden_dim,
            "memory": config.model.use_memory,
            "seq_len": config.data.seq_len,
            "effective_batch": config.data.batch_size * accum_steps,
            "lr": config.training.learning_rate
        }
    )

    # 2. DATA PIPELINE
    # Daha büyük bir veri seti kullanın (Örn: TinyStories veya birleştirilmiş textler)
    dataset_path = 'data/tinystories.txt' # Changed to tinystories.txt as recommended
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found: {dataset_path}. Falling back to shakespeare.txt if available.")
        dataset_path = 'data/shakespeare.txt'
        if not os.path.exists(dataset_path):
             logger.error(f"Dataset not found: {dataset_path}")
             return

    logger.info(f"Loading dataset: {dataset_path}")
    train_loader, val_loader, vocab_size, char_to_idx, idx_to_char = get_text_dataloaders(
        filepath=dataset_path,
        seq_len=config.data.seq_len,
        batch_size=config.data.batch_size,
        train_split=0.95
    )
    config.data.vocab_size = vocab_size
    
    # 3. MODEL INITIALIZATION
    rng = jax.random.PRNGKey(config.training.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_generative_train_state(init_rng, config)
    
    # Checkpoint Manager
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", "pro_model")
    best_dir = os.path.join(ckpt_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    # Restore if available
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    start_step = int(state.step)
    logger.info(f"Starting training from step: {start_step}")

    # 4. TRAINING LOOP
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    progress_bar = tqdm(range(start_step, config.training.num_steps), initial=start_step, total=config.training.num_steps)

    for step in progress_bar:
        step_start = time.time()
        
        # Gradient Accumulation Batching
        accum_inputs = []
        for _ in range(accum_steps):
            try:
                inputs, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, _ = next(train_iter)
            accum_inputs.append(inputs)
            
        # Stack: (Accum, Batch, Seq)
        batch = {'input': jnp.stack(accum_inputs, axis=0)}
        
        # Train Step
        state, loss, acc, rng = train_step_generative(state, batch, rng)
        
        # Metrics
        loss_val = float(loss)
        acc_val = float(acc)
        lr_val = float(config.training.learning_rate) # Scheduler varsa burası dinamik olmalı
        
        # WandB Log (Every 10 steps)
        if step % 10 == 0:
            wandb.log({
                "train/loss": loss_val,
                "train/accuracy": acc_val,
                "train/step_time": time.time() - step_start,
                "train/step": step
            })
            progress_bar.set_description(f"Loss: {loss_val:.4f} | Acc: {acc_val:.4f}")

        # Validation & Generation (Every 500 steps)
        if step % 500 == 0 and step > 0:
            logger.info("Running Validation...")
            val_losses = []
            for val_inputs, _ in val_loader:
                metrics = eval_step_generative(state, {'input': val_inputs})
                val_losses.append(metrics['loss'])
                if len(val_losses) > 50: break # Sadece 50 batch test et (hız için)
            
            mean_val_loss = float(jnp.mean(jnp.array(val_losses)))
            wandb.log({"val/loss": mean_val_loss, "val/step": step})
            
            # Save Best
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                checkpoints.save_checkpoint(best_dir, state, step, keep=1, overwrite=True)
                logger.info(f"New Best Model! Val Loss: {best_val_loss:.4f}")
            
            # Save Regular
            checkpoints.save_checkpoint(ckpt_dir, state, step, keep=2, overwrite=True)

            # Sample Generation
            rng, gen_rng = jax.random.split(rng)
            prompt_text = "The meaning of"
            prompt_ids = [char_to_idx.get(c, 0) for c in prompt_text]
            prompt_tensor = jnp.array([prompt_ids], dtype=jnp.int32)
            
            gen_seq = generate(state, prompt_tensor, gen_rng, max_new_tokens=100, temperature=0.8)
            gen_text = decode_text(gen_seq[0], idx_to_char)
            
            logger.info(f"\nSample Generation:\n{gen_text}\n")
            
            # Log sample to WandB
            wandb.log({
                "sample_generation": wandb.Html(f"<pre>{gen_text}</pre>"),
                "val/step": step
            })

    wandb.finish()
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
