# Spectral-JAX Training Guide (L4 Cluster)

This guide details how to train the **Infinite Context Spectral-JAX** model using the new professional infrastructure.

## 1. Environment Setup

Ensure you have the required dependencies installed on the cluster node:
```bash
pip install -r requirements.txt
pip install wandb
```

## 2. Data Preparation (The "Turkish-First" ASI Curriculum)

We follow a 3-Phase Curriculum to build ASI, leveraging Turkish morphology as a logical foundation.

### Phase 1: Morphological Foundation (The "Logic" Phase)
*   **Goal**: Learn strict compositional logic via agglutinative morphology.
*   **Data**: 80% High-Quality Turkish (Academic/Literature), 20% Code.
*   **Command**:
    ```bash
    python train.py \
        --data_paths data/turkish_academic.txt,data/github_code.txt \
        --data_weights 0.8,0.2 \
        ...
    ```

### Phase 2: World Knowledge (The "Expansion" Phase)
*   **Goal**: Absorb facts and world model.
*   **Data**: 40% English (The Pile), 30% Turkish, 30% Code.
*   **Command**:
    ```bash
    python train.py \
        --data_paths data/english_pile.txt,data/turkish_all.txt,data/github_code.txt \
        --data_weights 0.4,0.3,0.3 \
        ...
    ```

### Phase 3: Reasoning & ASI (The "Super" Phase)
*   **Goal**: Complex reasoning and generalization.
*   **Data**: 50% Math/Code, 25% English, 25% Turkish.

**Note**: The script automatically converts `.txt` files to `.bin` (memmap) on the first run.

## 3. Running Training

Use the unified `train.py` script. It supports command-line arguments for all key hyperparameters.

### Scaling Run (L4 GPU Recommended)
For a serious training run on an L4 GPU (24GB VRAM), use these settings:

```bash
python train.py \
    --run_name scale_phase_1 \
    --data_paths data/turkish_academic.txt,data/github_code.txt \
    --data_weights 0.8,0.2 \
    --hidden_dim 512 \
    --num_layers 12 \
    --seq_len 2048 \
    --batch_size 16 \
    --accum_steps 4 \
    --lr 3e-4 \
    --steps 50000
```

*   **Effective Batch Size**: `batch_size` * `accum_steps` = 16 * 4 = 64.
*   **Context**: 2048 tokens (extrapolates to infinite).

## 4. Monitoring

### WandB
Training metrics are logged to Weights & Biases automatically.
*   **Project**: `spectral-jax-scaling`
*   **Metrics**: Loss, Extrapolation Success (2x, 4x length).

### Checkpoints
Checkpoints are saved to `checkpoints/<run_name>/`.
*   `checkpoint_0`: Latest step.
*   `checkpoint_1`: Previous step.

## 5. Scaling Strategy

To reach ASI capabilities, follow this roadmap:

1.  **Level 1 (Current)**: 512 dim, 12 layers. Train on 1GB data.
2.  **Level 2 (Medium)**: 1024 dim, 24 layers. Train on 10GB data.
3.  **Level 3 (Large)**: 2048 dim, 48 layers. Train on 100GB+ data. *Requires Multi-GPU.*

## 6. Troubleshooting

*   **OOM (Out of Memory)**: Reduce `--batch_size` and increase `--accum_steps` to keep the effective batch size constant.
*   **NaN Loss**: Reduce `--lr` (Learning Rate) or increase `warmup_steps`.
