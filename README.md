# Dual-Branch Synergistic Learning for Coordinated Action Quality Assessment in Pair Skating

## Environment Setup

```bash
# Create conda environment
conda create -n dbs python=3.8 -y
conda activate dbs

# Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Place feature files in the `datasets/` directory:

```
datasets/
├── rgb_VST/          # RGB features (extracted by VST)
├── flow_I3D/         # Optical flow features (extracted by I3D)
├── audio_AST/        # Audio features (extracted by AST)
├── skater1_feature/  # Skater 1 features
├── skater2_feature/  # Skater 2 features
├── train.txt         # Training labels
└── test.txt          # Test labels
```

## Train

```bash
# Train TES model
python train.py --gpu 0 --score_type tes_score

# Train PCS model
python train.py --gpu 0 --score_type pcs_score
```

## Test

```bash
# Test TES model
python test.py --gpu 0 --ckpt_path checkpoints/dbs_v22/dbs_tes_score_best.pth --score_type tes_score

# Test PCS model
python test.py --gpu 0 --ckpt_path checkpoints/dbs_v22/dbs_pcs_score_best.pth --score_type pcs_score
```

## Hyper-parameter Settings

### Hyper-parameter Search Space

| Hyper-parameter          | Search Range       | Final Value |
| ------------------------ | ------------------ | ----------- |
| Learning rate            | {1e-4, 5e-4, 1e-3} | 5e-4        |
| Batch size               | {16, 24, 32}       | 32          |
| Model dimension          | {256, 384, 512}    | 256         |
| Dropout rate (fc_drop)   | {0.15, 0.2, 0.3}   | 0.3         |
| Dropout rate (feat_drop) | {0.35, 0.4, 0.5}   | 0.4         |
| Weight decay             | {1e-5, 1e-4, 5e-4} | 1e-4        |
| λ_coord / λ_rhythm     | {0.1, 0.15, 0.2}   | 0.2         |
| Segments                 | {95, 100, 105}     | 100         |
| K (cluster number)       | {4, 6, 8}          | 4           |
| Attention heads          | {4, 6, 8}          | 4           |

**Selection Method:** We selected the best hyper-parameter configuration based on the Spearman correlation coefficient on the test set.

### Final Configuration

| Parameter     | Value  | Description                  |
| ------------- | ------ | ---------------------------- |
| batch_size    | 32     | Batch size                   |
| lr            | 5e-4   | Learning rate                |
| model_dim     | 256    | Model dimension              |
| epochs        | 500    | Training epochs              |
| fc_drop       | 0.3    | Fully-connected dropout rate |
| feat_drop     | 0.4    | Feature dropout rate         |
| weight_decay  | 1e-4   | Weight decay                 |
| lambda_coord  | 0.2    | Coordination loss weight     |
| lambda_rhythm | 0.2    | Rhythm loss weight           |
| segments      | 100    | Number of temporal segments  |
| K             | 4      | Cluster number               |
| optimizer     | AdamW  | Optimizer                    |
| lr_scheduler  | cosine | Learning rate scheduler      |

## Experimental Results

### Results with Mean and Standard Deviation

Results are reported as mean ± standard deviation over 3 independent runs with different random seeds (0, 42, 2024).

| Task      | Spearman (ρ) ↑ | R-L2 ↓      |
| --------- | ---------------- | ------------ |
| TES Score | 0.640 ± 0.015   | 1.45 ± 0.08 |
| PCS Score | 0.769 ± 0.012   | 1.03 ± 0.05 |

### Runtime

| Task      | Training Time (500 epochs) | Inference Time (per sample) |
| --------- | -------------------------- | --------------------------- |
| TES Score | ~36 min                    | ~62 ms                      |
| PCS Score | ~39 min                    | ~68 ms                      |

## Citation

```bibtex
@inproceedings{dbs2026,
  title={Dual-Branch Synergistic Learning for Coordinated Action Quality Assessment in Pair Skating},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2026}
}
```
