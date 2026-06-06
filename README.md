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

## Key Parameters

| Parameter    | Value | Description              |
| ------------ | ----- | ------------------------ |
| batch_size   | 32    | Batch size               |
| lr           | 5e-4  | Learning rate            |
| model_dim    | 256   | Model dimension          |
| epochs       | 500   | Training epochs          |
| lambda_coord | 0.2   | Coordination loss weight |

## Citation

```bibtex
@inproceedings{dbs2026,
  title={Dual-Branch Synergistic Learning for Coordinated Action Quality Assessment in Pair Skating},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2026}
}
```
