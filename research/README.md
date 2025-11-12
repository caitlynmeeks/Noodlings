# Research: Training Pipeline

This directory contains the training pipeline for Noodlings architectures.

## Contents

- **training/** - Multi-stage training pipeline
- **evaluate_checkpoints.py** - Checkpoint validation script

## Training Pipeline

**Requirements**: M3 Ultra (512GB RAM) recommended for full pipeline

```bash
cd training
./train.sh
```

**Stages**:
1. Generate synthetic data (affective arcs)
2. Train Theory of Mind module (~2-3 hours)
3. Train Relationship model (~1 hour)
4. Train full system (~4-6 hours)

**Output**: Checkpoints saved to `training/checkpoints/`

## Training Data

Training data is NOT included in this repository (.gitignore excludes):
- `training/data/cmush_real/` - Real agent interactions (privacy)
- `training/data/synthetic/*.jsonl` - Generated affective arcs

**To generate synthetic data**:
```bash
cd training/scripts
python 00_generate_synthetic_data.py
```

## Status Monitoring

```bash
cd training
./status.sh
```

Shows current training stage, epoch progress, loss metrics, and ETA.

---

**Note**: This is research code. Training requires significant compute (M3 Ultra recommended). Most users should use pre-trained checkpoints from the main applications.
