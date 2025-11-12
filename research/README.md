# Research: Training & Validation

This directory contains training pipelines and ablation studies for Noodlings architectures.

## Contents

- **training/** - Multi-stage training pipeline for Phase 4-6 architectures
- **evaluation/** - Ablation studies comparing architectural variants
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
4. Train Phase 4 full system (~4-6 hours)

**Output**: Checkpoints saved to `training/checkpoints/`

## Ablation Studies

Located in `evaluation/ablation_studies/`.

**Architectures Compared**:
1. Baseline: LLM only (no temporal model)
2. Control: LLM + random states
3. Single-layer: LLM + single LSTM
4. Hierarchical: Fast/Medium/Slow (no observers)
5. With Observers: Full system (75 loops) - **DEPRECATED**
6. Dense Observers: 2x observer density - **DEPRECATED**

**Note**: Observer loops were removed in Phase 4. Studies focus on hierarchical vs. flat architectures.

## Metrics

See `noodlings/metrics/temporal_metrics.py`:

- **TPH** (Temporal Prediction Horizon): Accuracy at 1/5/10/20/50 timestep predictions
- **SNC** (Surprise-Novelty Correlation): Correlation between model surprise and entropy
- **HSI** (Hierarchical Separation Index): Variance ratios between fast/medium/slow layers
- **PCS** (Personality Consistency Score): Consistency of agent responses across scenarios

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

Shows:
- Current training stage
- Epoch progress
- Loss metrics
- ETA

---

**For public release**: This is research code. Training requires significant compute (M3 Ultra recommended). Most users should use pre-trained checkpoints from the main applications.
