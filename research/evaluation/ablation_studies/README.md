# Noodlings Ablation Study - Phase 5

Scientific validation of the hierarchical architecture through systematic comparison.

## 🔬 Major Discovery: Observer Loops Prevent Hierarchical Collapse

**Key Finding**: Without observer loops, multi-timescale architectures **catastrophically collapse** during training. All layers converge to the same timescale (HSI: 0.004 → 11.423 after 50 epochs).

**The Solution**: Dense observer loops (150+) act as "gradient sinks" that stabilize the hierarchy.

See detailed analysis in:
- `EXECUTIVE_SUMMARY.md` - High-level overview
- `STABILITY_ANALYSIS.md` - Technical deep dive
- `OBSERVER_ARCHITECTURE_EXPLORATION.md` - Research proposals

## 6 Architecture Variants

| # | Name | Description | Parameters | Purpose |
|---|------|-------------|------------|---------|
| 1 | **Baseline** | No temporal model (zeros) | 0 | Floor performance |
| 2 | **Control** | Random states | 0 | Prove structure matters |
| 3 | **SingleLayer** | One 40-D LSTM | ~6.7K | Show benefit of hierarchy |
| 4 | **Hierarchical** | Fast + Medium + Slow | ~4.5K | Core architecture |
| 5 | **Phase4** | Hierarchical + 75 observers | ~132K | With integrated information (Φ) |
| 6 | **DenseObservers** | Hierarchical + 150 observers | ~264K | Test observer density |

## Evaluation Metrics

All architectures are evaluated with:

1. **TPH (Temporal Prediction Horizon)**: Prediction accuracy at 1, 5, 10, 20, 50 timesteps
   - Lower MSE = better long-term prediction

2. **HSI (Hierarchical Separation Index)**: Timescale separation via variance ratios
   - Slow/Fast < 0.2 = excellent separation
   - Slow/Fast < 0.7 = moderate separation

3. **SNC (Surprise-Novelty Correlation)**: Alignment of surprise with entropy
   - r > 0.7 = strong correlation
   - r > 0.4 = moderate correlation

4. **PCS (Personality Consistency Score)**: Response stability across scenarios
   - > 0.8 = high consistency
   - > 0.6 = moderate consistency

## Usage

### Quick Test (5 minutes)
```bash
cd evaluation/ablation_studies
source /path/to/venv/bin/activate
python3 run_ablation.py --epochs 10 --train-all
```

### Full Training (~2-3 hours)
```bash
python3 run_ablation.py --epochs 50 --train-all --output results_full.json
```

### Evaluate Only (no training)
```bash
python3 run_ablation.py --evaluate-only
```

## Expected Results

**Hypothesis:**
- **Hierarchical** should show better HSI (timescale separation) than SingleLayer
- **Phase4** should show similar HSI but potentially better PCS (consistency)
- **DenseObservers** may show diminishing returns or overfitting

**Baseline/Control** establish floor performance - everything should beat them!

## Output

Results saved to `ablation_results.json`:

```json
[
  {
    "name": "hierarchical",
    "training": {"final_loss": 0.0234, "training_time": 456.7},
    "evaluation": {
      "tph": {1: 0.023, 5: 0.045, 10: 0.089},
      "hsi": {"slow/fast": 0.15, "interpretation": "Excellent separation"},
      "snc": 0.72,
      "pcs": {"overall": 0.85, "interpretation": "High consistency"}
    }
  },
  ...
]
```

## Results Summary (November 2025)

**Completed**: Initial 6-architecture ablation (50 epochs)

| Architecture | Observers | HSI | Status | Training Time |
|--------------|-----------|-----|--------|---------------|
| Hierarchical | 0 | 11.423 | ❌ **COLLAPSED** | 140s |
| Phase 4 | 75 | 2.619 | ⚠️ Unstable | 1431s (10x) |
| **Dense** | **150** | **0.113** | **✅ STABLE** | **2600s (18.5x)** |

**Key insight**: Only Dense Observers (150 loops) maintains hierarchical separation (HSI < 0.3).

**The "Valley of Death"**: 75 observers provide partial stabilization but are insufficient. There's a critical phase transition between 75-150 observers.

**Hypothesis**: Power law scaling
```
HSI(N) = k / N^β  where β ≈ 2
```

---

## Proposed Follow-up Experiments

### 1. Find N_critical (Critical Threshold)

Test N ∈ {75, 85, 95, 105, 115, 125, 135, 150} to find exact transition point.

```bash
python3 test_critical_threshold.py --epochs 50
python3 plot_threshold.py
```

### 2. Topology Ablation

Test flat, hierarchical, star, and ring observer topologies.

### 3. Parameter Ratio Test

Test if it's observer COUNT or TOTAL PARAMETERS that matters.

### 4. Collapse Reversibility

Can observers rescue an already-collapsed hierarchy?

### 5. Long-term Stability

Does 150 observers maintain stability for 200+ epochs?

See `OBSERVER_ARCHITECTURE_EXPLORATION.md` for detailed research proposals.

---

## Next Steps

1. ✅ ~~Run ablation study~~
2. ✅ ~~Analyze results~~ (Dense observers win!)
3. ⏳ **Current**: Test critical threshold (N_critical)
4. ⏳ Implement remaining experiments (topology, parameter ratio, etc.)
5. ⏳ Validate on real noodleMUSH data
6. ⏳ Prepare manuscript with falsifiable claims

## Files

- `architectures/` - 6 architecture implementations
- `test_architectures.py` - Verify all architectures work
- `run_ablation.py` - Train & evaluate all architectures
- `README.md` - This file

---

**Status**: Ready to run! 🎉
**Created**: November 2025
**Phase**: 5 (Scientific Validation)
