# Quick Start: Scientific Validation

This guide will help you rigorously validate the observer stabilization effect.

## TL;DR - Run This First

```bash
cd evaluation/reproducibility

# Quick test (3 seeds, 10 epochs, ~30 minutes)
python3 multi_seed_ablation.py --quick

# Full Phase 1 (10 seeds, 50 epochs, ~4 hours)
python3 multi_seed_ablation.py --seeds 10 --epochs 50
```

**What This Tests**: Whether the observer stabilization effect replicates across random seeds with statistical significance.

**Success Criteria**: p < 0.01, Cohen's d > 0.8, low variance

---

## The Validation Roadmap

### Phase 1: Reproducibility (CRITICAL) ⬅️ **START HERE**

**Goal**: Prove effect isn't a fluke

**Script**: `reproducibility/multi_seed_ablation.py`

**What it does**:
- Runs ablation study with 10 different random seeds
- Computes mean ± std for all metrics
- Tests statistical significance (t-test)
- Checks for red flags (high variance, low significance)

**Time**: ~4 hours on M3 Ultra

**Decision Point**:
- ✅ If p < 0.01 → Proceed to Phase 2
- ⚠️ If 0.01 < p < 0.05 → Run more seeds
- ❌ If p > 0.05 → STOP, effect is spurious

### Phase 2: Confound Controls

**Goal**: Rule out alternative explanations

**Experiments**:
1. **Parameter-matched baseline**: Does wide hierarchical (132K params) still collapse?
2. **Training time control**: Does hierarchical trained 10x longer still collapse?
3. **Random skip connections**: Do non-prediction connections stabilize?
4. **Dropout regularization**: Does heavy dropout prevent collapse?

**Implementation**: `confounds/run_confound_experiments.py` (TODO)

**Time**: ~6 hours

**Decision Point**:
- ✅ If effect persists → Real phenomenon
- ❌ If confound explains it → Adjust claim

### Phase 3: Mechanistic Validation

**Goal**: Understand WHY it works

**Experiments**:
1. **Gradient flow analysis**: Track gradient norms per layer
2. **Information flow analysis**: Measure mutual information
3. **Observer ablation**: Remove specific observer connections

**Implementation**: `mechanisms/analyze_mechanisms.py` (TODO)

**Time**: ~8 hours

### Phase 4: Scaling Analysis

**Goal**: Find boundaries of the effect

**Experiments**:
1. **Observer count sweep**: 0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300
2. **Architecture scale**: 1x, 2x, 4x, 10x model size
3. **Timescale ratio sweep**: Vary learning rate ratios

**Implementation**: `scaling/observer_sweep.py` (TODO)

**Time**: ~12 hours

### Phase 5: Real-World Validation

**Goal**: Prove it's not just synthetic data

**Datasets**:
- CMU-MOSEI (multimodal sentiment)
- RECOLA (continuous affect)
- AVEC (depression detection)

**Implementation**: `benchmarks/run_benchmarks.py` (TODO)

**Time**: ~20 hours

---

## Quick Test (30 minutes)

Want to quickly check if things work before committing 4+ hours?

```bash
cd evaluation/reproducibility
python3 multi_seed_ablation.py --quick
```

This runs:
- 3 seeds (instead of 10)
- 10 epochs (instead of 50)
- Same 6 architectures

**Output**:
- Summary table with mean ± std
- Statistical test results
- Red flag warnings
- Pass/Fail verdict

**Interpretation**:
- If quick test shows p < 0.05, full test likely works
- If quick test shows p > 0.1, full test may fail
- If borderline, must run full test

---

## Understanding the Output

### Key Metrics

**HSI (Hierarchical Separation Index)**:
- Lower = better hierarchical separation
- < 0.3: Good (layers maintain different timescales)
- 0.3 - 1.0: Poor (layers converging)
- \> 1.0: Collapsed (layers nearly identical)

**Expected Results**:
- Dense observers: HSI ≈ 0.1 ± 0.05
- Hierarchical: HSI ≈ 10 ± 3
- Phase4 (75 obs): HSI ≈ 2.5 ± 0.5

### Statistical Tests

**T-statistic**: Difference in means relative to variance
- Larger magnitude = stronger effect

**P-value (one-tailed)**: Probability effect is due to chance
- p < 0.01: Very strong evidence
- p < 0.05: Strong evidence
- p > 0.05: Weak/no evidence

**Cohen's d**: Standardized effect size
- d > 0.8: Large effect
- d > 0.5: Medium effect
- d > 0.2: Small effect
- d < 0.2: Negligible

### Red Flags 🚩

**High Variance**: std > 50% of mean
- Means effect is unstable across seeds
- Could indicate initialization sensitivity

**No Significance**: p > 0.05
- Effect could be random noise
- Need more data or better controls

**Small Effect Size**: d < 0.5
- Effect exists but is weak
- May not be practically important

---

## FAQ

### Q: How long does full validation take?

**A**: ~50 hours of compute across all phases
- Phase 1: 4 hours
- Phase 2: 6 hours
- Phase 3: 8 hours
- Phase 4: 12 hours
- Phase 5: 20 hours

You can run phases in parallel on multiple machines.

### Q: What if Phase 1 fails?

**A**: STOP. Don't proceed to other phases.

Either:
1. Original finding was spurious (initialization luck)
2. Bug in validation code (double-check implementation)
3. Need more seeds (try 20 instead of 10)

Investigate before moving forward.

### Q: Can I run validation on smaller model?

**A**: Yes, but effect might change with scale.

Try:
- Fast=8, Medium=8, Slow=4 (20D state)
- 50 observers instead of 150
- 25 epochs instead of 50

This tests if effect is scale-dependent.

### Q: What statistical test should I use?

**A**: We use two-sample t-test because:
- Comparing two groups (dense vs hierarchical)
- Independent samples (different random seeds)
- Metric is continuous (HSI)

Alternative: Mann-Whitney U test (non-parametric) if HSI distribution is highly skewed.

### Q: How do I know if I have enough seeds?

**A**: Power analysis

If effect size (Cohen's d) is large (d > 0.8):
- 10 seeds gives 95% power
- 5 seeds gives 80% power

If effect size is medium (d ≈ 0.5):
- 20 seeds gives 95% power
- 10 seeds gives 80% power

Current results show d ≈ 2-3 (very large), so 10 seeds is adequate.

### Q: Should I preregister hypotheses?

**A**: Ideally, yes!

We've already observed the effect (not preregistered), but you can preregister Phase 2-5 predictions before running them.

Use OSF (Open Science Framework) to timestamp predictions.

---

## Next Steps After Phase 1

### If Phase 1 PASSES (p < 0.01, no red flags):

1. **Celebrate** 🎉 (but cautiously)
2. **Run Phase 2** (confound controls)
3. **Start documenting** for paper/whitepaper
4. **Share preliminary results** with trusted colleagues

### If Phase 1 is MARGINAL (0.01 < p < 0.05):

1. **Run more seeds** (20 instead of 10)
2. **Check for bugs** in implementation
3. **Examine outliers** (which seeds failed?)
4. **Consider covariates** (training time, final loss)

### If Phase 1 FAILS (p > 0.05):

1. **STOP immediately**
2. **Investigate**:
   - Was original result initialization luck?
   - Bug in validation code?
   - Hyperparameter sensitivity?
3. **Consider negative result publication**
   - "Observer networks do NOT consistently stabilize hierarchies"
   - Still valuable to community

---

## Epistemic Humility Checklist

Before sharing results publicly:

- [ ] Replicated across ≥10 seeds
- [ ] Statistically significant (p < 0.01)
- [ ] Large effect size (d > 0.8)
- [ ] Survived confound controls
- [ ] Tested on real data (not just synthetic)
- [ ] Full code/data released
- [ ] Limitations clearly stated
- [ ] Claims are modest (no "solved consciousness")

**Remember**: We're exploring functional correlates, not claiming to build real consciousness. Stay humble!

---

## Getting Help

- See `VALIDATION_PROTOCOL.md` for detailed methodology
- Check `evaluation/reproducibility/multi_seed_ablation.py` for implementation
- Review Phase 5 plan in `PHASE5_REORGANIZATION_PLAN.md`

**Questions?** Open an issue on GitHub or reach out to project maintainers.

---

**Status**: Phase 1 implementation complete, ready to run
**Last Updated**: November 7, 2025
