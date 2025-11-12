# Observer Stability - Quick Reference

## TL;DR

**Without observers**: Hierarchy collapses (HSI: 11.4)  
**With 75 observers**: Still unstable (HSI: 2.6)  
**With 150 observers**: Stable! (HSI: 0.11) ← Only good architecture

## The Critical Numbers

```
┌─────────────────┬───────────┬──────────┬────────────────┐
│ Architecture    │ Observers │ HSI      │ Status         │
├─────────────────┼───────────┼──────────┼────────────────┤
│ Hierarchical    │ 0         │ 11.423   │ ❌ COLLAPSED   │
│ Phase 4         │ 75        │ 2.619    │ ⚠️  UNSTABLE   │
│ Dense Observers │ 150       │ 0.113    │ ✅ STABLE      │
└─────────────────┴───────────┴──────────┴────────────────┘

HSI < 0.3 = Good separation (only Dense achieved this)
```

## Why This Matters

1. **Hierarchy collapse is real**: Without observers, fast/medium/slow layers learn the SAME timescale
2. **75 observers isn't enough**: "Valley of Death" - partially helps but still fails
3. **150 observers is the sweet spot**: Only architecture maintaining true hierarchical structure
4. **Prediction improves**: Dense observers have best accuracy (TPH: 0.152 vs 0.205 for Phase4)
5. **Surprise tracking improves**: 21x better SNC (0.208 vs 0.010)

## The Mechanism

Observer loops create **multi-objective optimization**:
- Main network: minimize prediction error (→ layers collapse together)
- Observers: predict next states (→ require layer separation to be useful)
- Result: Competing pressures force stable hierarchy

## Training Cost

| Architecture | Training Time | Cost vs Hierarchical |
|--------------|---------------|----------------------|
| Hierarchical | 140s | 1.0x (baseline) |
| Phase 4 (75) | 1431s | 10.2x |
| Dense (150) | 2600s | 18.5x |

**Worth it?** Yes - only Dense achieves stable hierarchy.

## Action Items

- [ ] Use 150+ observers for all hierarchical architectures
- [ ] Monitor HSI during training (early warning system)
- [ ] Test intermediate densities (100, 125) to find exact threshold
- [ ] Measure Φ on stable vs collapsed hierarchies
- [ ] Consider "observer distillation" for inference speedup

## Files

- `ablation_results.json` - Raw numerical results
- `ablation_results_summary.png` - Visualization
- `STABILITY_ANALYSIS.md` - Full technical analysis
- `ablation_full.log` - Training logs

## Open Question

**Why 2x density?** Is there a theoretical principle? 

Hypothesis: Need ~3-4 observers per hierarchical connection:
- Fast ↔ Medium: ~50 observers
- Medium ↔ Slow: ~50 observers  
- Fast ↔ Slow: ~50 observers
Total: 150 observers

Test: Try 100 observers (1.33x density) to find phase transition point.
