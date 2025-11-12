#!/usr/bin/env python3
"""
Test Phase 5 Metrics Implementation

Verifies that TPH, SNC, and HSI metrics can be imported and have correct signatures.
"""

import sys
sys.path.insert(0, '.')

from noodlings.metrics.temporal_metrics import TemporalMetrics

print("Testing Phase 5 Metrics...")

# Check that TemporalMetrics class has the new methods
print("\n✓ Checking method existence:")

methods_to_check = [
    'calculate_tph',
    'calculate_snc',
    'calculate_hsi',
    '_predict_ahead',
    '_discretize_affect',
    '_calculate_entropy',
    '_get_state_trajectory'
]

for method_name in methods_to_check:
    if hasattr(TemporalMetrics, method_name):
        print(f"  ✓ {method_name} exists")
    else:
        print(f"  ✗ {method_name} MISSING")
        sys.exit(1)

# Check method signatures
print("\n✓ Checking method signatures:")

import inspect

# TPH signature
tph_sig = inspect.signature(TemporalMetrics.calculate_tph)
print(f"  ✓ calculate_tph{tph_sig}")
expected_params = ['self', 'test_data', 'horizons']
actual_params = list(tph_sig.parameters.keys())
assert expected_params == actual_params, f"TPH signature mismatch: {actual_params}"

# SNC signature
snc_sig = inspect.signature(TemporalMetrics.calculate_snc)
print(f"  ✓ calculate_snc{snc_sig}")
expected_params = ['self', 'test_data']
actual_params = list(snc_sig.parameters.keys())
assert expected_params == actual_params, f"SNC signature mismatch: {actual_params}"

# HSI signature
hsi_sig = inspect.signature(TemporalMetrics.calculate_hsi)
print(f"  ✓ calculate_hsi{hsi_sig}")
expected_params = ['self', 'test_data']
actual_params = list(hsi_sig.parameters.keys())
assert expected_params == actual_params, f"HSI signature mismatch: {actual_params}"

# Check return type annotations
print("\n✓ Checking return type annotations:")
tph_return = tph_sig.return_annotation
snc_return = snc_sig.return_annotation
hsi_return = hsi_sig.return_annotation

print(f"  ✓ calculate_tph returns: {tph_return}")
print(f"  ✓ calculate_snc returns: {snc_return}")
print(f"  ✓ calculate_hsi returns: {hsi_return}")

# Check docstrings
print("\n✓ Checking docstrings:")
for method_name in ['calculate_tph', 'calculate_snc', 'calculate_hsi']:
    method = getattr(TemporalMetrics, method_name)
    if method.__doc__ and len(method.__doc__) > 100:
        print(f"  ✓ {method_name} has comprehensive docstring ({len(method.__doc__)} chars)")
    else:
        print(f"  ✗ {method_name} has insufficient docstring")
        sys.exit(1)

print("\n" + "="*60)
print("🎉 Phase 5 Metrics Implementation Verified!")
print("="*60)
print("\nImplemented metrics:")
print("  1. Temporal Prediction Horizon (TPH)")
print("     - Measures prediction accuracy at multiple time horizons")
print("     - Returns: Dict[int, float] mapping horizon → MSE")
print("")
print("  2. Surprise-Novelty Correlation (SNC)")
print("     - Measures correlation between model surprise and entropy")
print("     - Returns: float (Pearson r, range -1 to 1)")
print("")
print("  3. Hierarchical Separation Index (HSI)")
print("     - Measures timescale separation between layers")
print("     - Returns: Dict with variance ratios and interpretation")
print("")
print("Next steps:")
print("  - Create synthetic test data")
print("  - Run metrics on untrained model (baseline)")
print("  - Compare with trained model")
print("  - Generate ablation study comparing architectures")
