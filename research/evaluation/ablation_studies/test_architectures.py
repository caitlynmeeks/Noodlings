#!/usr/bin/env python3
"""
Test Ablation Study Architectures

Verifies that all 6 architecture variants can instantiate,
forward pass, and reports parameter counts.
"""

import sys
sys.path.insert(0, '../..')

import mlx.core as mx
from architectures import (
    BaselineArchitecture,
    ControlArchitecture,
    SingleLayerArchitecture,
    HierarchicalArchitecture,
    Phase4Architecture,
    DenseObserversArchitecture
)


def test_architecture(arch_class, name):
    """Test a single architecture."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)

    try:
        # Instantiate
        model = arch_class()
        print(f"✓ Instantiation successful")

        # Check parameter count
        param_count = model.parameter_count()
        print(f"✓ Parameters: {param_count:,}")

        # Test forward pass
        test_affect = mx.array([0.5, 0.6, 0.2, 0.3, 0.1], dtype=mx.float32)
        phenomenal_state, surprise = model(test_affect)

        print(f"✓ Forward pass successful")
        print(f"  - Phenomenal state shape: {phenomenal_state.shape}")
        print(f"  - Surprise: {surprise:.4f}")

        # Test reset
        model.reset_states()
        print(f"✓ State reset successful")

        # Test get_phenomenal_state
        state = model.get_phenomenal_state()
        assert state.shape == (1, 40), f"Expected (1, 40), got {state.shape}"
        print(f"✓ get_phenomenal_state returns correct shape")

        # Print description
        print(f"\nDescription:")
        print(model.architecture_description())

        return True, param_count

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """Test all architectures."""
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Ablation Study Architecture Testing                             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    architectures = [
        (BaselineArchitecture, "Baseline (No Temporal Model)"),
        (ControlArchitecture, "Control (Random States)"),
        (SingleLayerArchitecture, "Single Layer LSTM"),
        (HierarchicalArchitecture, "Hierarchical (No Observers)"),
        (Phase4Architecture, "Phase 4 (75 Observers)"),
        (DenseObserversArchitecture, "Dense Observers (150 Observers)")
    ]

    results = []
    all_passed = True

    for arch_class, name in architectures:
        passed, param_count = test_architecture(arch_class, name)
        results.append((name, passed, param_count))
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Architecture':<40} {'Status':<10} {'Parameters':>15}")
    print("-" * 70)

    for name, passed, param_count in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<40} {status:<10} {param_count:>15,}")

    print("\n" + "="*70)

    if all_passed:
        print("✓ All architectures tested successfully!")
        print("\nParameter Budget Comparison:")
        print("  - Baseline:      0 params (no learning)")
        print("  - Control:       0 params (random)")
        print("  - Single Layer:  ~6.7K params")
        print("  - Hierarchical:  ~4.5K params")
        print("  - Phase 4:       ~132K params")
        print("  - Dense:         ~264K params")
        print("\nReady for ablation study training!")
        return 0
    else:
        print("✗ Some architectures failed. Fix errors before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
