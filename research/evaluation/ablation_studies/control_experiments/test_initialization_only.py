#!/usr/bin/env python3
"""
Control Experiment: Initialization-Only Effect

CRITICAL QUESTION: Do observers help at INITIALIZATION or only through LEARNING?

Tests HSI at random initialization (NO TRAINING) to determine if the observer
effect is architectural or learned.

If observers help at init: Effect is architectural/initialization lottery
If observers only help after training: Effect is learned (but we're skeptical)
"""

import sys
sys.path.insert(0, '..')

import mlx.core as mx
import numpy as np
import json
from datetime import datetime
from test_harmonic_ratios import HarmonicObserverArchitecture, calculate_hsi


def test_initialization_effect(num_observers, num_trials=50, seq_length=20):
    """Test HSI at initialization (no training)"""
    print(f"\n{'='*70}")
    print(f"Testing N={num_observers} observers at INITIALIZATION")
    print(f"({'='*70})")

    hsi_values = []

    # Generate test sequence
    np.random.seed(999)  # Fixed seed for test data
    valence = np.random.uniform(-1, 1, seq_length)
    arousal = np.random.uniform(0, 1, seq_length)
    fear = np.random.uniform(0, 0.5, seq_length)
    sorrow = np.random.uniform(0, 0.3, seq_length)
    boredom = np.random.uniform(0, 0.2, seq_length)
    sequence = np.stack([valence, arousal, fear, sorrow, boredom], axis=1)
    test_sequence = mx.array(sequence, dtype=mx.float32)

    for trial in range(num_trials):
        # Create model with unique random seed
        mx.random.seed(trial)
        np.random.seed(trial)

        model = HarmonicObserverArchitecture(
            affect_dim=5,
            num_observers=num_observers
        )

        # NO TRAINING - just measure HSI at initialization
        model.reset_states()
        hsi = calculate_hsi(model, test_sequence, seq_length)
        hsi_values.append(hsi['slow/fast'])

        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1}/{num_trials}: HSI={hsi['slow/fast']:.6f}")

    mean_hsi = np.mean(hsi_values)
    std_hsi = np.std(hsi_values)
    min_hsi = np.min(hsi_values)
    max_hsi = np.max(hsi_values)
    stable_count = sum(1 for h in hsi_values if h < 0.3)
    stable_rate = stable_count / len(hsi_values)

    print(f"\nResults (NO TRAINING):")
    print(f"  Mean HSI:       {mean_hsi:.6f}")
    print(f"  Std Dev:        {std_hsi:.6f}")
    print(f"  Min HSI:        {min_hsi:.6f}")
    print(f"  Max HSI:        {max_hsi:.6f}")
    print(f"  Stable rate:    {stable_rate*100:.1f}% ({stable_count}/{num_trials})")

    return {
        'num_observers': num_observers,
        'mean_hsi': float(mean_hsi),
        'std_hsi': float(std_hsi),
        'min_hsi': float(min_hsi),
        'max_hsi': float(max_hsi),
        'stable_rate': float(stable_rate),
        'hsi_values': [float(h) for h in hsi_values]
    }


def main():
    print("="*80)
    print("CONTROL EXPERIMENT: Initialization-Only Effect")
    print("="*80)
    print()
    print("Testing whether observers help BEFORE training (architectural effect)")
    print("vs. only AFTER training (learned effect)")
    print()

    configs = [0, 50, 100]
    results = {}

    for n_obs in configs:
        result = test_initialization_effect(n_obs, num_trials=50)
        results[n_obs] = result

    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    print()

    print(f"{'N Observers':<15} {'Mean HSI':<15} {'Std Dev':<15} {'Stable Rate':<15}")
    print("-"*70)

    for n_obs in configs:
        r = results[n_obs]
        print(f"{n_obs:<15} {r['mean_hsi']:<15.6f} {r['std_hsi']:<15.6f} {r['stable_rate']*100:<14.1f}%")

    # Hypothesis test
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    no_obs_mean = results[0]['mean_hsi']
    obs_50_mean = results[50]['mean_hsi']
    obs_100_mean = results[100]['mean_hsi']

    # Check if observers help at initialization
    improvement_50 = (no_obs_mean - obs_50_mean) / no_obs_mean * 100
    improvement_100 = (no_obs_mean - obs_100_mean) / no_obs_mean * 100

    observers_help_at_init = (obs_100_mean < no_obs_mean * 0.7)

    print(f"Improvement at N=50:  {improvement_50:+.1f}%")
    print(f"Improvement at N=100: {improvement_100:+.1f}%")
    print()

    if observers_help_at_init:
        print("✓ ARCHITECTURAL EFFECT:")
        print("  Observers provide lower HSI at initialization (before learning)")
        print("  → Effect is architectural or initialization lottery")
        print("  → More parameters = more chances for favorable init")
        interpretation = "architectural_effect"
    else:
        print("✗ NO INITIALIZATION EFFECT:")
        print("  All configurations have similar HSI at initialization")
        print("  → Effect (if real) only appears through training")
        print("  → But our training on noise suggests no real learning...")
        interpretation = "no_init_effect"

    # Check stability rates
    print("\n" + "="*80)
    print("STABILITY RATES AT INITIALIZATION")
    print("="*80)
    print()

    for n_obs in configs:
        r = results[n_obs]
        print(f"N={n_obs}: {r['stable_rate']*100:.1f}% stable at init (HSI < 0.3)")

    if results[100]['stable_rate'] > 0.2:  # >20% stable at init
        print("\n⚠️  WARNING: High stability rate at initialization suggests")
        print("   the 'observer effect' may be mostly initialization lottery!")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'initialization_only',
        'hypothesis': 'Do observers help at initialization or only through learning?',
        'num_trials_per_config': 50,
        'results': {str(k): v for k, v in results.items()},
        'interpretation': interpretation,
        'improvement_50': float(improvement_50),
        'improvement_100': float(improvement_100)
    }

    with open('initialization_only_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to initialization_only_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
