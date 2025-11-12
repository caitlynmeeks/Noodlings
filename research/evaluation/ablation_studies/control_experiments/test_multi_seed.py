#!/usr/bin/env python3
"""
Control Experiment 3: Multi-Seed Replication

HYPOTHESIS: Observer effect is reproducible across different random seeds.

CONCERN: High variance - same config gives different results on different runs.

Tests:
- No observers (N=0): 10 replications
- Few observers (N=50): 10 replications
- Many observers (N=100): 10 replications

If variance is LOW: Effect is real and robust
If variance is HIGH: Effect is stochastic/unreliable
"""

import sys
sys.path.insert(0, '../..')

import mlx.core as mx
import numpy as np
import json
from datetime import datetime

# Import the architecture from test_harmonic_ratios
sys.path.insert(0, '..')
from test_harmonic_ratios import HarmonicObserverArchitecture, calculate_hsi, train_and_evaluate


def replicate_experiment(num_observers, num_replications=10, epochs=50):
    """Run same experiment multiple times with different seeds"""
    print(f"\n{'='*70}")
    print(f"Replicating N={num_observers} observers × {num_replications} runs")
    print(f"{'='*70}")

    results = []

    for rep in range(num_replications):
        print(f"\n--- Replication {rep+1}/{num_replications} ---")

        # Set unique seed for this replication
        seed = 1000 + rep
        mx.random.seed(seed)
        np.random.seed(seed)

        result = train_and_evaluate(num_observers, epochs=epochs)
        result['replication'] = rep
        result['seed'] = seed
        results.append(result)

        print(f"HSI: {result['final_hsi']:.6f}")

    return results


def main():
    print("="*80)
    print("CONTROL EXPERIMENT: Multi-Seed Replication")
    print("="*80)
    print()
    print("Testing reproducibility across different random initializations")
    print()

    # Test configurations
    configurations = [0, 50, 100]

    all_results = {}

    for n_obs in configurations:
        results = replicate_experiment(n_obs, num_replications=10, epochs=50)
        all_results[n_obs] = results

    # Analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    print()

    print(f"{'N Observers':<15} {'Mean HSI':<15} {'Std Dev':<15} {'Min':<10} {'Max':<10} {'CV':<10}")
    print("-"*80)

    for n_obs in configurations:
        results = all_results[n_obs]
        hsi_values = [r['final_hsi'] for r in results]

        mean_hsi = np.mean(hsi_values)
        std_hsi = np.std(hsi_values)
        min_hsi = np.min(hsi_values)
        max_hsi = np.max(hsi_values)
        cv = std_hsi / mean_hsi if mean_hsi > 0 else float('nan')  # Coefficient of variation

        print(f"{n_obs:<15} {mean_hsi:<15.6f} {std_hsi:<15.6f} {min_hsi:<10.6f} {max_hsi:<10.6f} {cv:<10.2f}")

    # Hypothesis test
    print("\n" + "="*80)
    print("REPRODUCIBILITY ASSESSMENT")
    print("="*80)
    print()

    # Calculate coefficient of variation for each condition
    cvs = {}
    for n_obs in configurations:
        results = all_results[n_obs]
        hsi_values = [r['final_hsi'] for r in results]
        mean_hsi = np.mean(hsi_values)
        std_hsi = np.std(hsi_values)
        cvs[n_obs] = std_hsi / mean_hsi if mean_hsi > 0 else float('nan')

    # Check if observer conditions have LOWER variance
    no_obs_cv = cvs[0]
    obs_cvs = [cvs[50], cvs[100]]
    obs_reduce_variance = all(cv < no_obs_cv * 0.8 for cv in obs_cvs)

    # Check if means are significantly different
    no_obs_mean = np.mean([r['final_hsi'] for r in all_results[0]])
    obs_50_mean = np.mean([r['final_hsi'] for r in all_results[50]])
    obs_100_mean = np.mean([r['final_hsi'] for r in all_results[100]])

    effect_is_large = (obs_100_mean < no_obs_mean * 0.5)

    if effect_is_large and obs_reduce_variance:
        print("✓ HIGHLY REPRODUCIBLE:")
        print("  Large effect size AND low variance with observers")
        print("  → Observer effect is real and robust")
        assessment = "reproducible"
    elif effect_is_large and not obs_reduce_variance:
        print("⚠ EFFECT EXISTS BUT VARIABLE:")
        print("  Large mean difference but high variance")
        print("  → Effect is real but context-dependent")
        assessment = "variable_effect"
    elif not effect_is_large:
        print("✗ WEAK OR NO EFFECT:")
        print("  Means don't differ much across conditions")
        print("  → Effect may not be real")
        assessment = "weak_effect"
    else:
        print("? UNCLEAR:")
        print("  Mixed signals from variance and means")
        assessment = "unclear"

    # Additional: Check stability rates
    print("\n" + "="*80)
    print("STABILITY RATES (HSI < 0.3)")
    print("="*80)
    print()

    for n_obs in configurations:
        results = all_results[n_obs]
        stable_count = sum(1 for r in results if r['is_stable'])
        stability_rate = stable_count / len(results) * 100
        print(f"N={n_obs}: {stability_rate:.1f}% stable ({stable_count}/{len(results)} runs)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'multi_seed_replication',
        'hypothesis': 'Is the observer effect reproducible across seeds?',
        'configurations': configurations,
        'num_replications': 10,
        'results': {str(k): v for k, v in all_results.items()},
        'assessment': assessment,
        'statistics': {
            'no_obs_mean': float(no_obs_mean),
            'obs_50_mean': float(obs_50_mean),
            'obs_100_mean': float(obs_100_mean),
            'no_obs_cv': float(no_obs_cv),
            'obs_50_cv': float(cvs[50]),
            'obs_100_cv': float(cvs[100])
        }
    }

    with open('multi_seed_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to multi_seed_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
