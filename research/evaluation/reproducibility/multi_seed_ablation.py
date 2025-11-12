#!/usr/bin/env python3
"""
Phase 1: Multi-Seed Reproducibility Test

Tests whether the observer stabilization effect replicates across multiple random seeds.

CRITICAL: This is the first and most important validation step.
If this fails, the entire finding is suspect.

Usage:
    python3 multi_seed_ablation.py --epochs 50 --seeds 10
    python3 multi_seed_ablation.py --quick  # 3 seeds, 10 epochs (fast test)
"""

import sys
sys.path.insert(0, '../..')

import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import time
from scipy import stats

# Use existing architectures
sys.path.insert(0, '../ablation_studies')
from architectures import (
    BaselineArchitecture,
    ControlArchitecture,
    SingleLayerArchitecture,
    HierarchicalArchitecture,
    Phase4Architecture,
    DenseObserversArchitecture
)

# Use existing training/eval functions
from run_ablation import (
    generate_synthetic_data,
    train_architecture,
    evaluate_architecture,
    generate_test_scenarios
)


def run_single_seed_ablation(
    seed: int,
    architectures: List[tuple],
    train_data: List,
    test_data: List,
    test_scenarios: Dict,
    epochs: int
) -> Dict:
    """
    Run full ablation study with a single random seed.

    Returns:
        results: Dict mapping architecture name to evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    # Set random seeds
    mx.random.seed(seed)
    np.random.seed(seed)

    results = {}

    for name, model_class in architectures:
        print(f"\n--- {name} (seed={seed}) ---")

        # Create fresh model instance
        model = model_class()

        # Train (skip for baseline/control)
        if name in ['baseline', 'control']:
            print(f"Skipping training for {name}")
            training_time = 0
            final_loss = None
        else:
            start = time.time()
            history = train_architecture(model, train_data, epochs=epochs, learning_rate=1e-3)
            training_time = time.time() - start
            final_loss = history['loss'][-1]
            print(f"  Training time: {training_time:.1f}s, Final loss: {final_loss:.6f}")

        # Evaluate
        eval_results = evaluate_architecture(model, test_data, test_scenarios)

        results[name] = {
            'seed': seed,
            'training_time': training_time,
            'final_loss': final_loss,
            'evaluation': eval_results
        }

    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across seeds and compute statistics.

    Args:
        all_results: List of result dicts, one per seed

    Returns:
        aggregated: Dict with mean, std, and per-seed values
    """
    # Get architecture names
    arch_names = list(all_results[0].keys())

    aggregated = {}

    for arch_name in arch_names:
        # Collect values across seeds
        hsi_sf_values = []
        tph_1_values = []
        tph_10_values = []
        snc_values = []
        pcs_values = []

        for seed_results in all_results:
            result = seed_results[arch_name]

            # HSI slow/fast
            hsi_sf = result['evaluation'].get('hsi', {}).get('slow/fast', float('nan'))
            hsi_sf_values.append(hsi_sf)

            # TPH
            tph = result['evaluation'].get('tph', {})
            tph_1 = float(tph.get('1', float('nan')))
            tph_10 = float(tph.get('10', float('nan')))
            tph_1_values.append(tph_1)
            tph_10_values.append(tph_10)

            # SNC
            snc = result['evaluation'].get('snc', float('nan'))
            snc_values.append(snc)

            # PCS
            pcs = result['evaluation'].get('pcs', {}).get('overall', float('nan'))
            pcs_values.append(pcs)

        # Compute statistics (ignoring NaNs)
        aggregated[arch_name] = {
            'hsi_sf': {
                'mean': float(np.nanmean(hsi_sf_values)),
                'std': float(np.nanstd(hsi_sf_values)),
                'min': float(np.nanmin(hsi_sf_values)),
                'max': float(np.nanmax(hsi_sf_values)),
                'values': hsi_sf_values
            },
            'tph_1': {
                'mean': float(np.nanmean(tph_1_values)),
                'std': float(np.nanstd(tph_1_values)),
                'values': tph_1_values
            },
            'tph_10': {
                'mean': float(np.nanmean(tph_10_values)),
                'std': float(np.nanstd(tph_10_values)),
                'values': tph_10_values
            },
            'snc': {
                'mean': float(np.nanmean(snc_values)),
                'std': float(np.nanstd(snc_values)),
                'values': snc_values
            },
            'pcs': {
                'mean': float(np.nanmean(pcs_values)),
                'std': float(np.nanstd(pcs_values)),
                'values': pcs_values
            }
        }

    return aggregated


def statistical_analysis(aggregated: Dict) -> Dict:
    """
    Perform statistical hypothesis testing.

    Tests:
    - H0: Dense observers HSI = Hierarchical HSI
    - H1: Dense observers HSI < Hierarchical HSI
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Extract HSI values
    dense_hsi = aggregated['dense_observers']['hsi_sf']['values']
    hier_hsi = aggregated['hierarchical']['hsi_sf']['values']

    # Remove NaNs
    dense_hsi = [x for x in dense_hsi if not np.isnan(x)]
    hier_hsi = [x for x in hier_hsi if not np.isnan(x)]

    if len(dense_hsi) == 0 or len(hier_hsi) == 0:
        print("ERROR: No valid HSI values for statistical test!")
        return {'error': 'No valid values'}

    # Two-sample t-test (two-tailed)
    t_stat, p_value_twotail = stats.ttest_ind(dense_hsi, hier_hsi)

    # One-tailed test (dense < hier)
    p_value_onetail = p_value_twotail / 2 if t_stat < 0 else 1 - (p_value_twotail / 2)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(dense_hsi) + np.var(hier_hsi)) / 2)
    cohens_d = (np.mean(hier_hsi) - np.mean(dense_hsi)) / pooled_std

    # Summary
    results = {
        't_statistic': float(t_stat),
        'p_value_twotail': float(p_value_twotail),
        'p_value_onetail': float(p_value_onetail),
        'cohens_d': float(cohens_d),
        'dense_mean': float(np.mean(dense_hsi)),
        'dense_std': float(np.std(dense_hsi)),
        'hier_mean': float(np.mean(hier_hsi)),
        'hier_std': float(np.std(hier_hsi)),
        'significant_001': p_value_onetail < 0.01,
        'significant_005': p_value_onetail < 0.05
    }

    print(f"\nH0: Dense observers HSI = Hierarchical HSI")
    print(f"H1: Dense observers HSI < Hierarchical HSI (better separation)\n")
    print(f"Dense observers:  HSI = {results['dense_mean']:.3f} ± {results['dense_std']:.3f}")
    print(f"Hierarchical:     HSI = {results['hier_mean']:.3f} ± {results['hier_std']:.3f}")
    print(f"\nT-statistic:      {results['t_statistic']:.3f}")
    print(f"P-value (1-tail): {results['p_value_onetail']:.6f}")
    print(f"Cohen's d:        {results['cohens_d']:.3f}")
    print(f"\nSignificance:")
    print(f"  p < 0.05: {'✓ YES' if results['significant_005'] else '✗ NO'}")
    print(f"  p < 0.01: {'✓ YES' if results['significant_001'] else '✗ NO'}")

    if results['cohens_d'] > 0.8:
        effect_size_interp = "LARGE effect"
    elif results['cohens_d'] > 0.5:
        effect_size_interp = "MEDIUM effect"
    elif results['cohens_d'] > 0.2:
        effect_size_interp = "SMALL effect"
    else:
        effect_size_interp = "NEGLIGIBLE effect"

    print(f"  Effect size: {effect_size_interp}")

    # RED FLAG check
    print("\n" + "-"*70)
    print("RED FLAG CHECK:")
    print("-"*70)

    red_flags = []

    # Check 1: High variance
    cv_dense = results['dense_std'] / results['dense_mean'] if results['dense_mean'] > 0 else 0
    cv_hier = results['hier_std'] / results['hier_mean'] if results['hier_mean'] > 0 else 0

    if cv_dense > 0.5:
        red_flags.append(f"High variance in dense observers (CV={cv_dense:.2f})")
    if cv_hier > 0.5:
        red_flags.append(f"High variance in hierarchical (CV={cv_hier:.2f})")

    # Check 2: No significance
    if not results['significant_005']:
        red_flags.append("No statistical significance (p > 0.05)")

    # Check 3: Small effect size
    if results['cohens_d'] < 0.5:
        red_flags.append(f"Small effect size (d={results['cohens_d']:.2f})")

    if red_flags:
        print("🚩 RED FLAGS DETECTED:")
        for flag in red_flags:
            print(f"  - {flag}")
        print("\nClaim may be suspect. Investigate further.")
    else:
        print("✓ No red flags detected. Results look solid.")

    results['red_flags'] = red_flags

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-Seed Reproducibility Test')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test (3 seeds, 10 epochs)')
    parser.add_argument('--output', type=str, default='multi_seed_results.json', help='Output file')

    args = parser.parse_args()

    if args.quick:
        args.seeds = 3
        args.epochs = 10
        print("🏃 QUICK MODE: 3 seeds, 10 epochs")

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 1: Multi-Seed Reproducibility Test                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"\nSeeds: {args.seeds}")
    print(f"Epochs per architecture: {args.epochs}")
    print()

    # Seeds
    seed_list = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768][:args.seeds]

    # Architectures
    architectures = [
        ('baseline', BaselineArchitecture),
        ('control', ControlArchitecture),
        ('single_layer', SingleLayerArchitecture),
        ('hierarchical', HierarchicalArchitecture),
        ('phase4', Phase4Architecture),
        ('dense_observers', DenseObserversArchitecture)
    ]

    # Generate data ONCE (use same data for all seeds for fair comparison)
    print("Generating synthetic data...")
    train_data = generate_synthetic_data(num_conversations=50, conversation_length=30)
    test_data = generate_synthetic_data(num_conversations=10, conversation_length=30)
    test_scenarios = generate_test_scenarios()
    print()

    # Run ablation for each seed
    all_results = []
    start_time = time.time()

    for seed in seed_list:
        seed_results = run_single_seed_ablation(
            seed=seed,
            architectures=architectures,
            train_data=train_data,
            test_data=test_data,
            test_scenarios=test_scenarios,
            epochs=args.epochs
        )
        all_results.append(seed_results)

    total_time = time.time() - start_time

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)

    aggregated = aggregate_results(all_results)

    # Print summary table
    print("\n📊 SUMMARY ACROSS SEEDS")
    print("-" * 70)
    print(f"{'Architecture':<20} {'HSI (S/F)':<20} {'TPH(1)':<15} {'SNC':<12}")
    print("-" * 70)

    for arch_name in aggregated.keys():
        hsi = aggregated[arch_name]['hsi_sf']
        tph = aggregated[arch_name]['tph_1']
        snc = aggregated[arch_name]['snc']

        hsi_str = f"{hsi['mean']:.3f} ± {hsi['std']:.3f}"
        tph_str = f"{tph['mean']:.4f} ± {tph['std']:.4f}"
        snc_str = f"{snc['mean']:.3f} ± {snc['std']:.3f}"

        print(f"{arch_name:<20} {hsi_str:<20} {tph_str:<15} {snc_str:<12}")

    # Statistical analysis
    stats_results = statistical_analysis(aggregated)

    # Save results
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'seeds': seed_list,
            'num_seeds': args.seeds,
            'epochs': args.epochs,
            'total_time_minutes': total_time / 60
        },
        'per_seed_results': all_results,
        'aggregated_results': aggregated,
        'statistical_analysis': stats_results
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Final verdict
    print("\n" + "="*70)
    print("PHASE 1 VERDICT")
    print("="*70)

    if stats_results.get('significant_001') and not stats_results.get('red_flags'):
        print("✅ PASS: Effect is statistically significant and reproducible")
        print("   → Proceed to Phase 2 (Confound Analysis)")
    elif stats_results.get('significant_005'):
        print("⚠️  MARGINAL: Effect is significant at p<0.05 but not p<0.01")
        print("   → Consider running more seeds or investigating variance")
    else:
        print("❌ FAIL: Effect is not statistically significant")
        print("   → STOP. Original finding may be spurious.")
        if stats_results.get('red_flags'):
            print("\n   Red flags detected:")
            for flag in stats_results['red_flags']:
                print(f"     - {flag}")


if __name__ == '__main__':
    main()
