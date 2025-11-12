#!/usr/bin/env python3
"""
Evaluate Trained Checkpoints with Phase 5 Metrics

Tests Theory of Mind and Relationships checkpoints against:
- TPH (Temporal Prediction Horizon)
- SNC (Surprise-Novelty Correlation)
- HSI (Hierarchical Separation Index)
- PCS (Personality Consistency Score)

Usage:
    python3 evaluate_checkpoints.py
"""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from noodlings.metrics.temporal_metrics import TemporalMetrics
from noodlings.models.noodling_phase4 import NoodlingModelPhase4

# Checkpoint paths
CHECKPOINTS = {
    'theory_of_mind': '/Users/thistlequell/git/consilience/training/checkpoints/theory_of_mind/best.npz',
    'relationships': '/Users/thistlequell/git/consilience/training/checkpoints/relationships/best.npz',
}

# Test data paths
TEST_DATA_DIR = Path('/Users/thistlequell/git/noodlings/training/data/cmush_real')


def load_model_from_checkpoint(checkpoint_path: str) -> NoodlingModelPhase4:
    """Load a trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create model instance
    model = NoodlingModelPhase4(
        affect_dim=5,
        fast_hidden=16,
        medium_hidden=16,
        slow_hidden=8
    )

    # Load weights
    weights = mx.load(checkpoint_path)
    model.update(weights)

    print(f"  ✓ Loaded {sum(p.size for p in model.parameters()['trainable'])} parameters")

    return model


def load_test_data() -> list:
    """Load test data from cmush_real sessions."""
    test_conversations = []

    # Load session files
    session_files = sorted(TEST_DATA_DIR.glob('session_*.jsonl'))

    print(f"Loading test data from {len(session_files)} sessions...")

    for session_file in session_files[:5]:  # Use first 5 sessions
        conversation = []

        with open(session_file, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Extract affect vector if present
                if 'affect' in data:
                    affect = data['affect']
                    # Convert to mx.array: [valence, arousal, fear, sorrow, boredom]
                    affect_vec = mx.array([
                        affect.get('valence', 0.0),
                        affect.get('arousal', 0.0),
                        affect.get('fear', 0.0),
                        affect.get('sorrow', 0.0),
                        affect.get('boredom', 0.0)
                    ], dtype=mx.float32)
                    conversation.append(affect_vec)

        if len(conversation) > 10:  # Only use conversations with sufficient length
            test_conversations.append(conversation)

    print(f"  ✓ Loaded {len(test_conversations)} conversations")
    print(f"  ✓ Average length: {np.mean([len(c) for c in test_conversations]):.1f} timesteps")

    return test_conversations


def generate_synthetic_scenarios() -> dict:
    """Generate test scenarios for PCS metric."""
    return {
        'greeting': [
            mx.array([0.7, 0.6, 0.1, 0.1, 0.2], dtype=mx.float32),  # Happy greeting
            mx.array([0.8, 0.5, 0.1, 0.1, 0.1], dtype=mx.float32),  # Warm greeting
            mx.array([0.6, 0.7, 0.2, 0.1, 0.1], dtype=mx.float32),  # Excited greeting
        ],
        'conflict': [
            mx.array([-0.6, 0.8, 0.7, 0.3, 0.2], dtype=mx.float32),  # Angry
            mx.array([-0.5, 0.7, 0.6, 0.4, 0.1], dtype=mx.float32),  # Tense
            mx.array([-0.7, 0.6, 0.8, 0.2, 0.1], dtype=mx.float32),  # Fearful conflict
        ],
        'praise': [
            mx.array([0.9, 0.7, 0.1, 0.0, 0.0], dtype=mx.float32),  # Joyful
            mx.array([0.8, 0.6, 0.1, 0.1, 0.1], dtype=mx.float32),  # Pleased
            mx.array([0.7, 0.8, 0.2, 0.0, 0.1], dtype=mx.float32),  # Excited happiness
        ],
        'sadness': [
            mx.array([-0.5, 0.3, 0.3, 0.8, 0.3], dtype=mx.float32),  # Deep sadness
            mx.array([-0.4, 0.2, 0.2, 0.7, 0.4], dtype=mx.float32),  # Melancholy
            mx.array([-0.6, 0.4, 0.4, 0.9, 0.2], dtype=mx.float32),  # Grief
        ]
    }


def evaluate_checkpoint(checkpoint_name: str, checkpoint_path: str):
    """Evaluate a single checkpoint with all metrics."""
    print("\n" + "="*70)
    print(f"Evaluating: {checkpoint_name.upper()}")
    print("="*70)

    # Load model
    model = load_model_from_checkpoint(checkpoint_path)

    # Load test data
    test_data = load_test_data()
    test_scenarios = generate_synthetic_scenarios()

    # Initialize metrics calculator
    metrics = TemporalMetrics(model)

    # Results dictionary
    results = {
        'checkpoint': checkpoint_name,
        'path': checkpoint_path,
        'timestamp': datetime.now().isoformat(),
        'metrics': {}
    }

    # 1. TPH - Temporal Prediction Horizon
    print("\n1. Calculating TPH (Temporal Prediction Horizon)...")
    try:
        tph = metrics.calculate_tph(test_data, horizons=[1, 5, 10, 20])
        results['metrics']['tph'] = tph
        print(f"   TPH Results:")
        for horizon, mse in sorted(tph.items()):
            print(f"     {horizon:2d} steps: MSE = {mse:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['metrics']['tph'] = {'error': str(e)}

    # 2. SNC - Surprise-Novelty Correlation
    print("\n2. Calculating SNC (Surprise-Novelty Correlation)...")
    try:
        snc = metrics.calculate_snc(test_data)
        results['metrics']['snc'] = snc
        print(f"   SNC: r = {snc:.3f}")
        if snc > 0.7:
            print("   ✓ Strong correlation: surprise aligns with novelty")
        elif snc > 0.4:
            print("   ~ Moderate correlation")
        else:
            print("   ✗ Weak correlation")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['metrics']['snc'] = {'error': str(e)}

    # 3. HSI - Hierarchical Separation Index
    print("\n3. Calculating HSI (Hierarchical Separation Index)...")
    try:
        hsi = metrics.calculate_hsi(test_data)
        results['metrics']['hsi'] = hsi
        print(f"   HSI Results:")
        print(f"     Slow/Fast:   {hsi['slow/fast']:.3f}")
        print(f"     Slow/Medium: {hsi['slow/medium']:.3f}")
        print(f"     Medium/Fast: {hsi['medium/fast']:.3f}")
        print(f"   {hsi['interpretation']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['metrics']['hsi'] = {'error': str(e)}

    # 4. PCS - Personality Consistency Score
    print("\n4. Calculating PCS (Personality Consistency Score)...")
    try:
        pcs = metrics.calculate_pcs(test_scenarios, num_trials=3)
        results['metrics']['pcs'] = pcs
        print(f"   PCS Overall: {pcs['overall']:.3f}")
        print(f"   By Scenario:")
        for scenario, score in pcs['by_scenario'].items():
            print(f"     {scenario:10s}: {score:.3f}")
        print(f"   {pcs['interpretation']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['metrics']['pcs'] = {'error': str(e)}

    return results


def main():
    """Main evaluation routine."""
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 5 Checkpoint Evaluation - Noodlings Metrics Suite         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    all_results = []

    # Evaluate each checkpoint
    for name, path in CHECKPOINTS.items():
        if not Path(path).exists():
            print(f"\n⚠ Checkpoint not found: {path}")
            continue

        results = evaluate_checkpoint(name, path)
        all_results.append(results)

    # Save results
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print(f"✓ Evaluation complete! Results saved to: {output_file}")
    print("="*70)

    # Summary comparison
    print("\n📊 SUMMARY COMPARISON")
    print("-" * 70)

    for result in all_results:
        name = result['checkpoint']
        metrics_data = result['metrics']

        print(f"\n{name.upper()}:")

        # TPH - show 1-step and 10-step
        if 'tph' in metrics_data and not isinstance(metrics_data['tph'], dict) or 'error' not in metrics_data['tph']:
            tph = metrics_data['tph']
            print(f"  TPH (1-step):  {tph.get(1, float('nan')):.4f}")
            print(f"  TPH (10-step): {tph.get(10, float('nan')):.4f}")

        # SNC
        if 'snc' in metrics_data:
            print(f"  SNC:           {metrics_data['snc']:.3f}")

        # HSI
        if 'hsi' in metrics_data and isinstance(metrics_data['hsi'], dict):
            hsi = metrics_data['hsi']
            print(f"  HSI (S/F):     {hsi.get('slow/fast', float('nan')):.3f}")

        # PCS
        if 'pcs' in metrics_data and isinstance(metrics_data['pcs'], dict):
            pcs = metrics_data['pcs']
            print(f"  PCS:           {pcs.get('overall', float('nan')):.3f}")

    print("\n" + "="*70)
    print("Next step: Design ablation study architectures!")
    print("="*70)


if __name__ == '__main__':
    main()
