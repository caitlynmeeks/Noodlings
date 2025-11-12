#!/usr/bin/env python3
"""
Ablation Study Training & Evaluation

Trains all 6 architecture variants and compares them using:
- TPH (Temporal Prediction Horizon)
- HSI (Hierarchical Separation Index)
- SNC (Surprise-Novelty Correlation)
- PCS (Personality Consistency Score)

Usage:
    python3 run_ablation.py --epochs 50 --train-all
    python3 run_ablation.py --evaluate-only   # Use existing checkpoints
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

from architectures import (
    BaselineArchitecture,
    ControlArchitecture,
    SingleLayerArchitecture,
    HierarchicalArchitecture,
    Phase4Architecture,
    DenseObserversArchitecture
)

from noodlings.metrics.temporal_metrics import TemporalMetrics


def generate_synthetic_data(num_conversations=100, conversation_length=30):
    """Generate synthetic affective conversation data."""
    print("Generating synthetic training data...")

    conversations = []

    for i in range(num_conversations):
        conversation = []

        # Random personality baseline
        baseline_valence = np.random.uniform(-0.3, 0.3)
        baseline_arousal = np.random.uniform(0.3, 0.7)

        for t in range(conversation_length):
            # Drift + noise
            valence = baseline_valence + 0.5 * np.sin(t * 0.1) + np.random.normal(0, 0.1)
            arousal = baseline_arousal + 0.3 * np.cos(t * 0.15) + np.random.normal(0, 0.1)
            fear = np.abs(np.random.normal(0, 0.2))
            sorrow = np.abs(np.random.normal(0, 0.2))
            boredom = np.clip(t / conversation_length + np.random.normal(0, 0.1), 0, 1)

            affect = mx.array([valence, arousal, fear, sorrow, boredom], dtype=mx.float32)
            conversation.append(affect)

        conversations.append(conversation)

    print(f"  ✓ Generated {len(conversations)} conversations")
    return conversations


def train_architecture(
    model,
    train_data: List[List[mx.array]],
    epochs: int = 50,
    learning_rate: float = 1e-3
):
    """
    Train a single architecture.

    Args:
        model: Architecture instance
        train_data: List of conversations
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        training_history: Dict with loss history
    """
    optimizer = optim.Adam(learning_rate=learning_rate)

    history = {'loss': []}

    print(f"Training {model.architecture_name()} for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_losses = []

        for conversation in train_data:
            model.reset_states()

            conversation_loss = 0.0

            for t, affect in enumerate(conversation):
                if affect.ndim == 1:
                    affect = affect[None, :]  # Add batch dim

                # Forward pass
                def loss_fn():
                    phenomenal_state, surprise = model(affect)

                    # Prediction loss (minimize surprise)
                    pred_loss = surprise

                    # Regularization (prevent state explosion)
                    state_norm = mx.mean(phenomenal_state ** 2)
                    reg_loss = 0.01 * state_norm

                    total_loss = pred_loss + reg_loss
                    return total_loss

                # Compute gradients
                loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad_fn()

                # Update parameters
                optimizer.update(model, grads)

                # Evaluate to free memory
                mx.eval(model.parameters())

                # Track loss
                conversation_loss += float(loss)

            epoch_losses.append(conversation_loss / len(conversation))

        # Clean up memory between epochs
        mx.eval(model.parameters())

        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")

    print(f"  ✓ Training complete. Final loss: {history['loss'][-1]:.4f}")

    return history


def evaluate_architecture(
    model,
    test_data: List[List[mx.array]],
    test_scenarios: Dict[str, List[mx.array]]
) -> Dict:
    """
    Evaluate architecture with all metrics.

    Returns:
        results: Dict with all metric scores
    """
    print(f"Evaluating {model.architecture_name()}...")

    metrics = TemporalMetrics(model)
    results = {}

    # TPH
    try:
        tph = metrics.calculate_tph(test_data, horizons=[1, 5, 10])
        results['tph'] = tph
        print(f"  TPH (1-step): {tph[1]:.4f}, (10-step): {tph[10]:.4f}")
    except Exception as e:
        print(f"  TPH error: {e}")
        results['tph'] = {'error': str(e)}

    # HSI
    try:
        hsi = metrics.calculate_hsi(test_data)
        results['hsi'] = hsi
        print(f"  HSI (Slow/Fast): {hsi['slow/fast']:.3f} - {hsi['interpretation']}")
    except Exception as e:
        print(f"  HSI error: {e}")
        results['hsi'] = {'error': str(e)}

    # SNC
    try:
        snc = metrics.calculate_snc(test_data)
        results['snc'] = snc
        print(f"  SNC: {snc:.3f}")
    except Exception as e:
        print(f"  SNC error: {e}")
        results['snc'] = float('nan')

    # PCS
    try:
        pcs = metrics.calculate_pcs(test_scenarios, num_trials=3)
        results['pcs'] = pcs
        print(f"  PCS: {pcs['overall']:.3f} - {pcs['interpretation']}")
    except Exception as e:
        print(f"  PCS error: {e}")
        results['pcs'] = {'error': str(e)}

    return results


def generate_test_scenarios():
    """Generate test scenarios for PCS."""
    return {
        'greeting': [
            mx.array([0.7, 0.6, 0.1, 0.1, 0.2], dtype=mx.float32),
            mx.array([0.8, 0.5, 0.1, 0.1, 0.1], dtype=mx.float32),
            mx.array([0.6, 0.7, 0.2, 0.1, 0.1], dtype=mx.float32),
        ],
        'conflict': [
            mx.array([-0.6, 0.8, 0.7, 0.3, 0.2], dtype=mx.float32),
            mx.array([-0.5, 0.7, 0.6, 0.4, 0.1], dtype=mx.float32),
        ],
        'praise': [
            mx.array([0.9, 0.7, 0.1, 0.0, 0.0], dtype=mx.float32),
            mx.array([0.8, 0.6, 0.1, 0.1, 0.1], dtype=mx.float32),
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Training & Evaluation')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--train-all', action='store_true', help='Train all architectures')
    parser.add_argument('--evaluate-only', action='store_true', help='Evaluate existing checkpoints only')
    parser.add_argument('--output', type=str, default='ablation_results.json', help='Output file')

    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Noodlings Ablation Study                                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # Generate data
    train_data = generate_synthetic_data(num_conversations=50, conversation_length=30)
    test_data = generate_synthetic_data(num_conversations=10, conversation_length=30)
    test_scenarios = generate_test_scenarios()

    # Define architectures
    architectures = [
        ('baseline', BaselineArchitecture()),
        ('control', ControlArchitecture()),
        ('single_layer', SingleLayerArchitecture()),
        ('hierarchical', HierarchicalArchitecture()),
        ('phase4', Phase4Architecture()),
        ('dense_observers', DenseObserversArchitecture())
    ]

    all_results = []

    for name, model in architectures:
        print("\n" + "="*70)
        print(f"Architecture: {name.upper()}")
        print("="*70)
        print(model.architecture_description())
        print()

        result = {
            'name': name,
            'architecture': model.architecture_name(),
            'description': model.architecture_description(),
            'timestamp': datetime.now().isoformat()
        }

        # Training
        if args.train_all and not args.evaluate_only:
            if name in ['baseline', 'control']:
                print(f"Skipping training for {name} (non-trainable)")
                result['training'] = {'skipped': True}
            else:
                start_time = time.time()
                history = train_architecture(model, train_data, epochs=args.epochs)
                training_time = time.time() - start_time

                result['training'] = {
                    'history': history,
                    'final_loss': history['loss'][-1],
                    'training_time': training_time
                }

                print(f"  Training time: {training_time/60:.1f} minutes")

        # Evaluation
        print()
        eval_results = evaluate_architecture(model, test_data, test_scenarios)
        result['evaluation'] = eval_results

        all_results.append(result)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n📊 SUMMARY COMPARISON")
    print("-" * 70)
    print(f"{'Architecture':<20} {'TPH(1)':<10} {'HSI(S/F)':<10} {'SNC':<8} {'PCS':<8}")
    print("-" * 70)

    for result in all_results:
        name = result['name']
        eval_res = result['evaluation']

        tph_1 = eval_res.get('tph', {}).get(1, float('nan'))
        hsi_sf = eval_res.get('hsi', {}).get('slow/fast', float('nan'))
        snc = eval_res.get('snc', float('nan'))
        pcs = eval_res.get('pcs', {}).get('overall', float('nan'))

        if isinstance(tph_1, dict):
            tph_1 = float('nan')
        if isinstance(hsi_sf, dict):
            hsi_sf = float('nan')
        if isinstance(pcs, dict):
            pcs = float('nan')

        print(f"{name:<20} {tph_1:<10.4f} {hsi_sf:<10.3f} {snc:<8.3f} {pcs:<8.3f}")

    print("\n✓ Run complete! Now you can analyze results and create visualizations.")


if __name__ == '__main__':
    main()
