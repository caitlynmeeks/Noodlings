#!/usr/bin/env python3
"""
Experiment Set 1: Find Critical Observer Threshold

Tests observer counts from 75 to 150 to find exact N_critical where HSI < 0.3

Falsifiable predictions:
- Power law: HSI(N) = k / N^β where β ≈ 2
- If β significantly differs from 2, power law hypothesis is rejected
- If curve is linear, power law hypothesis is falsified
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
import time

from architectures.base import AblationArchitecture
from noodlings.metrics.temporal_metrics import TemporalMetrics


class ParametricObserverArchitecture(AblationArchitecture):
    """
    Hierarchical architecture with configurable observer count.

    Observer distribution follows 2:1:0.2 ratio across 3 levels:
    - Level 0: 2/3.2 of total
    - Level 1: 1/3.2 of total
    - Level 2: 0.2/3.2 of total
    """

    def __init__(self, affect_dim: int = 5, num_observers: int = 150):
        super().__init__(affect_dim)

        self.num_observers = num_observers

        # Core hierarchy
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=16)
        self.c_fast = mx.zeros((1, 16))

        self.medium_lstm = nn.LSTM(input_size=16, hidden_size=16)
        self.c_medium = mx.zeros((1, 16))

        self.slow_gru = nn.GRU(input_size=16, hidden_size=8)

        # Main predictor
        joint_dim = 16 + 16 + 8  # 40
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_dim)
        )

        # Observer networks with proportional distribution
        self.observers = self._create_observers(num_observers)

    def _create_observers(self, total: int):
        """Create observers with 2:1:0.2 ratio across levels."""
        observers = []

        # Calculate distribution
        ratio_sum = 2.0 + 1.0 + 0.2  # 3.2
        level0_count = int(total * (2.0 / ratio_sum))
        level1_count = int(total * (1.0 / ratio_sum))
        level2_count = total - level0_count - level1_count  # Remainder

        # Level 0 observers
        for i in range(level0_count):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        # Level 1 observers
        for i in range(level1_count):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        # Level 2 observers
        for i in range(level2_count):
            observer = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            observers.append(observer)

        self.level_split = (level0_count, level1_count, level2_count)
        return observers

    def __call__(self, affect: mx.array):
        """Forward pass with configurable observer cascade."""
        # Ensure correct shape
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Fast layer
        h_fast_seq, c_fast_seq = self.fast_lstm(
            affect, hidden=self.h_fast, cell=self.c_fast
        )
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        # Medium layer
        h_med_seq, c_med_seq = self.medium_lstm(
            self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium
        )
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        # Slow layer
        h_slow_seq = self.slow_gru(
            self.h_medium[:, None, :], hidden=self.h_slow
        )
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Get current state
        current_state = self.get_phenomenal_state()

        # Main predictor
        predicted_state = self.predictor(current_state)

        # Observer cascade
        observer_errors = []
        level0, level1, level2 = self.level_split

        # Level 0 observers
        for i in range(level0):
            obs_pred = self.observers[i](predicted_state)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Level 1 observers
        for i in range(level0, level0 + level1):
            # Sample from level 0
            if level0 > 0:
                level0_sample = self.observers[i % level0](predicted_state)
            else:
                level0_sample = predicted_state
            obs_pred = self.observers[i](level0_sample)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Level 2 observers
        for i in range(level0 + level1, level0 + level1 + level2):
            # Sample from level 1
            if level1 > 0:
                level1_idx = level0 + (i % level1)
                level1_sample = self.observers[level1_idx](predicted_state)
            else:
                level1_sample = predicted_state
            obs_pred = self.observers[i](level1_sample)
            obs_error = mx.mean((obs_pred - current_state) ** 2)
            observer_errors.append(obs_error)

        # Total surprise
        main_error = mx.mean((predicted_state - current_state) ** 2)
        if observer_errors:
            observer_error = mx.mean(mx.array(observer_errors))
            surprise = float(main_error + 0.1 * observer_error)
        else:
            surprise = float(main_error)

        return current_state, surprise

    def reset_states(self):
        """Reset all layer states."""
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def architecture_name(self) -> str:
        return f"Parametric_{self.num_observers}obs"

    def architecture_description(self) -> str:
        level0, level1, level2 = self.level_split
        return (
            f"Hierarchical with {self.num_observers} observers\n"
            f"Distribution: L0={level0}, L1={level1}, L2={level2}\n"
            f"Parameters: ~{self.parameter_count():,}"
        )


def generate_synthetic_data(num_conversations=50, conversation_length=30):
    """Generate synthetic affective conversation data."""
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

    return conversations


def train_architecture(model, train_data, epochs=50, learning_rate=1e-3):
    """Train a single architecture."""
    optimizer = optim.Adam(learning_rate=learning_rate)
    history = {'loss': []}

    for epoch in range(epochs):
        epoch_losses = []

        for conversation in train_data:
            model.reset_states()
            conversation_loss = 0.0

            for affect in conversation:
                if affect.ndim == 1:
                    affect = affect[None, :]

                def loss_fn():
                    phenomenal_state, surprise = model(affect)
                    pred_loss = surprise
                    state_norm = mx.mean(phenomenal_state ** 2)
                    reg_loss = 0.01 * state_norm
                    return pred_loss + reg_loss

                loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad_fn()
                optimizer.update(model, grads)
                mx.eval(model.parameters())

                conversation_loss += float(loss)

            epoch_losses.append(conversation_loss / len(conversation))

        mx.eval(model.parameters())
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")

    return history


def evaluate_architecture(model, test_data):
    """Evaluate architecture with HSI metric."""
    metrics = TemporalMetrics(model)

    try:
        hsi = metrics.calculate_hsi(test_data)
        return hsi
    except Exception as e:
        print(f"  HSI error: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Critical Threshold Experiment')
    parser.add_argument('--counts', type=int, nargs='+',
                       default=[75, 85, 95, 105, 115, 125, 135, 150],
                       help='Observer counts to test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output', type=str, default='threshold_results.json')

    args = parser.parse_args()

    print("="*80)
    print("EXPERIMENT SET 1: CRITICAL THRESHOLD DETECTION")
    print("="*80)
    print()
    print("Hypothesis: HSI(N) = k / N^β where β ≈ 2")
    print(f"Testing observer counts: {args.counts}")
    print()

    # Generate data
    train_data = generate_synthetic_data(num_conversations=50, conversation_length=30)
    test_data = generate_synthetic_data(num_conversations=10, conversation_length=30)

    results = []

    for num_obs in args.counts:
        print(f"\n{'='*80}")
        print(f"Testing {num_obs} observers")
        print(f"{'='*80}")

        model = ParametricObserverArchitecture(num_observers=num_obs)
        print(model.architecture_description())
        print()

        # Train
        start_time = time.time()
        history = train_architecture(model, train_data, epochs=args.epochs)
        training_time = time.time() - start_time

        # Evaluate
        hsi = evaluate_architecture(model, test_data)

        result = {
            'num_observers': num_obs,
            'hsi': hsi,
            'final_loss': history['loss'][-1],
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }

        results.append(result)

        # Print results
        if 'slow/fast' in hsi:
            hsi_sf = hsi['slow/fast']
            status = "✅ STABLE" if hsi_sf < 0.3 else "⚠️ UNSTABLE" if hsi_sf < 1.5 else "❌ COLLAPSED"
            print(f"\nResult: HSI = {hsi_sf:.3f} {status}")
        else:
            print(f"\nResult: HSI calculation failed")

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

    # Analyze results
    print("\nRESULTS SUMMARY:")
    print("-"*80)
    print(f"{'Observers':<12} {'HSI':<10} {'Status':<15} {'Time (s)':<10}")
    print("-"*80)

    valid_results = []
    for r in results:
        num_obs = r['num_observers']
        hsi_val = r['hsi'].get('slow/fast', float('nan'))
        training_time = r['training_time']

        if not np.isnan(hsi_val):
            status = "✅ STABLE" if hsi_val < 0.3 else "⚠️ UNSTABLE" if hsi_val < 1.5 else "❌ COLLAPSED"
            valid_results.append((num_obs, hsi_val))
        else:
            status = "ERROR"

        print(f"{num_obs:<12} {hsi_val:<10.3f} {status:<15} {training_time:<10.1f}")

    # Fit power law
    if len(valid_results) >= 3:
        print("\n" + "="*80)
        print("POWER LAW FIT: HSI(N) = k / N^β")
        print("="*80)

        N_vals = np.array([n for n, _ in valid_results])
        HSI_vals = np.array([h for _, h in valid_results])

        # Log-space linear fit: log(HSI) = log(k) - β*log(N)
        log_N = np.log(N_vals)
        log_HSI = np.log(HSI_vals)

        # Linear regression
        A = np.vstack([np.ones(len(log_N)), -log_N]).T
        coeffs, residuals, _, _ = np.linalg.lstsq(A, log_HSI, rcond=None)

        log_k = coeffs[0]
        beta = coeffs[1]
        k = np.exp(log_k)

        print(f"\nFitted parameters:")
        print(f"  k = {k:.2f}")
        print(f"  β = {beta:.3f}")
        print()

        if abs(beta - 2.0) < 0.5:
            print("✅ Power law β ≈ 2.0 confirmed!")
        else:
            print(f"⚠️ Power law β = {beta:.3f} differs from predicted 2.0")

        # Predict critical threshold (HSI = 0.3)
        N_critical = (k / 0.3) ** (1 / beta)
        print(f"\nPredicted N_critical (HSI < 0.3): {N_critical:.0f} observers")

        # R² goodness of fit
        HSI_pred = k / (N_vals ** beta)
        ss_res = np.sum((HSI_vals - HSI_pred) ** 2)
        ss_tot = np.sum((HSI_vals - np.mean(HSI_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"R² = {r_squared:.3f}", end=" ")
        if r_squared > 0.9:
            print("(excellent fit)")
        elif r_squared > 0.7:
            print("(good fit)")
        else:
            print("(poor fit - power law may be wrong)")

    print(f"\n✓ Results saved to: {output_path}")
    print("\nNext: Run 'python3 plot_threshold.py' to visualize")


if __name__ == '__main__':
    main()
