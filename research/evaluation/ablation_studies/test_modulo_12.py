#!/usr/bin/env python3
"""
Modulo 12 Experiment - Testing Phase Equivalence

If the period ≈ 12 oscillation is real, then observer counts that are equivalent
modulo 12 should show similar stability patterns.

HYPOTHESIS: N₁ ≡ N₂ (mod 12) → HSI(N₁) ≈ HSI(N₂)

Example: 50, 62, 74, 86, 98 all ≡ 2 (mod 12)
         If hypothesis is true, these should have similar HSI values.

This tests whether the "phase" (position in 12-observer cycle) matters more
than the absolute observer count.
"""

import sys
sys.path.insert(0, '..')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from architectures.base import AblationArchitecture
import numpy as np
import json
from datetime import datetime

class ParametricObserverArchitecture(AblationArchitecture):
    """Architecture with configurable observer count"""

    def __init__(self, affect_dim=5, num_observers=100):
        super().__init__(affect_dim)
        self.num_observers = num_observers

        # Core hierarchy
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=16)
        self.c_fast = mx.zeros((1, 16))
        self.medium_lstm = nn.LSTM(input_size=16, hidden_size=16)
        self.c_medium = mx.zeros((1, 16))
        self.slow_gru = nn.GRU(input_size=16, hidden_size=8)

        # Main predictor
        joint_dim = 40
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_dim)
        )

        # Observers
        self.observers = []
        for _ in range(num_observers):
            obs = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            self.observers.append(obs)

        self.prev_state = mx.zeros((1, joint_dim))

    def reset_states(self):
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def __call__(self, affect):
        """Forward pass with observer corrections"""
        # Ensure affect has correct shape
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Fast layer
        h_fast_seq, c_fast_seq = self.fast_lstm(affect, hidden=self.h_fast, cell=self.c_fast)
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        # Medium layer
        h_med_seq, c_med_seq = self.medium_lstm(self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium)
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        # Slow layer
        h_slow_seq = self.slow_gru(self.h_medium[:, None, :], hidden=self.h_slow)
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Phenomenal state (40-D)
        phenomenal_state = mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)

        # Predictive processing
        predicted_state = self.predictor(phenomenal_state)

        # Observer errors (included in loss for gradient flow)
        observer_errors = []
        if self.num_observers > 0:
            for observer in self.observers:
                obs_pred = observer(predicted_state)
                observer_errors.append(mx.mean((obs_pred - phenomenal_state) ** 2))

        # Total surprise (main prediction error + observer errors)
        main_error = mx.mean((predicted_state - phenomenal_state) ** 2)
        observer_error = mx.mean(mx.array(observer_errors)) if observer_errors else mx.array(0.0)
        surprise = float(main_error + 0.1 * observer_error)

        return phenomenal_state, surprise


def calculate_hsi(model, sequence, seq_length=20):
    """Calculate HSI over a sequence"""
    fast_states = []
    medium_states = []
    slow_states = []

    for t in range(seq_length):
        affect = sequence[t:t+1, :]
        _, _ = model(affect)

        fast_states.append(model.h_fast.reshape(-1))
        medium_states.append(model.h_medium.reshape(-1))
        slow_states.append(model.h_slow.reshape(-1))

    # Calculate variances
    fast_array = mx.stack(fast_states)
    slow_array = mx.stack(slow_states)

    fast_var = float(mx.var(fast_array))
    slow_var = float(mx.var(slow_array))

    hsi_slow_fast = slow_var / fast_var if fast_var > 1e-10 else float('nan')

    return {
        'slow/fast': hsi_slow_fast,
        'fast_var': fast_var,
        'slow_var': slow_var
    }


def train_and_evaluate(num_observers, epochs=50, seq_length=20):
    """Train architecture and return final HSI"""
    print(f"\n{'='*70}")
    print(f"Testing N={num_observers} observers (N mod 12 = {num_observers % 12})")
    print(f"{'='*70}")

    model = ParametricObserverArchitecture(
        affect_dim=5,
        num_observers=num_observers
    )

    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-5)

    # Generate synthetic data
    np.random.seed(42)
    sequences = []
    for _ in range(10):
        valence = np.random.uniform(-1, 1, seq_length)
        arousal = np.random.uniform(0, 1, seq_length)
        fear = np.random.uniform(0, 0.5, seq_length)
        sorrow = np.random.uniform(0, 0.3, seq_length)
        boredom = np.random.uniform(0, 0.2, seq_length)

        sequence = np.stack([valence, arousal, fear, sorrow, boredom], axis=1)
        sequences.append(mx.array(sequence, dtype=mx.float32))

    # Training loop
    loss_history = []

    for epoch in range(epochs):
        epoch_losses = []

        for seq_idx, sequence in enumerate(sequences):
            model.reset_states()

            seq_loss = 0.0
            for t in range(seq_length):
                affect = sequence[t:t+1, :]

                def loss_fn():
                    state, surprise = model(affect)
                    return mx.array(surprise)

                loss_and_grad = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad()

                optimizer.update(model, grads)
                mx.eval(model.parameters())

                seq_loss += float(loss)

            epoch_losses.append(seq_loss / seq_length)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Evaluate HSI
    model.reset_states()
    final_hsi = calculate_hsi(model, sequences[0], seq_length)

    print(f"\nFinal HSI: {final_hsi['slow/fast']:.6f}")

    is_stable = final_hsi['slow/fast'] < 0.3
    status = "✓ STABLE" if is_stable else "✗ UNSTABLE"
    print(f"Status: {status}")

    return {
        'num_observers': num_observers,
        'modulo_12': num_observers % 12,
        'final_hsi': final_hsi['slow/fast'],
        'final_loss': loss_history[-1],
        'is_stable': is_stable,
        'loss_history': loss_history
    }


def main():
    print("="*80)
    print("MODULO 12 EXPERIMENT")
    print("Testing Phase Equivalence Hypothesis")
    print("="*80)
    print()
    print("HYPOTHESIS: If period ≈ 12, then N₁ ≡ N₂ (mod 12) → HSI(N₁) ≈ HSI(N₂)")
    print()
    print("Testing 5 replications for each residue class (mod 12)")
    print("="*80)
    print()

    # Test configurations: 5 examples for each residue class mod 12
    configurations = []

    # For each residue class 0-11, test 5 different observer counts
    for residue in range(12):
        # Start at residue + 50, then add 12 four more times
        base = 50 + residue
        for i in range(5):
            n = base + (i * 12)
            if n > 0:  # Make sure we don't have negative or zero observers
                configurations.append(n)

    # Sort for cleaner output
    configurations.sort()

    print(f"Testing {len(configurations)} configurations...")
    print()

    results = []
    for num_obs in configurations:
        result = train_and_evaluate(num_obs, epochs=50)
        results.append(result)

    # Analysis by residue class
    print("\n" + "="*80)
    print("PHASE ANALYSIS (Grouped by N mod 12)")
    print("="*80)
    print()

    residue_groups = {}
    for r in range(12):
        residue_groups[r] = [res for res in results if res['modulo_12'] == r]

    print(f"{'Phase':<8} {'Count':<8} {'Avg HSI':<12} {'Std HSI':<12} {'Stability %':<15} {'Range'}")
    print("-"*80)

    for residue in range(12):
        group = residue_groups[residue]
        if group:
            hsi_values = [r['final_hsi'] for r in group]
            avg_hsi = np.mean(hsi_values)
            std_hsi = np.std(hsi_values)
            stable_pct = 100 * sum(1 for r in group if r['is_stable']) / len(group)
            hsi_range = max(hsi_values) - min(hsi_values)

            print(f"{residue:<8} {len(group):<8} {avg_hsi:<12.6f} {std_hsi:<12.6f} {stable_pct:<15.1f} {hsi_range:.6f}")

    # Statistical test: Is variance WITHIN residue classes smaller than BETWEEN?
    print("\n" + "="*80)
    print("STATISTICAL HYPOTHESIS TEST")
    print("="*80)
    print()

    # Calculate within-group variance
    within_var = 0
    total_n = 0
    for residue in range(12):
        group = residue_groups[residue]
        if len(group) > 1:
            hsi_values = [r['final_hsi'] for r in group]
            within_var += np.var(hsi_values) * len(group)
            total_n += len(group)

    within_var /= total_n if total_n > 0 else 1

    # Calculate between-group variance
    all_hsi = [r['final_hsi'] for r in results]
    grand_mean = np.mean(all_hsi)

    between_var = 0
    for residue in range(12):
        group = residue_groups[residue]
        if group:
            group_hsi = [r['final_hsi'] for r in group]
            group_mean = np.mean(group_hsi)
            between_var += len(group) * (group_mean - grand_mean) ** 2

    between_var /= total_n if total_n > 0 else 1

    # F-ratio
    f_ratio = between_var / within_var if within_var > 0 else float('inf')

    print(f"Within-phase variance:  {within_var:.8f}")
    print(f"Between-phase variance: {between_var:.8f}")
    print(f"F-ratio (between/within): {f_ratio:.4f}")
    print()

    if f_ratio > 1.5:
        print("✓ PHASE MATTERS: Between-phase variance > within-phase variance")
        print("  → Phase (N mod 12) is a significant predictor of HSI")
    else:
        print("✗ PHASE DOESN'T MATTER: Within-phase variance is as large as between-phase")
        print("  → Phase (N mod 12) does NOT predict HSI")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'modulo_12',
        'hypothesis': 'Observer counts equivalent mod 12 have similar HSI',
        'results': results,
        'residue_analysis': {
            r: {
                'count': len(residue_groups[r]),
                'avg_hsi': float(np.mean([res['final_hsi'] for res in residue_groups[r]])) if residue_groups[r] else None,
                'std_hsi': float(np.std([res['final_hsi'] for res in residue_groups[r]])) if residue_groups[r] else None,
                'stability_rate': float(sum(1 for res in residue_groups[r] if res['is_stable']) / len(residue_groups[r])) if residue_groups[r] else None
            }
            for r in range(12)
        },
        'statistics': {
            'within_variance': float(within_var),
            'between_variance': float(between_var),
            'f_ratio': float(f_ratio),
            'hypothesis_supported': f_ratio > 1.5
        }
    }

    with open('modulo_12_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to modulo_12_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
