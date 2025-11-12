#!/usr/bin/env python3
"""
Control Experiment 2: Parameter-Matched Baseline

HYPOTHESIS: Maybe stability comes from having MORE PARAMETERS, not observers specifically.

NULL HYPOTHESIS: Any architecture with ~same parameter count will show similar stability.

Tests:
A. Baseline (no observers): ~6K params
B. 100 observers: ~330K params
C. Wider layers (no observers): ~330K params (matched)
D. Deeper hierarchy (no observers): ~330K params (matched)

If B better than C&D: Observer structure matters (not just param count)
If B ≈ C ≈ D: It's just about capacity → Null hypothesis true
"""

import sys
sys.path.insert(0, '../..')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from architectures.base import AblationArchitecture
import numpy as np
import json
from datetime import datetime


class WideArchitecture(AblationArchitecture):
    """Wider layers, matched parameter count"""

    def __init__(self, affect_dim=5):
        super().__init__(affect_dim)

        # Make layers much wider to match observer param count
        # Original: 16+16+8 = 40-D state
        # Wide: 64+64+32 = 160-D state (4x wider)
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=64)
        self.c_fast = mx.zeros((1, 64))
        self.medium_lstm = nn.LSTM(input_size=64, hidden_size=64)
        self.c_medium = mx.zeros((1, 64))
        self.slow_gru = nn.GRU(input_size=64, hidden_size=32)

        # Predictor
        joint_dim = 160
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 256),
            nn.ReLU(),
            nn.Linear(256, joint_dim)
        )

    def reset_states(self):
        self.h_fast = mx.zeros((1, 64))
        self.c_fast = mx.zeros((1, 64))
        self.h_medium = mx.zeros((1, 64))
        self.c_medium = mx.zeros((1, 64))
        self.h_slow = mx.zeros((1, 32))

    def __call__(self, affect):
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Process layers
        h_fast_seq, c_fast_seq = self.fast_lstm(affect, hidden=self.h_fast, cell=self.c_fast)
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 64)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 64)

        h_med_seq, c_med_seq = self.medium_lstm(self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium)
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 64)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 64)

        h_slow_seq = self.slow_gru(self.h_medium[:, None, :], hidden=self.h_slow)
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 32)

        # Phenomenal state
        phenomenal_state = mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)

        # Predictive processing
        predicted_state = self.predictor(phenomenal_state)
        main_error = mx.mean((predicted_state - phenomenal_state) ** 2)
        surprise = float(main_error)

        return phenomenal_state, surprise


class DeepArchitecture(AblationArchitecture):
    """More layers, matched parameter count"""

    def __init__(self, affect_dim=5):
        super().__init__(affect_dim)

        # Add more layers instead of wider
        self.layer1 = nn.LSTM(input_size=affect_dim, hidden_size=32)
        self.c1 = mx.zeros((1, 32))
        self.layer2 = nn.LSTM(input_size=32, hidden_size=32)
        self.c2 = mx.zeros((1, 32))
        self.layer3 = nn.LSTM(input_size=32, hidden_size=16)
        self.c3 = mx.zeros((1, 16))
        self.layer4 = nn.LSTM(input_size=16, hidden_size=16)
        self.c4 = mx.zeros((1, 16))
        self.layer5 = nn.GRU(input_size=16, hidden_size=8)

        # Store states for HSI calculation
        self.h_fast = mx.zeros((1, 32))  # layer1
        self.h_medium = mx.zeros((1, 16))  # layer3
        self.h_slow = mx.zeros((1, 8))  # layer5

        # Predictor
        joint_dim = 56  # 32+16+8
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, joint_dim)
        )

    def reset_states(self):
        self.h1 = mx.zeros((1, 32))
        self.c1 = mx.zeros((1, 32))
        self.h2 = mx.zeros((1, 32))
        self.c2 = mx.zeros((1, 32))
        self.h3 = mx.zeros((1, 16))
        self.c3 = mx.zeros((1, 16))
        self.h4 = mx.zeros((1, 16))
        self.c4 = mx.zeros((1, 16))
        self.h5 = mx.zeros((1, 8))

        # For HSI
        self.h_fast = self.h1
        self.h_medium = self.h3
        self.h_slow = self.h5

    def __call__(self, affect):
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Deep processing
        h1_seq, c1_seq = self.layer1(affect, hidden=self.h1, cell=self.c1)
        self.h1 = h1_seq[:, -1, :].reshape(1, 32)
        self.c1 = c1_seq[:, -1, :].reshape(1, 32)

        h2_seq, c2_seq = self.layer2(self.h1[:, None, :], hidden=self.h2, cell=self.c2)
        self.h2 = h2_seq[:, -1, :].reshape(1, 32)
        self.c2 = c2_seq[:, -1, :].reshape(1, 32)

        h3_seq, c3_seq = self.layer3(self.h2[:, None, :], hidden=self.h3, cell=self.c3)
        self.h3 = h3_seq[:, -1, :].reshape(1, 16)
        self.c3 = c3_seq[:, -1, :].reshape(1, 16)

        h4_seq, c4_seq = self.layer4(self.h3[:, None, :], hidden=self.h4, cell=self.c4)
        self.h4 = h4_seq[:, -1, :].reshape(1, 16)
        self.c4 = c4_seq[:, -1, :].reshape(1, 16)

        h5_seq = self.layer5(self.h4[:, None, :], hidden=self.h5)
        self.h5 = h5_seq[:, -1, :].reshape(1, 8)

        # Update for HSI
        self.h_fast = self.h1
        self.h_medium = self.h3
        self.h_slow = self.h5

        # Phenomenal state
        phenomenal_state = mx.concatenate([self.h1, self.h3, self.h5], axis=1)

        # Predictive processing
        predicted_state = self.predictor(phenomenal_state)
        main_error = mx.mean((predicted_state - phenomenal_state) ** 2)
        surprise = float(main_error)

        return phenomenal_state, surprise


def calculate_hsi(model, sequence, seq_length=20):
    """Calculate HSI"""
    fast_states = []
    slow_states = []

    for t in range(seq_length):
        affect = sequence[t:t+1, :]
        _, _ = model(affect)
        fast_states.append(model.h_fast.reshape(-1))
        slow_states.append(model.h_slow.reshape(-1))

    fast_array = mx.stack(fast_states)
    slow_array = mx.stack(slow_states)

    fast_var = float(mx.var(fast_array))
    slow_var = float(mx.var(slow_array))

    return slow_var / fast_var if fast_var > 1e-10 else float('nan')


def count_parameters(model):
    """Count trainable parameters"""
    total = 0
    for params in model.parameters().values():
        for p in params:
            if hasattr(p, 'size'):
                total += p.size
    return total


def train_and_evaluate(arch_type, epochs=50, seq_length=20):
    """Train and evaluate architecture"""
    print(f"\n{'='*70}")
    print(f"Testing: {arch_type}")
    print(f"{'='*70}")

    # Create model based on type
    if arch_type == 'baseline':
        from test_harmonic_ratios import HarmonicObserverArchitecture
        model = HarmonicObserverArchitecture(affect_dim=5, num_observers=0)
    elif arch_type == 'observers':
        from test_harmonic_ratios import HarmonicObserverArchitecture
        model = HarmonicObserverArchitecture(affect_dim=5, num_observers=100)
    elif arch_type == 'wide':
        model = WideArchitecture(affect_dim=5)
    elif arch_type == 'deep':
        model = DeepArchitecture(affect_dim=5)

    # Count parameters
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-5)

    # Generate data
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

    # Training
    loss_history = []
    for epoch in range(epochs):
        epoch_losses = []
        for sequence in sequences:
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

    is_stable = final_hsi < 0.3
    status = "✓ STABLE" if is_stable else "✗ UNSTABLE"
    print(f"Final HSI: {final_hsi:.6f} - {status}")

    return {
        'arch_type': arch_type,
        'n_params': n_params,
        'final_hsi': final_hsi,
        'final_loss': loss_history[-1],
        'is_stable': is_stable
    }


def main():
    print("="*80)
    print("CONTROL EXPERIMENT: Parameter-Matched Baseline")
    print("="*80)
    print()
    print("Testing whether effect is due to parameter count vs. observer structure")
    print()

    arch_types = ['baseline', 'observers', 'wide', 'deep']
    results = []

    for arch_type in arch_types:
        result = train_and_evaluate(arch_type, epochs=50)
        results.append(result)

    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    print()

    print(f"{'Architecture':<20} {'Parameters':<15} {'Final HSI':<15} {'Stable?'}")
    print("-"*70)

    for r in results:
        stable_str = "✓" if r['is_stable'] else "✗"
        print(f"{r['arch_type']:<20} {r['n_params']:<15,} {r['final_hsi']:<15.6f} {stable_str}")

    # Hypothesis test
    print("\n" + "="*80)
    print("HYPOTHESIS TEST")
    print("="*80)
    print()

    obs_result = next(r for r in results if r['arch_type'] == 'observers')
    wide_result = next(r for r in results if r['arch_type'] == 'wide')
    deep_result = next(r for r in results if r['arch_type'] == 'deep')

    obs_better_than_wide = obs_result['final_hsi'] < wide_result['final_hsi'] * 0.8
    obs_better_than_deep = obs_result['final_hsi'] < deep_result['final_hsi'] * 0.8

    if obs_better_than_wide and obs_better_than_deep:
        print("✓ OBSERVER STRUCTURE MATTERS:")
        print("  Observers outperform parameter-matched alternatives")
        print("  → Effect is architectural, not just about capacity")
        interpretation = "structure_matters"
    else:
        print("✗ NULL HYPOTHESIS:")
        print("  Parameter-matched baselines perform similarly")
        print("  → It's just about having more parameters")
        interpretation = "just_parameters"

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'parameter_matched',
        'hypothesis': 'Is it about observer structure or just parameter count?',
        'results': results,
        'interpretation': interpretation
    }

    with open('parameter_matched_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to parameter_matched_results.json")
    print()
    print("="*80)


if __name__ == '__main__':
    main()
