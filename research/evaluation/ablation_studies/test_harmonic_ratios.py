#!/usr/bin/env python3
"""
Harmonic Ratio Experiment - Testing Musical Interval Hypothesis

Tests whether observer network stability follows harmonic principles similar to musical intervals.
Hypothesis: "Consonant" ratios (octave 2:1, fifth 3:2) show correlated stability,
while "dissonant" ratios (tritone √2:1) show anti-correlated or unstable behavior.

Musical Intervals Tested:
- Octave (2:1): Most consonant
- Perfect Fifth (3:2): Very consonant
- Perfect Fourth (4:3): Consonant
- Major Third (5:4): Moderately consonant
- Tritone (45:32 ≈ √2:1): Maximum dissonance
- Minor Second (16:15): Very dissonant
- Harmonic Series (1×, 2×, 3×, 4×): Integer multiples
"""

import sys
import os
sys.path.insert(0, '..')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from architectures.base import AblationArchitecture
import numpy as np
import json
from datetime import datetime
import math

class HarmonicObserverArchitecture(AblationArchitecture):
    """Parametric observer architecture for testing harmonic ratios"""

    def __init__(self, affect_dim=5, num_observers=100):
        super().__init__(affect_dim)
        self.num_observers = num_observers

        # Core hierarchy (matching base architecture)
        self.fast_lstm = nn.LSTM(input_size=affect_dim, hidden_size=16)
        self.c_fast = mx.zeros((1, 16))
        self.medium_lstm = nn.LSTM(input_size=16, hidden_size=16)
        self.c_medium = mx.zeros((1, 16))
        self.slow_gru = nn.GRU(input_size=16, hidden_size=8)

        # Main predictor
        joint_dim = 40  # 16 + 16 + 8
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_dim)
        )

        # Observer distribution: 62% L0, 31% L1, 7% L2
        n_L0 = int(num_observers * 0.62)
        n_L1 = int(num_observers * 0.31)
        n_L2 = num_observers - n_L0 - n_L1

        # Create all observers as simple predictors
        self.observers = []
        for _ in range(num_observers):
            obs = nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            )
            self.observers.append(obs)

    def reset_states(self):
        """Reset all hidden states"""
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

        # Extract layer states
        fast_states.append(model.h_fast.reshape(-1))
        medium_states.append(model.h_medium.reshape(-1))
        slow_states.append(model.h_slow.reshape(-1))

    # Calculate variances
    fast_array = mx.stack(fast_states)
    medium_array = mx.stack(medium_states)
    slow_array = mx.stack(slow_states)

    fast_var = float(mx.var(fast_array))
    medium_var = float(mx.var(medium_array))
    slow_var = float(mx.var(slow_array))

    # HSI ratios
    hsi_slow_fast = slow_var / fast_var if fast_var > 1e-10 else float('nan')
    hsi_medium_fast = medium_var / fast_var if fast_var > 1e-10 else float('nan')

    return {
        'slow/fast': hsi_slow_fast,
        'medium/fast': hsi_medium_fast,
        'fast_var': fast_var,
        'medium_var': medium_var,
        'slow_var': slow_var
    }


def train_and_evaluate(num_observers, epochs=50, seq_length=20):
    """Train architecture and return final HSI"""
    print(f"\n{'='*70}")
    print(f"Testing N={num_observers} observers")
    print(f"{'='*70}")

    model = HarmonicObserverArchitecture(
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
            # Reset states
            model.reset_states()

            # Process sequence
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

    print(f"\nFinal HSI:")
    print(f"  Slow/Fast:   {final_hsi['slow/fast']:.6f}")
    print(f"  Medium/Fast: {final_hsi['medium/fast']:.6f}")

    # Determine stability
    is_stable = final_hsi['slow/fast'] < 0.3
    stability_str = "✓ STABLE" if is_stable else "✗ UNSTABLE"
    print(f"  Status: {stability_str}")

    return {
        'num_observers': num_observers,
        'final_hsi': final_hsi['slow/fast'],
        'final_loss': loss_history[-1],
        'is_stable': is_stable,
        'loss_history': loss_history
    }


def calculate_musical_interval(n1, n2):
    """Calculate the musical interval between two observer counts"""
    ratio = n2 / n1 if n2 > n1 else n1 / n2

    # Convert to cents (1200 cents = 1 octave)
    cents = 1200 * math.log2(ratio)

    # Classify interval
    intervals = {
        'Unison': (0, 50),
        'Minor Second': (50, 150),
        'Major Second': (150, 250),
        'Minor Third': (250, 350),
        'Major Third': (350, 450),
        'Perfect Fourth': (450, 550),
        'Tritone': (550, 650),
        'Perfect Fifth': (650, 750),
        'Minor Sixth': (750, 850),
        'Major Sixth': (850, 950),
        'Minor Seventh': (950, 1050),
        'Major Seventh': (1050, 1150),
        'Octave': (1150, 1250),
        'Beyond Octave': (1250, 10000)
    }

    for name, (low, high) in intervals.items():
        if low <= cents < high:
            return name, cents, ratio

    return 'Unknown', cents, ratio


def main():
    print("="*80)
    print("HARMONIC RATIO EXPERIMENT")
    print("Testing Musical Interval Hypothesis for Observer Networks")
    print("="*80)
    print()
    print("Hypothesis: Observer networks follow harmonic principles.")
    print("Consonant intervals (octave, fifth) → correlated stability")
    print("Dissonant intervals (tritone, minor 2nd) → instability")
    print()

    # Define test configurations based on musical intervals
    # Base frequency: 50 observers
    base = 50

    configurations = [
        # Harmonic series (1×, 2×, 3×, 4×)
        ('Harmonic_1x', base * 1),      # 50
        ('Harmonic_2x', base * 2),      # 100 (octave above)
        ('Harmonic_3x', base * 3),      # 150 (fifth + octave)
        ('Harmonic_4x', base * 4),      # 200 (double octave)

        # Octave pairs (2:1)
        ('Octave_60', 60),
        ('Octave_120', 120),

        # Perfect fifth pairs (3:2)
        ('Fifth_60', 60),
        ('Fifth_90', 90),
        ('Fifth_80', 80),

        # Perfect fourth pairs (4:3)
        ('Fourth_75', 75),
        ('Fourth_100', 100),

        # Major third pairs (5:4)
        ('MajorThird_80', 80),
        ('MajorThird_100', 100),

        # Tritone pairs (45:32 ≈ 1.41)
        ('Tritone_70', 70),
        ('Tritone_99', 99),  # 70 * 1.414 ≈ 99

        # Minor second pairs (16:15 ≈ 1.067)
        ('MinorSecond_75', 75),
        ('MinorSecond_80', 80),  # 75 * 1.067 ≈ 80
    ]

    results = []

    for name, num_obs in configurations:
        result = train_and_evaluate(num_obs, epochs=50)
        result['config_name'] = name
        results.append(result)

    # Analyze harmonic relationships
    print("\n" + "="*80)
    print("HARMONIC RELATIONSHIP ANALYSIS")
    print("="*80)

    # Group by interval type
    interval_groups = {
        'Harmonic Series': [],
        'Octave (2:1)': [],
        'Perfect Fifth (3:2)': [],
        'Perfect Fourth (4:3)': [],
        'Major Third (5:4)': [],
        'Tritone (√2:1)': [],
        'Minor Second (16:15)': []
    }

    # Analyze pairs
    print("\nInterval Analysis:")
    print(f"{'Pair':<30} {'Interval':<20} {'Ratio':<10} {'N1→N2':<15} {'HSI1→HSI2':<20} {'Both Stable?'}")
    print("-"*120)

    tested_pairs = [
        # Harmonic series
        (50, 100, 'Octave'),
        (50, 150, 'Fifth+Octave'),
        (100, 150, 'Perfect Fifth'),

        # Other intervals
        (60, 120, 'Octave'),
        (60, 90, 'Perfect Fifth'),
        (75, 100, 'Perfect Fourth'),
        (80, 100, 'Major Third'),
        (70, 99, 'Tritone'),
        (75, 80, 'Minor Second'),
    ]

    for n1, n2, expected_interval in tested_pairs:
        # Find results
        r1 = next((r for r in results if r['num_observers'] == n1), None)
        r2 = next((r for r in results if r['num_observers'] == n2), None)

        if r1 and r2:
            interval_name, cents, ratio = calculate_musical_interval(n1, n2)

            hsi_str = f"{r1['final_hsi']:.3f} → {r2['final_hsi']:.3f}"
            both_stable = "✓ YES" if (r1['is_stable'] and r2['is_stable']) else "✗ NO"

            print(f"{n1}→{n2:<27} {interval_name:<20} {ratio:.3f}      {n1}→{n2:<12} {hsi_str:<20} {both_stable}")

    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)

    # Consonant intervals (octave, fifth, fourth)
    consonant_configs = [50, 60, 80, 90, 100, 120, 150, 200]
    consonant_results = [r for r in results if r['num_observers'] in consonant_configs]
    consonant_stable_rate = sum(1 for r in consonant_results if r['is_stable']) / len(consonant_results)
    consonant_avg_hsi = np.mean([r['final_hsi'] for r in consonant_results])

    # Dissonant intervals (tritone, minor second)
    dissonant_configs = [70, 75, 99]
    dissonant_results = [r for r in results if r['num_observers'] in dissonant_configs]
    dissonant_stable_rate = sum(1 for r in dissonant_results if r['is_stable']) / len(dissonant_results) if dissonant_results else 0
    dissonant_avg_hsi = np.mean([r['final_hsi'] for r in dissonant_results]) if dissonant_results else 0

    print(f"\nConsonant Intervals (octave, fifth, fourth):")
    print(f"  Configs tested: {consonant_configs}")
    print(f"  Stability rate: {consonant_stable_rate*100:.1f}%")
    print(f"  Average HSI: {consonant_avg_hsi:.4f}")

    print(f"\nDissonant Intervals (tritone, minor second):")
    print(f"  Configs tested: {dissonant_configs}")
    print(f"  Stability rate: {dissonant_stable_rate*100:.1f}%")
    print(f"  Average HSI: {dissonant_avg_hsi:.4f}")

    # Hypothesis test
    print(f"\n{'='*80}")
    print("HYPOTHESIS TEST")
    print("="*80)
    print(f"\nH0: Consonant and dissonant intervals show no difference in stability")
    print(f"H1: Consonant intervals are more stable than dissonant intervals")
    print()

    if consonant_stable_rate > dissonant_stable_rate:
        print(f"✓ SUPPORTED: Consonant stability ({consonant_stable_rate*100:.1f}%) > Dissonant stability ({dissonant_stable_rate*100:.1f}%)")
    else:
        print(f"✗ NOT SUPPORTED: Consonant stability ({consonant_stable_rate*100:.1f}%) ≤ Dissonant stability ({dissonant_stable_rate*100:.1f}%)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'harmonic_ratios',
        'hypothesis': 'Musical interval relationships predict observer network stability',
        'results': results,
        'summary': {
            'consonant_stability_rate': float(consonant_stable_rate),
            'consonant_avg_hsi': float(consonant_avg_hsi),
            'dissonant_stability_rate': float(dissonant_stable_rate),
            'dissonant_avg_hsi': float(dissonant_avg_hsi),
            'hypothesis_supported': consonant_stable_rate > dissonant_stable_rate
        }
    }

    with open('harmonic_ratios_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to harmonic_ratios_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
