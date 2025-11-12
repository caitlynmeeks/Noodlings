#!/usr/bin/env python3
"""
Phase Spacing Experiment - Testing Interval Distance Hypothesis

Musical intervals are defined by RATIOS or DIFFERENCES in frequency.
If observer networks follow harmonic principles, the SPACING between
observer counts should matter.

HYPOTHESIS: Pairs of configurations separated by musically significant
intervals (12, 6, 4, 3) should show predictable stability relationships.

Example:
- Pairs differing by 12 (octave): both should be stable
- Pairs differing by 6 (tritone): should show maximum contrast
- Pairs differing by 4 (major third): moderate correlation
- Pairs differing by 3 (minor third): moderate correlation

This tests whether the RELATIVE difference matters more than absolute counts.
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
from itertools import combinations

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

    hsi_slow_fast = slow_var / fast_var if fast_var > 1e-10 else float('nan')

    return hsi_slow_fast


def train_and_evaluate(num_observers, epochs=50, seq_length=20):
    """Train architecture and return final HSI"""
    print(f"  N={num_observers}: ", end='', flush=True)

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

    # Training loop (quiet mode)
    for epoch in range(epochs):
        for seq_idx, sequence in enumerate(sequences):
            model.reset_states()

            for t in range(seq_length):
                affect = sequence[t:t+1, :]

                def loss_fn():
                    state, surprise = model(affect)
                    return mx.array(surprise)

                loss_and_grad = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad()

                optimizer.update(model, grads)
                mx.eval(model.parameters())

    # Evaluate HSI
    model.reset_states()
    final_hsi = calculate_hsi(model, sequences[0], seq_length)

    is_stable = final_hsi < 0.3
    status = "✓" if is_stable else "✗"
    print(f"HSI={final_hsi:.4f} {status}")

    return {
        'num_observers': num_observers,
        'final_hsi': final_hsi,
        'is_stable': is_stable
    }


def main():
    print("="*80)
    print("PHASE SPACING EXPERIMENT")
    print("Testing Musical Interval Distance Hypothesis")
    print("="*80)
    print()
    print("HYPOTHESIS: Pairs separated by musically significant intervals")
    print("            show predictable stability relationships")
    print()

    # Test spacings: 1, 2, 3, 4, 6, 12
    # These correspond to: semitone, whole tone, minor third, major third, tritone, octave
    spacings_to_test = [1, 2, 3, 4, 6, 12]

    # Base configurations to start from
    base_configs = [60, 80, 100, 120]

    # Generate all pairs
    all_configs = set()
    test_pairs = {spacing: [] for spacing in spacings_to_test}

    for base in base_configs:
        for spacing in spacings_to_test:
            n1 = base
            n2 = base + spacing
            all_configs.add(n1)
            all_configs.add(n2)
            test_pairs[spacing].append((n1, n2))

    all_configs = sorted(list(all_configs))

    print(f"Testing {len(all_configs)} unique configurations...")
    print()

    # Train all configurations
    results_dict = {}
    for num_obs in all_configs:
        result = train_and_evaluate(num_obs, epochs=50)
        results_dict[num_obs] = result

    # Analyze by spacing
    print("\n" + "="*80)
    print("SPACING ANALYSIS")
    print("="*80)
    print()

    spacing_analysis = {}

    for spacing in spacings_to_test:
        pairs = test_pairs[spacing]

        hsi_correlations = []
        stability_concordance = 0
        total_pairs = 0

        print(f"\nSpacing = {spacing} observers:")
        print(f"{'Pair':<15} {'HSI₁':<10} {'HSI₂':<10} {'ΔHS I':<10} {'Both Stable?'}")
        print("-"*60)

        for n1, n2 in pairs:
            r1 = results_dict[n1]
            r2 = results_dict[n2]

            hsi1 = r1['final_hsi']
            hsi2 = r2['final_hsi']
            delta_hsi = abs(hsi2 - hsi1)

            # Check if both stable or both unstable
            both_stable = (r1['is_stable'] and r2['is_stable'])
            both_unstable = (not r1['is_stable']) and (not r2['is_stable'])
            concordant = both_stable or both_unstable

            if concordant:
                stability_concordance += 1
            total_pairs += 1

            hsi_correlations.append(delta_hsi)

            concordance_str = "✓ YES" if concordant else "✗ NO"
            print(f"{n1}→{n2:<11} {hsi1:<10.4f} {hsi2:<10.4f} {delta_hsi:<10.4f} {concordance_str}")

        avg_delta = np.mean(hsi_correlations)
        std_delta = np.std(hsi_correlations)
        concordance_rate = stability_concordance / total_pairs if total_pairs > 0 else 0

        print(f"\nSummary:")
        print(f"  Average ΔHSI: {avg_delta:.4f} ± {std_delta:.4f}")
        print(f"  Concordance rate: {concordance_rate*100:.1f}%")

        spacing_analysis[spacing] = {
            'avg_delta_hsi': float(avg_delta),
            'std_delta_hsi': float(std_delta),
            'concordance_rate': float(concordance_rate),
            'pairs': pairs
        }

    # Overall analysis
    print("\n" + "="*80)
    print("OVERALL PATTERN ANALYSIS")
    print("="*80)
    print()

    print(f"{'Spacing':<10} {'Avg ΔHSI':<15} {'Std ΔHSI':<15} {'Concordance %':<15} {'Interpretation'}")
    print("-"*80)

    for spacing in spacings_to_test:
        data = spacing_analysis[spacing]
        avg = data['avg_delta_hsi']
        std = data['std_delta_hsi']
        conc = data['concordance_rate'] * 100

        # Interpretation
        if spacing == 12:
            expected = "Octave: high concordance"
        elif spacing == 6:
            expected = "Tritone: low concordance"
        elif spacing in [3, 4]:
            expected = "Third: moderate"
        else:
            expected = "Small interval"

        print(f"{spacing:<10} {avg:<15.4f} {std:<15.4f} {conc:<15.1f} {expected}")

    # Hypothesis test
    print("\n" + "="*80)
    print("HYPOTHESIS TEST")
    print("="*80)
    print()

    # Octave (12) should have high concordance
    octave_concordance = spacing_analysis[12]['concordance_rate']
    # Tritone (6) should have low concordance
    tritone_concordance = spacing_analysis[6]['concordance_rate']

    print(f"Octave (spacing=12) concordance: {octave_concordance*100:.1f}%")
    print(f"Tritone (spacing=6) concordance: {tritone_concordance*100:.1f}%")
    print()

    if octave_concordance > tritone_concordance:
        print("✓ HYPOTHESIS SUPPORTED:")
        print("  Octave intervals show higher stability concordance than tritone")
        print("  → Musical spacing relationships matter!")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED:")
        print("  No clear difference between octave and tritone spacing")
        print("  → Musical analogy may be spurious")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'phase_spacing',
        'hypothesis': 'Observer count spacing follows musical interval principles',
        'results': list(results_dict.values()),
        'spacing_analysis': spacing_analysis,
        'hypothesis_test': {
            'octave_concordance': float(octave_concordance),
            'tritone_concordance': float(tritone_concordance),
            'supported': octave_concordance > tritone_concordance
        }
    }

    with open('phase_spacing_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to phase_spacing_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
