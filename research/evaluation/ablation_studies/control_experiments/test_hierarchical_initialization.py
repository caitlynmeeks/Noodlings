#!/usr/bin/env python3
"""
Control Experiment: Hierarchical Initialization

HYPOTHESIS: Can we achieve good HSI through DESIGNED initialization
            rather than observers or random luck?

Tests whether amplifying fast layer weights and dampening slow layer weights
gives reliably good HSI without observers.

If YES: Observer effect is just initialization, we can design it directly
If NO: Something more complex is happening
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


class HierarchicalInitArchitecture(AblationArchitecture):
    """Architecture with designed hierarchical initialization"""

    def __init__(self, affect_dim=5, fast_scale=1.0, slow_scale=1.0):
        super().__init__(affect_dim)

        self.fast_scale = fast_scale
        self.slow_scale = slow_scale

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

        # Apply hierarchical scaling to initialization (after all layers created)
        self._apply_hierarchical_init()

    def _apply_hierarchical_init(self):
        """Scale layer weights to bias toward hierarchical separation"""

        # Get all parameters and scale them
        params = self.parameters()

        # Scale fast LSTM weights
        for key in params['fast_lstm']:
            if isinstance(params['fast_lstm'][key], dict):
                for gate in params['fast_lstm'][key]:
                    params['fast_lstm'][key][gate] = params['fast_lstm'][key][gate] * self.fast_scale

        # Scale slow GRU weights
        for key in params['slow_gru']:
            if isinstance(params['slow_gru'][key], dict):
                for gate in params['slow_gru'][key]:
                    params['slow_gru'][key][gate] = params['slow_gru'][key][gate] * self.slow_scale

        # Update model with scaled params
        self.update(params)

    def reset_states(self):
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def __call__(self, affect):
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

        # Phenomenal state
        phenomenal_state = mx.concatenate([self.h_fast, self.h_medium, self.h_slow], axis=1)

        # Predictive processing
        predicted_state = self.predictor(phenomenal_state)
        main_error = mx.mean((predicted_state - phenomenal_state) ** 2)
        surprise = float(main_error)

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


def test_initialization_config(fast_scale, slow_scale, num_trials=20, train=False, epochs=50):
    """Test a specific initialization configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: fast_scale={fast_scale:.2f}, slow_scale={slow_scale:.2f}")
    print(f"Training: {'YES' if train else 'NO (init only)'}")
    print(f"{'='*70}")

    # Generate test sequence
    np.random.seed(999)
    valence = np.random.uniform(-1, 1, 20)
    arousal = np.random.uniform(0, 1, 20)
    fear = np.random.uniform(0, 0.5, 20)
    sorrow = np.random.uniform(0, 0.3, 20)
    boredom = np.random.uniform(0, 0.2, 20)
    sequence = np.stack([valence, arousal, fear, sorrow, boredom], axis=1)
    test_sequence = mx.array(sequence, dtype=mx.float32)

    hsi_values = []

    for trial in range(num_trials):
        mx.random.seed(trial)
        np.random.seed(trial)

        model = HierarchicalInitArchitecture(
            affect_dim=5,
            fast_scale=fast_scale,
            slow_scale=slow_scale
        )

        if train:
            # Quick training
            optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-5)

            for epoch in range(epochs):
                model.reset_states()
                for t in range(20):
                    affect = test_sequence[t:t+1, :]

                    def loss_fn():
                        state, surprise = model(affect)
                        return mx.array(surprise)

                    loss_and_grad = nn.value_and_grad(model, loss_fn)
                    loss, grads = loss_and_grad()
                    optimizer.update(model, grads)
                    mx.eval(model.parameters())

        # Measure HSI
        model.reset_states()
        hsi = calculate_hsi(model, test_sequence, 20)
        hsi_values.append(hsi)

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{num_trials}: HSI={hsi:.6f}")

    mean_hsi = np.mean(hsi_values)
    std_hsi = np.std(hsi_values)
    min_hsi = np.min(hsi_values)
    max_hsi = np.max(hsi_values)
    stable_count = sum(1 for h in hsi_values if h < 0.3)
    stable_rate = stable_count / len(hsi_values)

    print(f"\n  Results:")
    print(f"    Mean HSI:    {mean_hsi:.6f}")
    print(f"    Std Dev:     {std_hsi:.6f}")
    print(f"    Min:         {min_hsi:.6f}")
    print(f"    Max:         {max_hsi:.6f}")
    print(f"    Stable rate: {stable_rate*100:.1f}% ({stable_count}/{num_trials})")

    return {
        'fast_scale': fast_scale,
        'slow_scale': slow_scale,
        'trained': train,
        'mean_hsi': float(mean_hsi),
        'std_hsi': float(std_hsi),
        'min_hsi': float(min_hsi),
        'max_hsi': float(max_hsi),
        'stable_rate': float(stable_rate),
        'hsi_values': [float(h) for h in hsi_values]
    }


def main():
    print("="*80)
    print("CONTROL EXPERIMENT: Hierarchical Initialization")
    print("="*80)
    print()
    print("Testing whether we can DESIGN good HSI through initialization")
    print("without observers or random luck")
    print()

    results = []

    # Test configurations
    configs = [
        # Baseline (standard init)
        (1.0, 1.0, False, "Baseline (init only)"),
        (1.0, 1.0, True, "Baseline (trained)"),

        # Mild hierarchy bias
        (1.5, 0.7, False, "Mild hierarchy (init only)"),
        (1.5, 0.7, True, "Mild hierarchy (trained)"),

        # Strong hierarchy bias
        (2.0, 0.5, False, "Strong hierarchy (init only)"),
        (2.0, 0.5, True, "Strong hierarchy (trained)"),

        # Extreme hierarchy bias
        (3.0, 0.3, False, "Extreme hierarchy (init only)"),
        (3.0, 0.3, True, "Extreme hierarchy (trained)"),
    ]

    for fast_scale, slow_scale, train, label in configs:
        result = test_initialization_config(fast_scale, slow_scale, num_trials=20, train=train, epochs=50)
        result['label'] = label
        results.append(result)

    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    print()

    print(f"{'Configuration':<35} {'Mean HSI':<12} {'Stable Rate':<15} {'Effect'}")
    print("-"*80)

    baseline_init = next(r for r in results if r['fast_scale'] == 1.0 and not r['trained'])

    for r in results:
        improvement = (baseline_init['mean_hsi'] - r['mean_hsi']) / baseline_init['mean_hsi'] * 100
        effect = f"{improvement:+.0f}%" if r['fast_scale'] != 1.0 or r['trained'] else "baseline"

        print(f"{r['label']:<35} {r['mean_hsi']:<12.6f} {r['stable_rate']*100:<14.1f}% {effect}")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    extreme_init = next(r for r in results if r['fast_scale'] == 3.0 and not r['trained'])
    extreme_trained = next(r for r in results if r['fast_scale'] == 3.0 and r['trained'])

    print(f"1. INITIALIZATION CONTROL:")
    print(f"   Baseline:          HSI = {baseline_init['mean_hsi']:.3f}")
    print(f"   Extreme hierarchy: HSI = {extreme_init['mean_hsi']:.3f}")
    print(f"   Improvement: {(1 - extreme_init['mean_hsi']/baseline_init['mean_hsi'])*100:.1f}%")
    print()

    if extreme_init['stable_rate'] > 0.5:
        print("   ✓ HIERARCHICAL INITIALIZATION WORKS!")
        print("   → Can achieve good HSI through initialization design")
        print("   → No observers needed")
    else:
        print("   ✗ Hierarchical initialization helps but not enough")
        print("   → HSI still too high for stability")

    print()
    print(f"2. TRAINING EFFECT:")
    baseline_trained = next(r for r in results if r['fast_scale'] == 1.0 and r['trained'])
    print(f"   Baseline trained:  HSI = {baseline_trained['mean_hsi']:.3f}")
    print(f"   Extreme trained:   HSI = {extreme_trained['mean_hsi']:.3f}")

    if abs(baseline_trained['mean_hsi'] - extreme_init['mean_hsi']) < 0.5:
        print("   ⚠️  Training ERASES initialization advantage!")
        print("   → Confirms: training on noise just randomizes HSI")
    else:
        print("   ✓ Initialization advantage persists after training")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'hierarchical_initialization',
        'hypothesis': 'Can we design good HSI through initialization?',
        'results': results,
        'baseline_hsi': float(baseline_init['mean_hsi']),
        'extreme_hsi': float(extreme_init['mean_hsi']),
        'improvement': float((1 - extreme_init['mean_hsi']/baseline_init['mean_hsi'])*100)
    }

    with open('hierarchical_init_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to hierarchical_init_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
