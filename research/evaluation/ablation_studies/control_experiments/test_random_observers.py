#!/usr/bin/env python3
"""
Control Experiment 1: Random Observer Baseline

HYPOTHESIS: If observer effect is real and mechanistic (gradient sink, etc.),
            then RANDOM observers (no training) should still provide stability.

ALTERNATIVE: If stability requires LEARNED predictions, random observers won't help.

Tests 4 conditions:
A. No observers (baseline)
B. Trained observers (original condition)
C. Frozen random observers (never trained)
D. Pure noise injection (no observer networks)

If B > C ≈ D ≈ A: Effect requires learning → True observer effect
If B ≈ C > A: Effect is architectural → Any computation helps
If B ≈ C ≈ D > A: Effect is regularization → Noise helps
If B ≈ C ≈ D ≈ A: No effect → Something else is going on
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

class ControlObserverArchitecture(AblationArchitecture):
    """Architecture with configurable observer behavior"""

    def __init__(self, affect_dim=5, num_observers=100, observer_mode='trained'):
        super().__init__(affect_dim)
        self.num_observers = num_observers
        self.observer_mode = observer_mode  # 'trained', 'frozen', 'noise'

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

        # Observers (behavior depends on mode)
        if observer_mode == 'noise':
            # No observer networks, just inject noise
            self.observers = None
        else:
            self.observers = []
            for _ in range(num_observers):
                obs = nn.Sequential(
                    nn.Linear(40, 32),
                    nn.ReLU(),
                    nn.Linear(32, 40)
                )
                self.observers.append(obs)

            # Freeze observers if in frozen mode
            if observer_mode == 'frozen':
                for obs in self.observers:
                    obs.freeze()

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

        # Observer errors (mode-dependent)
        observer_errors = []

        if self.observer_mode == 'noise':
            # Pure noise injection
            if self.num_observers > 0:
                for _ in range(self.num_observers):
                    noise = mx.random.normal(phenomenal_state.shape) * 0.1
                    observer_errors.append(mx.mean(noise ** 2))

        elif self.observers:
            # Use actual observers (trained or frozen)
            for observer in self.observers:
                obs_pred = observer(predicted_state)
                observer_errors.append(mx.mean((obs_pred - phenomenal_state) ** 2))

        # Total surprise
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


def train_and_evaluate(observer_mode, num_observers, epochs=50, seq_length=20):
    """Train architecture and return final HSI"""
    print(f"\n{'='*70}")
    print(f"Testing: {observer_mode.upper()} observers (N={num_observers})")
    print(f"{'='*70}")

    model = ControlObserverArchitecture(
        affect_dim=5,
        num_observers=num_observers,
        observer_mode=observer_mode
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
        'observer_mode': observer_mode,
        'num_observers': num_observers,
        'final_hsi': final_hsi,
        'final_loss': loss_history[-1],
        'is_stable': is_stable,
        'loss_history': loss_history
    }


def main():
    print("="*80)
    print("CONTROL EXPERIMENT: Random Observer Baseline")
    print("="*80)
    print()
    print("Testing whether observers need to LEARN or just EXIST")
    print()

    # Test conditions with N=100 observers (known to be stable in original tests)
    conditions = [
        ('none', 0),           # A: No observers (baseline)
        ('trained', 100),      # B: Trained observers (original)
        ('frozen', 100),       # C: Frozen random observers
        ('noise', 100),        # D: Pure noise injection
    ]

    results = []

    for mode, n_obs in conditions:
        result = train_and_evaluate(mode, n_obs, epochs=50)
        results.append(result)

    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    print()

    print(f"{'Condition':<20} {'N':<8} {'Final HSI':<15} {'Stable?':<10} {'Interpretation'}")
    print("-"*80)

    for r in results:
        mode_label = {
            'none': 'A: No observers',
            'trained': 'B: Trained',
            'frozen': 'C: Frozen random',
            'noise': 'D: Pure noise'
        }[r['observer_mode']]

        stable_str = "✓" if r['is_stable'] else "✗"

        print(f"{mode_label:<20} {r['num_observers']:<8} {r['final_hsi']:<15.6f} {stable_str:<10}")

    # Hypothesis testing
    print("\n" + "="*80)
    print("HYPOTHESIS TEST")
    print("="*80)
    print()

    none_hsi = next(r['final_hsi'] for r in results if r['observer_mode'] == 'none')
    trained_hsi = next(r['final_hsi'] for r in results if r['observer_mode'] == 'trained')
    frozen_hsi = next(r['final_hsi'] for r in results if r['observer_mode'] == 'frozen')
    noise_hsi = next(r['final_hsi'] for r in results if r['observer_mode'] == 'noise')

    print(f"Baseline (no observers):     HSI = {none_hsi:.4f}")
    print(f"Trained observers:           HSI = {trained_hsi:.4f}")
    print(f"Frozen random observers:     HSI = {frozen_hsi:.4f}")
    print(f"Pure noise injection:        HSI = {noise_hsi:.4f}")
    print()

    # Determine which hypothesis is supported
    trained_better = trained_hsi < (none_hsi * 0.7)  # 30% improvement
    frozen_helps = frozen_hsi < (none_hsi * 0.7)
    noise_helps = noise_hsi < (none_hsi * 0.7)

    if trained_better and not frozen_helps and not noise_helps:
        print("✓ LEARNING REQUIRED:")
        print("  Only trained observers help → True observer effect")
        print("  Observers must learn meaningful predictions to stabilize")
        interpretation = "true_effect"
    elif trained_better and frozen_helps and not noise_helps:
        print("✓ ARCHITECTURAL EFFECT:")
        print("  Both trained and frozen help → Computation matters")
        print("  Any observer computation provides stability (gradient flow)")
        interpretation = "architectural"
    elif trained_better and frozen_helps and noise_helps:
        print("✓ REGULARIZATION EFFECT:")
        print("  All conditions help → It's about adding noise/regularization")
        print("  Not specific to observer architecture")
        interpretation = "regularization"
    else:
        print("✗ NO CLEAR EFFECT:")
        print("  No consistent improvement → Effect may not be real")
        print("  Or: very context-dependent")
        interpretation = "unclear"

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'random_observer_baseline',
        'hypothesis': 'Do observers need to learn or just exist?',
        'results': results,
        'interpretation': interpretation,
        'summary': {
            'none_hsi': float(none_hsi),
            'trained_hsi': float(trained_hsi),
            'frozen_hsi': float(frozen_hsi),
            'noise_hsi': float(noise_hsi),
            'trained_better': trained_better,
            'frozen_helps': frozen_helps,
            'noise_helps': noise_helps
        }
    }

    with open('random_observer_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to random_observer_results.json")
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
