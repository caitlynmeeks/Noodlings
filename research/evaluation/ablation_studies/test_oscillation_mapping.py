#!/usr/bin/env python3
"""
OSCILLATION MAPPING EXPERIMENT

After discovering non-monotonic HSI behavior, we now map the pattern precisely.

GOAL: Test every 5 observers from 75-155 to capture oscillation structure

HYPOTHESES TO TEST:
1. Is there a periodic pattern? (period = 10? 20? 30?)
2. Are there "resonant frequencies" between observer count and network structure?
3. Is there modular arithmetic relationship? (N mod K = stable zones)
4. Do stability zones cluster or are they random?

EXPERIMENTAL DESIGN:
- Observer counts: 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155
- 17 data points (vs. 8 before) → 2x resolution
- Same architecture, same epochs (50)
- Track: HSI, loss, training dynamics

FALSIFIABLE PREDICTIONS:
✓ If periodic: Adjacent stable points will be evenly spaced
✓ If resonant: Stability zones will cluster around specific ratios
✓ If random: No pattern will emerge (hypothesis rejected)

Author: Noodlings Project
Date: November 2025
"""

import sys
sys.path.insert(0, '..')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Import base class
from architectures.base import AblationArchitecture


class ParametricObserverArchitecture(AblationArchitecture):
    """
    Hierarchical architecture with configurable observer count.

    Observer distribution follows 2:1:0.2 ratio across 3 levels.
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
        joint_dim = 40
        self.predictor = nn.Sequential(nn.Linear(joint_dim, 64), nn.ReLU(), nn.Linear(64, joint_dim))

        # Observers
        self.observers = self._create_observers(num_observers)

    def _create_observers(self, total: int):
        observers = []
        ratio_sum = 3.2
        level0 = int(total * (2.0 / ratio_sum))
        level1 = int(total * (1.0 / ratio_sum))
        level2 = total - level0 - level1

        for _ in range(total):
            observers.append(nn.Sequential(nn.Linear(40, 32), nn.ReLU(), nn.Linear(32, 40)))

        self.level_split = (level0, level1, level2)
        return observers

    def __call__(self, affect: mx.array):
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Fast
        h_fast_seq, c_fast_seq = self.fast_lstm(affect, hidden=self.h_fast, cell=self.c_fast)
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        # Medium
        h_med_seq, c_med_seq = self.medium_lstm(self.h_fast[:, None, :], hidden=self.h_medium, cell=self.c_medium)
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        # Slow
        h_slow_seq = self.slow_gru(self.h_medium[:, None, :], hidden=self.h_slow)
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Predict and observe
        current_state = self.get_phenomenal_state()
        predicted_state = self.predictor(current_state)

        # Observer cascade
        observer_errors = []
        for obs in self.observers:
            obs_pred = obs(predicted_state)
            observer_errors.append(mx.mean((obs_pred - current_state) ** 2))

        main_error = mx.mean((predicted_state - current_state) ** 2)
        observer_error = mx.mean(mx.array(observer_errors)) if observer_errors else 0
        surprise = float(main_error + 0.1 * observer_error)

        return current_state, surprise

    def reset_states(self):
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def architecture_description(self) -> str:
        return f"Hierarchical + {self.num_observers} observers (L0/L1/L2: {self.level_split})"

print()
print("="*80)
print("EXPERIMENT: OBSERVER OSCILLATION MAPPING")
print("="*80)
print()
print("Discovered: Non-monotonic HSI pattern (not a smooth power law!)")
print("Hypothesis: Interference/resonance between observers and main network")
print()
print("Testing: Fine-grained observer counts (every 5 from 75-155)")
print("="*80)
print()


def calculate_hsi_from_states(
    fast_states: List[mx.array],
    medium_states: List[mx.array],
    slow_states: List[mx.array]
) -> Dict[str, float]:
    """Calculate HSI from state histories."""

    if len(fast_states) < 10:
        return {'slow/fast': np.nan, 'interpretation': 'Insufficient data'}

    # Convert to numpy
    fast_np = np.array([s.tolist() for s in fast_states[-100:]])
    medium_np = np.array([s.tolist() for s in medium_states[-100:]])
    slow_np = np.array([s.tolist() for s in slow_states[-100:]])

    # Calculate variances
    var_fast = np.var(fast_np, axis=0).mean()
    var_medium = np.var(medium_np, axis=0).mean()
    var_slow = np.var(slow_np, axis=0).mean()

    # HSI ratios
    hsi_slow_fast = var_slow / var_fast if var_fast > 1e-10 else np.nan

    # Interpret
    if np.isnan(hsi_slow_fast):
        interpretation = "Undefined"
    elif hsi_slow_fast < 0.1:
        interpretation = "Excellent separation"
    elif hsi_slow_fast < 0.3:
        interpretation = "Good separation"
    elif hsi_slow_fast < 1.0:
        interpretation = "Moderate separation"
    elif hsi_slow_fast < 2.0:
        interpretation = "Weak separation"
    else:
        interpretation = "COLLAPSED"

    return {
        'slow/fast': float(hsi_slow_fast),
        'interpretation': interpretation
    }


def train_and_measure(
    num_observers: int,
    epochs: int = 50,
    batch_size: int = 32,
    seq_length: int = 20
) -> Dict:
    """Train hierarchical model with N observers and measure HSI."""

    print(f"\n{'='*80}")
    print(f"Testing {num_observers} observers")
    print(f"{'='*80}")

    # Calculate hierarchical distribution (62% L0, 31% L1, 7% L2)
    l0_count = int(num_observers * 0.62)
    l1_count = int(num_observers * 0.31)
    l2_count = num_observers - l0_count - l1_count

    print(f"Hierarchical with {num_observers} observers")
    print(f"Distribution: L0={l0_count}, L1={l1_count}, L2={l2_count}")

    # Create model
    model = ParametricObserverArchitecture(
        affect_dim=5,
        num_observers=num_observers
    )

    print(f"Parameters: ~{model.parameter_count()}")

    # Optimizer with hierarchical learning rates
    optimizer = optim.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5
    )

    # Training history
    loss_history = []
    hsi_history = []

    # State tracking
    fast_states = []
    medium_states = []
    slow_states = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0

        # Generate synthetic sequence
        sequence = mx.random.normal((seq_length, 5))

        # Process sequence
        for t in range(seq_length):
            affect = sequence[t]

            # Define loss for this step
            def loss_fn():
                state, surprise = model(affect)
                return mx.array(surprise)

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

            # Update
            loss, grads = loss_and_grad_fn()
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)

        # Average loss for epoch
        avg_loss = epoch_loss / seq_length

        # Track states after epoch
        fast_states.append(model.h_fast)
        medium_states.append(model.h_medium)
        slow_states.append(model.h_slow)

        # Store loss
        loss_history.append(avg_loss)

        # Calculate HSI periodically
        if (epoch + 1) % 10 == 0:
            hsi = calculate_hsi_from_states(fast_states, medium_states, slow_states)
            hsi_history.append(hsi['slow/fast'])

            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}, HSI = {hsi['slow/fast']:.3f}")

    # Final HSI
    final_hsi = calculate_hsi_from_states(fast_states, medium_states, slow_states)

    # Status
    if final_hsi['slow/fast'] < 0.3:
        status = "✅ STABLE"
    elif final_hsi['slow/fast'] < 1.0:
        status = "⚠️ UNSTABLE"
    else:
        status = "❌ COLLAPSED"

    print(f"\nResult: HSI = {final_hsi['slow/fast']:.3f} {status}")

    return {
        'num_observers': num_observers,
        'observer_distribution': [l0_count, l1_count, l2_count],
        'final_hsi': final_hsi['slow/fast'],
        'final_loss': float(loss_history[-1]) if loss_history else 0.0,
        'hsi_history': hsi_history,
        'loss_history': loss_history,
        'status': status,
        'interpretation': final_hsi['interpretation']
    }


def analyze_oscillation_pattern(results: List[Dict]) -> Dict:
    """Analyze the oscillation pattern for periodicity and structure."""

    observer_counts = [r['num_observers'] for r in results]
    hsi_values = [r['final_hsi'] for r in results]

    # Filter out NaN
    valid_pairs = [(n, h) for n, h in zip(observer_counts, hsi_values) if not np.isnan(h)]
    if not valid_pairs:
        return {'error': 'No valid data'}

    observer_counts, hsi_values = zip(*valid_pairs)
    observer_counts = np.array(observer_counts)
    hsi_values = np.array(hsi_values)

    analysis = {}

    # 1. Identify stable zones (HSI < 0.3)
    stable_mask = hsi_values < 0.3
    stable_counts = observer_counts[stable_mask]
    unstable_counts = observer_counts[~stable_mask]

    analysis['stable_zones'] = stable_counts.tolist()
    analysis['unstable_zones'] = unstable_counts.tolist()
    analysis['stability_ratio'] = len(stable_counts) / len(observer_counts)

    # 2. Check for periodicity (FFT)
    if len(hsi_values) > 4:
        fft = np.fft.fft(hsi_values)
        freqs = np.fft.fftfreq(len(hsi_values), d=5)  # 5 observer spacing

        # Find dominant frequency
        power = np.abs(fft[1:len(fft)//2])  # Skip DC component
        if len(power) > 0:
            peak_idx = np.argmax(power)
            dominant_freq = freqs[1:len(freqs)//2][peak_idx]
            period = 1.0 / dominant_freq if dominant_freq != 0 else np.inf

            analysis['dominant_period'] = float(period)
            analysis['fft_peak_power'] = float(power[peak_idx])

    # 3. Check for modular patterns
    # Test various moduli
    best_mod = None
    best_separation = 0

    for mod in [5, 10, 15, 20, 25, 30]:
        remainders = observer_counts % mod

        # For each remainder, calculate mean HSI
        remainder_groups = {}
        for remainder, hsi in zip(remainders, hsi_values):
            if remainder not in remainder_groups:
                remainder_groups[remainder] = []
            remainder_groups[remainder].append(hsi)

        # Calculate variance between groups
        group_means = [np.mean(g) for g in remainder_groups.values()]
        if len(group_means) > 1:
            between_var = np.var(group_means)
            if between_var > best_separation:
                best_separation = between_var
                best_mod = mod

    analysis['best_modulus'] = best_mod
    analysis['modular_separation'] = float(best_separation)

    # 4. Gap analysis (consecutive stable zones)
    if len(stable_counts) > 1:
        gaps = np.diff(sorted(stable_counts))
        analysis['stable_gaps'] = gaps.tolist()
        analysis['mean_stable_gap'] = float(np.mean(gaps))
        analysis['gap_std'] = float(np.std(gaps))

    # 5. Clustering analysis
    # Are stable zones clustered or uniformly distributed?
    if len(stable_counts) > 2:
        # Nearest neighbor distances
        stable_sorted = np.sort(stable_counts)
        nn_distances = np.diff(stable_sorted)

        # Compare to uniform distribution
        expected_spacing = (stable_sorted[-1] - stable_sorted[0]) / (len(stable_sorted) - 1)
        clustering_score = np.std(nn_distances) / expected_spacing

        analysis['clustering_score'] = float(clustering_score)
        # clustering_score > 1 → clustered, < 1 → uniform

    return analysis


def main():
    """Run fine-grained oscillation mapping experiment."""

    # Observer counts to test (every 5 from 75-155)
    observer_counts = list(range(75, 160, 5))

    print(f"Testing {len(observer_counts)} observer configurations:")
    print(f"  Range: {min(observer_counts)} to {max(observer_counts)}")
    print(f"  Spacing: 5 observers")
    print(f"  Total experiments: {len(observer_counts)}")
    print()

    # Run experiments
    all_results = []

    start_time = time.time()

    for i, num_obs in enumerate(observer_counts, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(observer_counts)}")
        print(f"{'='*80}")

        result = train_and_measure(
            num_observers=num_obs,
            epochs=50
        )

        all_results.append(result)

        # Save intermediate results
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / 'oscillation_mapping_intermediate.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time

    # Analyze pattern
    print()
    print("="*80)
    print("OSCILLATION PATTERN ANALYSIS")
    print("="*80)
    print()

    pattern_analysis = analyze_oscillation_pattern(all_results)

    # Print analysis
    print("STABILITY ZONES:")
    print(f"  Stable configurations: {pattern_analysis['stable_zones']}")
    print(f"  Unstable configurations: {pattern_analysis['unstable_zones']}")
    print(f"  Stability ratio: {pattern_analysis['stability_ratio']:.1%}")
    print()

    if 'dominant_period' in pattern_analysis:
        print("PERIODICITY:")
        print(f"  Dominant period: {pattern_analysis['dominant_period']:.1f} observers")
        print(f"  FFT peak power: {pattern_analysis['fft_peak_power']:.3f}")
        print()

    if pattern_analysis['best_modulus']:
        print("MODULAR PATTERN:")
        print(f"  Best modulus: {pattern_analysis['best_modulus']}")
        print(f"  Separation strength: {pattern_analysis['modular_separation']:.3f}")
        print()

    if 'stable_gaps' in pattern_analysis:
        print("GAP ANALYSIS:")
        print(f"  Mean gap between stable zones: {pattern_analysis['mean_stable_gap']:.1f}")
        print(f"  Gap std deviation: {pattern_analysis['gap_std']:.1f}")
        print()

    if 'clustering_score' in pattern_analysis:
        print("CLUSTERING:")
        clustering = pattern_analysis['clustering_score']
        print(f"  Clustering score: {clustering:.2f}")
        if clustering > 1.5:
            print("  → Stable zones are CLUSTERED (non-uniform)")
        elif clustering < 0.5:
            print("  → Stable zones are UNIFORMLY distributed")
        else:
            print("  → Moderate clustering")
        print()

    # Summary table
    print("="*80)
    print("COMPLETE RESULTS")
    print("="*80)
    print()
    print(f"{'Observers':<12} {'HSI':<10} {'Status':<15} {'Loss':<10}")
    print("-"*80)

    for r in all_results:
        print(f"{r['num_observers']:<12} {r['final_hsi']:<10.3f} {r['status']:<15} {r['final_loss']:<10.6f}")

    print()
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()

    # Save final results
    final_output = {
        'experiment': 'oscillation_mapping',
        'observer_counts': observer_counts,
        'results': all_results,
        'pattern_analysis': pattern_analysis,
        'elapsed_time': elapsed
    }

    with open(output_dir / 'oscillation_mapping_results.json', 'w') as f:
        json.dump(final_output, f, indent=2)

    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print()
    print("✓ Results saved to: results/oscillation_mapping_results.json")
    print()
    print("SCIENTIFIC CONCLUSIONS:")
    print()

    # Draw conclusions
    if pattern_analysis['stability_ratio'] > 0.6:
        print("✅ Most configurations are STABLE (>60%)")
        print("   → Observer effect is robust across densities")
    else:
        print("⚠️  Many configurations are UNSTABLE")
        print("   → Critical to choose correct observer count")

    print()

    if 'dominant_period' in pattern_analysis and pattern_analysis['dominant_period'] < 40:
        print(f"🌊 PERIODIC OSCILLATION detected (period ≈ {pattern_analysis['dominant_period']:.0f})")
        print("   → Suggests resonance/interference mechanism")
    else:
        print("~ No clear periodic pattern")
        print("   → More complex dynamics at play")

    print()

    if pattern_analysis['best_modulus']:
        print(f"🔢 MODULAR PATTERN found (mod {pattern_analysis['best_modulus']})")
        print("   → Discrete stability zones")

    print()
    print("Next: Visualize with 'python3 plot_oscillation_mapping.py'")
    print()


if __name__ == '__main__':
    main()
