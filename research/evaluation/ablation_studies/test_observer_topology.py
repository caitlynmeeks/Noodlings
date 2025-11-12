#!/usr/bin/env python3
"""
OBSERVER TOPOLOGY ABLATION STUDY

After discovering non-monotonic HSI oscillations with hierarchical observers,
we now test: DOES TOPOLOGY MATTER?

RESEARCH QUESTION:
Is the oscillation pattern specific to hierarchical topology, or universal?

EXPERIMENTAL DESIGN:
==================
Fix observer count at N=100, vary topology:

1. FLAT: All 100 observers at Level 0 (no hierarchy)
2. SHALLOW: 80 L0, 20 L1 (minimal hierarchy)
3. MODERATE: 70 L0, 25 L1, 5 L2 (balanced)
4. STEEP: 50 L0, 40 L1, 10 L2 (deep hierarchy)
5. CURRENT: 62 L0, 31 L1, 7 L2 (2:1:0.2 ratio - control)
6. INVERTED: 30 L0, 50 L1, 20 L2 (top-heavy)

HYPOTHESES:
===========
H1: Flat topology will show DIFFERENT stability than hierarchical
H2: Steeper hierarchy → lower HSI (better separation)
H3: Topology affects WHERE in observer-count space stability occurs

FALSIFIABLE PREDICTIONS:
========================
✓ If H1: Flat will have different final HSI than hierarchical (p < 0.05)
✓ If H2: Correlation between hierarchy depth and HSI (r < -0.5)
✓ If H3: Different topologies will show stability at different N values

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

from architectures.base import AblationArchitecture


class TopologyObserverArchitecture(AblationArchitecture):
    """
    Hierarchical architecture with configurable observer TOPOLOGY.

    Unlike ParametricObserverArchitecture which fixes ratios,
    this allows arbitrary L0/L1/L2 distributions.
    """

    def __init__(
        self,
        affect_dim: int = 5,
        num_observers: int = 100,
        topology: Tuple[int, int, int] = (62, 31, 7)  # (L0, L1, L2)
    ):
        super().__init__(affect_dim)

        self.num_observers = num_observers
        self.topology = topology

        # Validate topology
        assert sum(topology) == num_observers, \
            f"Topology {topology} must sum to {num_observers}"

        # Core hierarchy (same for all)
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

        # Create observers with specified topology
        self.observers = self._create_observers(topology)

    def _create_observers(self, topology: Tuple[int, int, int]):
        """Create observers with specified topology distribution."""
        observers = []
        l0_count, l1_count, l2_count = topology

        # Level 0 observers (watch main predictor)
        for _ in range(l0_count):
            observers.append(nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            ))

        # Level 1 observers (watch L0)
        for _ in range(l1_count):
            observers.append(nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            ))

        # Level 2 observers (watch L1)
        for _ in range(l2_count):
            observers.append(nn.Sequential(
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 40)
            ))

        return observers

    def __call__(self, affect: mx.array):
        if affect.ndim == 1:
            affect = affect[None, None, :]
        elif affect.ndim == 2:
            affect = affect[:, None, :]

        # Forward through hierarchy
        h_fast_seq, c_fast_seq = self.fast_lstm(
            affect, hidden=self.h_fast, cell=self.c_fast
        )
        self.h_fast = h_fast_seq[:, -1, :].reshape(1, 16)
        self.c_fast = c_fast_seq[:, -1, :].reshape(1, 16)

        h_med_seq, c_med_seq = self.medium_lstm(
            self.h_fast[:, None, :],
            hidden=self.h_medium,
            cell=self.c_medium
        )
        self.h_medium = h_med_seq[:, -1, :].reshape(1, 16)
        self.c_medium = c_med_seq[:, -1, :].reshape(1, 16)

        h_slow_seq = self.slow_gru(
            self.h_medium[:, None, :],
            hidden=self.h_slow
        )
        self.h_slow = h_slow_seq[:, -1, :].reshape(1, 8)

        # Predict and observe
        current_state = self.get_phenomenal_state()
        predicted_state = self.predictor(current_state)

        # Observer cascade respects topology
        observer_errors = []
        l0_count, l1_count, l2_count = self.topology

        # L0 observers watch main prediction
        for i in range(l0_count):
            obs_pred = self.observers[i](predicted_state)
            observer_errors.append(mx.mean((obs_pred - current_state) ** 2))

        # L1 observers watch L0 predictions (sample from L0)
        if l1_count > 0 and l0_count > 0:
            for i in range(l0_count, l0_count + l1_count):
                # Sample an L0 prediction
                l0_idx = (i - l0_count) % l0_count
                l0_pred = self.observers[l0_idx](predicted_state)
                obs_pred = self.observers[i](l0_pred)
                observer_errors.append(mx.mean((obs_pred - current_state) ** 2))

        # L2 observers watch L1 predictions (sample from L1)
        if l2_count > 0 and l1_count > 0:
            for i in range(l0_count + l1_count, l0_count + l1_count + l2_count):
                # Sample an L1 prediction
                l1_idx = l0_count + ((i - l0_count - l1_count) % l1_count)
                l1_pred = self.observers[l1_idx](predicted_state)
                obs_pred = self.observers[i](l1_pred)
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
        l0, l1, l2 = self.topology
        return f"Topology: {l0} L0, {l1} L1, {l2} L2 (total={self.num_observers})"


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

    # HSI ratio
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


def calculate_hierarchy_depth(topology: Tuple[int, int, int]) -> float:
    """
    Calculate hierarchy "depth" score.

    Higher score = deeper hierarchy (more meta-levels)
    Flat topology (all L0) = 0.0
    Perfect pyramid = 1.0
    """
    l0, l1, l2 = topology
    total = sum(topology)

    if total == 0:
        return 0.0

    # Weighted by level
    depth = (0*l0 + 1*l1 + 2*l2) / (2*total)
    return depth


def train_and_evaluate_topology(
    topology: Tuple[int, int, int],
    topology_name: str,
    epochs: int = 50,
    seq_length: int = 20
) -> Dict:
    """Train model with specified topology and evaluate."""

    print(f"\n{'='*80}")
    print(f"Testing Topology: {topology_name}")
    print(f"{'='*80}")

    l0, l1, l2 = topology
    print(f"Distribution: {l0} L0, {l1} L1, {l2} L2")
    print(f"Hierarchy depth: {calculate_hierarchy_depth(topology):.3f}")

    # Create model
    model = TopologyObserverArchitecture(
        affect_dim=5,
        num_observers=sum(topology),
        topology=topology
    )

    print(f"Parameters: ~{model.parameter_count()}")
    print()

    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-5)

    # State tracking
    fast_states = []
    medium_states = []
    slow_states = []
    loss_history = []
    hsi_history = []

    # Training
    for epoch in range(epochs):
        epoch_loss = 0
        sequence = mx.random.normal((seq_length, 5))

        for t in range(seq_length):
            affect = sequence[t]

            def loss_fn():
                state, surprise = model(affect)
                return mx.array(surprise)

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn()
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)

        avg_loss = epoch_loss / seq_length
        loss_history.append(avg_loss)

        # Track states
        fast_states.append(model.h_fast)
        medium_states.append(model.h_medium)
        slow_states.append(model.h_slow)

        # Calculate HSI
        if (epoch + 1) % 10 == 0:
            hsi = calculate_hsi_from_states(fast_states, medium_states, slow_states)
            hsi_history.append(hsi['slow/fast'])
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss = {avg_loss:.6f}, HSI = {hsi['slow/fast']:.3f}")

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
        'topology_name': topology_name,
        'topology': topology,
        'hierarchy_depth': calculate_hierarchy_depth(topology),
        'final_hsi': final_hsi['slow/fast'],
        'final_loss': loss_history[-1] if loss_history else 0.0,
        'hsi_history': hsi_history,
        'loss_history': loss_history,
        'status': status,
        'interpretation': final_hsi['interpretation']
    }


def main():
    """Run topology ablation study."""

    print()
    print("="*80)
    print("OBSERVER TOPOLOGY ABLATION STUDY")
    print("="*80)
    print()
    print("Research Question: Does observer network topology affect stability?")
    print()
    print("Fixed: N = 100 observers, 50 epochs")
    print("Varied: L0/L1/L2 distribution")
    print("="*80)
    print()

    # Define topologies to test
    topologies = {
        'A_flat': (100, 0, 0),           # All L0, no hierarchy
        'B_shallow': (80, 20, 0),        # Minimal hierarchy
        'C_moderate': (70, 25, 5),       # Balanced
        'D_steep': (50, 40, 10),         # Deep hierarchy
        'E_current': (62, 31, 7),        # Control (2:1:0.2 ratio)
        'F_inverted': (30, 50, 20),      # Top-heavy (unusual)
    }

    print("Topologies to test:")
    for name, topo in topologies.items():
        depth = calculate_hierarchy_depth(topo)
        print(f"  {name}: L0={topo[0]}, L1={topo[1]}, L2={topo[2]} "
              f"(depth={depth:.3f})")
    print()
    print("Starting experiments...")
    print()

    # Run experiments
    all_results = []
    start_time = time.time()

    for i, (name, topo) in enumerate(topologies.items(), 1):
        print()
        print("="*80)
        print(f"EXPERIMENT {i}/{len(topologies)}")
        print("="*80)

        result = train_and_evaluate_topology(
            topology=topo,
            topology_name=name,
            epochs=50
        )

        all_results.append(result)

        # Save intermediate
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / 'topology_intermediate.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time

    # Analysis
    print()
    print("="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    print()

    # Sort by hierarchy depth
    results_sorted = sorted(all_results, key=lambda r: r['hierarchy_depth'])

    print(f"{'Topology':<20} {'Depth':<8} {'HSI':<10} {'Status':<15} {'Loss':<10}")
    print("-"*80)

    for r in results_sorted:
        print(f"{r['topology_name']:<20} "
              f"{r['hierarchy_depth']:<8.3f} "
              f"{r['final_hsi']:<10.3f} "
              f"{r['status']:<15} "
              f"{r['final_loss']:<10.6f}")

    print()

    # Statistical analysis
    print("="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)
    print()

    # H1: Flat vs. Hierarchical
    flat_hsi = [r['final_hsi'] for r in all_results if r['topology_name'] == 'A_flat'][0]
    hierarchical_hsi = [r['final_hsi'] for r in all_results
                       if r['hierarchy_depth'] > 0]

    if hierarchical_hsi:
        avg_hierarchical = np.mean(hierarchical_hsi)
        diff = flat_hsi - avg_hierarchical
        pct_diff = (diff / flat_hsi * 100) if flat_hsi > 0 else 0

        print(f"H1: Flat vs. Hierarchical")
        print(f"  Flat HSI: {flat_hsi:.3f}")
        print(f"  Hierarchical avg HSI: {avg_hierarchical:.3f}")
        print(f"  Difference: {diff:+.3f} ({pct_diff:+.1f}%)")

        if abs(diff) > 0.05:
            print(f"  → H1 SUPPORTED: Topology affects stability!")
        else:
            print(f"  → H1 REJECTED: No significant difference")
        print()

    # H2: Correlation between depth and HSI
    depths = [r['hierarchy_depth'] for r in all_results]
    hsis = [r['final_hsi'] for r in all_results]

    if len(depths) > 2:
        correlation = np.corrcoef(depths, hsis)[0, 1]

        print(f"H2: Hierarchy Depth → Lower HSI")
        print(f"  Correlation (depth, HSI): r = {correlation:.3f}")

        if correlation < -0.5:
            print(f"  → H2 SUPPORTED: Deeper hierarchy → lower HSI")
        elif correlation > 0.5:
            print(f"  → H2 REVERSED: Deeper hierarchy → HIGHER HSI!")
        else:
            print(f"  → H2 REJECTED: No strong correlation")
        print()

    # H3: Best and worst topologies
    best = min(all_results, key=lambda r: r['final_hsi'])
    worst = max(all_results, key=lambda r: r['final_hsi'])

    print(f"H3: Topology Matters")
    print(f"  Best topology: {best['topology_name']} (HSI={best['final_hsi']:.3f})")
    print(f"  Worst topology: {worst['topology_name']} (HSI={worst['final_hsi']:.3f})")
    print(f"  Range: {worst['final_hsi'] - best['final_hsi']:.3f}")

    if worst['final_hsi'] - best['final_hsi'] > 0.1:
        print(f"  → H3 SUPPORTED: Topology significantly affects stability!")
    else:
        print(f"  → H3 REJECTED: Topology has minimal effect")

    print()

    # Save final results
    final_output = {
        'experiment': 'topology_ablation',
        'topologies': topologies,
        'results': all_results,
        'elapsed_time': elapsed,
        'summary': {
            'best_topology': best['topology_name'],
            'worst_topology': worst['topology_name'],
            'correlation_depth_hsi': float(correlation) if len(depths) > 2 else None
        }
    }

    with open(output_dir / 'topology_results.json', 'w') as f:
        json.dump(final_output, f, indent=2)

    print("="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print()
    print(f"✓ Results saved to: results/topology_results.json")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print()
    print("SCIENTIFIC CONCLUSIONS:")
    print()

    # Draw conclusions
    if abs(diff) > 0.05:
        print("🎯 TOPOLOGY MATTERS!")
        print("  Observer network structure significantly affects stability.")
        print()

    if correlation < -0.5:
        print("📊 HIERARCHY HELPS!")
        print("  Deeper observer hierarchies lead to better separation.")
        print()
    elif correlation > 0.5:
        print("⚠️  UNEXPECTED: Hierarchy HURTS!")
        print("  Deeper hierarchies actually increase HSI (worse).")
        print("  This contradicts our intuition and needs investigation!")
        print()

    print("Next: Visualize with 'python3 plot_topology.py'")
    print()


if __name__ == '__main__':
    main()
