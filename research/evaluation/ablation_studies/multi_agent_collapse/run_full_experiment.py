#!/usr/bin/env python3
"""
FULL MULTI-AGENT COLLAPSE EXPERIMENT

Tests the General Hierarchical Collapse Principle:
"Multi-timescale systems require observer diversity to prevent collapse"

EXPERIMENTAL CONDITIONS:
========================

Condition A: NO OBSERVERS (Control)
  - 10 active agents, 0 observers
  - Prediction: HSI increases, agents collapse into reactive behavior

Condition B: FEW OBSERVERS (1:3 ratio)
  - 10 active agents, 3 observers (Level 0)
  - Prediction: Mixed results, some agents stable, some collapse

Condition C: BALANCED OBSERVERS (1:1 ratio)
  - 10 active agents, 10 observers (8 L0, 2 L1)
  - Prediction: Most agents stable, low HSI

Condition D: DENSE OBSERVERS (hierarchical)
  - 10 active agents, 15 observers (10 L0, 4 L1, 1 L2)
  - Prediction: All agents stable, very low HSI

HYPOTHESIS:
===========
HSI will be significantly lower (p < 0.05) in conditions with more observers.
Observer density follows power law: HSI ~ k / N_observers^β

FALSIFIABLE:
============
If observer conditions show NO difference in HSI, hypothesis is rejected.

Author: Noodlings Project
Date: November 2025
"""

import sys
sys.path.insert(0, '..')

import mlx.core as mx
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import time

from agents import NoodlingAgent, ObserverAgent
from environment import ResourceAllocationGame


class MultiAgentExperiment:
    """Orchestrates a complete multi-agent + observer experiment."""

    def __init__(
        self,
        num_active_agents: int,
        num_observers: int,
        observer_hierarchy: List[int],  # e.g., [10, 4, 1] for L0/L1/L2
        num_rounds: int = 200,
        experiment_name: str = "unnamed"
    ):
        """
        Initialize experiment.

        Args:
            num_active_agents: Number of agents playing the game
            num_observers: Total observers
            observer_hierarchy: Distribution across levels [L0, L1, L2, ...]
            num_rounds: Game length
            experiment_name: Name for results
        """
        self.num_active_agents = num_active_agents
        self.num_observers = num_observers
        self.observer_hierarchy = observer_hierarchy
        self.num_rounds = num_rounds
        self.experiment_name = experiment_name

        # Will be created on run
        self.agents = []
        self.observers = []
        self.game = None

        # Results storage
        self.results = {
            'config': {
                'name': experiment_name,
                'num_active_agents': num_active_agents,
                'num_observers': num_observers,
                'observer_hierarchy': observer_hierarchy,
                'num_rounds': num_rounds,
                'timestamp': datetime.now().isoformat()
            },
            'agent_hsi_history': {},
            'agent_final_hsi': {},
            'game_stats': {},
            'observer_quality': {}
        }

    def setup(self):
        """Create all agents, observers, and environment."""
        print(f"Setting up experiment: {self.experiment_name}")
        print(f"  Active agents: {self.num_active_agents}")
        print(f"  Observers: {self.num_observers} {self.observer_hierarchy}")

        # Create active agents
        self.agents = []
        for i in range(self.num_active_agents):
            agent = NoodlingAgent(
                agent_id=f"agent_{i:03d}",
                input_dim=10,
                action_dim=3,
                message_dim=8
            )
            self.agents.append(agent)

        # Create observers at different levels
        self.observers = []
        observer_count = 0

        for level, count in enumerate(self.observer_hierarchy):
            for i in range(count):
                observer = ObserverAgent(
                    agent_id=f"observer_L{level}_{observer_count:03d}",
                    observation_level=level,
                    input_dim=10,
                    message_dim=8,
                    prediction_weight=0.15  # Slightly stronger than default
                )
                self.observers.append(observer)
                observer_count += 1

        # Create game
        self.game = ResourceAllocationGame(
            num_agents=self.num_active_agents,
            base_resources=self.num_active_agents * 2,
            scarcity_cycle=50
        )

        print(f"✓ Setup complete")
        print()

    def run(self, verbose: bool = False):
        """Run the full experiment."""

        print("="*80)
        print(f"RUNNING: {self.experiment_name}")
        print("="*80)
        print()

        start_time = time.time()

        # Storage for metrics
        agent_hsi_history = {i: [] for i in range(self.num_active_agents)}
        game_stats_list = []

        # Main simulation loop
        for round_num in range(self.num_rounds):

            # 1. Get observations for all agents
            observations = {i: self.game.get_observation(i)
                          for i in range(self.num_active_agents)}

            # 2. Observers generate corrections
            corrections = {}
            if self.observers:
                for observer in self.observers:
                    # Observer watches all agents
                    agent_corrections = observer.observe_agents(
                        self.agents,
                        observations[0]  # Use first agent's obs as env state
                    )
                    corrections.update(agent_corrections)

            # 3. Agents act (with corrections if present)
            actions = {}
            for i, agent in enumerate(self.agents):
                correction = corrections.get(agent.agent_id, None)

                state, action_logits, message = agent.perceive_and_update(
                    observations[i],
                    observer_correction=correction
                )

                # Choose action (argmax for deterministic, sample for stochastic)
                action = int(mx.argmax(action_logits, axis=-1))
                actions[i] = action

            # 4. Environment step
            rewards, info, done = self.game.step(actions)

            # 5. Update agents with rewards
            for i, agent in enumerate(self.agents):
                agent.update_from_reward(rewards[i])

            # 6. Track metrics
            for i, agent in enumerate(self.agents):
                hsi = agent.calculate_hsi()
                if not np.isnan(hsi['slow/fast']):
                    agent_hsi_history[i].append(hsi['slow/fast'])

            game_stats_list.append(info)

            # 7. Progress reporting
            if verbose and (round_num + 1) % 50 == 0:
                valid_hsi = [h[-1] for h in agent_hsi_history.values() if h]
                avg_hsi = np.mean(valid_hsi) if valid_hsi else np.nan

                print(f"Round {round_num + 1}/{self.num_rounds}: "
                      f"avg_HSI={avg_hsi:.3f}, "
                      f"trust={info['avg_trust']:.2f}, "
                      f"coop={info['cooperation_events']}")

            if done:
                break

        elapsed_time = time.time() - start_time

        # Store results
        self.results['agent_hsi_history'] = {
            i: [float(h) for h in history]
            for i, history in agent_hsi_history.items()
        }

        self.results['agent_final_hsi'] = {
            i: float(agent.calculate_hsi()['slow/fast'])
            for i, agent in enumerate(self.agents)
        }

        self.results['game_stats'] = self.game.get_game_statistics()

        if self.observers:
            self.results['observer_quality'] = {
                obs.agent_id: obs.get_observation_quality()
                for obs in self.observers
            }

        self.results['elapsed_time'] = elapsed_time

        print()
        print(f"✓ Experiment complete ({elapsed_time:.1f}s)")
        print()

        return self.results

    def analyze(self):
        """Analyze and summarize results."""

        print("="*80)
        print("ANALYSIS")
        print("="*80)
        print()

        # HSI Analysis
        final_hsi_values = list(self.results['agent_final_hsi'].values())
        final_hsi_values = [h for h in final_hsi_values if not np.isnan(h)]

        if final_hsi_values:
            avg_hsi = np.mean(final_hsi_values)
            std_hsi = np.std(final_hsi_values)
            min_hsi = np.min(final_hsi_values)
            max_hsi = np.max(final_hsi_values)

            print("Hierarchical Separation Index (HSI):")
            print(f"  Average: {avg_hsi:.4f} ± {std_hsi:.4f}")
            print(f"  Range: [{min_hsi:.4f}, {max_hsi:.4f}]")

            # Count stable vs collapsed
            stable = sum(1 for h in final_hsi_values if h < 0.3)
            moderate = sum(1 for h in final_hsi_values if 0.3 <= h < 1.0)
            collapsed = sum(1 for h in final_hsi_values if h >= 1.0)

            print(f"  Stable (HSI < 0.3): {stable}/{len(final_hsi_values)}")
            print(f"  Moderate (0.3-1.0): {moderate}/{len(final_hsi_values)}")
            print(f"  Collapsed (> 1.0): {collapsed}/{len(final_hsi_values)}")
        else:
            print("⚠️ No valid HSI measurements")
            avg_hsi = np.nan

        print()

        # Game performance
        stats = self.results['game_stats']
        print("Game Performance:")
        print(f"  Cooperation rate: {stats['share_rate']:.1%}")
        print(f"  Conflict rate: {stats['conflict_events']/max(stats['total_rounds'],1):.1%}")
        print(f"  Avg score: {stats['avg_score']:.2f} ± {stats['score_std']:.2f}")
        print(f"  Final trust: {stats['avg_trust']:.2f}")
        print()

        # Observer effectiveness (if present)
        if self.observers:
            print("Observer Effectiveness:")
            for obs_id, quality in self.results['observer_quality'].items():
                print(f"  {obs_id}:")
                print(f"    Corrections: {quality['total_corrections']}")
                print(f"    Prediction error: {quality['avg_prediction_error']:.4f}")
            print()

        # Summary assessment
        print("="*80)
        print("ASSESSMENT")
        print("="*80)
        print()

        if avg_hsi < 0.3:
            print("✅ EXCELLENT: Hierarchical separation maintained")
        elif avg_hsi < 0.5:
            print("✓ GOOD: Moderate separation maintained")
        elif avg_hsi < 1.0:
            print("⚠️ FAIR: Some collapse occurring")
        else:
            print("❌ POOR: Significant hierarchical collapse")

        print()

        return avg_hsi, stats


def run_condition(condition_name: str, config: Dict, replications: int = 1):
    """Run a single experimental condition multiple times."""

    print()
    print("╔" + "="*78 + "╗")
    print(f"║ CONDITION: {condition_name:^66} ║")
    print("╚" + "="*78 + "╝")
    print()

    all_results = []

    for rep in range(replications):
        print(f"\n--- Replication {rep + 1}/{replications} ---\n")

        exp = MultiAgentExperiment(
            num_active_agents=config['num_agents'],
            num_observers=config['num_observers'],
            observer_hierarchy=config['observer_hierarchy'],
            num_rounds=config['num_rounds'],
            experiment_name=f"{condition_name}_rep{rep+1}"
        )

        exp.setup()
        results = exp.run(verbose=True)
        avg_hsi, stats = exp.analyze()

        results['summary'] = {
            'avg_hsi': float(avg_hsi) if not np.isnan(avg_hsi) else None,
            'avg_score': stats['avg_score'],
            'cooperation_rate': stats['share_rate'],
            'final_trust': stats['avg_trust']
        }

        all_results.append(results)

    return all_results


def main():
    """Run the complete ablation study."""

    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║     MULTI-AGENT HIERARCHICAL COLLAPSE ABLATION STUDY              ║")
    print("║                                                                    ║")
    print("║  Testing: Do observers prevent collapse in multi-agent systems?   ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    # Experimental design
    conditions = {
        'A_no_observers': {
            'num_agents': 10,
            'num_observers': 0,
            'observer_hierarchy': [],
            'num_rounds': 200
        },
        'B_few_observers': {
            'num_agents': 10,
            'num_observers': 3,
            'observer_hierarchy': [3],  # 3 Level-0 observers
            'num_rounds': 200
        },
        'C_balanced_observers': {
            'num_agents': 10,
            'num_observers': 10,
            'observer_hierarchy': [8, 2],  # 8 L0, 2 L1
            'num_rounds': 200
        },
        'D_dense_observers': {
            'num_agents': 10,
            'num_observers': 15,
            'observer_hierarchy': [10, 4, 1],  # 10 L0, 4 L1, 1 L2
            'num_rounds': 200
        }
    }

    # Number of replications per condition
    replications = 3  # Start with 3, can increase to 10+ for final study

    print(f"Experimental Design:")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Replications per condition: {replications}")
    print(f"  Total experiments: {len(conditions) * replications}")
    print()
    print("Starting experiments...")
    print()

    # Run all conditions
    all_condition_results = {}

    for cond_name, config in conditions.items():
        results = run_condition(cond_name, config, replications)
        all_condition_results[cond_name] = results

        # Save intermediate results
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / f'{cond_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Final comparative analysis
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                     COMPARATIVE ANALYSIS                           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    summary_table = []

    for cond_name, results in all_condition_results.items():
        # Average across replications
        hsi_values = [r['summary']['avg_hsi'] for r in results
                     if r['summary']['avg_hsi'] is not None]
        scores = [r['summary']['avg_score'] for r in results]
        coop_rates = [r['summary']['cooperation_rate'] for r in results]

        if hsi_values:
            summary_table.append({
                'condition': cond_name,
                'avg_hsi': np.mean(hsi_values),
                'std_hsi': np.std(hsi_values),
                'avg_score': np.mean(scores),
                'avg_coop': np.mean(coop_rates)
            })

    # Print summary table
    print(f"{'Condition':<25} {'HSI':<15} {'Score':<12} {'Cooperation':<12}")
    print("-" * 80)

    for row in summary_table:
        print(f"{row['condition']:<25} "
              f"{row['avg_hsi']:.3f} ± {row['std_hsi']:.3f}   "
              f"{row['avg_score']:>7.1f}    "
              f"{row['avg_coop']:>8.1%}")

    print()

    # Statistical test (simple for now)
    if len(summary_table) >= 2:
        print("Key Comparisons:")
        baseline_hsi = summary_table[0]['avg_hsi']

        for i, row in enumerate(summary_table[1:], 1):
            diff = baseline_hsi - row['avg_hsi']
            pct_improvement = (diff / baseline_hsi) * 100 if baseline_hsi > 0 else 0

            print(f"  {row['condition']} vs {summary_table[0]['condition']}:")
            print(f"    HSI change: {diff:+.3f} ({pct_improvement:+.1f}%)")

    print()
    print("="*80)
    print()
    print("🎉 ABLATION STUDY COMPLETE!")
    print()
    print("Results saved to: results/")
    print("Next: Analyze statistical significance and create visualizations")
    print()


if __name__ == '__main__':
    main()
