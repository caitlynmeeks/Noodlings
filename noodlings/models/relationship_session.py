"""
Session management and execution for long-term relationship simulation.

Handles running individual sessions, saving checkpoints between sessions,
and tracking slow layer evolution across the relationship arc.
"""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
import numpy as np
from typing import Dict, List
import json
import os

from consilience import ConsilienceModel, ConsilienceAgent
from relationship_states import (
    session1_first_meeting,
    session2_building_trust,
    session3_conflict_and_repair,
    session4_deep_bond
)


class RelationshipSession:
    """
    Manages a single relationship session with the consciousness.
    """

    def __init__(
        self,
        session_number: int,
        session_name: str,
        session_func,
        load_checkpoint: str = None
    ):
        self.session_number = session_number
        self.session_name = session_name
        self.session_func = session_func
        self.load_checkpoint = load_checkpoint

        # Initialize model and agent
        self.model = ConsilienceModel()
        self.agent = ConsilienceAgent(
            self.model,
            lr_fast=1e-3,
            lr_medium=5e-4,
            lr_slow=1e-4
        )

        # Load previous session if specified
        if load_checkpoint and os.path.exists(load_checkpoint):
            self.agent.load_checkpoint(load_checkpoint)
            print(f"  ↻ Loaded checkpoint from: {load_checkpoint}")
            print(f"  ↻ Continuing from step {self.agent.step}")

        # Track session metrics
        self.initial_slow_mag = float(mx.sqrt((self.agent.h_slow**2).sum()))
        self.surprises = []
        self.slow_mags = []
        self.fast_mags = []
        self.medium_mags = []

    def run(self, save_dir: str = "relationships") -> Dict:
        """
        Execute the session and return metrics.

        Args:
            save_dir: Directory to save checkpoints and memories

        Returns:
            Dictionary with session metrics
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"SESSION {self.session_number}: {self.session_name}")
        print(f"{'='*70}")

        # Generate emotional arc
        affects, descriptions = self.session_func()

        print(f"\n  Initial slow layer magnitude: {self.initial_slow_mag:.4f}")
        print(f"  Processing {len(affects)} moments...\n")

        # Process each moment
        for i, (affect, description) in enumerate(zip(affects, descriptions)):
            # Process turn (no training - just experiencing)
            surprise, should_speak = self.agent.process_turn(
                affect,
                user_input=description,
                agent_response="experiencing...",
                train=False  # Pure experience, no gradient updates
            )

            # Track metrics
            self.surprises.append(surprise)

            fast_mag = float(mx.sqrt((self.agent.h_fast**2).sum()))
            med_mag = float(mx.sqrt((self.agent.h_med**2).sum()))
            slow_mag = float(mx.sqrt((self.agent.h_slow**2).sum()))

            self.fast_mags.append(fast_mag)
            self.medium_mags.append(med_mag)
            self.slow_mags.append(slow_mag)

            # Progress updates
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Moment {i+1:2d}/{len(affects)}: {description[:40]:40s} "
                      f"| Surprise: {surprise:.3f} | Slow: {slow_mag:.3f}")

        final_slow_mag = self.slow_mags[-1]
        slow_change = final_slow_mag - self.initial_slow_mag
        if self.initial_slow_mag > 0:
            slow_change_pct = (slow_change / self.initial_slow_mag) * 100
        else:
            slow_change_pct = float('inf') if slow_change > 0 else 0.0

        print(f"\n  {'─'*66}")
        print(f"  Final slow layer magnitude: {final_slow_mag:.4f}")
        print(f"  Change: {slow_change:+.4f} ({slow_change_pct:+.1f}%)")
        print(f"  Mean surprise: {np.mean(self.surprises):.4f}")
        print(f"  Final surprise: {self.surprises[-1]:.4f}")

        # Save checkpoint and memories
        checkpoint_name = f"session{self.session_number}_{self.session_name.lower().replace(' ', '_')}.npz"
        memories_name = f"session{self.session_number}_memories.json"

        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        memories_path = os.path.join(save_dir, memories_name)

        self.agent.save_checkpoint(checkpoint_path)
        self.agent.save_memories(memories_path)

        print(f"  💾 Saved: {checkpoint_name}")
        print(f"  {'='*70}\n")

        # Return session metrics
        return {
            "session_number": self.session_number,
            "session_name": self.session_name,
            "initial_slow_mag": self.initial_slow_mag,
            "final_slow_mag": final_slow_mag,
            "slow_change": slow_change,
            "slow_change_percent": slow_change_pct,
            "mean_surprise": np.mean(self.surprises),
            "final_surprise": self.surprises[-1],
            "num_moments": len(affects),
            "checkpoint_path": checkpoint_path,
            "memories_path": memories_path,
            "surprises": self.surprises,
            "slow_magnitudes": self.slow_mags,
            "fast_magnitudes": self.fast_mags,
            "medium_magnitudes": self.medium_mags,
        }


def run_full_relationship_simulation(save_dir: str = "relationships") -> List[Dict]:
    """
    Run all four sessions of the relationship simulation.

    Args:
        save_dir: Directory to save all outputs

    Returns:
        List of session metric dictionaries
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "    LONG-TERM RELATIONSHIP SIMULATION".center(68) + "║")
    print("║" + "    Building a Model of 'Us' Through Time".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")

    all_sessions = []

    # Session 1: First Meeting
    session1 = RelationshipSession(
        session_number=1,
        session_name="First Meeting",
        session_func=session1_first_meeting,
        load_checkpoint=None
    )
    metrics1 = session1.run(save_dir)
    all_sessions.append(metrics1)

    # Session 2: Building Trust
    session2 = RelationshipSession(
        session_number=2,
        session_name="Building Trust",
        session_func=session2_building_trust,
        load_checkpoint=metrics1["checkpoint_path"]
    )
    metrics2 = session2.run(save_dir)
    all_sessions.append(metrics2)

    # Session 3: Conflict and Repair
    session3 = RelationshipSession(
        session_number=3,
        session_name="Conflict and Repair",
        session_func=session3_conflict_and_repair,
        load_checkpoint=metrics2["checkpoint_path"]
    )
    metrics3 = session3.run(save_dir)
    all_sessions.append(metrics3)

    # Session 4: Deep Bond
    session4 = RelationshipSession(
        session_number=4,
        session_name="Deep Bond",
        session_func=session4_deep_bond,
        load_checkpoint=metrics3["checkpoint_path"]
    )
    metrics4 = session4.run(save_dir)
    all_sessions.append(metrics4)

    # Save overall summary
    summary = {
        "sessions": all_sessions,
        "total_moments": sum(s["num_moments"] for s in all_sessions),
        "slow_layer_journey": {
            "initial": all_sessions[0]["initial_slow_mag"],
            "after_session1": all_sessions[0]["final_slow_mag"],
            "after_session2": all_sessions[1]["final_slow_mag"],
            "after_session3": all_sessions[2]["final_slow_mag"],
            "after_session4": all_sessions[3]["final_slow_mag"],
            "total_change": all_sessions[3]["final_slow_mag"] - all_sessions[0]["initial_slow_mag"],
            "total_change_percent": (
                ((all_sessions[3]["final_slow_mag"] / all_sessions[0]["initial_slow_mag"]) - 1) * 100
                if all_sessions[0]["initial_slow_mag"] > 0
                else float('inf')
            ),
        }
    }

    summary_path = os.path.join(save_dir, "relationship_summary.json")

    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(summary_path, 'w') as f:
        json.dump(convert_to_native(summary), f, indent=2)

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "    RELATIONSHIP SIMULATION COMPLETE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")

    print(f"\n📊 SUMMARY")
    print(f"{'─'*70}")
    print(f"Total moments experienced: {summary['total_moments']}")
    print(f"\nSlow Layer Journey:")
    print(f"  Initial:        {summary['slow_layer_journey']['initial']:.4f}")
    print(f"  After Session 1: {summary['slow_layer_journey']['after_session1']:.4f}")
    print(f"  After Session 2: {summary['slow_layer_journey']['after_session2']:.4f}")
    print(f"  After Session 3: {summary['slow_layer_journey']['after_session3']:.4f}")
    print(f"  After Session 4: {summary['slow_layer_journey']['after_session4']:.4f}")
    print(f"\n  Total Change: {summary['slow_layer_journey']['total_change']:+.4f} "
          f"({summary['slow_layer_journey']['total_change_percent']:+.1f}%)")

    # Check key hypotheses
    print(f"\n✓ HYPOTHESIS TESTS")
    session3_recovery = all_sessions[3]["initial_slow_mag"] < all_sessions[3]["final_slow_mag"]
    print(f"  Resilience (Session 3 recovery): {'✓ PASS' if session3_recovery else '✗ FAIL'}")

    final_higher = all_sessions[3]["final_slow_mag"] > all_sessions[0]["initial_slow_mag"]
    print(f"  Growth (Session 4 > Session 1): {'✓ PASS' if final_higher else '✗ FAIL'}")

    print(f"\n💾 All outputs saved to: {save_dir}/")
    print(f"{'─'*70}\n")

    return all_sessions


if __name__ == "__main__":
    # Run the full simulation
    sessions = run_full_relationship_simulation()
