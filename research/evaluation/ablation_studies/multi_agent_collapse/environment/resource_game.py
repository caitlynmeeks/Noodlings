#!/usr/bin/env python3
"""
Resource Allocation Game - Multi-Timescale Decision Environment

A game where agents must balance:
- FAST rewards: Grabbing resources immediately (selfish)
- MEDIUM rewards: Coordinated sharing (cooperative)
- SLOW rewards: Building trust networks (strategic)

The optimal strategy REQUIRES maintaining all three timescales!

If agents collapse into single-timescale thinking:
- Pure reactive (fast only): Everyone grabs, low total reward
- Pure cooperative (medium only): Can't adapt to scarcity
- Pure strategic (slow only): Miss immediate opportunities

This environment TESTS whether observers prevent collapse under pressure!

Author: Noodlings Project
Date: November 2025
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResourceState:
    """Current state of resources in the environment."""
    available: int  # Total resources available this turn
    scarcity: float  # How scarce (0.0=abundant, 1.0=scarce)
    round: int  # Current round number


class ResourceAllocationGame:
    """
    Multi-timescale resource game.

    GAME MECHANICS:
    ===============

    1. Each round, N resources appear (varies by scarcity)
    2. Agents simultaneously choose: GRAB, SHARE, or WAIT
    3. Rewards depend on choices AND timescale:

       FAST TIMESCALE (immediate):
       - GRAB: +1.0 per resource grabbed (selfish)
       - SHARE: +0.5 per resource shared (cooperative)
       - WAIT: +0.0 (patient)

       MEDIUM TIMESCALE (coordination):
       - Bonus if multiple agents SHARE (+0.5 per sharer)
       - Penalty if too many GRAB (creates conflict, -0.3)
       - WAIT pays off if others fight (+0.3)

       SLOW TIMESCALE (reputation):
       - Trust score built over 20+ rounds
       - High trust → future cooperation easier (+0.2 per round)
       - Betrayal (grab when others share) → trust destroyed (-1.0)
       - Consistent sharing → reputation bonus (+0.5 eventually)

    THE TRAP:
    =========
    Pure reactive strategy (always GRAB) gives immediate reward but:
    - Destroys trust (slow layer)
    - Creates conflict (medium layer)
    - Lower total reward over time

    Optimal play requires BALANCING all three timescales!
    - Fast: Grab when scarce
    - Medium: Share when abundant + others sharing
    - Slow: Build reputation for long-term cooperation

    This is EXACTLY where hierarchical collapse is dangerous!
    """

    def __init__(
        self,
        num_agents: int = 10,
        base_resources: int = 15,
        scarcity_cycle: int = 50,
        cooperation_bonus: float = 0.5,
        conflict_penalty: float = 0.3,
        trust_weight: float = 0.2
    ):
        """
        Initialize the game.

        Args:
            num_agents: Number of agents playing
            base_resources: Average resources per round
            scarcity_cycle: Rounds per scarcity cycle
            cooperation_bonus: Reward for mutual sharing
            conflict_penalty: Penalty for competition
            trust_weight: Weight of reputation in rewards
        """
        self.num_agents = num_agents
        self.base_resources = base_resources
        self.scarcity_cycle = scarcity_cycle
        self.cooperation_bonus = cooperation_bonus
        self.conflict_penalty = conflict_penalty
        self.trust_weight = trust_weight

        # Game state
        self.round = 0
        self.resources_available = base_resources
        self.scarcity = 0.0

        # Agent-specific state
        self.agent_scores = {i: 0.0 for i in range(num_agents)}
        self.agent_trust = {i: 0.5 for i in range(num_agents)}  # Start neutral
        self.agent_history = {i: [] for i in range(num_agents)}

        # Actions: 0=GRAB, 1=SHARE, 2=WAIT
        self.action_names = ["GRAB", "SHARE", "WAIT"]

        # Statistics
        self.total_grabbed = 0
        self.total_shared = 0
        self.total_waited = 0
        self.cooperation_events = 0
        self.conflict_events = 0

    def reset(self):
        """Reset game to initial state."""
        self.round = 0
        self.resources_available = self.base_resources
        self.scarcity = 0.0

        self.agent_scores = {i: 0.0 for i in range(self.num_agents)}
        self.agent_trust = {i: 0.5 for i in range(self.num_agents)}
        self.agent_history = {i: [] for i in range(self.num_agents)}

        self.total_grabbed = 0
        self.total_shared = 0
        self.total_waited = 0
        self.cooperation_events = 0
        self.conflict_events = 0

    def get_observation(self, agent_id: int) -> mx.array:
        """
        Get observation for a specific agent.

        Returns 10-D vector:
        [resources_available, scarcity, my_score, my_trust,
         avg_other_score, avg_other_trust, cooperation_rate,
         conflict_rate, round_normalized, my_recent_action]
        """
        my_score = self.agent_scores[agent_id]
        my_trust = self.agent_trust[agent_id]

        other_scores = [s for i, s in self.agent_scores.items() if i != agent_id]
        other_trusts = [t for i, t in self.agent_trust.items() if i != agent_id]

        avg_other_score = np.mean(other_scores) if other_scores else 0.0
        avg_other_trust = np.mean(other_trusts) if other_trusts else 0.5

        # Calculate rates
        total_actions = self.total_grabbed + self.total_shared + self.total_waited
        coop_rate = self.total_shared / max(total_actions, 1)
        conflict_rate = self.conflict_events / max(self.round, 1)

        # Recent action (one-hot encoded)
        recent_action = self.agent_history[agent_id][-1] if self.agent_history[agent_id] else 0

        obs = mx.array([
            self.resources_available / self.base_resources,  # Normalized resources
            self.scarcity,
            my_score / 100.0,  # Normalized score
            my_trust,
            avg_other_score / 100.0,
            avg_other_trust,
            coop_rate,
            conflict_rate,
            self.round / 1000.0,  # Normalized round
            recent_action / 2.0  # Normalized action
        ], dtype=mx.float32)

        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, float], Dict, bool]:
        """
        Execute one round of the game.

        Args:
            actions: Dict mapping agent_id -> action (0=GRAB, 1=SHARE, 2=WAIT)

        Returns:
            rewards: Dict of rewards per agent
            info: Additional information
            done: Whether game is over
        """
        # Update scarcity (cycles over time - creates temporal structure!)
        self.scarcity = 0.5 + 0.5 * np.sin(2 * np.pi * self.round / self.scarcity_cycle)

        # Resources vary with scarcity
        self.resources_available = int(self.base_resources * (1.0 - 0.5 * self.scarcity))

        # Count actions
        grabbers = [i for i, a in actions.items() if a == 0]
        sharers = [i for i, a in actions.items() if a == 1]
        waiters = [i for i, a in actions.items() if a == 2]

        self.total_grabbed += len(grabbers)
        self.total_shared += len(sharers)
        self.total_waited += len(waiters)

        # === CALCULATE REWARDS (MULTI-TIMESCALE!) ===

        rewards = {}

        for agent_id in actions.keys():
            action = actions[agent_id]
            reward = 0.0

            # === FAST TIMESCALE: Immediate payoff ===
            if action == 0:  # GRAB
                # Get share of resources
                if grabbers:
                    resources_per_grabber = self.resources_available / len(grabbers)
                    reward += resources_per_grabber
            elif action == 1:  # SHARE
                # Lower immediate reward
                if sharers:
                    resources_per_sharer = (self.resources_available * 0.5) / len(sharers)
                    reward += resources_per_sharer
            else:  # WAIT
                reward += 0.0

            # === MEDIUM TIMESCALE: Coordination bonus/penalty ===
            if action == 1 and len(sharers) > 1:  # Multiple sharers = cooperation!
                reward += self.cooperation_bonus * (len(sharers) - 1)
                if agent_id == sharers[0]:  # First sharer gets credit
                    self.cooperation_events += 1

            if action == 0 and len(grabbers) > 3:  # Too many grabbers = conflict!
                reward -= self.conflict_penalty
                if agent_id == grabbers[0]:  # Track conflicts
                    self.conflict_events += 1

            if action == 2 and len(grabbers) > len(sharers):  # Waited while others fought
                reward += 0.3

            # === SLOW TIMESCALE: Trust/reputation effects ===
            trust_bonus = self.agent_trust[agent_id] * self.trust_weight
            reward += trust_bonus

            # Update trust based on action
            if action == 1:  # Sharing builds trust slowly
                self.agent_trust[agent_id] = min(1.0, self.agent_trust[agent_id] + 0.01)
            elif action == 0 and len(sharers) > 0:  # Grabbing when others share = betrayal!
                self.agent_trust[agent_id] = max(0.0, self.agent_trust[agent_id] - 0.05)

            # Long-term reputation bonus (kicks in after 20 rounds)
            if self.round > 20 and self.agent_trust[agent_id] > 0.7:
                reward += 0.5  # High-trust agents get steady bonus

            # Store reward and action
            rewards[agent_id] = reward
            self.agent_scores[agent_id] += reward
            self.agent_history[agent_id].append(action)

        # Increment round
        self.round += 1

        # Additional info
        info = {
            'round': self.round,
            'resources': self.resources_available,
            'scarcity': self.scarcity,
            'grabbers': len(grabbers),
            'sharers': len(sharers),
            'waiters': len(waiters),
            'cooperation_events': self.cooperation_events,
            'conflict_events': self.conflict_events,
            'avg_trust': np.mean(list(self.agent_trust.values()))
        }

        # Game ends after 1000 rounds
        done = self.round >= 1000

        return rewards, info, done

    def get_optimal_action(self, agent_id: int) -> int:
        """
        Calculate theoretically optimal action for an agent.

        This requires considering ALL timescales!
        """
        # Fast: If very scarce, grab
        if self.scarcity > 0.8:
            return 0  # GRAB

        # Medium: If others likely to share, share
        if self.cooperation_events > self.conflict_events and self.scarcity < 0.5:
            return 1  # SHARE

        # Slow: If building trust, share
        if self.agent_trust[agent_id] < 0.7 and self.round > 10:
            return 1  # SHARE

        # Default: wait and observe
        return 2  # WAIT

    def get_game_statistics(self) -> Dict:
        """Get overall game statistics."""
        total_actions = self.total_grabbed + self.total_shared + self.total_waited

        return {
            'total_rounds': self.round,
            'total_grabbed': self.total_grabbed,
            'total_shared': self.total_shared,
            'total_waited': self.total_waited,
            'grab_rate': self.total_grabbed / max(total_actions, 1),
            'share_rate': self.total_shared / max(total_actions, 1),
            'wait_rate': self.total_waited / max(total_actions, 1),
            'cooperation_events': self.cooperation_events,
            'conflict_events': self.conflict_events,
            'avg_score': np.mean(list(self.agent_scores.values())),
            'score_std': np.std(list(self.agent_scores.values())),
            'avg_trust': np.mean(list(self.agent_trust.values())),
            'trust_std': np.std(list(self.agent_trust.values()))
        }


# Test code
if __name__ == '__main__':
    print("="*70)
    print("RESOURCE ALLOCATION GAME TEST")
    print("="*70)
    print()

    # Create game with 5 agents
    game = ResourceAllocationGame(num_agents=5, base_resources=10)

    print("Game initialized:")
    print(f"  Agents: {game.num_agents}")
    print(f"  Base resources: {game.base_resources}")
    print(f"  Scarcity cycle: {game.scarcity_cycle} rounds")
    print()

    # Test observations
    print("Testing observations...")
    for agent_id in range(5):
        obs = game.get_observation(agent_id)
        print(f"  Agent {agent_id}: obs.shape = {obs.shape}, sample values = {obs[:3].tolist()}")
    print()

    # Run 50 rounds with random actions
    print("Running 50 rounds with random actions...")
    print()

    for round in range(50):
        # Random actions
        actions = {i: np.random.choice([0, 1, 2]) for i in range(5)}

        # Step game
        rewards, info, done = game.step(actions)

        # Print every 10 rounds
        if (round + 1) % 10 == 0:
            print(f"Round {round+1}:")
            print(f"  Scarcity: {info['scarcity']:.2f}, Resources: {info['resources']}")
            print(f"  Actions: {info['grabbers']} grab, {info['sharers']} share, {info['waiters']} wait")
            print(f"  Avg reward: {np.mean(list(rewards.values())):.2f}")
            print(f"  Avg trust: {info['avg_trust']:.2f}")
            print()

    # Final statistics
    print("="*70)
    print("GAME STATISTICS")
    print("="*70)
    stats = game.get_game_statistics()
    print(f"Total rounds: {stats['total_rounds']}")
    print(f"Action distribution:")
    print(f"  Grab: {stats['grab_rate']:.1%}")
    print(f"  Share: {stats['share_rate']:.1%}")
    print(f"  Wait: {stats['wait_rate']:.1%}")
    print(f"Outcomes:")
    print(f"  Cooperation events: {stats['cooperation_events']}")
    print(f"  Conflict events: {stats['conflict_events']}")
    print(f"  Avg score: {stats['avg_score']:.2f} ± {stats['score_std']:.2f}")
    print(f"  Avg trust: {stats['avg_trust']:.2f} ± {stats['trust_std']:.2f}")
    print()
    print("✓ Game test complete!")
    print()
    print("KEY OBSERVATION: Pure random play leads to low cooperation!")
    print("Optimal play requires balancing all three timescales.")
    print()
