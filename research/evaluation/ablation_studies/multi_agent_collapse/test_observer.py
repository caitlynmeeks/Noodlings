#!/usr/bin/env python3
"""
Test script for ObserverAgent.

Tests if observers can watch and correct active agents to prevent collapse.
"""

import sys
sys.path.insert(0, '..')

import mlx.core as mx
from agents import NoodlingAgent, ObserverAgent

print("="*70)
print("OBSERVER AGENT TEST")
print("="*70)
print()

# Create a regular agent and an observer
print("Creating test agents...")
active_agent = NoodlingAgent(agent_id="active_001", input_dim=10)
observer = ObserverAgent(
    agent_id="observer_L0_001",
    observation_level=0,
    input_dim=10
)

print(f"✓ Active agent: {active_agent.agent_id}")
print(f"✓ Observer: {observer.agent_id} (Level {observer.observation_level})")
print()

# Run test episode: observer watches active agent
print("Running 100-step episode with observer intervention...")
print()

for step in range(100):
    # Random environment
    env_obs = mx.random.normal((10,))

    # Active agent acts WITHOUT observer first
    active_state_before, action, msg = active_agent.perceive_and_update(env_obs)

    # Observer watches and generates correction
    corrections = observer.observe_agents([active_agent], env_obs)
    correction = corrections.get(active_agent.agent_id)

    # Apply observer correction to active agent
    if correction is not None:
        active_state_after, _, _ = active_agent.perceive_and_update(
            env_obs,
            observer_correction=correction
        )

        # Validate observer's prediction
        observer.validate_prediction(
            active_agent.agent_id,
            active_state_before,  # What observer thought would happen
            active_state_after    # What actually happened
        )

    # Print progress
    if (step + 1) % 25 == 0:
        agent_hsi = active_agent.calculate_hsi()
        obs_quality = observer.get_observation_quality()

        print(f"Step {step+1:3d}:")
        print(f"  Agent HSI: {agent_hsi['slow/fast']:.4f} - {agent_hsi['interpretation']}")
        print(f"  Observer: {obs_quality['total_corrections']} corrections, "
              f"error={obs_quality['avg_prediction_error']:.4f}")
        print()

print("="*70)
print("FINAL ANALYSIS")
print("="*70)
print()

# Agent final state
agent_hsi = active_agent.calculate_hsi()
print("Active Agent:")
print(f"  HSI (slow/fast): {agent_hsi['slow/fast']:.4f}")
print(f"  Interpretation: {agent_hsi['interpretation']}")
print()

# Observer quality
obs_quality = observer.get_observation_quality()
print("Observer Performance:")
print(f"  Observation level: {obs_quality['observation_level']}")
print(f"  Agents watched: {obs_quality['agents_observed']}")
print(f"  Total corrections: {obs_quality['total_corrections']}")
print(f"  Avg prediction error: {obs_quality['avg_prediction_error']:.4f}")
print()

print("✓ Observer test complete!")
print()
print("KEY OBSERVATION: Did observer maintain agent's low HSI?")
print("Compare with agent running alone (HSI ≈ 0.16 from earlier test)")
print()
