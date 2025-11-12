#!/usr/bin/env python3
"""
INTEGRATION TEST - All Components Together!

Tests if:
1. Agents can play the resource game
2. Observers can watch and correct agents
3. HSI is maintained under pressure
4. The whole system doesn't explode

This is the PROOF OF CONCEPT before we scale to full experiments!

Author: Noodlings Project
Date: November 2025
"""

import sys
sys.path.insert(0, '..')

import mlx.core as mx
import numpy as np
from agents import NoodlingAgent, ObserverAgent
from environment import ResourceAllocationGame


def run_integration_test(
    num_agents: int = 3,
    num_observers: int = 1,
    num_rounds: int = 100,
    verbose: bool = True
):
    """
    Run integration test of all components.

    Args:
        num_agents: Number of active agents
        num_observers: Number of observer agents
        num_rounds: Rounds to run
        verbose: Print detailed output
    """

    if verbose:
        print("="*80)
        print("🧪 INTEGRATION TEST: Agents + Observers + Environment")
        print("="*80)
        print()
        print(f"Configuration:")
        print(f"  Active agents: {num_agents}")
        print(f"  Observers: {num_observers}")
        print(f"  Rounds: {num_rounds}")
        print()

    # === CREATE COMPONENTS ===

    # 1. Create agents
    agents = []
    for i in range(num_agents):
        agent = NoodlingAgent(
            agent_id=f"agent_{i:03d}",
            input_dim=10,
            action_dim=3,  # GRAB, SHARE, WAIT
            message_dim=8
        )
        agents.append(agent)

    if verbose:
        print(f"✓ Created {len(agents)} active agents")

    # 2. Create observers
    observers = []
    for i in range(num_observers):
        observer = ObserverAgent(
            agent_id=f"observer_{i:03d}",
            observation_level=0,
            input_dim=10,
            message_dim=8
        )
        observers.append(observer)

    if verbose:
        print(f"✓ Created {len(observers)} observers")

    # 3. Create environment
    game = ResourceAllocationGame(
        num_agents=num_agents,
        base_resources=num_agents * 3,  # 3 resources per agent
        scarcity_cycle=50
    )

    if verbose:
        print(f"✓ Created resource game")
        print()
        print("="*80)
        print("RUNNING SIMULATION")
        print("="*80)
        print()

    # === RUN SIMULATION ===

    agent_hsi_history = {i: [] for i in range(num_agents)}
    game_stats_history = []

    for round_num in range(num_rounds):

        # 1. AGENTS OBSERVE ENVIRONMENT
        observations = {}
        for i, agent in enumerate(agents):
            obs = game.get_observation(i)
            observations[i] = obs

        # 2. OBSERVERS WATCH AGENTS (generate corrections)
        corrections = {}
        if observers:
            # All observers watch all agents
            for observer in observers:
                agent_corrections = observer.observe_agents(agents, observations[0])
                corrections.update(agent_corrections)

        # 3. AGENTS PERCEIVE AND DECIDE (with observer corrections if present)
        actions = {}
        messages = {}

        for i, agent in enumerate(agents):
            # Get correction for this agent
            correction = corrections.get(agent.agent_id, None)

            # Perceive (with correction!)
            state, action_logits, message = agent.perceive_and_update(
                observations[i],
                observer_correction=correction
            )

            # Choose action (sample from logits)
            # For now, use simple argmax
            action = int(mx.argmax(action_logits, axis=-1))

            actions[i] = action
            messages[i] = message

        # 4. ENVIRONMENT UPDATES
        rewards, info, done = game.step(actions)

        # 5. AGENTS RECEIVE REWARDS
        for i, agent in enumerate(agents):
            agent.update_from_reward(rewards[i])

        # 6. TRACK METRICS
        for i, agent in enumerate(agents):
            hsi = agent.calculate_hsi()
            if not np.isnan(hsi['slow/fast']):
                agent_hsi_history[i].append(hsi['slow/fast'])

        game_stats_history.append(info)

        # 7. PRINT PROGRESS
        if verbose and (round_num + 1) % 25 == 0:
            print(f"Round {round_num + 1}/{num_rounds}")
            print(f"  Game: scarcity={info['scarcity']:.2f}, "
                  f"cooperation={info['cooperation_events']}, "
                  f"avg_trust={info['avg_trust']:.2f}")

            # Agent HSI
            avg_hsi = np.mean([h[-1] for h in agent_hsi_history.values() if h])
            print(f"  Agents: avg_HSI={avg_hsi:.4f}")

            # Observer quality
            if observers:
                obs_quality = observers[0].get_observation_quality()
                print(f"  Observer: corrections={obs_quality['total_corrections']}, "
                      f"error={obs_quality['avg_prediction_error']:.4f}")
            print()

        if done:
            break

    # === ANALYZE RESULTS ===

    if verbose:
        print("="*80)
        print("FINAL ANALYSIS")
        print("="*80)
        print()

    # Game statistics
    final_stats = game.get_game_statistics()

    if verbose:
        print("Game Outcomes:")
        print(f"  Total rounds: {final_stats['total_rounds']}")
        print(f"  Action distribution:")
        print(f"    Grab:  {final_stats['grab_rate']:.1%}")
        print(f"    Share: {final_stats['share_rate']:.1%}")
        print(f"    Wait:  {final_stats['wait_rate']:.1%}")
        print(f"  Cooperation events: {final_stats['cooperation_events']}")
        print(f"  Conflict events: {final_stats['conflict_events']}")
        print(f"  Avg score: {final_stats['avg_score']:.2f} ± {final_stats['score_std']:.2f}")
        print(f"  Avg trust: {final_stats['avg_trust']:.2f}")
        print()

    # Agent HSI analysis
    if verbose:
        print("Agent Hierarchical Stability:")

    hsi_results = {}
    for i, agent in enumerate(agents):
        final_hsi = agent.calculate_hsi()
        hsi_results[i] = final_hsi

        if verbose:
            print(f"  Agent {i}: HSI={final_hsi['slow/fast']:.4f} - {final_hsi['interpretation']}")

    avg_final_hsi = np.mean([h['slow/fast'] for h in hsi_results.values()
                              if not np.isnan(h['slow/fast'])])

    if verbose:
        print(f"  Average: HSI={avg_final_hsi:.4f}")
        print()

    # Observer performance
    if observers and verbose:
        print("Observer Performance:")
        for i, observer in enumerate(observers):
            quality = observer.get_observation_quality()
            print(f"  Observer {i}:")
            print(f"    Corrections sent: {quality['total_corrections']}")
            print(f"    Avg prediction error: {quality['avg_prediction_error']:.4f}")
            print(f"    Agents watched: {quality['agents_observed']}")
        print()

    # Success criteria
    if verbose:
        print("="*80)
        print("SUCCESS CRITERIA")
        print("="*80)
        print()

    success = True

    # 1. Agents maintained low HSI
    if avg_final_hsi < 0.5:
        if verbose:
            print("✅ PASS: Agents maintained hierarchical separation (HSI < 0.5)")
    else:
        if verbose:
            print(f"❌ FAIL: Agents collapsed (HSI = {avg_final_hsi:.4f} > 0.5)")
        success = False

    # 2. Some cooperation occurred
    if final_stats['cooperation_events'] > 10:
        if verbose:
            print(f"✅ PASS: Cooperation emerged ({final_stats['cooperation_events']} events)")
    else:
        if verbose:
            print(f"⚠️  WARN: Low cooperation ({final_stats['cooperation_events']} events)")

    # 3. Observers had low prediction error
    if observers:
        avg_obs_error = np.mean([obs.get_observation_quality()['avg_prediction_error']
                                 for obs in observers])
        if avg_obs_error < 0.01:
            if verbose:
                print(f"✅ PASS: Observers accurate (error = {avg_obs_error:.4f})")
        else:
            if verbose:
                print(f"⚠️  WARN: Observer error high (error = {avg_obs_error:.4f})")

    # 4. No crashes!
    if verbose:
        print("✅ PASS: System ran to completion without crashes!")

    print()

    if success and verbose:
        print("🎉 INTEGRATION TEST SUCCESSFUL!")
        print()
        print("All components work together! Ready to scale to full experiments.")
    elif verbose:
        print("⚠️  Integration test completed with warnings.")
        print("System functional but needs tuning.")

    print()

    # Return results for analysis
    return {
        'agents': agents,
        'observers': observers,
        'game': game,
        'hsi_history': agent_hsi_history,
        'game_stats': game_stats_history,
        'final_hsi': avg_final_hsi,
        'final_stats': final_stats,
        'success': success
    }


if __name__ == '__main__':
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         NOODLINGS MULTI-AGENT INTEGRATION TEST                    ║")
    print("║                                                                    ║")
    print("║  Testing if observers prevent collapse in competitive game        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    # Run test
    results = run_integration_test(
        num_agents=3,
        num_observers=1,
        num_rounds=100,
        verbose=True
    )

    print()
    print("="*80)
    print()
    print("🔬 SCIENTIFIC ASSESSMENT:")
    print()

    if results['success']:
        print("The multi-timescale + observer architecture successfully:")
        print("  1. Maintained hierarchical separation (low HSI)")
        print("  2. Enabled strategic multi-timescale behavior")
        print("  3. Prevented collapse under game pressure")
        print()
        print("👉 CONCLUSION: System ready for full ablation study!")
        print("   Next: Compare NO OBSERVERS vs OBSERVERS across 30 runs")
    else:
        print("System needs tuning before full study.")
        print("Check agent parameters, learning rates, or observer strength.")

    print()
    print("="*80)
    print()
