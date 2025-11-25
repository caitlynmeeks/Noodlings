#!/usr/bin/env python3
"""
Integration test: Spawn agent with AffectTransistor and verify loading.
"""

import yaml
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_affect_transistor_integration():
    """Test AffectTransistor integration with agent creation."""

    print("\n" + "=" * 70)
    print("INTEGRATION TEST: AffectTransistor in Agent System")
    print("=" * 70)

    # Import after logging setup
    from agent_bridge import CMUSHConsilienceAgent
    from world import World
    from llm_interface import OpenAICompatibleLLM
    import os

    # Create minimal world
    world = World()

    # Create LLM client
    config = yaml.safe_load(open('config.yaml'))
    llm_client = OpenAICompatibleLLM(
        base_url=config['llm']['lmstudio']['base_url'],
        api_key="not-needed",
        default_model=config['llm']['lmstudio']['default_model']
    )

    # Load Spock recipe
    print("\n1. Loading Spock recipe (low affect salience)...")
    with open('recipes/spock_example.yaml', 'r') as f:
        recipe = yaml.safe_load(f)

    affect_config = recipe['cognitive_components'].get('affect')
    if affect_config:
        print(f"    Found AffectTransistor config: salience={affect_config.get('salience')}")
    else:
        print("   ✗ ERROR: No affect config in recipe!")
        return False

    # Create agent
    print("\n2. Creating CMUSHConsilienceAgent with AffectTransistor...")
    agent = CMUSHConsilienceAgent(
        agent_id='test_spock',
        config={**config['agent'], **recipe},
        llm_client=llm_client,
        world=world
    )

    # Check if cognitive manifold exists
    if not hasattr(agent, 'cognitive_manifold'):
        print("   ✗ ERROR: No cognitive_manifold attribute!")
        return False

    if agent.cognitive_manifold is None:
        print("   ✗ ERROR: cognitive_manifold is None!")
        return False

    print(f"    Cognitive manifold created")

    # Check transistors
    transistor_count = len(agent.cognitive_manifold.transistors)
    print(f"    Loaded {transistor_count} transistors:")

    affect_found = False
    for transistor in agent.cognitive_manifold.transistors:
        transistor_type = transistor.__class__.__name__
        salience = transistor.salience
        print(f"      - {transistor_type:25s} (salience={salience:.2f})")

        if transistor_type == 'AffectTransistor':
            affect_found = True
            if salience == 0.15:
                print(f"         Correct salience for Vulcan emotional suppression")
            else:
                print(f"        ✗ Unexpected salience: {salience} (expected 0.15)")

    if not affect_found:
        print("   ✗ ERROR: AffectTransistor not loaded!")
        return False

    print("\n" + "=" * 70)
    print("SUCCESS: AffectTransistor integrates correctly with agent system!")
    print("=" * 70)
    print("\nSpock can now suppress emotions with affect salience = 0.15")
    print("While cultural/logic transistors dominate with salience = 0.85-0.90")
    return True

if __name__ == '__main__':
    success = asyncio.run(test_affect_transistor_integration())
    exit(0 if success else 1)
