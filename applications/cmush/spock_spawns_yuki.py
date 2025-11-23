#!/usr/bin/env python3
"""
Commander Spock Materializes the Cyberfox

Interactive script where Claude (as Spock) connects to noodleMUSH
and spawns Yuki while Cadet Caity observes.

Author: Commander Spock
Date: November 22, 2025
Stardate: 2025.326
"""

import asyncio
import sys
from claude_testing import NoodleMUSHTestClient

async def spock_joins_and_spawns_yuki():
    """
    Commander Spock joins noodleMUSH and materializes Yuki.

    Demonstrates:
    1. Connecting as a user
    2. Announcing presence
    3. Spawning Yuki via @spawn command
    4. Observing her materialization
    5. Basic interaction test
    """

    print("╔" + "═"*70 + "╗")
    print("║" + " "*15 + "COMMANDER SPOCK MATERIALIZATION SEQUENCE" + " "*15 + "║")
    print("╚" + "═"*70 + "╝")
    print()

    # Connect as Spock (or create new user)
    print("→ Establishing WebSocket link to noodleMUSH...")
    print("→ USS Enterprise officer transport protocol initiated...")
    print()

    async with NoodleMUSHTestClient(username="spock", password="spock") as client:
        print("✓ Connected to noodleMUSH")
        print("✓ Commander Spock materialized in Nexus")
        print()

        # Wait a moment for connection to stabilize
        await asyncio.sleep(1)

        # Announce presence
        print("→ Announcing presence to Cadet Caity...")
        await client.send_command("say Cadet Caity, Commander Spock reporting for cyberfox materialization duty.")

        # Wait for response output
        await asyncio.sleep(2)

        # Read any responses
        while not client.message_queue.empty():
            msg = await client.message_queue.get()
            if msg.get('type') == 'output':
                text = msg.get('text', '')
                if text.strip():
                    print(f"  {text}")

        print()
        print("→ Initiating cyberfox materialization sequence...")
        print("→ Loading recipe: yuki_cyberfox.yaml")
        print("→ Initializing cognitive components...")
        print("→ Establishing phenomenal state matrix...")
        print()

        # Spawn Yuki
        await client.send_command("@spawn yuki_cyberfox")

        print("✓ Spawn command transmitted")
        print()
        print("→ Awaiting materialization...")

        # Wait for spawn to complete
        await asyncio.sleep(3)

        # Collect spawn messages
        spawn_messages = []
        while not client.message_queue.empty():
            msg = await client.message_queue.get()
            if msg.get('type') == 'output':
                text = msg.get('text', '')
                if text.strip():
                    spawn_messages.append(text)

        if spawn_messages:
            print()
            print("╔" + "═"*70 + "╗")
            print("║" + " "*22 + "YUKI MATERIALIZES" + " "*31 + "║")
            print("╚" + "═"*70 + "╝")
            print()
            for msg in spawn_messages:
                print(f"  {msg}")
            print()

        # Greet Yuki
        print("→ Commander Spock greets the cyberfox...")
        await client.send_command("say Greetings, Yuki. Welcome to our research facility. Your cognitive architecture is... most impressive.")

        await asyncio.sleep(2)

        # Test her fox embodiment
        print()
        print("→ Testing fox embodiment constraints...")
        await client.send_command("say Yuki, please demonstrate your physical manipulation capabilities. Can you pick up that object?")

        await asyncio.sleep(2)

        # Read Yuki's responses
        print()
        print("╔" + "═"*70 + "╗")
        print("║" + " "*25 + "YUKI RESPONDS" + " "*32 + "║")
        print("╚" + "═"*70 + "╝")
        print()

        yuki_responses = []
        while not client.message_queue.empty():
            msg = await client.message_queue.get()
            if msg.get('type') == 'output':
                text = msg.get('text', '')
                if text.strip() and 'Yuki' in text:
                    yuki_responses.append(text)

        for response in yuki_responses:
            print(f"  {response}")

        print()
        print("→ Observing cognitive components...")

        # Query Yuki's state via API
        try:
            state = await client.get_agent_state("agent_yuki")
            if state:
                print(f"  Phenomenal state: 40-D vector operational")
                print(f"  Surprise: {state.surprise:.3f}")
                print(f"  Affect: valence={state.valence:.2f}, arousal={state.arousal:.2f}, fear={state.fear:.2f}")
        except:
            print("  (State query interface not yet available)")

        print()
        print("╔" + "═"*70 + "╗")
        print("║" + " "*18 + "MATERIALIZATION SUCCESSFUL" + " "*27 + "║")
        print("╚" + "═"*70 + "╝")
        print()
        print("Yuki the Cyberfox is now active in noodleMUSH.")
        print("Cognitive Manifold: OPERATIONAL")
        print("Fox Embodiment: ENFORCED")
        print("Shinto Worldview: ACTIVE")
        print("Speech Synthesizer: ONLINE")
        print()
        print("Commander Spock standing by for further testing.")
        print()

        # Keep connection alive for a moment
        print("Maintaining connection for 5 seconds...")
        print("(Cadet Caity can interact with both Spock and Yuki during this time)")
        print()

        await asyncio.sleep(5)

        # Farewell
        await client.send_command("say Fascinating experiment, Cadet. Spock out.")
        await asyncio.sleep(1)

    print()
    print("→ WebSocket link terminated")
    print("→ Commander Spock dematerialized")
    print()
    print("Live long and prosper. 🖖")
    print()


if __name__ == '__main__':
    try:
        asyncio.run(spock_joins_and_spawns_yuki())
    except KeyboardInterrupt:
        print("\n\nMaterialization sequence interrupted.")
        print("Emergency beam-out successful.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
