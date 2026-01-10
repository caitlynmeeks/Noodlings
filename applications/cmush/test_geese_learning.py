# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.test_geese_learning
# PURPOSE:  Test geese affective learning over turns
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Test script for geese affective learning

Spawns fresh geese with affective reinforcement, then interacts
to observe affect changes over 10-20 conversational turns.

Expected behavior:
- When geese do comedy (HONK, trip, fumble): valence rises
- When geese do mysticism (quiet, stillness): valence drops, boredom rises
- After 10-15 turns: slow LSTM layer learns "I prefer comedy"

Usage:
    python test_geese_learning.py
"""

import asyncio
import aiohttp
import json
import time

API_BASE = "http://localhost:8080/api"

async def remove_agent(agent_name: str):
    """Remove an existing agent."""
    async with aiohttp.ClientSession() as session:
        url = f"{API_BASE}/command"
        payload = {
            "command": f"@remove {agent_name}",
            "user_id": "spock"
        }
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            print(f"Remove agent: {result.get('output', result)}")

async def spawn_agent(agent_name: str, recipe: str = "mysterious_stranger", fresh: bool = True):
    """Spawn a new agent from recipe."""
    async with aiohttp.ClientSession() as session:
        url = f"{API_BASE}/command"
        fresh_flag = " -f" if fresh else ""
        payload = {
            "command": f"@spawn {agent_name} -r {recipe}{fresh_flag}",
            "user_id": "spock"
        }
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            print(f"Spawn agent: {result.get('output', result)}")
            return result.get('success', False)

async def get_agent_state(agent_id: str):
    """Get current agent phenomenal state."""
    async with aiohttp.ClientSession() as session:
        url = f"{API_BASE}/agents/{agent_id}/state"
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                phenom = data.get('phenomenal_state', [])
                if len(phenom) >= 5:
                    valence, arousal, fear, sorrow, boredom = phenom[:5]
                    return {
                        'valence': valence,
                        'arousal': arousal,
                        'fear': fear,
                        'sorrow': sorrow,
                        'boredom': boredom,
                        'surprise': data.get('surprise', 0.0)
                    }
            return None

async def say_to_geese(text: str, user: str = "spock"):
    """Send a message to the geese."""
    async with aiohttp.ClientSession() as session:
        url = f"{API_BASE}/command"
        payload = {
            "command": f"say {text}",
            "user_id": user
        }
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            events = result.get('events', [])

            # Find geese response
            for event in events:
                if event.get('agent_id') == 'agent_geese':
                    return event.get('text', '(no response)')

            return "(no response)"

async def run_learning_test():
    """
    Run 20-turn conversation with geese, tracking affect changes.
    """
    print("=" * 70)
    print("AFFECTIVE REINFORCEMENT LEARNING TEST")
    print("Testing with Mysterious_Stranger (fugitive geese)")
    print("=" * 70)
    print()

    # Step 1: Remove old agent
    print("[1/4] Removing old geese agent...")
    await remove_agent("geese")
    await asyncio.sleep(1)

    # Step 2: Spawn fresh with new recipe
    print("[2/4] Spawning fresh geese with affective reinforcement...")
    success = await spawn_agent("geese", recipe="mysterious_stranger", fresh=True)
    if not success:
        print(" Failed to spawn geese!")
        return

    await asyncio.sleep(3)  # Let agent initialize

    # Step 3: Get initial state
    print("[3/4] Recording initial affect state...")
    initial_state = await get_agent_state("agent_geese")
    if initial_state:
        print(f"  Initial valence: {initial_state['valence']:.2f}")
        print(f"  Initial boredom: {initial_state['boredom']:.2f}")
    print()

    # Step 4: Run conversation loop
    print("[4/4] Running 20-turn conversation...")
    print("-" * 70)
    print()

    test_messages = [
        # Comedy-inducing messages (should boost valence)
        "Hey there! Want some bread?",
        "*holds out a fresh baguette*",
        "You can sit down if you want!",
        "*offers root beer*",
        "Nice coat! Very human-like!",
        # Neutral messages
        "How are you doing?",
        "What brings you here?",
        "Tell me about yourself",
        # More comedy prompts
        "Do a little dance!",
        "*tosses bread in air*",
        # Comedy-inducing
        "Show me your best waddle!",
        "*applauds enthusiastically* That was great!",
        "You're so funny!",
        "*giggles* Do that again!",
        "You remind me of a cartoon character!",
        # Final rounds
        "Want more bread?",
        "*offers entire bakery box*",
        "You're adorable!",
        "Best performance ever!",
        "*standing ovation*"
    ]

    affect_history = []

    for turn, message in enumerate(test_messages, 1):
        print(f"Turn {turn}/20: {message}")

        # Send message
        response = await say_to_geese(message)
        print(f"  Geese: {response[:100]}{'...' if len(response) > 100 else ''}")

        # Get affect after response
        await asyncio.sleep(1)
        state = await get_agent_state("agent_geese")

        if state:
            affect_history.append(state)
            valence = state['valence']
            boredom = state['boredom']

            # Check for comedy markers in response
            comedy_markers = ['*honk*', '*HONK*', 'trip', 'fumble', 'waddle', 'feather']
            has_comedy = any(marker.lower() in response.lower() for marker in comedy_markers)

            # Check for mysticism markers
            mysticism_markers = ['quiet', 'stillness', 'silence', 'calm', 'gentle']
            has_mysticism = any(marker.lower() in response.lower() for marker in mysticism_markers)

            marker_str = " COMEDY" if has_comedy else ("😴 MYSTICISM" if has_mysticism else "")
            print(f"  Affect: valence={valence:.2f}, boredom={boredom:.2f} {marker_str}")

        print()
        await asyncio.sleep(2)  # Conversation pacing

    # Step 5: Analyze results
    print("=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    print()

    if len(affect_history) < 2:
        print(" Not enough data collected")
        return

    # Calculate trends
    valences = [s['valence'] for s in affect_history]
    boredoms = [s['boredom'] for s in affect_history]

    initial_valence = valences[0]
    final_valence = valences[-1]
    valence_change = final_valence - initial_valence

    initial_boredom = boredoms[0]
    final_boredom = boredoms[-1]
    boredom_change = final_boredom - initial_boredom

    print(f"Initial valence: {initial_valence:.2f}")
    print(f"Final valence:   {final_valence:.2f}")
    print(f"Change:          {valence_change:+.2f}")
    print()
    print(f"Initial boredom: {initial_boredom:.2f}")
    print(f"Final boredom:   {final_boredom:.2f}")
    print(f"Change:          {boredom_change:+.2f}")
    print()

    # Check if learning occurred
    if valence_change > 0.1:
        print(" POSITIVE LEARNING: Geese valence increased!")
        print("  Affective reinforcement may be working.")
    elif valence_change < -0.1:
        print("⚠ NEGATIVE LEARNING: Geese valence decreased")
        print("  May indicate mysticism is still present")
    else:
        print("→ NEUTRAL: No significant valence change")
        print("  May need more turns or stronger intensity")

    print()
    print("Recommended next steps:")
    print("1. Review logs for ' COMEDY REWARD' and '😴 MYSTICISM PENALTY' messages")
    print("2. If no rewards firing, check if markers are detected correctly")
    print("3. If rewards firing but no learning, increase intensity in recipe")
    print("4. If mysticism persists, may need additional architectural changes")

if __name__ == '__main__':
    try:
        asyncio.run(run_learning_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
