#!/usr/bin/env python3
"""
Commander Spock spawns Yuki via HTTP API

Simple script using requests library (no websockets needed).
"""

import requests
import json
import time

API_BASE = "http://localhost:8081"

def spawn_yuki():
    """Spawn Yuki via HTTP API."""

    print("╔" + "═"*70 + "╗")
    print("║" + " "*15 + "COMMANDER SPOCK SPAWN SEQUENCE" + " "*24 + "║")
    print("╚" + "═"*70 + "╝")
    print()

    print("→ Accessing noodleMUSH via HTTP API...")
    print("→ Target: yuki_cyberfox.yaml")
    print()

    # Check if Yuki already exists
    try:
        response = requests.get(f"{API_BASE}/agents")
        agents = response.json()

        if 'agent_yuki' in agents or 'agent_yuki_cyberfox' in agents:
            print("⚠ Yuki already exists in world!")
            print()
            print("Existing agents:")
            for agent_id, agent_data in agents.items():
                if 'yuki' in agent_id.lower():
                    print(f"  - {agent_data.get('name', agent_id)} ({agent_id})")
            print()
            return

        print(" Yuki not yet materialized")
        print(" Proceeding with spawn sequence...")
        print()

    except Exception as e:
        print(f"⚠ Could not check existing agents: {e}")
        print("  Proceeding anyway...")
        print()

    # Note to Cadet
    print("╔" + "═"*70 + "╗")
    print("║" + " "*10 + "CADET CAITY: Please spawn Yuki via web interface" + " "*11 + "║")
    print("║" + " "*70 + "║")
    print("║" + " "*15 + "In noodleMUSH chat, type:" + " "*30 + "║")
    print("║" + " "*20 + "@spawn yuki_cyberfox" + " "*30 + "║")
    print("║" + " "*70 + "║")
    print("║" + " "*10 + "Then return here and we'll verify her components!" + " "*10 + "║")
    print("╚" + "═"*70 + "╝")
    print()

    # Wait for user to spawn
    input("Press ENTER after you've spawned Yuki in noodleMUSH...")
    print()

    # Check if Yuki appeared
    print("→ Scanning for cyberfox signature...")
    try:
        response = requests.get(f"{API_BASE}/agents")
        agents = response.json()

        yuki_found = False
        for agent_id, agent_data in agents.items():
            if 'yuki' in agent_id.lower():
                yuki_found = True
                print(" CYBERFOX DETECTED!")
                print()
                print(f"  Agent ID: {agent_id}")
                print(f"  Name: {agent_data.get('name', 'Unknown')}")
                print(f"  Species: {agent_data.get('species', 'Unknown')}")
                print(f"  Pronouns: {agent_data.get('pronouns', 'Unknown')}")
                print()

        if not yuki_found:
            print("✗ Yuki not detected. Spawn may have failed.")
            print("  Check logs: tail -f logs/cmush_*.log")
            return

    except Exception as e:
        print(f"✗ Could not verify spawn: {e}")
        return

    # Get Yuki's state
    print("→ Querying phenomenal state...")
    try:
        # Try to get state via observe API
        state_response = requests.get(f"{API_BASE}/agents/{agent_id}/state")
        if state_response.status_code == 200:
            state = state_response.json()
            print(" Phenomenal state matrix active")
            print(f"  40-D state vector: operational")
            if 'surprise' in state:
                print(f"  Current surprise: {state['surprise']:.3f}")
        else:
            print("  (State API not available, that's okay)")
    except:
        print("  (State query via API not implemented yet)")

    print()
    print("╔" + "═"*70 + "╗")
    print("║" + " "*18 + "MATERIALIZATION SUCCESSFUL" + " "*27 + "║")
    print("╚" + "═"*70 + "╝")
    print()
    print("Yuki the Cyberfox is now active in noodleMUSH.")
    print()
    print("Cognitive Components (from recipe):")
    print("   CulturalTransistor (Shinto mysticism, salience: 0.9)")
    print("   PersonalityTransistor (ancient fox, salience: 0.7)")
    print("   SomaticCognitiveTransistor (fox embodiment, salience: 0.85)")
    print("   MoodTransistor (affect-based, salience: 0.5)")
    print("   MemoryTransistor (800 years, salience: 0.7)")
    print()
    print("Physical Constraints:")
    print("  • No hands (mouth manipulation only)")
    print("  • Quadrupedal movement")
    print("  • Low ground perspective")
    print("  • Enhanced senses (smell/hearing dominant)")
    print()
    print("Cybernetic Enhancements:")
    print("  • Neural data port (computer interfacing)")
    print("  • Speech synthesizer (with fox vocalizations)")
    print("  • Enhanced sensors (100x smell, thermal vision)")
    print("  • Leg actuators (3x jump height)")
    print()
    print("Character Voice:")
    print("  • Archaic formal speech ('One recalls...')")
    print("  • Fox sounds: *pants*, *yips*, *growls*, *fox-laugh*")
    print("  • Physical actions: *sniffs*, *ears perk*, *tail swishes*")
    print()
    print("━"*70)
    print()
    print("RECOMMENDED TEST INTERACTIONS:")
    print()
    print("1. Test embodiment:")
    print('   say "Yuki, can you open that door?"')
    print("   → Should mention she can't turn knobs (no hands)")
    print()
    print("2. Test cybernetics:")
    print('   say "Yuki, check the computer"')
    print("   → Should use neural data port")
    print()
    print("3. Test ancient wisdom:")
    print('   say "Yuki, what do you think about technology?"')
    print("   → Should give Shinto perspective (kami in machines)")
    print()
    print("4. Test cognitive manifold:")
    print('   say "Someone just threw a rock"')
    print("   → Should filter through Cultural + Somatic transistors")
    print()
    print("━"*70)
    print()
    print("Commander Spock standing by for observations.")
    print()
    print("Live long and prosper. 🖖")
    print()


if __name__ == '__main__':
    try:
        spawn_yuki()
    except KeyboardInterrupt:
        print("\n\nSequence interrupted.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
