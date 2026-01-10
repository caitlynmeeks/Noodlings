#!/usr/bin/env python3
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
# MODULE:   applications.cmush.test_checkpoint_loading
# PURPOSE:  Test checkpoint loading from recipes
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Test Checkpoint Loading Fix

Verifies that:
1. Recipe checkpoint paths are passed to agent creation
2. Checkpoints load successfully
3. 40-D phenomenal states are populated
"""

import asyncio
import websockets
import json
import time
import subprocess
from typing import Dict, Any

# Connection settings
WS_URL = "ws://localhost:8765"
API_URL = "http://localhost:8081"
USERNAME = "caity"
PASSWORD = "j33k13p13"

async def send_command(websocket, command: str) -> Dict[str, Any]:
    """Send command and wait for response."""
    await websocket.send(json.dumps({
        'type': 'command',
        'command': command
    }))

    response = await websocket.recv()
    return json.loads(response)

async def test_checkpoint_loading():
    """Test that checkpoints are loaded when spawning agents."""
    print("=" * 70)
    print("CHECKPOINT LOADING TEST")
    print("=" * 70)

    # Connect to server
    async with websockets.connect(WS_URL) as websocket:
        print(f"\n Connected to {WS_URL}")

        # Authenticate
        await websocket.send(json.dumps({
            'type': 'login',
            'username': USERNAME,
            'password': PASSWORD
        }))

        login_response = await websocket.recv()
        login_data = json.loads(login_response)

        if login_data.get('success'):
            print(f" Authenticated as {USERNAME}")
        else:
            print(f"✗ Authentication failed: {login_data.get('output')}")
            return

        # Step 1: Remove existing testsubject if present
        print("\n--- Step 1: Removing existing agent ---")
        response = await send_command(websocket, "@remove -s testsubject")
        if response.get('success'):
            print(" Removed agent_testsubject")
        else:
            print(f"  (Agent may not exist: {response.get('output', 'unknown error')})")

        time.sleep(1)

        # Step 2: Spawn fresh testsubject
        print("\n--- Step 2: Spawning fresh agent ---")
        print("Command: @spawn testsubject")
        response = await send_command(websocket, "@spawn testsubject")

        if response.get('success'):
            print(f" Spawn successful: {response.get('output', '')}")
        else:
            print(f"✗ Spawn failed: {response.get('output', 'unknown error')}")
            return

        time.sleep(2)  # Wait for initialization

        # Step 3: Query API for phenomenal state
        print("\n--- Step 3: Querying phenomenal state ---")
        result = subprocess.run(
            ['curl', '-s', f"{API_URL}/api/agents/agent_testsubject/state"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            state_data = json.loads(result.stdout)

            fast_state = state_data.get('fast_state', [])
            medium_state = state_data.get('medium_state', [])
            slow_state = state_data.get('slow_state', [])

            fast_dim = len(fast_state)
            medium_dim = len(medium_state)
            slow_dim = len(slow_state)
            total_dim = fast_dim + medium_dim + slow_dim

            print(f"Fast state: {fast_dim}-D")
            print(f"Medium state: {medium_dim}-D")
            print(f"Slow state: {slow_dim}-D")
            print(f"Total: {total_dim}-D")

            # Verify dimensions
            if fast_dim == 16 and medium_dim == 16 and slow_dim == 8:
                print("\n SUCCESS: 40-D phenomenal state loaded correctly!")
                print("  Fast layer: 16-D ")
                print("  Medium layer: 16-D ")
                print("  Slow layer: 8-D ")

                # Show sample values
                if fast_state:
                    print(f"\n  Sample fast_state values: {fast_state[:3]}...")
                if medium_state:
                    print(f"  Sample medium_state values: {medium_state[:3]}...")
                if slow_state:
                    print(f"  Sample slow_state values: {slow_state[:3]}...")

                return True
            else:
                print("\n✗ FAILURE: Phenomenal state has wrong dimensions")
                print(f"  Expected: 16 + 16 + 8 = 40-D")
                print(f"  Got: {fast_dim} + {medium_dim} + {slow_dim} = {total_dim}-D")

                if total_dim == 0:
                    print("\n    This indicates checkpoint was NOT loaded!")
                    print("     Check server logs for ' Loaded checkpoint' message")

                return False
        else:
            print(f"✗ API request failed")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False

if __name__ == '__main__':
    try:
        success = asyncio.run(test_checkpoint_loading())

        print("\n" + "=" * 70)
        if success:
            print("TEST PASSED")
            print("Checkpoint loading is working correctly!")
        else:
            print("TEST FAILED")
            print("Checkpoint loading is NOT working")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
