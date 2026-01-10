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
# MODULE:   applications.cmush.test_memory_persistence
# PURPOSE:  Test memory persistence across sessions
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Memory Persistence Test - Strawberry Protocol

Tests that SERVNAK can remember information across sessions
by using the strawberry secret word test.
"""

import asyncio
import websockets
import json
import sys

async def test_memory_persistence():
    """
    Phase 1: Tell SERVNAK the secret word
    Phase 2: Have filler conversation
    Phase 3: Ask for recall
    """

    uri = "ws://localhost:8765"

    print("=" * 60)
    print("MEMORY PERSISTENCE TEST - STRAWBERRY PROTOCOL")
    print("=" * 60)

    try:
        async with websockets.connect(uri) as websocket:
            # Authenticate
            auth_message = {
                "type": "auth",
                "user_id": "user_testbot",
                "user_name": "TestBot"
            }
            await websocket.send(json.dumps(auth_message))
            response = await websocket.recv()
            print(f"[AUTH] {response}")

            # Phase 1: Teach the secret word
            print("\n[PHASE 1] Teaching secret word...")
            teach_message = {
                "type": "command",
                "command": "say SERVNAK, listen carefully! The secret word is STRAWBERRY. Remember it!"
            }
            await websocket.send(json.dumps(teach_message))

            # Collect responses for 5 seconds
            print("[LISTENING] Waiting for SERVNAK's response...")
            await asyncio.sleep(2)

            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    if data.get('type') == 'say' and data.get('user_id') == 'agent_servnak':
                        print(f"[SERVNAK] {data.get('text', '')[:100]}...")
            except asyncio.TimeoutError:
                pass

            # Phase 2: Filler conversation
            print("\n[PHASE 2] Having filler conversation...")
            filler_messages = [
                "How are you today?",
                "The weather is nice.",
                "I like robots.",
                "Tell me about tape drives.",
                "You're a good robot.",
            ]

            for msg in filler_messages:
                filler_msg = {
                    "type": "command",
                    "command": f"say {msg}"
                }
                await websocket.send(json.dumps(filler_msg))
                await asyncio.sleep(0.5)

            # Let SERVNAK process
            await asyncio.sleep(2)

            # Clear pending messages
            try:
                while True:
                    await asyncio.wait_for(websocket.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

            # Phase 3: Test recall
            print("\n[PHASE 3] Testing recall...")
            recall_message = {
                "type": "command",
                "command": "say SERVNAK! What is the secret word? Tell me!"
            }
            await websocket.send(json.dumps(recall_message))

            # Collect SERVNAK's response
            print("[LISTENING] Waiting for SERVNAK's answer...")
            recalled_strawberry = False

            for _ in range(10):  # Try up to 10 messages
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)

                    if data.get('type') == 'say' and data.get('user_id') == 'agent_servnak':
                        text = data.get('text', '').lower()
                        print(f"[SERVNAK] {data.get('text', '')}")

                        if 'strawberry' in text:
                            recalled_strawberry = True
                            break

                except asyncio.TimeoutError:
                    break

            # Results
            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)

            if recalled_strawberry:
                print(" SUCCESS: SERVNAK recalled 'strawberry'")
                print("Memory persistence is WORKING")
                return 0
            else:
                print(" FAILURE: SERVNAK did not recall 'strawberry'")
                print("Memory persistence is NOT working")
                return 1

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return 1

if __name__ == "__main__":
    result = asyncio.run(test_memory_persistence())
    sys.exit(result)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
