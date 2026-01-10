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
# MODULE:   applications.cmush.test_live_interaction
# PURPOSE:  Live message send/receive test
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Live interaction test - send a message and wait longer for SERVNAK to respond.
"""
import asyncio
import websockets
import json

async def test():
    uri = 'ws://localhost:8765'
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        print("Connected!")

        # Login
        print("Sending login...")
        await ws.send(json.dumps({'type': 'login', 'username': 'caity', 'password': 'j33k13p13'}))

        # Clear history messages
        print("Clearing history messages...")
        for _ in range(50):
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                break

        print("\n" + "=" * 60)
        print("Sending: 'say hello SERVNAK!'")
        print("=" * 60)

        # Send a greeting
        await ws.send(json.dumps({'type': 'command', 'command': 'say hello SERVNAK!'}))

        # Wait up to 15 seconds for ANY response
        print("\nWaiting for responses (up to 15 seconds)...")
        deadline = asyncio.get_event_loop().time() + 15.0

        response_count = 0
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                msg_type = data.get('type')

                if msg_type == 'output':
                    response_count += 1
                    text = data.get('text', '')
                    print(f"\n[OUTPUT] {text}")
                elif msg_type == 'system':
                    text = data.get('text', '')
                    print(f"\n[SYSTEM] {text}")
                elif msg_type not in ['history']:  # Skip history
                    print(f"\n[{msg_type.upper()}] {data}")

            except asyncio.TimeoutError:
                print(".", end="", flush=True)
                continue

        print(f"\n\nReceived {response_count} output messages")

asyncio.run(test())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
