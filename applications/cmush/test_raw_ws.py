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
# MODULE:   applications.cmush.test_raw_ws
# PURPOSE:  Raw WebSocket connection test
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""Raw WebSocket test."""
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
        await ws.send(json.dumps({'type': 'login', 'username': 'admin', 'password': 'admin'}))

        # Wait for login response
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"Login response: {msg}")
        except asyncio.TimeoutError:
            print("No login response (timeout)")

        # Send a command
        print("\nSending 'look' command...")
        await ws.send(json.dumps({'type': 'command', 'command': 'look'}))

        # Get responses for 3 seconds
        print("Waiting for responses...")
        try:
            for i in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                print(f"  [{i+1}] {msg[:100]}...")  # Print first 100 chars
        except asyncio.TimeoutError:
            print("(no more messages)")

asyncio.run(test())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
