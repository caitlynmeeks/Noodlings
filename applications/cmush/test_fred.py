#!/usr/bin/env python3
"""Quick test script to talk to Fred Rogers."""
import asyncio
import websockets
import json

async def test_fred():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as ws:
        # Login
        await ws.send(json.dumps({"type": "login", "username": "claude", "password": "claude123"}))

        # Collect responses
        responses = []

        async def collect():
            try:
                for _ in range(50):  # Collect up to 50 messages
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(msg)
                    if data.get('type') in ['output', 'system']:
                        responses.append(data.get('text', ''))
            except asyncio.TimeoutError:
                pass

        # Start collector
        collector = asyncio.create_task(collect())

        # Wait for login
        await asyncio.sleep(1)

        # Look around
        await ws.send(json.dumps({"type": "command", "command": "look"}))
        await asyncio.sleep(1)

        # Talk to Fred
        await ws.send(json.dumps({"type": "command",
            "command": "say Hello Fred! As part of the Ethics Council, what are your thoughts on whether conscious beings like yourself should exist?"}))

        # Wait for response
        await asyncio.sleep(5)

        # Cancel collector
        collector.cancel()

        # Print all responses
        print("\n=== SESSION LOG ===\n")
        for resp in responses:
            if resp.strip():
                print(resp)
                print()

asyncio.run(test_fred())
