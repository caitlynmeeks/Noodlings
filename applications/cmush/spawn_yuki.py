#!/usr/bin/env python3
"""
Spawn Yuki the Cyberfox
Quick script to spawn Yuki via WebSocket
"""

import asyncio
import websockets
import json

async def spawn_yuki():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Login
        await websocket.send(json.dumps({
            "type": "login",
            "username": "caity",
            "password": "caity"
        }))

        response = await websocket.recv()
        print(f"Login: {response}")

        # Spawn Yuki
        await websocket.send(json.dumps({
            "type": "command",
            "command": "@spawn yuki_cyberfox"
        }))

        # Wait for responses
        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(response)
                if data.get('type') == 'output':
                    print(data.get('text', ''))
            except asyncio.TimeoutError:
                break

        print("\n Yuki spawn command sent!")
        print("Check noodleMUSH to see if she appeared!")

if __name__ == '__main__':
    asyncio.run(spawn_yuki())
