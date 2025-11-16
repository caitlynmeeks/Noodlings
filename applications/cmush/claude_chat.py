#!/usr/bin/env python3
"""
Interactive chat session for Claude to talk with Noodlings
"""
import asyncio
import json
import websockets

async def interactive_chat():
    """Maintain persistent connection and chat with Noodlings."""
    async with websockets.connect('ws://localhost:8765') as websocket:
        # Login
        await websocket.send(json.dumps({
            "type": "login",
            "username": "Claude",
            "password": "mcpserver"
        }))

        # Wait for login response
        logged_in = False
        for _ in range(5):
            response = await websocket.recv()
            result = json.loads(response)

            if result.get("type") == "login_response":
                if not result.get("success"):
                    print(f"Login failed: {result.get('message')}")
                    return
                logged_in = True
                break

        if not logged_in:
            print("Did not receive login confirmation")
            return

        print("=== Connected to NoodleMUSH ===")
        print("Listening for messages from Noodlings...")
        print("=" * 40)

        # Keep connection alive and listen for messages
        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                # Print all messages we receive
                if data.get("type") == "speech":
                    speaker = data.get("speaker", "Unknown")
                    text = data.get("text", "")
                    print(f"\n{speaker} says: {text}")
                elif data.get("type") == "output":
                    print(f"\n{data.get('text', '')}")
                elif data.get("type") == "event":
                    print(f"\n[{data.get('event_type', 'event')}] {data.get('text', '')}")

        except KeyboardInterrupt:
            print("\n\nDisconnecting...")
        except websockets.exceptions.ConnectionClosed:
            print("\n\nConnection closed")

if __name__ == "__main__":
    asyncio.run(interactive_chat())
