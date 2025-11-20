#!/usr/bin/env python3
"""
Simple WebSocket client for Claude to interact with noodleMUSH.
"""
import asyncio
import websockets
import json
import sys

async def connect_to_mush():
    """Connect to noodleMUSH and interact."""
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("[Connected to noodleMUSH]")

        # Login as admin
        login_msg = {
            "type": "login",
            "username": "admin",
            "password": "admin123"
        }
        await websocket.send(json.dumps(login_msg))

        # Start listening for messages
        async def listen():
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    if data.get('type') == 'output':
                        print(data.get('text', ''))
                    elif data.get('type') == 'system':
                        print(f"[SYSTEM] {data.get('text', '')}")
                    elif data.get('type') == 'history':
                        # Skip history for now
                        pass
                    elif data.get('type') == 'login':
                        if data.get('success'):
                            print(f"[Logged in as {data.get('username')}]")
                        else:
                            print(f"[Login failed: {data.get('message')}]")
                except websockets.exceptions.ConnectionClosed:
                    print("[Connection closed]")
                    break
                except Exception as e:
                    print(f"[Error: {e}]")

        # Start listener task
        listen_task = asyncio.create_task(listen())

        # Wait a moment for login
        await asyncio.sleep(1)

        # Send commands from stdin
        print("\n[Ready to send commands. Type 'quit' to exit.]")
        print("[Commands: say <text>, @observe <agent>, look, etc.]\n")

        try:
            while True:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                line = line.strip()

                if not line:
                    continue

                if line.lower() == 'quit':
                    break

                # Send command
                command_msg = {
                    "type": "command",
                    "command": line
                }
                await websocket.send(json.dumps(command_msg))

                # Give time for response
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            listen_task.cancel()

if __name__ == "__main__":
    asyncio.run(connect_to_mush())
