#!/usr/bin/env python3
"""
Quick script for Claude Code to interact with NoodleMUSH
"""
import asyncio
import json
import sys
import websockets
import aiohttp

async def send_message(message: str):
    """Send a message to NoodleMUSH and get responses."""
    async with websockets.connect('ws://localhost:8765') as websocket:
        # Login
        await websocket.send(json.dumps({
            "type": "login",
            "username": "Claude",
            "password": "mcpserver"
        }))

        # Wait for login response (may receive system messages first)
        logged_in = False
        for _ in range(5):
            response = await websocket.recv()
            result = json.loads(response)

            if result.get("type") == "login_response":
                if not result.get("success"):
                    print(f"Login failed: {result.get('message')}")
                    return None
                logged_in = True
                break

        if not logged_in:
            print("Did not receive login confirmation")
            return None

        # Send message
        await websocket.send(json.dumps({
            "type": "command",
            "command": f"say {message}"
        }))

        # Collect responses for 2 seconds
        responses = []
        try:
            async with asyncio.timeout(2.0):
                while True:
                    response = await websocket.recv()
                    responses.append(json.loads(response))
        except asyncio.TimeoutError:
            pass

        return responses

async def get_agent_state(agent_id: str):
    """Get agent's phenomenal state."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://localhost:8081/api/profiler/realtime/{agent_id}?last_n=1"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}

async def list_agents():
    """List active agents."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:8081/api/profiler/live-session"
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("metadata", {}).get("agents", [])
            else:
                return {"error": f"HTTP {response.status}"}

async def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ./claude_interact.py list")
        print("  ./claude_interact.py say <message>")
        print("  ./claude_interact.py state <agent_id>")
        return

    command = sys.argv[1]

    if command == "list":
        agents = await list_agents()
        print(json.dumps(agents, indent=2))

    elif command == "say":
        message = " ".join(sys.argv[2:])
        responses = await send_message(message)
        print(json.dumps(responses, indent=2))

    elif command == "state":
        agent_id = sys.argv[2]
        state = await get_agent_state(agent_id)
        print(json.dumps(state, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
