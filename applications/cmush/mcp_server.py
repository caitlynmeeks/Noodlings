#!/usr/bin/env python3
"""
NoodleMUSH MCP Server

Allows Claude instances to interact with NoodleMUSH and observe Noodling consciousness.

Tools provided:
- noodlemush_send_message: Send a message to the MUD
- noodlemush_get_agent_state: Get an agent's current phenomenal state
- noodlemush_query_profiler: Query session profiler timeline data
- noodlemush_ask_kimmie: Get @Kimmie's interpretation of consciousness data
- noodlemush_list_agents: List all active agents in the world

Author: NoodleMUSH Project
Date: November 2025
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import websockets
import aiohttp

# MCP SDK imports
from mcp.server import Server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoodleMUSHMCPServer:
    """MCP server for NoodleMUSH integration."""

    def __init__(
        self,
        websocket_url: str = "ws://localhost:8765",
        api_url: str = "http://localhost:8081"
    ):
        """
        Initialize NoodleMUSH MCP server.

        Args:
            websocket_url: WebSocket URL for NoodleMUSH
            api_url: HTTP API URL for profiler/Kimmie
        """
        self.websocket_url = websocket_url
        self.api_url = api_url
        self.server = Server("noodlemush")

        # Register tool handlers
        self._register_handlers()

        logger.info(f"NoodleMUSH MCP server initialized")
        logger.info(f"WebSocket: {websocket_url}")
        logger.info(f"API: {api_url}")

    def _register_handlers(self):
        """Register MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available NoodleMUSH tools."""
            return [
                Tool(
                    name="noodlemush_send_message",
                    description="Send a message to NoodleMUSH. You can chat with Noodlings and see their responses.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to send (e.g., 'hi callie' or 'hello everyone')"
                            }
                        },
                        "required": ["message"]
                    }
                ),
                Tool(
                    name="noodlemush_get_agent_state",
                    description="Get the current phenomenal state and affect of a Noodling agent. Returns 40-D phenomenal state, 5-D affect vector, surprise level, and recent activity.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID (e.g., 'agent_callie', 'agent_desobelle')"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="noodlemush_query_profiler",
                    description="Query the session profiler for timeline data. Get consciousness evolution over time, including phenomenal states, affect trajectories, surprise spikes, HSI metrics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID to query (optional, returns all if not specified)"
                            },
                            "last_n": {
                                "type": "number",
                                "description": "Number of recent timesteps to return (default: 10)"
                            }
                        }
                    }
                ),
                Tool(
                    name="noodlemush_ask_kimmie",
                    description="Ask @Kimmie (the phenomenal state interpreter) to explain what's happening in an agent's consciousness during a specific time period.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID to analyze"
                            },
                            "question": {
                                "type": "string",
                                "description": "Question to ask @Kimmie (e.g., 'What caused that surprise spike?')"
                            },
                            "start_time": {
                                "type": "number",
                                "description": "Start timestamp (optional)"
                            },
                            "end_time": {
                                "type": "number",
                                "description": "End timestamp (optional)"
                            }
                        },
                        "required": ["agent_id", "question"]
                    }
                ),
                Tool(
                    name="noodlemush_list_agents",
                    description="List all active Noodling agents in the world with their current locations and status.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "noodlemush_send_message":
                    result = await self._send_message(arguments["message"])
                elif name == "noodlemush_get_agent_state":
                    result = await self._get_agent_state(arguments["agent_id"])
                elif name == "noodlemush_query_profiler":
                    result = await self._query_profiler(
                        arguments.get("agent_id"),
                        arguments.get("last_n", 10)
                    )
                elif name == "noodlemush_ask_kimmie":
                    result = await self._ask_kimmie(
                        arguments["agent_id"],
                        arguments["question"],
                        arguments.get("start_time"),
                        arguments.get("end_time")
                    )
                elif name == "noodlemush_list_agents":
                    result = await self._list_agents()
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

    async def _send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to NoodleMUSH via WebSocket."""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                # Try to login as Claude
                login_msg = {
                    "type": "login",
                    "username": "Claude",
                    "password": "mcpserver"
                }
                await websocket.send(json.dumps(login_msg))

                # Wait for login response
                response = await websocket.recv()
                login_result = json.loads(response)

                # If login fails, try to register
                if login_result.get("type") != "login_success":
                    register_msg = {
                        "type": "register",
                        "username": "Claude",
                        "password": "mcpserver"
                    }
                    await websocket.send(json.dumps(register_msg))

                    response = await websocket.recv()
                    register_result = json.loads(response)

                    if not register_result.get("success"):
                        return {"error": "Registration failed", "details": register_result}

                    # Now login with new account
                    await websocket.send(json.dumps(login_msg))
                    response = await websocket.recv()
                    login_result = json.loads(response)

                    if login_result.get("type") != "login_success":
                        return {"error": "Login failed after registration", "details": login_result}

                # Send the message
                say_msg = {
                    "type": "command",
                    "command": f"say {message}"
                }
                await websocket.send(json.dumps(say_msg))

                # Collect responses for 2 seconds
                responses = []
                try:
                    async with asyncio.timeout(2.0):
                        while True:
                            response = await websocket.recv()
                            responses.append(json.loads(response))
                except asyncio.TimeoutError:
                    pass

                return {
                    "success": True,
                    "message_sent": message,
                    "responses": responses
                }

        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            return {"error": str(e)}

    async def _get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent's current phenomenal state."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/api/profiler/realtime/{agent_id}?last_n=1"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            latest = data[-1]
                            return {
                                "success": True,
                                "agent_id": agent_id,
                                "current_state": latest
                            }
                        else:
                            return {
                                "success": False,
                                "error": "No state data available yet"
                            }
                    else:
                        return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Error getting agent state: {e}", exc_info=True)
            return {"error": str(e)}

    async def _query_profiler(
        self,
        agent_id: Optional[str] = None,
        last_n: int = 10
    ) -> Dict[str, Any]:
        """Query session profiler for timeline data."""
        try:
            async with aiohttp.ClientSession() as session:
                if agent_id:
                    url = f"{self.api_url}/api/profiler/realtime/{agent_id}?last_n={last_n}"
                else:
                    url = f"{self.api_url}/api/profiler/live-session"

                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "data": data
                        }
                    else:
                        return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Error querying profiler: {e}", exc_info=True)
            return {"error": str(e)}

    async def _ask_kimmie(
        self,
        agent_id: str,
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Ask @Kimmie to interpret consciousness data."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "agent_id": agent_id,
                    "user_message": question
                }
                if start_time is not None:
                    payload["start_time"] = start_time
                if end_time is not None:
                    payload["end_time"] = end_time

                async with session.post(
                    f"{self.api_url}/api/kimmie/interpret",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "interpretation": data.get("interpretation", "")
                        }
                    else:
                        return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Error asking Kimmie: {e}", exc_info=True)
            return {"error": str(e)}

    async def _list_agents(self) -> Dict[str, Any]:
        """List all active agents."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/api/profiler/live-session"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        agents = data.get("metadata", {}).get("agents", [])
                        return {
                            "success": True,
                            "agents": agents,
                            "count": len(agents)
                        }
                    else:
                        return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Error listing agents: {e}", exc_info=True)
            return {"error": str(e)}

    async def run(self):
        """Run the MCP server."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = NoodleMUSHMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
