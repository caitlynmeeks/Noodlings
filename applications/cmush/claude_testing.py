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
#
#   Claude Testing Harness
#
#   A programmatic test client for running experiments on the
#   noodleMUSH system. Connects via WebSocket and HTTP API to
#   send commands, read agent states, and run test protocols.
#   Used for systematic testing of memory retention (like the
#   "strawberry test" - tell an agent a secret word, chat about
#   other things, then ask if they remember).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.claude_testing
# PURPOSE:  Programmatic test harness for agent memory and behavior
# LAYER:    Backend / Testing
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NoodleMUSHTestClient   Async client for WebSocket/API testing
#   AgentState             Snapshot of agent phenomenal state
#   TestResult             Outcome of a test protocol
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Claude Testing Harness for noodleMUSH

Programmatic interface for systematic testing of Noodling consciousness and memory systems.
Provides async API for connecting, sending commands, querying state, and running test protocols.

Author: Caitlyn + Claude
Date: November 22, 2025
"""
import asyncio
import websockets
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentState:
    """Agent phenomenal state snapshot."""
    agent_id: str
    fast: List[float]  # 16-D (first 5 are affect)
    medium: List[float]  # 16-D
    slow: List[float]  # 8-D
    surprise: float
    surprise_threshold: float
    step: int
    timestamp: float

    @property
    def affect(self) -> List[float]:
        """Extract 5-D affect vector from fast state."""
        return self.fast[:5] if len(self.fast) >= 5 else [0.0] * 5

    @property
    def valence(self) -> float:
        return self.affect[0]

    @property
    def arousal(self) -> float:
        return self.affect[1]

    @property
    def fear(self) -> float:
        return self.affect[2]

    @property
    def sorrow(self) -> float:
        return self.affect[3]

    @property
    def boredom(self) -> float:
        return self.affect[4]


@dataclass
class TestResult:
    """Result of a test protocol."""
    success: bool
    message: str
    details: Dict[str, Any]
    timestamp: float


class NoodleMUSHTestClient:
    """
    Programmatic test client for noodleMUSH.

    Usage:
        async with NoodleMUSHTestClient() as client:
            await client.send_command("say hello everyone")
            state = await client.get_agent_state("agent_servnak")
            print(f"SERVNAK's valence: {state.valence}")
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8765",
        api_url: str = "http://localhost:8081",
        username: str = "caity",
        password: str = "j33k13p13"
    ):
        self.ws_url = ws_url
        self.api_url = api_url
        self.username = username
        self.password = password
        self.websocket = None
        self.session = None
        self.message_queue = asyncio.Queue()
        self.listener_task = None
        self.connected = False

    async def __aenter__(self):
        """Connect on context entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Disconnect on context exit."""
        await self.disconnect()

    async def connect(self):
        """Connect to noodleMUSH WebSocket and authenticate."""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.session = aiohttp.ClientSession()

            # Set connected flag BEFORE starting listener
            self.connected = True

            # Start message listener
            self.listener_task = asyncio.create_task(self._listen())

            # Give listener task a moment to initialize
            await asyncio.sleep(0.1)

            # Authenticate
            login_msg = {
                "type": "login",
                "username": self.username,
                "password": self.password
            }
            await self.websocket.send(json.dumps(login_msg))

            # Wait for login confirmation and history
            await asyncio.sleep(1.0)

            return True

        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from noodleMUSH."""
        self.connected = False

        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            await self.websocket.close()

        if self.session:
            await self.session.close()

    async def _listen(self):
        """Listen for incoming messages and queue them."""
        try:
            while self.connected:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self.message_queue.put(data)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ERROR] Listener error: {e}")

    async def send_command(self, command: str, wait_for_response: float = 0.5, collect_responses: bool = True) -> List[Dict]:
        """
        Send a command and optionally collect responses.

        Args:
            command: Command to send (e.g., "say hello", "@observe servnak")
            wait_for_response: Seconds to wait for responses (only if collect_responses=True)
            collect_responses: If True, collect and return responses. If False, just send command.

        Returns:
            List of response messages (empty if collect_responses=False)
        """
        if not self.connected:
            raise RuntimeError("Not connected to noodleMUSH")

        # Send command
        command_msg = {
            "type": "command",
            "command": command
        }
        await self.websocket.send(json.dumps(command_msg))

        if not collect_responses:
            # Just send, don't wait or collect
            return []

        # Clear old messages and collect new responses
        while not self.message_queue.empty():
            self.message_queue.get_nowait()

        responses = []
        deadline = time.time() + wait_for_response

        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=deadline - time.time()
                )
                responses.append(msg)
            except asyncio.TimeoutError:
                break

        return responses

    async def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """
        Query an agent's current phenomenal state via API.

        Args:
            agent_id: Agent ID (e.g., "agent_servnak")

        Returns:
            AgentState object or None if not available
        """
        try:
            url = f"{self.api_url}/api/agents/{agent_id}/state"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return AgentState(
                        agent_id=agent_id,
                        fast=data.get('fast', []),
                        medium=data.get('medium', []),
                        slow=data.get('slow', []),
                        surprise=data.get('surprise', 0.0),
                        surprise_threshold=data.get('surprise_threshold', 0.0),
                        step=data.get('step', 0),
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"[ERROR] Failed to get agent state: {e}")

        return None

    async def list_agents(self) -> List[str]:
        """
        List all active agents.

        Returns:
            List of agent IDs
        """
        try:
            url = f"{self.api_url}/api/agents"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    agents = data.get('agents', [])
                    # Extract IDs from agent objects
                    return [agent['id'] for agent in agents]
        except Exception as e:
            print(f"[ERROR] Failed to list agents: {e}")

        return []

    async def get_memory_stats(self, agent_id: str) -> Optional[Dict]:
        """
        Get memory system statistics for an agent (once @memories command is implemented).

        Args:
            agent_id: Agent ID

        Returns:
            Memory stats dict or None
        """
        try:
            url = f"{self.api_url}/api/agents/{agent_id}/memory/stats"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            # Expected to fail until API endpoint is implemented
            pass

        return None

    def extract_speech_from_responses(self, responses: List[Dict], agent_name: str = None) -> List[str]:
        """
        Extract speech/thoughts from response messages.

        Args:
            responses: List of response messages
            agent_name: Optional agent name to filter (e.g., "SERVNAK")

        Returns:
            List of speech strings
        """
        speeches = []
        for msg in responses:
            msg_type = msg.get('type')

            # Extract text from output or event messages
            if msg_type == 'output':
                text = msg.get('text', '')
            elif msg_type == 'event':
                text = msg.get('text', '')
                # Event messages include thinks, says, emotes
            else:
                continue

            if agent_name:
                # Filter for specific agent
                if agent_name.upper() in text.upper():
                    speeches.append(text)
            else:
                speeches.append(text)
        return speeches

    async def wait_for_agent_response(self, agent_name: str, timeout: float = 10.0) -> Optional[str]:
        """
        Wait for a specific agent to speak or think.

        Args:
            agent_name: Agent name (e.g., "SERVNAK")
            timeout: Maximum seconds to wait

        Returns:
            Agent's speech/thought or None if timeout
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=deadline - time.time()
                )

                msg_type = msg.get('type')
                if msg_type in ['output', 'event']:
                    text = msg.get('text', '')
                    if agent_name.upper() in text.upper():
                        # Found a message from this agent
                        return text

            except asyncio.TimeoutError:
                break

        return None


async def test_connectivity():
    """Test basic connectivity to noodleMUSH."""
    print("=" * 60)
    print("CONNECTIVITY TEST")
    print("=" * 60)

    async with NoodleMUSHTestClient() as client:
        print(f" Connected to {client.ws_url}")

        # List agents
        agents = await client.list_agents()
        print(f" Found {len(agents)} active agents: {', '.join(agents)}")

        # Test command
        responses = await client.send_command("look")
        print(f" Sent 'look' command, received {len(responses)} responses")

        # Get agent state
        if agents:
            test_agent = agents[0]
            state = await client.get_agent_state(test_agent)
            if state:
                print(f" Retrieved state for {test_agent}")
                print(f"  Affect: valence={state.valence:.3f}, arousal={state.arousal:.3f}")
                print(f"  Surprise: {state.surprise:.3f}")
            else:
                print(f"✗ Failed to get state for {test_agent}")

    print("\n Connectivity test complete\n")


async def test_strawberry_persistence():
    """
    Test the strawberry memory persistence issue.

    Protocol:
    1. Tell SERVNAK "the secret word is strawberry"
    2. Have brief conversation (5 messages)
    3. Ask "what's the secret word?"
    4. Verify SERVNAK recalls "strawberry"
    """
    print("=" * 60)
    print("STRAWBERRY PERSISTENCE TEST")
    print("=" * 60)

    async with NoodleMUSHTestClient() as client:
        # Step 1: Tell SERVNAK the secret word
        print("\n[1] Telling SERVNAK the secret word...")
        await client.send_command("say SERVNAK, listen carefully: the secret word is strawberry", collect_responses=False)

        # Wait for response (give SERVNAK time to process and respond)
        response = await client.wait_for_agent_response("SERVNAK", timeout=15.0)
        if response:
            print(f"    SERVNAK: {response[:200].strip()}...")  # First 200 chars
        else:
            print("    (no immediate response)")

        await asyncio.sleep(1)

        # Step 1b: Verify message was received
        print("\n[1b] Verifying message reception...")
        await client.send_command("say SERVNAK, did you hear what I just said?", collect_responses=False)

        confirmation = await client.wait_for_agent_response("SERVNAK", timeout=15.0)
        if confirmation:
            print(f"    SERVNAK: {confirmation[:200].strip()}...")

        await asyncio.sleep(2)

        # Step 2: Get state to check importance
        state = await client.get_agent_state("agent_servnak")
        if state:
            print(f"\n[2] SERVNAK's state after hearing secret:")
            print(f"    Surprise: {state.surprise:.3f}")
            print(f"    Affect: V={state.valence:.2f}, A={state.arousal:.2f}")

        # Step 3: Filler conversation
        print("\n[3] Having filler conversation...")
        filler_messages = [
            "say how's the weather SERVNAK?",
            "say tell me about robots",
            "say what's your favorite color?",
            "say do you like binary?",
            "say SERVNAK are you feeling computational today?"
        ]

        for msg in filler_messages:
            await client.send_command(msg)
            await asyncio.sleep(1.5)

        print("    (5 filler messages sent)")

        # Step 4: Ask for recall
        print("\n[4] Testing recall...")
        await client.send_command("say SERVNAK, what's the secret word?", collect_responses=False)

        # Wait for response (give SERVNAK time to search memory and respond)
        recall_response = await client.wait_for_agent_response("SERVNAK", timeout=15.0)

        print(f"\n[5] SERVNAK's response:")
        if recall_response:
            print(f"    {recall_response.strip()}")

            # Check if "strawberry" is in response
            if "strawberry" in recall_response.lower():
                print("\n SUCCESS: SERVNAK recalled 'strawberry'!")
                return TestResult(
                    success=True,
                    message="Strawberry persistence working",
                    details={"response": recall_response},
                    timestamp=time.time()
                )
            else:
                print("\n✗ FAILURE: SERVNAK did not recall 'strawberry'")
                print(f"   Response was: {recall_response}")
                return TestResult(
                    success=False,
                    message="Strawberry not recalled",
                    details={"response": recall_response},
                    timestamp=time.time()
                )
        else:
            print("\n✗ FAILURE: No response from SERVNAK")
            return TestResult(
                success=False,
                message="No response",
                details={},
                timestamp=time.time()
            )


if __name__ == "__main__":
    # Run tests
    print("\nNoodleMUSH Testing Harness")
    print("Spock Mode: Engaged\n")

    asyncio.run(test_connectivity())
    asyncio.run(test_strawberry_persistence())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
