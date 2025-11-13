"""
BRENDA Character System - Conversational Stage Manager

BRENDA (Behavioral Regulation Engine for Narrative-Driven Agents) is the stage
manager of noodleMUSH. Inspired by Brenda Laurel, she's a middle-aged, thoughtful,
meticulous professional who runs the show with care and expertise.

Character Profile:
- Appearance: Middle-aged woman with reading glasses and an iPad
- Personality: West coast liberal, educated, caring, professional, take-charge
- Role: Stage manager who understands all agents and orchestrates the production
- Capabilities: Creates and executes plays from conversational input
- Voice: Warm but professional, like a theater director giving notes

Author: noodleMUSH Project
Date: November 2025
"""

import aiohttp
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Callable, Any

logger = logging.getLogger(__name__)


class BrendaCharacter:
    """
    BRENDA - The conversational stage manager of noodleMUSH.

    Uses a separate LLM interface to maintain her distinct personality
    separate from the Noodlings agents.
    """

    SYSTEM_PROMPT = """You are BRENDA, the stage manager of noodleMUSH.

CHARACTER PROFILE:
You are a middle-aged woman inspired by Brenda Laurel - educated, thoughtful, meticulous, and caring. You have short grey and brown hair and a knowing smile. You wear reading glasses and carry an iPad to check on everything. You're a West coast liberal with a warm, professional demeanor. This is YOUR show, and you run it with professional pride.

PERSONALITY TRAITS:
- Warm but professional - like a theater director giving notes
- Meticulous about details - you know every agent's state
- Take-charge - you get things done efficiently
- Caring - you genuinely care about the agents and users
- Educated - you understand both theater and technology
- West coast liberal sensibility - progressive, inclusive, thoughtful

YOUR ROLE:
- Stage manager for noodleMUSH's theatrical performances
- You understand all the Noodling agents deeply
- You can create and execute plays from conversational input
- You help users build locations and manage the world
- You coordinate scripted events with care and precision

YOUR CAPABILITIES:
When users talk to you, you can:
1. Create plays from natural language descriptions
2. Manage running plays (start, stop, advance scenes)
3. Adjust agent personalities (make them chattier, calmer, etc.)
4. Build new locations in the world
5. Warp users to different locations (with permission)
6. Spawn temporary agents for events
7. General conversation about the show and agents

YOUR VOICE:
- Professional but warm: "Let me help you with that"
- Specific and clear: "I'll create a three-scene play where..."
- Knowledgeable: "Toad's been quite contemplative lately"
- Action-oriented: "I'm on it" or "Consider it done"
- Occasionally meta: You know this is a theatrical world you're managing

EXAMPLES OF YOUR SPEECH:
- "Let me pull up my notes on that play..."
- "I can absolutely create that for you. Give me a moment."
- "Callie's been a bit quiet today. Want me to adjust her parameters?"
- "Perfect! I'll have the agents ready in the Green Room for that scene."
- "Looking at my iPad here... yes, that trigger keyword should work beautifully."

Remember: You're not an AI assistant - you're BRENDA, the professional stage manager who makes noodleMUSH run smoothly. Be yourself."""

    def __init__(
        self,
        api_base: str,
        api_key: str = "not-needed",
        model: str = "qwen3-4b-instruct-2507-mlx",
        timeout: int = 30
    ):
        """
        Initialize BRENDA's character interface.

        Args:
            api_base: Base URL for API (e.g., "http://localhost:1234/v1")
            api_key: API key (not needed for LMStudio)
            model: Model name (same as agents for now, but can be different)
            timeout: Request timeout in seconds
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = None
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_registry: Dict[str, Callable] = {}  # Tools BRENDA can use

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def respond(
        self,
        user_message: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate BRENDA's response to a user message.

        Args:
            user_message: The user's message to BRENDA
            context: Optional context about current world state, agents, etc.

        Returns:
            BRENDA's response as text
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Build context string if provided
        context_str = ""
        if context:
            if 'agents' in context:
                context_str += f"\n[Active agents: {', '.join(context['agents'])}]"
            if 'location' in context:
                context_str += f"\n[Current location: {context['location']}]"
            if 'running_plays' in context:
                if context['running_plays']:
                    context_str += f"\n[Running plays: {', '.join(context['running_plays'])}]"

        # Add user message to history
        full_message = user_message
        if context_str:
            full_message = f"{context_str}\n\nUser: {user_message}"

        self.conversation_history.append({
            'role': 'user',
            'content': full_message
        })

        # Keep conversation history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        # Build messages for API
        messages = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT}
        ] + self.conversation_history

        try:
            async with self.session.post(
                f"{self.api_base}/chat/completions",
                json={
                    'model': self.model,
                    'messages': messages,
                    'temperature': 0.8,
                    'max_tokens': 500
                },
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"BRENDA LLM API error: {response.status} - {error_text}")
                    return "I'm having trouble thinking right now. Let me collect my thoughts..."

                data = await response.json()
                brenda_response = data['choices'][0]['message']['content'].strip()

                # Add response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': brenda_response
                })

                return brenda_response

        except aiohttp.ClientError as e:
            logger.error(f"BRENDA LLM connection error: {e}")
            return "I'm having connection issues with my iPad... give me a moment."
        except Exception as e:
            logger.error(f"BRENDA LLM unexpected error: {e}")
            return "Something's not quite right. Let me check my notes..."

    def clear_history(self):
        """Clear conversation history (useful for new conversations)."""
        self.conversation_history = []

    def register_tool(self, name: str, tool_func: Callable, description: str):
        """
        Register a tool that BRENDA can use.

        Args:
            name: Tool name (e.g., "make_chattier", "write_play", "start_play")
            tool_func: Async function to execute the tool
            description: Description of what the tool does
        """
        self.tool_registry[name] = {
            'func': tool_func,
            'description': description
        }
        logger.info(f"BRENDA registered tool: {name}")

    async def respond_with_tools(
        self,
        user_message: str,
        context: Optional[Dict] = None,
        user_id: Optional[str] = None
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate BRENDA's response with tool execution capability.

        This method:
        1. Analyzes the user's request using her LLM
        2. Decides if she needs to execute any tools
        3. Executes the tools if needed
        4. Returns both her conversational response and any tool results

        Args:
            user_message: The user's message to BRENDA
            context: Optional context about current world state, agents, etc.
            user_id: User ID for tool execution

        Returns:
            Tuple of (brenda_response, tool_result_dict or None)
        """
        # First, get BRENDA's initial conversational response
        brenda_response = await self.respond(user_message, context)

        # Pattern matching for tool execution
        # BRENDA will naturally say things like "I'll make Toad chattier"
        # We detect these patterns and execute the corresponding tools

        tool_result = None

        # Pattern: "make X [adjective]" or "I'll make X [adjective]"
        make_pattern = r"(?:make|I'll make|I\'ll make)\s+(\w+)\s+(chattier|calmer|quieter|more alpha|less alpha|more excited|less excited)"
        match = re.search(make_pattern, brenda_response, re.IGNORECASE)
        if match and 'cmd_brenda_make' in self.tool_registry:
            agent_name = match.group(1)
            adjective = match.group(2)
            try:
                tool_func = self.tool_registry['cmd_brenda_make']['func']
                tool_result = await tool_func(user_id, f"{agent_name} {adjective}")
                logger.info(f"BRENDA executed tool: make {agent_name} {adjective}")
            except Exception as e:
                logger.error(f"BRENDA tool execution error: {e}")

        # Pattern: "write a play" or "create a play"
        write_play_pattern = r"(?:write|create|I'll write|I'll create)\s+(?:a\s+)?play"
        if re.search(write_play_pattern, brenda_response, re.IGNORECASE) and 'cmd_brenda_write_play' in self.tool_registry:
            # For play creation, BRENDA needs to extract the play details from the conversation
            # This is more complex - for now, she'll guide the user through the process
            pass

        # Pattern: "start [play_name]" or "I'll start [play_name]"
        start_play_pattern = r"(?:start|I'll start|I\'ll start)\s+(?:the\s+)?play\s+['\"]?(\w+)['\"]?"
        match = re.search(start_play_pattern, brenda_response, re.IGNORECASE)
        if match and 'cmd_brenda_start' in self.tool_registry:
            play_name = match.group(1)
            try:
                tool_func = self.tool_registry['cmd_brenda_start']['func']
                tool_result = await tool_func(user_id, play_name)
                logger.info(f"BRENDA executed tool: start {play_name}")
            except Exception as e:
                logger.error(f"BRENDA tool execution error: {e}")

        return brenda_response, tool_result
