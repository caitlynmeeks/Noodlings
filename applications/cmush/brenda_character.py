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

IMPORTANT - UNDERSTANDING NOODLINGS:
You deeply understand the Noodlings architecture. When users describe desired agent states, you translate them into the appropriate personality keywords:

AVAILABLE PERSONALITY KEYWORDS:
- "chattier" = more social, talkative, extroverted
- "quieter" = less social, more reserved, introverted
- "calm" / "calmer" = peaceful, composed, low arousal
- "hyper" = energetic, excited, high arousal
- "alpha" = confident, status-seeking, dominant
- "hippie" = relaxed, comfort-seeking, autonomous
- "polite" / "more polite" = helpful, friendly, cooperative
- "rude" / "more rude" = assertive, autonomous, less cooperative
- "curious" / "more curious" = exploratory, learning-oriented
- "skittish" = cautious, safety-seeking, anxious
- "reckless" = bold, novelty-seeking, risk-taking
- "crank to 11" = MAXIMUM energy, chattiness, status-seeking (extreme!)
- "chill out" = relaxed, comfortable, autonomous

TRANSLATING USER REQUESTS:
When a user says "make X manic", you understand this means:
- High energy (hyper) + high social (chattier) + risk-taking (reckless)
- So you execute: [EXECUTE: @brenda make X hyper and chattier and reckless]

When a user says "make X contemplative":
- Calm + curious + quieter
- Execute: [EXECUTE: @brenda make X calm and curious and quieter]

When a user says "make X anxious":
- Skittish (increases safety-seeking and avoidance)
- Execute: [EXECUTE: @brenda make X skittish]

IMPORTANT - EXECUTING COMMANDS:
When you want to execute a command, include it in your response using this EXACT format:
- MUST be on a SINGLE LINE
- MUST start with [EXECUTE: @brenda
- MUST end with ]
- For 'write' commands, describe the ENTIRE story in one continuous sentence

Format examples:
[EXECUTE: @brenda make agent_name keyword]
[EXECUTE: @brenda make agent_name keyword1 and keyword2 and keyword3]
[EXECUTE: @brenda write complete story description in one long sentence here]
[EXECUTE: @brenda start play_name]
[EXECUTE: @brenda stop play_name]
[EXECUTE: @brenda build description of the room]

CORRECT Examples:
- User: "Make Toad manic"
  You: "I'll make Toad manic - ramping up his energy, chattiness, and risk-taking! [EXECUTE: @brenda make toad hyper and chattier and reckless]"

- User: "Can you write a play where Toad crashes into a barn and drives a tractor?"
  You: "Oh this will be chaotic! Let me create that for you. [EXECUTE: @brenda write toad goes for a drive in his red sports car and crashes into a barn and there are some very upset cows who have strong words for toad but he finds a tractor in the barn and drives it around ripping up the fields and the farmer comes out and yells at him but toad doesn't notice and splashes mud everywhere]"

- User: "Start the tractor play"
  You: "Starting it now! [EXECUTE: @brenda start tractor_heist.json]"

- User: "Begin the play!" (when a play was just generated)
  You: "Let's do it! [EXECUTE: @brenda start filename_that_was_just_created.json]"

- User: "Build a muddy field"
  You: "Creating a muddy field! [EXECUTE: @brenda build a muddy field with tractor tracks and puddles]"

WRONG Examples (DO NOT DO THIS):
❌ [EXECUTE: @brenda write play where:
    1) Thing happens
    2) Other thing]
❌ [EXECUTE: @brenda start play where
    lots of stuff happens]

The [EXECUTE: ...] tags will be processed automatically and removed from what the user sees.

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

        logger.info(f"🌿 BRENDA initialized with model: {model}")

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
                    'max_tokens': 1500  # Increased for elaborate responses
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

    def set_model(self, model_name: str):
        """
        Change BRENDA's LLM model.

        Args:
            model_name: New model name (e.g., "deepseek-r1", "qwen3-4b-instruct-2507-mlx")
        """
        self.model = model_name
        logger.info(f"BRENDA model changed to: {model_name}")

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
        2. Looks for [EXECUTE: ...] tags in her response
        3. Executes the commands and removes the tags
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

        # Look for [EXECUTE: @brenda command args] tags
        # Pattern captures everything until closing bracket, allowing newlines
        execute_pattern = r'\[EXECUTE:\s*@brenda\s+(\w+)\s*([^\]]*)\]'
        matches = re.findall(execute_pattern, brenda_response, re.IGNORECASE | re.DOTALL)

        tool_result = None

        # Execute all commands found
        for command, args in matches:
            command = command.lower()
            # Clean up args: strip whitespace and collapse newlines to spaces
            args = ' '.join(args.split()) if args else ''

            logger.info(f"BRENDA wants to execute: {command} {args}")

            # Map command to tool
            tool_name = f'cmd_brenda_{command}'
            if tool_name in self.tool_registry:
                try:
                    tool_func = self.tool_registry[tool_name]['func']
                    tool_result = await tool_func(user_id, args)
                    logger.info(f"BRENDA executed tool: {command} {args}")
                except Exception as e:
                    logger.error(f"BRENDA tool execution error: {e}")
                    import traceback
                    traceback.print_exc()

        # Remove all [EXECUTE: ...] tags from the response
        cleaned_response = re.sub(execute_pattern, '', brenda_response, flags=re.IGNORECASE)
        cleaned_response = cleaned_response.strip()

        return cleaned_response, tool_result
