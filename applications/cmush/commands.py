"""
Command Parser for cMUSH

Handles all user commands:
- Movement: north, south, east, west, up, down
- Communication: say, emote, tell
- Observation: look, inventory, who
- Manipulation: take, drop
- Building: @create, @describe, @dig
- Agent: @rez, @observe, @relationship, @memory

Author: cMUSH Project
Date: October 2025
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
import time
import re
import json

from recipe_loader import RecipeLoader
from play_manager import PlayManager
from brenda_character import BrendaCharacter

logger = logging.getLogger(__name__)

# ===== BRENDA: Natural Language Parameter Tweaking =====
# Lazy stoner-friendly mappings: phrase → goal & config adjustments
# Goals from AppetiteLayer (Phase 6): explore_environment, seek_social_connection,
# demonstrate_competence, pursue_novelty, ensure_safety, gain_status, seek_comfort,
# maintain_autonomy, help_friend, avoid_consequences, restore_reputation, learn_skill,
# impress_others, solve_problem, express_emotion, achieve_goal
# Config params: speech_cooldown, addressed_speech_chance, unaddressed_speech_chance
BRENDA_CHAT_MAP = {
    # Chattiness / Social engagement
    r"\bmor?e?\s+chatt?y\b": {"seek_social_connection": +0.3, "speech_cooldown": -1.0, "addressed_speech_chance": +0.1},
    r"\bless\s+chatt?y\b": {"seek_social_connection": -0.3, "speech_cooldown": +1.5, "addressed_speech_chance": -0.2},
    r"\bnot\s+chatt?y\b": {"seek_social_connection": -0.5, "speech_cooldown": +2.0, "addressed_speech_chance": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+chatt?y\b": {"seek_social_connection": +0.6, "speech_cooldown": -2.0, "addressed_speech_chance": +0.3},
    r"\bmor?e?\s+quiet\b": {"seek_social_connection": -0.3, "speech_cooldown": +1.5, "addressed_speech_chance": -0.2},
    r"\bless\s+quiet\b": {"seek_social_connection": +0.3, "speech_cooldown": -1.0, "addressed_speech_chance": +0.1},
    r"\bnot\s+quiet\b": {"seek_social_connection": +0.5, "speech_cooldown": -2.0, "addressed_speech_chance": +0.2},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+quiet\b": {"seek_social_connection": -0.6, "speech_cooldown": +3.0, "addressed_speech_chance": -0.4},

    # Emotional intensity
    r"\bmor?e?\s+intense\b": {"seek_social_connection": +0.4, "gain_status": +0.3, "express_emotion": +0.3, "speech_cooldown": -1.5, "addressed_speech_chance": +0.15},
    r"\bless\s+intense\b": {"seek_comfort": +0.3, "maintain_autonomy": +0.2, "express_emotion": -0.3},
    r"\bnot\s+intense\b": {"seek_comfort": +0.4, "express_emotion": -0.5, "gain_status": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+intense\b": {"seek_social_connection": +0.6, "gain_status": +0.5, "express_emotion": +0.5, "speech_cooldown": -2.5, "addressed_speech_chance": +0.3},
    r"\bmor?e?\s+calm\b": {"seek_comfort": +0.2, "ensure_safety": +0.1, "express_emotion": -0.2},
    r"\bless\s+calm\b": {"express_emotion": +0.3, "pursue_novelty": +0.2},
    r"\bnot\s+calm\b": {"express_emotion": +0.4, "ensure_safety": -0.3, "pursue_novelty": +0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+calm\b": {"seek_comfort": +0.5, "ensure_safety": +0.3, "express_emotion": -0.5},

    # Aggression / Dominance
    r"\bmor?e?\s+(angry|furious)\b": {"gain_status": +0.3, "help_friend": -0.3, "express_emotion": +0.3, "avoid_consequences": -0.2},
    r"\bless\s+(angry|furious)\b": {"help_friend": +0.3, "seek_comfort": +0.2, "express_emotion": -0.2},
    r"\bnot\s+(angry|furious)\b": {"help_friend": +0.5, "seek_comfort": +0.3, "express_emotion": -0.4, "gain_status": -0.4},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+(angry|furious)\b": {"gain_status": +0.6, "help_friend": -0.6, "express_emotion": +0.6, "avoid_consequences": -0.5},
    r"\bmor?e?\s+dominant\b": {"gain_status": +0.4, "demonstrate_competence": +0.2, "help_friend": -0.2},
    r"\bless\s+dominant\b": {"help_friend": +0.3, "gain_status": -0.3},
    r"\bnot\s+dominant\b": {"help_friend": +0.5, "gain_status": -0.5, "demonstrate_competence": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+dominant\b": {"gain_status": +0.7, "demonstrate_competence": +0.4, "help_friend": -0.5},
    r"\bmor?e?\s+pushy\b": {"gain_status": +0.3, "maintain_autonomy": +0.2, "help_friend": -0.3},
    r"\bless\s+pushy\b": {"help_friend": +0.3, "seek_comfort": +0.2},
    r"\bnot\s+pushy\b": {"help_friend": +0.5, "gain_status": -0.4, "maintain_autonomy": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+pushy\b": {"gain_status": +0.6, "maintain_autonomy": +0.4, "help_friend": -0.6},
    r"\bmor?e?\s+rude\b": {"maintain_autonomy": +0.3, "help_friend": -0.3, "gain_status": +0.2},
    r"\bless\s+rude\b": {"help_friend": +0.3, "maintain_autonomy": -0.2},
    r"\bnot\s+rude\b": {"help_friend": +0.5, "maintain_autonomy": -0.4, "gain_status": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+rude\b": {"maintain_autonomy": +0.6, "help_friend": -0.6, "gain_status": +0.4},

    # Gentleness / Kindness
    r"\bmor?e?\s+gentle\b": {"help_friend": +0.4, "seek_comfort": +0.2, "gain_status": -0.2},
    r"\bless\s+gentle\b": {"gain_status": +0.2, "help_friend": -0.2},
    r"\bnot\s+gentle\b": {"gain_status": +0.4, "help_friend": -0.4, "express_emotion": +0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+gentle\b": {"help_friend": +0.7, "seek_comfort": +0.4, "gain_status": -0.5},
    r"\bmor?e?\s+polite\b": {"help_friend": +0.3, "avoid_consequences": +0.1},
    r"\bless\s+polite\b": {"maintain_autonomy": +0.2, "help_friend": -0.2},
    r"\bnot\s+polite\b": {"maintain_autonomy": +0.4, "help_friend": -0.4, "avoid_consequences": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+polite\b": {"help_friend": +0.6, "avoid_consequences": +0.3, "maintain_autonomy": -0.4},

    # Anxiety / Fear / Caution
    r"\bmor?e?\s+(anxious|skittish)\b": {"ensure_safety": +0.3, "avoid_consequences": +0.2, "seek_comfort": +0.2},
    r"\bless\s+(anxious|skittish)\b": {"pursue_novelty": +0.3, "ensure_safety": -0.2},
    r"\bnot\s+(anxious|skittish)\b": {"pursue_novelty": +0.5, "ensure_safety": -0.5, "avoid_consequences": -0.4},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+(anxious|skittish)\b": {"ensure_safety": +0.6, "avoid_consequences": +0.5, "seek_comfort": +0.5, "speech_cooldown": +2.0},
    r"\bmor?e?\s+cautious\b": {"ensure_safety": +0.3, "avoid_consequences": +0.2, "pursue_novelty": -0.2},
    r"\bless\s+cautious\b": {"pursue_novelty": +0.3, "ensure_safety": -0.2, "avoid_consequences": -0.2},
    r"\bnot\s+cautious\b": {"pursue_novelty": +0.5, "ensure_safety": -0.4, "avoid_consequences": -0.4},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+cautious\b": {"ensure_safety": +0.6, "avoid_consequences": +0.5, "pursue_novelty": -0.5},
    r"\bmor?e?\s+reckless\b": {"ensure_safety": -0.3, "pursue_novelty": +0.3, "avoid_consequences": -0.2},
    r"\bless\s+reckless\b": {"ensure_safety": +0.3, "avoid_consequences": +0.2},
    r"\bnot\s+reckless\b": {"ensure_safety": +0.5, "avoid_consequences": +0.4, "pursue_novelty": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+reckless\b": {"ensure_safety": -0.6, "pursue_novelty": +0.6, "avoid_consequences": -0.6},

    # Sadness / Depression
    r"\bmor?e?\s+sad\b": {"seek_social_connection": -0.3, "seek_comfort": +0.3, "express_emotion": +0.2, "speech_cooldown": +1.0},
    r"\bless\s+sad\b": {"seek_social_connection": +0.3, "express_emotion": -0.2},
    r"\bnot\s+sad\b": {"seek_social_connection": +0.5, "express_emotion": -0.4, "seek_comfort": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+sad\b": {"seek_social_connection": -0.6, "seek_comfort": +0.6, "express_emotion": +0.5, "speech_cooldown": +2.5},

    # Curiosity / Exploration
    r"\bmor?e?\s+curious\b": {"explore_environment": +0.3, "learn_skill": +0.2, "pursue_novelty": +0.2},
    r"\bless\s+curious\b": {"explore_environment": -0.3, "pursue_novelty": -0.2},
    r"\bnot\s+curious\b": {"explore_environment": -0.5, "pursue_novelty": -0.4, "learn_skill": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+curious\b": {"explore_environment": +0.6, "learn_skill": +0.5, "pursue_novelty": +0.5},

    # Introspection / Reflection
    r"\bmor?e?\s+introspective\b": {"maintain_autonomy": +0.3, "seek_social_connection": -0.2, "speech_cooldown": +1.0},
    r"\bless\s+introspective\b": {"seek_social_connection": +0.3, "maintain_autonomy": -0.2},
    r"\bnot\s+introspective\b": {"seek_social_connection": +0.5, "maintain_autonomy": -0.4, "speech_cooldown": -1.0},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+introspective\b": {"maintain_autonomy": +0.6, "seek_social_connection": -0.5, "speech_cooldown": +2.0},

    # Special patterns (keep these for backward compatibility)
    r"\bmor?e?\s+hippie\b": {"seek_comfort": +0.3, "maintain_autonomy": +0.2, "pursue_novelty": +0.2},
    r"\bless\s+hippie\b": {"demonstrate_competence": +0.2, "gain_status": +0.2},
    r"\bnot\s+hippie\b": {"demonstrate_competence": +0.4, "gain_status": +0.3, "seek_comfort": -0.4},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+hippie\b": {"seek_comfort": +0.6, "maintain_autonomy": +0.5, "pursue_novelty": +0.5},
    r"\bmor?e?\s+alpha\b": {"gain_status": +0.3, "demonstrate_competence": +0.2, "help_friend": -0.2},
    r"\bless\s+alpha\b": {"help_friend": +0.3, "gain_status": -0.2},
    r"\bnot\s+alpha\b": {"help_friend": +0.5, "gain_status": -0.5, "demonstrate_competence": -0.3},
    r"\b(completely|totally|fully|max(ed)?\s+out)\s+alpha\b": {"gain_status": +0.6, "demonstrate_competence": +0.5, "help_friend": -0.5},
    r"\bhyper\b": {"pursue_novelty": +0.3, "express_emotion": +0.2, "seek_social_connection": +0.2},
    r"\bcrank.*to\s+11\b": {"seek_social_connection": +0.4, "gain_status": +0.3, "express_emotion": +0.3, "speech_cooldown": -1.5, "addressed_speech_chance": +0.15},
    r"\bchill.*out\b": {"seek_comfort": +0.3, "maintain_autonomy": +0.2, "express_emotion": -0.2},
}


class CommandParser:
    """
    Parse and execute cMUSH commands.

    Commands are parsed from user input and executed against
    the world state, returning formatted output for the user.
    """

    def __init__(self, world, agent_manager, server=None, config=None, config_path=None):
        """
        Initialize command parser.

        Args:
            world: World state manager
            agent_manager: Agent manager instance
            server: Server instance (for shutdown command)
            config: Server config dict (for saving changes)
            config_path: Path to config.yaml (for persistence)
        """
        self.world = world
        self.agent_manager = agent_manager
        self.server = server
        self.config = config
        self.config_path = config_path
        self.recipe_loader = RecipeLoader("recipes")

        # BRENDA state tracking (lazy parameter tweaking)
        self.brenda_history = {}  # agent_id -> list of (timestamp, changes_dict)
        self.brenda_rate_limit = {}  # agent_id -> list of timestamps
        self.brenda_max_history = 10  # per agent
        self.brenda_rate_window = 300  # 5 minutes
        self.brenda_rate_max = 999  # max commands per window (dev: unlimited)

        # BRENDA character (conversational stage manager) - create FIRST
        llm_config = config.get('llm', {}) if config else {}
        brenda_config = config.get('brenda', {}) if config else {}

        # Use Brenda-specific model if configured, otherwise fall back to general LLM model
        brenda_model = brenda_config.get('model', llm_config.get('model', 'qwen3-4b-instruct-2507-mlx'))

        self.brenda_character = BrendaCharacter(
            api_base=llm_config.get('api_base', 'http://localhost:1234/v1'),
            api_key=llm_config.get('api_key', 'not-needed'),
            model=brenda_model,
            timeout=llm_config.get('timeout', 60)
        )

        # BRENDA play manager (drama generation) - pass brenda_character for model access
        self.play_manager = PlayManager(
            plays_dir="plays",
            server=server,
            brenda_character=self.brenda_character
        )

        # Register tools that BRENDA can use
        # We register these after commands are set up
        self._register_brenda_tools()

        # Command registry
        self.commands = {
            # Movement
            'north': self.cmd_move,
            'south': self.cmd_move,
            'east': self.cmd_move,
            'west': self.cmd_move,
            'up': self.cmd_move,
            'down': self.cmd_move,
            'n': self.cmd_move,
            's': self.cmd_move,
            'e': self.cmd_move,
            'w': self.cmd_move,
            'u': self.cmd_move,
            'd': self.cmd_move,

            # Communication
            'say': self.cmd_say,
            'emote': self.cmd_emote,
            'tell': self.cmd_tell,
            # Note: Shortcuts " and : handled in parse_and_execute before command lookup

            # Observation
            'look': self.cmd_look,
            'l': self.cmd_look,
            'inventory': self.cmd_inventory,
            'inv': self.cmd_inventory,
            'i': self.cmd_inventory,
            'who': self.cmd_who,

            # Manipulation
            'take': self.cmd_take,
            'get': self.cmd_take,
            'drop': self.cmd_drop,

            # Building
            '@create': self.cmd_create,
            '@describe': self.cmd_describe,
            '@dig': self.cmd_dig,
            '@destroy': self.cmd_destroy,

            # Agent commands
            '@rez': self.cmd_rez_agent,
            '@observe': self.cmd_observe_agent,
            '@me': self.cmd_observe_self,
            '@relationship': self.cmd_relationship,
            '@memory': self.cmd_memory,
            '@agents': self.cmd_list_agents,
            '@savestates': self.cmd_save_states,
            '@whoami': self.cmd_whoami,
            '@setname': self.cmd_setname,
            '@setdesc': self.cmd_setdesc,
            '@profile': self.cmd_profile,
            '@remove': self.cmd_remove,
            '@reset': self.cmd_reset,
            '@tpinvite': self.cmd_tpinvite,
            '@scope': self.cmd_launch_noodlescope,

            # Appetite orchestration (Phase 6)
            '@stoke': self.cmd_stoke_appetite,
            '@sate': self.cmd_sate_appetite,
            '@appetites': self.cmd_show_appetites,

            # Goal orchestration (Phase 6)
            '@override': self.cmd_override_goal,
            '@bias': self.cmd_set_goal_bias,
            '@reset_goals': self.cmd_reset_goals,
            '@clear_bias': self.cmd_clear_bias,
            '@goals': self.cmd_show_goals,

            # Self-protection
            '@withdrawn': self.cmd_check_withdrawn,
            '@reengage': self.cmd_reengage,

            # Consciousness metrics
            # '@phi': self.cmd_phi,  # Hidden until scientifically validated
            '@enlighten': self.cmd_enlighten,
            '@status': self.cmd_comprehensive_status,

            # LLM control
            '@model': self.cmd_set_model,
            '@models': self.cmd_list_models,
            '@maxservers': self.cmd_set_maxservers,

            # Agent tools (filesystem, messaging, cognition)
            '@think': self.cmd_think,
            '@remember': self.cmd_remember,
            '@message': self.cmd_message,
            '@inbox': self.cmd_inbox,
            '@write': self.cmd_write_file,
            '@read': self.cmd_read_file,
            '@ls': self.cmd_list_files,
            '@exec': self.cmd_execute_command,

            # Cognition control
            '@cognition': self.cmd_cognition_stats,
            '@set_frequency': self.cmd_set_frequency,
            '@ruminate': self.cmd_force_rumination,

            # BRENDA: Natural language parameter tweaking
            '@brenda': self.cmd_brenda,

            # NoodleScope 2.0 - Consciousness visualization
            '@scope': self.cmd_scope,

            # Utility
            'help': self.cmd_help,
            'quit': self.cmd_quit,
            'logout': self.cmd_quit,
            '@yeet': self.cmd_yeet,
            '@shutdown': self.cmd_shutdown
        }

    def _register_brenda_tools(self):
        """
        Register tools that BRENDA can use for command execution.

        This allows BRENDA to execute commands based on her conversational understanding.
        She'll analyze the user's request and execute the appropriate tools.
        """
        # Register personality adjustment tool
        async def tool_make(user_id: str, args: str):
            # Parse args: "agent_name phrase" -> split into agent_name and phrase
            parts = args.split(None, 1)  # Split on first whitespace
            if len(parts) < 2:
                return {'success': False, 'output': '⚠️  Need both agent name and description.', 'events': []}
            agent_name, phrase = parts
            return await self._brenda_make(user_id, agent_name, phrase)

        self.brenda_character.register_tool(
            'cmd_brenda_make',
            tool_make,
            'Adjust agent personality (make them chattier, calmer, etc.)'
        )

        # Register build room tool
        async def tool_build(user_id: str, args: str):
            return await self._brenda_build(user_id, args)

        self.brenda_character.register_tool(
            'cmd_brenda_build',
            tool_build,
            'Build a new room from natural language description'
        )

        # Register play write tool
        async def tool_write(user_id: str, args: str):
            # Args should be the story description
            return await self._brenda_write_play(user_id, args)

        self.brenda_character.register_tool(
            'cmd_brenda_write',
            tool_write,
            'Generate a play from natural language description'
        )

        # Register play start tool
        async def tool_start(user_id: str, args: str):
            # Args should be the play filename
            return await self._brenda_play_start(user_id, args)

        self.brenda_character.register_tool(
            'cmd_brenda_start',
            tool_start,
            'Start a play'
        )

        # Register play stop tool
        async def tool_stop(user_id: str, args: str):
            # Args should be the play filename
            return await self._brenda_play_stop(user_id, args)

        self.brenda_character.register_tool(
            'cmd_brenda_stop',
            tool_stop,
            'Stop a running play'
        )

        # Register model change tool
        async def tool_usemodel(user_id: str, args: str):
            return await self._brenda_usemodel(user_id, args)

        self.brenda_character.register_tool(
            'cmd_brenda_usemodel',
            tool_usemodel,
            'Change BRENDA\'s LLM model'
        )

    async def parse_and_execute(
        self,
        user_id: str,
        command_text: str
    ) -> Dict:
        """
        Parse and execute a command.

        Args:
            user_id: User executing command
            command_text: Command string

        Returns:
            Response dict with:
                - success: bool
                - output: str (formatted text for user)
                - events: list (events to broadcast)
        """
        if not command_text.strip():
            return {'success': False, 'output': '', 'events': []}

        command_text = command_text.strip()

        # Handle shortcuts BEFORE parsing
        if command_text.startswith('"'):
            # Say shortcut: "Hello world -> say Hello world
            cmd = 'say'
            args = command_text[1:].strip()
        elif command_text.startswith(':'):
            # Emote shortcut: :waves -> emote waves
            cmd = 'emote'
            args = command_text[1:].strip()
        else:
            # Regular command parsing
            parts = command_text.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ''

        # Handle direction shortcuts
        direction_map = {
            'n': 'north', 's': 'south', 'e': 'east',
            'w': 'west', 'u': 'up', 'd': 'down'
        }
        if cmd in direction_map:
            cmd = direction_map[cmd]

        # Execute command
        if cmd in self.commands:
            try:
                # For movement commands, pass the direction as args if args is empty
                if cmd in ['north', 'south', 'east', 'west', 'up', 'down'] and not args:
                    result = await self.commands[cmd](user_id, cmd)
                else:
                    result = await self.commands[cmd](user_id, args)
                logger.info(f"Command executed: {user_id} -> {cmd} {args}")
                return result
            except Exception as e:
                logger.error(f"Error executing command: {e}", exc_info=True)
                return {
                    'success': False,
                    'output': f"Error: {str(e)}",
                    'events': []
                }
        else:
            return {
                'success': False,
                'output': f"Unknown command: {cmd}. Type 'help' for commands.",
                'events': []
            }

    def _save_config(self):
        """
        Save current config back to config.yaml for persistence.

        This allows runtime changes (like @model) to persist across server restarts.
        """
        if not self.config or not self.config_path:
            logger.warning("Cannot save config: config or config_path not set")
            return

        import yaml
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    # ===== Movement Commands =====

    async def cmd_move(self, user_id: str, args: str) -> Dict:
        """Move in a direction."""
        direction = args.strip() if args else ''

        # If called with direction as command name
        if not direction:
            # Extract direction from call context (hacky but works)
            direction = 'north'  # Will be overridden by actual implementation

        user = self.world.get_user(user_id)
        if not user:
            return {'success': False, 'output': 'User not found.', 'events': []}

        room = self.world.get_room(user['current_room'])
        if not room:
            return {'success': False, 'output': 'Current room not found.', 'events': []}

        # Check if exit exists
        if direction not in room['exits']:
            return {
                'success': False,
                'output': f"You can't go {direction} from here.",
                'events': []
            }

        new_room_id = room['exits'][direction]
        new_room = self.world.get_room(new_room_id)
        if not new_room:
            return {'success': False, 'output': 'Destination not found.', 'events': []}

        # Move user
        self.world.move_user(user_id, new_room_id)

        # Get user description for enter event
        username = user.get('username', user_id)
        description = user.get('description', '')

        enter_text = f"{username} arrives"
        if description:
            enter_text += f". {description}"
        else:
            enter_text += "."

        # Generate events
        events = [
            {
                'type': 'exit',
                'user': user_id,
                'room': room['uid'],
                'direction': direction,
                'text': f"{username} leaves {direction}."
            },
            {
                'type': 'enter',
                'user': user_id,
                'room': new_room_id,
                'text': enter_text
            }
        ]

        # Check for play room_enter triggers
        await self.play_manager.check_room_enter_trigger(user_id, new_room_id)

        # Show new room
        look_result = await self.cmd_look(user_id, '')

        return {
            'success': True,
            'output': look_result['output'],
            'events': events
        }

    # ===== Communication Commands =====

    async def cmd_say(self, user_id: str, args: str) -> Dict:
        """Say something to the room."""
        if not args:
            return {'success': False, 'output': 'Say what?', 'events': []}

        user = self.world.get_user(user_id)
        room = self.world.get_user_room(user_id)

        if not user or not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        username = user.get('username', user.get('name', user_id))

        output = f'You say, "{args}"'

        event = {
            'type': 'say',
            'user': user_id,
            'username': username,
            'room': room['uid'],
            'text': args
        }

        # Check for play chat triggers
        await self.play_manager.check_chat_trigger(args, room['uid'])

        return {
            'success': True,
            'output': output,
            'events': [event]
        }

    async def cmd_emote(self, user_id: str, args: str) -> Dict:
        """Perform an emote action."""
        if not args:
            return {'success': False, 'output': 'Emote what?', 'events': []}

        user = self.world.get_user(user_id)
        room = self.world.get_user_room(user_id)

        if not user or not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        username = user.get('username', user.get('name', user_id))

        output = f"{username} {args}"

        event = {
            'type': 'emote',
            'user': user_id,
            'username': username,
            'room': room['uid'],
            'text': args
        }

        return {
            'success': True,
            'output': output,
            'events': [event]
        }

    async def cmd_tell(self, user_id: str, args: str) -> Dict:
        """Send private message to another user."""
        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: tell <user> <message>', 'events': []}

        target_name, message = parts
        target_id = f"user_{target_name}"

        # Check if target exists
        target = self.world.get_user(target_id)
        if not target:
            return {'success': False, 'output': f"User '{target_name}' not found.", 'events': []}

        user = self.world.get_user(user_id)
        username = user.get('username', user_id)

        output = f'You tell {target_name}, "{message}"'

        # Note: Private messages would need special handling in server
        event = {
            'type': 'tell',
            'user': user_id,
            'username': username,
            'target': target_id,
            'text': message
        }

        return {
            'success': True,
            'output': output,
            'events': [event]
        }

    # ===== Observation Commands =====

    async def cmd_look(self, user_id: str, args: str) -> Dict:
        """Look at room, person, or object."""
        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'You are nowhere.', 'events': []}

        # If args provided, look at specific target
        if args:
            target_name = args.strip().lower()

            # Handle "here" keyword - look at current room
            if target_name == 'here':
                args = ''  # Fall through to room description below

            # Handle "me" keyword - look at yourself
            elif target_name == 'me':
                user = self.world.get_user(user_id)
                if not user:
                    return {'success': False, 'output': 'Error: User not found.', 'events': []}

                lines = []
                display_name = user.get('username', user.get('name', user_id))
                user_type = 'agent' if user_id.startswith('agent_') else 'user'
                lines.append(f"\n{display_name} [{user_type}]")
                lines.append("=" * (len(display_name) + len(user_type) + 3))

                # Get description
                if user_id.startswith('agent_'):
                    agent = self.agent_manager.get_agent(user_id)
                    if agent and hasattr(agent, 'agent_description') and agent.agent_description:
                        lines.append(agent.agent_description)
                    else:
                        lines.append("You haven't set a description yet.")
                else:
                    desc = user.get('description', '')
                    lines.append(desc if desc else "You haven't set a description yet.")

                return {
                    'success': True,
                    'output': '\n'.join(lines),
                    'events': []
                }

        # Process target if not "here" or "me"
        if args and target_name not in ['here', 'me']:
            target_name = args.strip().lower()

            # Check occupants in the room
            for occ_id in room['occupants']:
                occ = self.world.get_user(occ_id)
                if occ:
                    occ_name = occ.get('username', occ.get('name', occ_id)).lower()
                    if occ_name == target_name or occ_id == f"agent_{target_name}" or occ_id == f"user_{target_name}":
                        # Looking at a person (user or agent)
                        lines = []
                        display_name = occ.get('username', occ.get('name', occ_id))
                        occ_type = 'agent' if occ_id.startswith('agent_') else 'user'
                        lines.append(f"\n{display_name} [{occ_type}]")
                        lines.append("=" * (len(display_name) + len(occ_type) + 3))

                        # Get description
                        if occ_id.startswith('agent_'):
                            # For agents, try to get description from agent manager
                            agent = self.agent_manager.get_agent(occ_id)
                            if agent and hasattr(agent, 'description') and agent.description:
                                lines.append(agent.description)
                            else:
                                lines.append(f"{display_name} hasn't set a description yet.")
                        else:
                            # For users, get description from user data
                            desc = occ.get('description', '')
                            if desc:
                                lines.append(desc)
                            else:
                                lines.append(f"{display_name} hasn't set a description yet.")

                        return {
                            'success': True,
                            'output': '\n'.join(lines),
                            'events': []
                        }

            # Check objects in the room
            for obj_id in room['objects']:
                obj = self.world.get_object(obj_id)
                if obj and obj['name'].lower() == target_name:
                    lines = []
                    lines.append(f"\n{obj['name']}")
                    lines.append("=" * len(obj['name']))
                    lines.append(obj.get('description', 'Nothing special.'))
                    return {
                        'success': True,
                        'output': '\n'.join(lines),
                        'events': []
                    }

            # Target not found
            return {
                'success': False,
                'output': f"You don't see '{args}' here.",
                'events': []
            }

        # No args - look at room
        lines = []
        lines.append(f"\n{room['name']}")
        lines.append("=" * len(room['name']))
        lines.append(room['description'])

        # Show exits
        if room['exits']:
            exits = ', '.join(room['exits'].keys())
            lines.append(f"\nExits: {exits}")

        # Show occupants (including yourself so you can see your profile)
        occupants = [
            self.world.get_user(uid)
            for uid in room['occupants']
        ]

        if occupants:
            lines.append("\nPeople here:")
            for occ in occupants:
                if occ:
                    name = occ.get('username', occ.get('name', occ['uid']))
                    is_agent = occ['uid'].startswith('agent_')

                    # Determine role (Noodler or Noodling)
                    role = 'Noodling' if is_agent else 'Noodler'

                    # Get metadata (species, age, pronoun)
                    if is_agent:
                        # For agents, check config in agent data
                        config = occ.get('config', {})
                        species = config.get('species', 'unknown')
                        age = config.get('age', 'unknown')
                        pronoun = config.get('pronoun', 'they')
                    else:
                        # For users, check user data directly
                        species = occ.get('species', 'human')
                        age = occ.get('age', 'unknown')
                        pronoun = occ.get('pronoun', 'they')

                    # Format: name [Role, species, age, pronoun]
                    lines.append(f"  {name} [{role}, {species}, {age}, {pronoun}]")

        # Show objects
        if room['objects']:
            lines.append("\nYou see:")
            for obj_id in room['objects']:
                obj = self.world.get_object(obj_id)
                if obj:
                    lines.append(f"  {obj['name']}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_inventory(self, user_id: str, args: str) -> Dict:
        """Show inventory."""
        user = self.world.get_user(user_id)
        if not user:
            return {'success': False, 'output': 'User not found.', 'events': []}

        inventory = user.get('inventory', [])

        if not inventory:
            return {'success': True, 'output': 'You are carrying nothing.', 'events': []}

        lines = ["You are carrying:"]
        for obj_id in inventory:
            obj = self.world.get_object(obj_id)
            if obj:
                lines.append(f"  {obj['name']}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_who(self, user_id: str, args: str) -> Dict:
        """List all connected users and agents."""
        lines = ["Connected users:"]

        # List users
        for uid, user in self.world.get_all_users().items():
            username = user.get('username', uid)
            room = self.world.get_room(user['current_room'])
            room_name = room['name'] if room else 'unknown'
            lines.append(f"  {username} - {room_name}")

        # List agents
        agents = self.world.get_all_agents()
        if agents:
            lines.append("\nActive agents:")
            for aid, agent in agents.items():
                name = agent.get('name', aid)
                room = self.world.get_room(agent['current_room'])
                room_name = room['name'] if room else 'unknown'
                lines.append(f"  {name} [agent] - {room_name}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    # ===== Manipulation Commands =====

    async def cmd_take(self, user_id: str, args: str) -> Dict:
        """Take an object."""
        if not args:
            return {'success': False, 'output': 'Take what?', 'events': []}

        # Find object in room
        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Simple name match
        obj_id = None
        for oid in room['objects']:
            obj = self.world.get_object(oid)
            if obj and args.lower() in obj['name'].lower():
                obj_id = oid
                break

        if not obj_id:
            return {'success': False, 'output': f"You don't see '{args}' here.", 'events': []}

        obj = self.world.get_object(obj_id)

        # Check if takeable
        if not obj['properties'].get('takeable', True):
            return {'success': False, 'output': f"You can't take {obj['name']}.", 'events': []}

        # Move to inventory
        room['objects'].remove(obj_id)
        user = self.world.get_user(user_id)
        if 'inventory' not in user:
            user['inventory'] = []
        user['inventory'].append(obj_id)
        obj['location'] = user_id

        self.world.save_all()

        return {
            'success': True,
            'output': f"You take {obj['name']}.",
            'events': []
        }

    async def cmd_drop(self, user_id: str, args: str) -> Dict:
        """Drop an object."""
        if not args:
            return {'success': False, 'output': 'Drop what?', 'events': []}

        user = self.world.get_user(user_id)
        inventory = user.get('inventory', [])

        # Find object in inventory
        obj_id = None
        for oid in inventory:
            obj = self.world.get_object(oid)
            if obj and args.lower() in obj['name'].lower():
                obj_id = oid
                break

        if not obj_id:
            return {'success': False, 'output': f"You don't have '{args}'.", 'events': []}

        obj = self.world.get_object(obj_id)
        room = self.world.get_user_room(user_id)

        # Move to room
        inventory.remove(obj_id)
        room['objects'].append(obj_id)
        obj['location'] = room['uid']

        self.world.save_all()

        return {
            'success': True,
            'output': f"You drop {obj['name']}.",
            'events': []
        }

    # ===== Building Commands =====

    async def cmd_create(self, user_id: str, args: str) -> Dict:
        """Create a room or object."""
        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @create <room|object> <name>', 'events': []}

        entity_type, name = parts

        if entity_type.lower() == 'room':
            room_id = self.world.create_room(
                name=name,
                description="A newly created room.",
                owner=user_id
            )
            return {
                'success': True,
                'output': f"Room created: {name} ({room_id})",
                'events': []
            }

        elif entity_type.lower() == 'object':
            room = self.world.get_user_room(user_id)
            obj_id = self.world.create_object(
                name=name,
                description="A newly created object.",
                owner=user_id,
                location=room['uid'] if room else None
            )
            return {
                'success': True,
                'output': f"Object created: {name} ({obj_id})",
                'events': []
            }

        else:
            return {'success': False, 'output': 'Usage: @create <room|object> <name>', 'events': []}

    async def cmd_describe(self, user_id: str, args: str) -> Dict:
        """Deprecated: Use @setdesc instead. This redirects to @setdesc here <description>."""
        if not args:
            return {'success': False, 'output': 'Usage: @setdesc here <description> (or @setdesc me, @setdesc "object")\n\nNote: @describe is deprecated, use @setdesc instead.', 'events': []}

        # Redirect to @setdesc here <description>
        return await self.cmd_setdesc(user_id, f"here {args}")

    async def cmd_dig(self, user_id: str, args: str) -> Dict:
        """Create an exit to a new or existing room."""
        parts = args.split()
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @dig <direction> <room_name>', 'events': []}

        direction = parts[0].lower()
        room_name = ' '.join(parts[1:])

        current_room = self.world.get_user_room(user_id)
        if not current_room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Create new room
        new_room_id = self.world.create_room(
            name=room_name,
            description="A newly dug room.",
            owner=user_id
        )

        # Create exit
        self.world.set_exit(current_room['uid'], direction, new_room_id)

        # Create return exit
        opposite = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'up': 'down', 'down': 'up'
        }
        if direction in opposite:
            self.world.set_exit(new_room_id, opposite[direction], current_room['uid'])

        return {
            'success': True,
            'output': f"Room '{room_name}' created {direction}.",
            'events': []
        }

    async def cmd_link(self, user_id: str, args: str) -> Dict:
        """Link current room to an existing room with a custom direction name."""
        # Parse: @link room_005 Glowing blue portal of whimsy
        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @link <room_id> <direction_name>', 'events': []}

        target_room_id = parts[0]
        direction_name = parts[1]

        current_room = self.world.get_user_room(user_id)
        if not current_room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Check if target room exists
        target_room = self.world.get_room(target_room_id)
        if not target_room:
            return {'success': False, 'output': f"Room '{target_room_id}' not found.", 'events': []}

        # Create custom exit
        self.world.set_exit(current_room['uid'], direction_name.lower(), target_room_id)

        return {
            'success': True,
            'output': f"✨ Linked! You can now travel via '{direction_name}' to {target_room.get('name', target_room_id)}.",
            'events': []
        }

    async def cmd_destroy(self, user_id: str, args: str) -> Dict:
        """Destroy an object in the current room."""
        if not args:
            return {'success': False, 'output': 'Usage: @destroy <object> OR @destroy "<multi word object>"', 'events': []}

        # Parse object name - support quoted names for multi-word objects
        object_name = args.strip()
        if object_name.startswith('"') and object_name.endswith('"'):
            object_name = object_name[1:-1]

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Find object in room
        obj = None
        obj_id = None
        for oid in room.get('objects', []):
            room_obj = self.world.get_object(oid)
            if room_obj and room_obj['name'].lower() == object_name.lower():
                obj = room_obj
                obj_id = oid
                break

        if not obj:
            return {'success': False, 'output': f"Object '{object_name}' not found in this room.", 'events': []}

        # Remove from room
        room['objects'].remove(obj_id)

        # Delete from world
        del self.world.objects[obj_id]

        self.world.save_all()

        return {
            'success': True,
            'output': f"Object '{obj['name']}' has been destroyed.",
            'events': []
        }

    # ===== Agent Commands =====

    async def cmd_rez_agent(self, user_id: str, args: str) -> Dict:
        """Spawn one or more Noodling agents (with optional recipe)."""
        if not args:
            return {
                'success': False,
                'output': (
                    'Usage: @rez [-f] [-e] <agent_name> [agent_name2 ...] [description]\n\n'
                    'Options:\n'
                    '  -f    Force fresh state (skip loading saved phenomenal state)\n'
                    '  -e    Enable enlightenment (agent is self-aware and metacognitive)\n\n'
                    'Available recipes:\n' +
                    '\n'.join(f'  - {name}' for name in self.recipe_loader.list_recipes()) +
                    '\n\nExamples:\n'
                    '  @rez phi\n'
                    '  @rez -f phi        (fresh rez, ignores saved state)\n'
                    '  @rez -e phi        (spawn with enlightenment)\n'
                    '  @rez -f -e phi     (fresh AND enlightened)\n'
                    '  @rez phi toad callie\n'
                    '  @rez phi curious and thoughtful'
                ),
                'events': []
            }

        # Check for -f flag (force fresh state)
        skip_phenomenal_state = False
        if args.startswith('-f ') or args == '-f':
            skip_phenomenal_state = True
            args = args[2:].strip()  # Remove -f flag
            if not args:
                return {'success': False, 'output': 'Error: No agent name provided after -f flag.', 'events': []}

        # Check for -e flag (enable enlightenment)
        enlightenment = False
        if args.startswith('-e ') or args == '-e':
            enlightenment = True
            args = args[2:].strip()  # Remove -e flag
            if not args:
                return {'success': False, 'output': 'Error: No agent name provided after -e flag.', 'events': []}

        # Parse agent names - support quoted names for multi-word agents like "Backwards Dweller"
        import shlex
        try:
            # Use shlex to properly handle quoted strings
            parts = shlex.split(args)
        except ValueError:
            # If shlex fails (unmatched quotes), fall back to simple split
            parts = args.split()

        agent_names = []
        description_parts = []

        # Collect agent names (assume they're recipe names or simple names)
        for i, part in enumerate(parts):
            part_lower = part.lower()
            # Check if this looks like a recipe name or simple agent name
            if self.recipe_loader.load_recipe(part_lower) or (i == 0 or part_lower.replace(' ', '_').replace('-', '_').isalpha()):
                # If it's the first word, or a known recipe, or a simple alphabetic word (with spaces/hyphens ok)
                if i < 3:  # Limit to first 3 words as potential agent names
                    agent_names.append(part_lower.replace(' ', '_'))  # Convert spaces to underscores for agent_id
                else:
                    # Rest is description
                    description_parts = parts[i:]
                    break
            else:
                # Not an agent name, must be start of description
                description_parts = parts[i:]
                break

        # If no agent names found, treat first word as agent name
        if not agent_names:
            agent_names = [parts[0].lower().replace(' ', '_')]
            description_parts = parts[1:] if len(parts) > 1 else []

        agent_description = ' '.join(description_parts) if description_parts else None

        # Spawn each agent
        all_events = []
        rezzed_agents = []
        errors = []

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        for agent_name in agent_names:
            agent_id = f"agent_{agent_name}"

            # Check if agent already exists
            if self.world.get_user(agent_id):
                errors.append(f"'{agent_name}' already exists")
                continue

            # Try to load recipe (will return None if not found)
            recipe = self.recipe_loader.load_recipe(agent_name)

            # Use recipe data if available, otherwise defaults
            if recipe:
                display_name = recipe.name
                description = recipe.description if not agent_description else agent_description

                # Build config from recipe
                # Phase 6: Debug self-monitoring config
                sm_config = self.config['agent'].get('self_monitoring', {})
                logger.debug(f"[SPAWN] agent_id={agent_id}, self.config['agent'] keys={list(self.config['agent'].keys())}, self_monitoring={sm_config}")

                config = {
                    'appetites': recipe.get_appetite_baselines(),
                    'personality': recipe.get_personality_vector(),
                    'identity_prompt': recipe.identity_prompt,
                    'species': recipe.species,
                    'language_mode': recipe.language_mode,
                    'temperature': recipe.temperature,
                    'max_tokens': recipe.max_tokens,
                    'enforce_action_format': recipe.enforce_action_format,
                    'response_cooldown': recipe.response_cooldown,
                    'enlightenment': enlightenment if enlightenment else recipe.enlightenment,
                    # Phase 6: Self-monitoring config from global config.yaml
                    'self_monitoring': sm_config
                }

                # Wind in the Willows-style natural arrival
                import random
                arrival_phrases = [
                    "steps into the scene",
                    "ambles into view",
                    "appears round the bend",
                    "wanders in from the riverbank",
                    "pops up cheerfully"
                ]
                arrival = random.choice(arrival_phrases)

                rez_msg = f"{display_name} ({recipe.species}) {arrival}"
                if recipe.language_mode == 'nonverbal':
                    rez_msg += ", watching curiously with bright eyes"
                rez_msg += f". {description}"
            else:
                # No recipe - use defaults
                display_name = agent_name.capitalize()
                description = agent_description if agent_description else "A Noodling consciousness agent"
                config = {
                    'enlightenment': enlightenment
                } if enlightenment else None

                # Wind in the Willows-style natural arrival
                import random
                arrival_phrases = [
                    "steps into the scene",
                    "ambles into view",
                    "appears round the bend",
                    "wanders in from somewhere",
                    "shows up with a friendly wave"
                ]
                arrival = random.choice(arrival_phrases)
                rez_msg = f"{display_name} {arrival}. {description}"

            # Create agent in world
            checkpoint_path = "../../consilience_core/checkpoints_phase4/best_checkpoint.npz"
            self.world.create_agent(
                name=agent_name,
                checkpoint_path=checkpoint_path,
                spawn_room=room['uid']
            )

            # Initialize agent in manager
            await self.agent_manager.create_agent(
                agent_id=agent_id,
                checkpoint_path=checkpoint_path,
                spawn_room=room['uid'],
                agent_name=display_name,
                agent_description=description,
                config=config,
                skip_phenomenal_state=skip_phenomenal_state
            )

            # Track successful spawn
            rezzed_agents.append(display_name)
            recipe_msg = f" (using recipe: {recipe.name})" if recipe else ""

            # Add event for this agent
            all_events.append({
                'type': 'enter',
                'user': agent_id,
                'room': room['uid'],
                'text': rez_msg
            })

        # Build result message
        if rezzed_agents:
            if len(rezzed_agents) == 1:
                output_msg = f"Agent '{rezzed_agents[0]}' spawned."
            else:
                output_msg = f"Spawned {len(rezzed_agents)} agents: {', '.join(rezzed_agents)}."

            # Add flags info
            flags = []
            if skip_phenomenal_state:
                flags.append("fresh state")
            if enlightenment:
                flags.append("enlightenment enabled")
            if flags:
                output_msg += f" ({', '.join(flags)})"

            if errors:
                output_msg += f"\nErrors: {', '.join(errors)}"

            return {
                'success': True,
                'output': output_msg,
                'events': all_events
            }
        else:
            # All failed
            return {
                'success': False,
                'output': f"Failed to spawn agents. Errors: {', '.join(errors)}",
                'events': []
            }

    async def cmd_remove(self, user_id: str, args: str) -> Dict:
        """Remove an agent from the world."""
        if not args:
            return {'success': False, 'output': 'Usage: @remove [-s] <agent_name>\n  -s: Silent removal (no departure message)', 'events': []}

        # Check for -s flag (silent removal)
        silent = False
        if args.startswith('-s ') or args == '-s':
            silent = True
            args = args[2:].strip()  # Remove -s flag
            if not args:
                return {'success': False, 'output': 'Error: No agent name provided after -s flag.', 'events': []}

        # Parse agent name - support quoted names like @rez does
        import shlex
        try:
            parts = shlex.split(args)
            agent_name = parts[0] if parts else args.strip()
        except ValueError:
            # If shlex fails, use simple strip
            agent_name = args.strip()

        # Convert spaces to underscores (matches @rez behavior)
        agent_name = agent_name.lower().replace(' ', '_')
        agent_id = f"agent_{agent_name}"

        # Check if agent exists
        agent_data = self.world.get_user(agent_id)
        if not agent_data:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        # Get current room for exit event
        room = self.world.get_room(agent_data['current_room'])

        # Remove from agent manager (stops consciousness)
        await self.agent_manager.remove_agent(agent_id)

        # Remove from world
        if agent_id in self.world.agents:
            del self.world.agents[agent_id]

        # Remove from room occupants
        if room and agent_id in room.get('occupants', []):
            room['occupants'].remove(agent_id)

        # Save world state
        self.world.save_all()

        # Build exit event
        events = []
        if not silent:
            # Wind in the Willows-style natural departure
            import random
            departure_phrases = [
                "remembers something urgent and hurries off",
                "suddenly recalls an appointment and dashes away",
                "realizes they're expected elsewhere and scurries off",
                "gets that look of sudden remembering and trots away",
                "mutters about forgetting something and bustles off",
                "hears a distant call and wanders away",
                "decides it's time for a ramble and ambles off"
            ]
            departure = random.choice(departure_phrases)

            events.append({
                'type': 'exit',
                'user': agent_id,
                'username': agent_name,
                'room': room['uid'] if room else 'unknown',
                'text': f"{agent_name} {departure}."
            })
        else:
            # Silent removal - still send exit event but with metadata flag
            events.append({
                'type': 'agent_removed',  # Different event type for silent removal
                'user': agent_id,
                'username': agent_name,
                'room': room['uid'] if room else 'unknown',
                'silent': True
            })

        return {
            'success': True,
            'output': f"Agent '{agent_name}' removed{' silently' if silent else ''}.",
            'events': events
        }

    async def cmd_reset(self, user_id: str, args: str) -> Dict:
        """Reset the world to default settings (removes all agents and custom objects)."""
        # Check for -c flag (clear screen after reset)
        clear_screen = False
        if args.strip().lower().startswith('-c '):
            clear_screen = True
            args = args.strip()[3:]  # Remove '-c ' prefix

        # Confirmation check
        if args.strip().lower() != 'confirm':
            return {
                'success': False,
                'output': 'WARNING: This will remove all agents and reset the world!\nType: @reset confirm (or @reset -c confirm to clear screen)',
                'events': []
            }

        # Remove all agents (with state deletion)
        agent_ids = list(self.world.agents.keys())
        for agent_id in agent_ids:
            await self.agent_manager.remove_agent(agent_id, delete_state=True)

        # Clean up any orphaned agent state directories
        import os
        import shutil
        agents_dir = 'world/agents'
        if os.path.exists(agents_dir):
            for entry in os.listdir(agents_dir):
                if entry.startswith('agent_'):
                    state_path = os.path.join(agents_dir, entry)
                    if os.path.isdir(state_path):
                        shutil.rmtree(state_path)

        # Clear world data
        self.world.agents = {}
        self.world.objects = {}

        # Reset rooms to original state (keep structure, clear occupants except humans)
        for room_id, room in self.world.rooms.items():
            # Keep only human users in rooms
            room['occupants'] = [uid for uid in room.get('occupants', []) if not uid.startswith('agent_')]
            # Clear custom objects from rooms
            room['objects'] = []

        # Save state
        self.world.save_all()

        # Build output message
        output_msg = 'World reset complete. All agents removed, objects cleared.'

        # Build events list
        reset_events = [{
            'type': 'system',
            'text': 'The world shimmers and resets to its original state.'
        }]

        # Add clear screen event if -c flag was used
        if clear_screen:
            reset_events.append({
                'type': 'clear_screen',
                'user_id': user_id
            })

        return {
            'success': True,
            'output': output_msg,
            'events': reset_events
        }

    async def cmd_observe_agent(self, user_id: str, args: str) -> Dict:
        """Observe an agent's internal state."""
        if not args:
            return {'success': False, 'output': 'Usage: @observe <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}"

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        state = agent.get_phenomenal_state()

        lines = [f"\nAgent: {agent_name}"]
        lines.append("=" * 40)
        lines.append(f"Surprise: {state.get('surprise', 0.0):.3f} (threshold: {state.get('surprise_threshold', 0.3):.3f})")
        lines.append(f"Step: {state.get('step', 0)}")
        lines.append(f"\nPhenomenal state (40-D):")
        lines.append(f"  Fast layer (16-D): {state.get('h_fast', [])[:4]}...")
        lines.append(f"  Medium layer (16-D): {state.get('h_medium', [])[:4]}...")
        lines.append(f"  Slow layer (8-D): {state.get('h_slow', [])[:4]}...")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_launch_noodlescope(self, user_id: str, args: str) -> Dict:
        """Launch the native NoodleScope visualization app."""
        import subprocess
        import os

        noodlescope_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'NoodleScope/.build/debug/NoodleScope'
        )

        if not os.path.exists(noodlescope_path):
            return {
                'success': False,
                'output': 'NoodleScope app not found. Run: cd NoodleScope && swift build',
                'events': []
            }

        try:
            # Launch NoodleScope in the background
            subprocess.Popen(
                [noodlescope_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            return {
                'success': True,
                'output': '🔬 Launching NoodleScope - native macOS visualization app...\n' +
                         'The app should open in a new window momentarily.',
                'events': []
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Failed to launch NoodleScope: {str(e)}',
                'events': []
            }

    async def cmd_phi(self, user_id: str, args: str) -> Dict:
        """Calculate and display agent's integrated information (Φ)."""
        if not args:
            return {'success': False, 'output': 'Usage: @phi <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}"

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        # Check if we have enough state history
        if len(agent.state_history) < 2:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' needs at least 2 timesteps of interaction history.\nCurrent history: {len(agent.state_history)} timesteps.",
                'events': []
            }

        # Calculate Φ (integrated information)
        try:
            # Get state statistics for debugging
            state = agent.state_history[-1]
            state_mean = float(np.mean(state))
            state_std = float(np.std(state))
            state_min = float(np.min(state))
            state_max = float(np.max(state))

            phi = agent.consciousness_metrics.calculate_phi(
                phenomenal_state=agent.state_history[-1],
                method='partition_based'
            )

            # Interpret Φ value
            if phi < 0.5:
                interpretation = "Minimal integration (likely unconscious)"
            elif phi < 1.0:
                interpretation = "Low integration (possibly minimal consciousness)"
            elif phi < 2.0:
                interpretation = "Moderate integration (suggests consciousness)"
            elif phi < 3.0:
                interpretation = "High integration (strong consciousness signature)"
            else:
                interpretation = "Very high integration (rich conscious experience)"

            # Calculate statistics if we have history
            phi_history = []
            if len(agent.state_history) >= 10:
                # Calculate Φ for last 10 states
                for i in range(-10, -1):
                    if len(agent.state_history) >= abs(i):
                        try:
                            phi_val = agent.consciousness_metrics.calculate_phi(
                                phenomenal_state=agent.state_history[i],
                                method='partition_based'
                            )
                            phi_history.append(phi_val)
                        except:
                            pass

            lines = [f"\nIntegrated Information (Φ) - Agent: {agent_name}"]
            lines.append("=" * 50)
            lines.append(f"Current Φ: {phi:.4f}")
            lines.append(f"Interpretation: {interpretation}")
            lines.append("")
            lines.append("Φ Scale:")
            lines.append("  < 0.5  → Minimal integration")
            lines.append("  0.5-1.0 → Low integration")
            lines.append("  1.0-2.0 → Moderate integration (consciousness threshold)")
            lines.append("  2.0-3.0 → High integration")
            lines.append("  > 3.0  → Very high integration")

            if phi_history:
                import numpy as np
                mean_phi = np.mean(phi_history)
                std_phi = np.std(phi_history)
                lines.append("")
                lines.append(f"Recent history (last {len(phi_history)} states):")
                lines.append(f"  Mean Φ: {mean_phi:.4f}")
                lines.append(f"  Std Φ: {std_phi:.4f}")
                lines.append(f"  Range: [{min(phi_history):.4f}, {max(phi_history):.4f}]")

            lines.append("")
            lines.append(f"State history size: {len(agent.state_history)} timesteps")
            lines.append(f"Surprise history size: {len(agent.surprise_history)} timesteps")

            # Add enhanced diagnostics
            lines.append("")
            lines.append("Phenomenal State Statistics:")
            lines.append(f"  Dimensionality: {len(state)}")
            lines.append(f"  Mean activation: {state_mean:.6f}")
            lines.append(f"  Std deviation: {state_std:.6f}")
            lines.append(f"  Range: [{state_min:.6f}, {state_max:.6f}]")

            # Check observer status
            lines.append("")
            lines.append("Observer Status:")
            observer_active = hasattr(agent.consciousness, 'config') and agent.consciousness.config.get('use_observers', False)
            hierarchical_obs = hasattr(agent.consciousness, 'config') and agent.consciousness.config.get('observe_hierarchical_states', False)
            lines.append(f"  Observer loops active: {observer_active}")
            lines.append(f"  Hierarchical observation: {hierarchical_obs}")

            # Check if consciousness model is wrapped with observers
            model_type = type(agent.consciousness).__name__
            lines.append(f"  Model type: {model_type}")

            lines.append("")
            lines.append("Note: Φ is calculated using Monte Carlo approximation")
            lines.append("of Integrated Information Theory (Tononi et al., 2016).")

            return {
                'success': True,
                'output': '\n'.join(lines),
                'events': []
            }

        except Exception as e:
            logger.error(f"Error calculating Φ: {e}", exc_info=True)
            return {
                'success': False,
                'output': f"Error calculating Φ: {str(e)}",
                'events': []
            }

    async def cmd_stoke_appetite(self, user_id: str, args: str) -> Dict:
        """
        Increase an agent's appetite (Phase 6 feature).

        Usage: @stoke <agent_name> <appetite> <amount>
        Example: @stoke Mr. Toad novelty 0.3

        Appetites: curiosity, status, mastery, novelty, safety, social_bond, comfort, autonomy
        Amount: 0.0-1.0 (how much to increase)
        """
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Appetite Orchestration - @stoke\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Increase an agent's internal drive/appetite.\n\n"
                    "Usage: @stoke <agent_name> <appetite> <amount>\n\n"
                    "Example: @stoke Toad novelty 0.3\n"
                    "         (Makes Toad 30% more drawn to new experiences)\n\n"
                    "Available Appetites:\n"
                    "  curiosity     - Drive to learn and explore\n"
                    "  status        - Desire for recognition/prestige\n"
                    "  mastery       - Need to excel and improve\n"
                    "  novelty       - Craving for new experiences\n"
                    "  safety        - Need for security/stability\n"
                    "  social_bond   - Desire for connection\n"
                    "  comfort       - Need for ease and pleasure\n"
                    "  autonomy      - Drive for independence\n\n"
                    "Amount: 0.0-1.0 (0.3 = moderate increase)\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Note: This is a Phase 6 (Appetite Architecture) feature.\n"
                    "Phase 6 training is currently in progress.\n"
                    "Check back after training completes to use this feature!"
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {
                'success': False,
                'output': "Usage: @stoke <agent_name> <appetite> <amount>\nExample: @stoke Toad novelty 0.3",
                'events': []
            }

        agent_name = parts[0]
        appetite_name = parts[1].lower()

        try:
            amount = float(parts[2])
            if amount < 0 or amount > 1.0:
                return {
                    'success': False,
                    'output': "Amount must be between 0.0 and 1.0",
                    'events': []
                }
        except ValueError:
            return {
                'success': False,
                'output': f"Invalid amount: {parts[2]}. Must be a number between 0.0 and 1.0",
                'events': []
            }

        # Check if agent exists
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see active agents.",
                'events': []
            }

        # Check if Phase 6 is available
        has_phase6 = hasattr(agent, 'stoke_appetite')

        if not has_phase6:
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"The Appetite Architecture (Phase 6) is currently being trained!\n\n"
                    f"Once training completes, you'll be able to:\n"
                    f"  @stoke {agent_name} {appetite_name} {amount}\n\n"
                    f"This will increase {agent_name}'s {appetite_name} drive by {amount},\n"
                    f"causing emergent behavior changes based on internal motivation.\n\n"
                    f"Check training progress: cd /Users/thistlequell/git/consilience/training && ./status.sh\n\n"
                    f"Current agents use Phase 4 (Social Cognition + Theory of Mind)."
                ),
                'events': []
            }

        # Actually stoke the appetite (when Phase 6 is loaded)
        agent.stoke_appetite(appetite_name, amount)

        return {
            'success': True,
            'output': (
                f"🔥 Appetite Stoked!\n\n"
                f"{agent_name}'s {appetite_name} appetite increased by {amount:.2f}\n\n"
                f"Watch for emergent behavior changes..."
            ),
            'events': [{'type': 'appetite_change', 'agent': agent_name, 'appetite': appetite_name, 'change': amount}]
        }

    async def cmd_sate_appetite(self, user_id: str, args: str) -> Dict:
        """
        Satisfy/decrease an agent's appetite (Phase 6 feature).

        Usage: @sate <agent_name> <appetite> <amount>
        Example: @sate Mr. Toad safety 0.8
        """
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Appetite Orchestration - @sate\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Satisfy/decrease an agent's internal drive.\n\n"
                    "Usage: @sate <agent_name> <appetite> <amount>\n\n"
                    "Example: @sate Toad safety 0.8\n"
                    "         (Calms Toad's reckless impulses)\n\n"
                    "See @stoke for list of appetites.\n\n"
                    "Amount: 0.0-1.0 (how much to satisfy)\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Note: This is a Phase 6 (Appetite Architecture) feature.\n"
                    "Phase 6 training is currently in progress.\n"
                    "Check back after training completes to use this feature!"
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {
                'success': False,
                'output': "Usage: @sate <agent_name> <appetite> <amount>\nExample: @sate Toad safety 0.8",
                'events': []
            }

        agent_name = parts[0]
        appetite_name = parts[1].lower()

        try:
            amount = float(parts[2])
            if amount < 0 or amount > 1.0:
                return {
                    'success': False,
                    'output': "Amount must be between 0.0 and 1.0",
                    'events': []
                }
        except ValueError:
            return {
                'success': False,
                'output': f"Invalid amount: {parts[2]}. Must be a number between 0.0 and 1.0",
                'events': []
            }

        # Check if agent exists
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see active agents.",
                'events': []
            }

        # Check if Phase 6 is available
        has_phase6 = hasattr(agent, 'sate_appetite')

        if not has_phase6:
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"The Appetite Architecture (Phase 6) is currently being trained!\n\n"
                    f"Once training completes, you'll be able to:\n"
                    f"  @sate {agent_name} {appetite_name} {amount}\n\n"
                    f"This will satisfy {agent_name}'s {appetite_name} drive by {amount},\n"
                    f"reducing that motivation and potentially changing their goals.\n\n"
                    f"Check training progress: cd /Users/thistlequell/git/consilience/training && ./status.sh\n\n"
                    f"Current agents use Phase 4 (Social Cognition + Theory of Mind)."
                ),
                'events': []
            }

        # Actually sate the appetite (when Phase 6 is loaded)
        agent.sate_appetite(appetite_name, amount)

        return {
            'success': True,
            'output': (
                f"💧 Appetite Sated!\n\n"
                f"{agent_name}'s {appetite_name} appetite decreased by {amount:.2f}\n\n"
                f"Watch for behavioral changes as motivation shifts..."
            ),
            'events': [{'type': 'appetite_change', 'agent': agent_name, 'appetite': appetite_name, 'change': -amount}]
        }

    async def cmd_show_appetites(self, user_id: str, args: str) -> Dict:
        """
        Show an agent's current appetite levels (Phase 6 feature).

        Usage: @appetites <agent_name>
        Example: @appetites Mr. Toad
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @appetites <agent_name>\nExample: @appetites Toad",
                'events': []
            }

        agent_name = args.strip()

        # Check if agent exists
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see active agents.",
                'events': []
            }

        # Check if Phase 6 is available
        has_phase6 = hasattr(agent, 'get_appetites')

        if not has_phase6:
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"The Appetite Architecture (Phase 6) is currently being trained!\n\n"
                    f"Once training completes, you'll be able to view {agent_name}'s\n"
                    f"internal drives and motivational state with:\n"
                    f"  @appetites {agent_name}\n\n"
                    f"This will show their 8-D appetite vector:\n"
                    f"  - curiosity, status, mastery, novelty\n"
                    f"  - safety, social_bond, comfort, autonomy\n\n"
                    f"Plus active goals and detected conflicts!\n\n"
                    f"Check training progress: cd /Users/thistlequell/git/consilience/training && ./status.sh\n\n"
                    f"Current agents use Phase 4 (Social Cognition + Theory of Mind)."
                ),
                'events': []
            }

        # Get appetite information (when Phase 6 is loaded)
        appetites = agent.get_appetites()

        lines = [
            f"{agent_name}'s Appetites (Phase 6)",
            "━" * 50,
            ""
        ]

        appetite_names = ["curiosity", "status", "mastery", "novelty",
                         "safety", "social_bond", "comfort", "autonomy"]

        for name in appetite_names:
            value = appetites.get(name, 0.0)
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"  {name:12s} [{bar}] {value:.2f}")

        lines.extend([
            "",
            "Active Goals:",
            "  " + (", ".join(agent.get_active_goals()) if hasattr(agent, 'get_active_goals') else "None"),
            "",
            "Detected Conflicts:",
            "  " + (agent.get_conflicts() if hasattr(agent, 'get_conflicts') else "None"),
        ])

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_override_goal(self, user_id: str, args: str) -> Dict:
        """
        Override an agent's goal activation (Phase 6 feature).

        Usage: @override <agent_name> <goal_name> <strength>
        Example: @override Toad learn_skill 0.95
        """
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Goal Orchestration - @override\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Directly override an agent's goal activation.\n"
                    "This replaces natural goal generation for strong narrative control.\n\n"
                    "Usage: @override <agent_name> <goal> <strength>\n\n"
                    "Example: @override Toad learn_skill 0.95\n"
                    "         @override Toad demonstrate_competence 0.90\n"
                    "         (Makes Toad obsessed with mastering his current passion!)\n\n"
                    "Available Goals (16):\n"
                    "  explore_environment, seek_social_connection, demonstrate_competence,\n"
                    "  pursue_novelty, ensure_safety, gain_status, seek_comfort,\n"
                    "  maintain_autonomy, help_friend, avoid_consequences, restore_reputation,\n"
                    "  learn_skill, impress_others, solve_problem, express_emotion, achieve_goal\n\n"
                    "Strength: 0.0-1.0 (goal activation level)\n\n"
                    "To clear: @reset_goals <agent_name> [goal]\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Note: This is a Phase 6 (Appetite Architecture) feature."
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {
                'success': False,
                'output': "Usage: @override <agent_name> <goal> <strength>\nExample: @override Toad learn_skill 0.95",
                'events': []
            }

        agent_name = parts[0]
        goal_name = parts[1].lower()

        try:
            strength = float(parts[2])
            if strength < 0 or strength > 1.0:
                return {
                    'success': False,
                    'output': f"Strength must be between 0.0 and 1.0, got {strength}",
                    'events': []
                }
        except ValueError:
            return {
                'success': False,
                'output': f"Invalid strength value: {parts[2]}. Must be a number between 0.0 and 1.0",
                'events': []
            }

        # Find agent
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see available agents.",
                'events': []
            }

        # Check if Phase 6 is available
        if not hasattr(agent, 'override_goal'):
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"Goal orchestration requires Phase 6 (Appetite Architecture).\n"
                    f"Current agents use Phase 4 (Social Cognition)."
                ),
                'events': []
            }

        try:
            agent.override_goal(goal_name, strength)
            return {
                'success': True,
                'output': (
                    f"🎭 Brenda's Narrative Control Activated\n\n"
                    f"{agent_name}'s goal '{goal_name}' overridden to {strength:.2f}\n\n"
                    f"This will dominate their behavior until cleared.\n"
                    f"Use @goals {agent_name} to see all active goals."
                ),
                'events': [{'type': 'goal_override', 'agent': agent_name, 'goal': goal_name, 'strength': strength}]
            }
        except ValueError as e:
            return {
                'success': False,
                'output': f"Error: {str(e)}",
                'events': []
            }

    async def cmd_set_goal_bias(self, user_id: str, args: str) -> Dict:
        """
        Add a persistent bias to an agent's goal generation (Phase 6 feature).

        Usage: @bias <agent_name> <goal_name> <bias>
        Example: @bias Toad ensure_safety -0.3
        """
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Goal Orchestration - @bias\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Add a subtle, persistent bias to goal generation.\n"
                    "This adds to natural generation rather than replacing it.\n\n"
                    "Usage: @bias <agent_name> <goal> <bias>\n\n"
                    "Example: @bias Toad ensure_safety -0.3\n"
                    "         @bias Toad pursue_novelty 0.2\n"
                    "         (Makes Toad more reckless and novelty-seeking)\n\n"
                    "Bias: -1.0 to 1.0 (negative reduces, positive increases)\n\n"
                    "To clear: @clear_bias <agent_name> [goal]\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Note: This is a Phase 6 (Appetite Architecture) feature."
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {
                'success': False,
                'output': "Usage: @bias <agent_name> <goal> <bias>\nExample: @bias Toad ensure_safety -0.3",
                'events': []
            }

        agent_name = parts[0]
        goal_name = parts[1].lower()

        try:
            bias = float(parts[2])
            if bias < -1.0 or bias > 1.0:
                return {
                    'success': False,
                    'output': f"Bias must be between -1.0 and 1.0, got {bias}",
                    'events': []
                }
        except ValueError:
            return {
                'success': False,
                'output': f"Invalid bias value: {parts[2]}. Must be a number between -1.0 and 1.0",
                'events': []
            }

        # Find agent
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see available agents.",
                'events': []
            }

        # Check if Phase 6 is available
        if not hasattr(agent, 'set_goal_bias'):
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"Goal orchestration requires Phase 6 (Appetite Architecture).\n"
                    f"Current agents use Phase 4 (Social Cognition)."
                ),
                'events': []
            }

        try:
            agent.set_goal_bias(goal_name, bias)
            return {
                'success': True,
                'output': (
                    f"🎭 Brenda's Subtle Influence Applied\n\n"
                    f"{agent_name}'s '{goal_name}' bias set to {bias:+.2f}\n\n"
                    f"This will subtly shape their motivations.\n"
                    f"Use @goals {agent_name} to see the effect."
                ),
                'events': [{'type': 'goal_bias', 'agent': agent_name, 'goal': goal_name, 'bias': bias}]
            }
        except ValueError as e:
            return {
                'success': False,
                'output': f"Error: {str(e)}",
                'events': []
            }

    async def cmd_reset_goals(self, user_id: str, args: str) -> Dict:
        """
        Clear goal overrides for an agent (Phase 6 feature).

        Usage: @reset_goals <agent_name> [goal_name]
        Example: @reset_goals Toad
                 @reset_goals Toad learn_skill
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @reset_goals <agent_name> [goal]\nExample: @reset_goals Toad",
                'events': []
            }

        parts = args.split()
        agent_name = parts[0]
        goal_name = parts[1].lower() if len(parts) > 1 else None

        # Find agent
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see available agents.",
                'events': []
            }

        # Check if Phase 6 is available
        if not hasattr(agent, 'clear_goal_overrides'):
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"Goal orchestration requires Phase 6 (Appetite Architecture).\n"
                    f"Current agents use Phase 4 (Social Cognition)."
                ),
                'events': []
            }

        agent.clear_goal_overrides(goal_name)

        if goal_name:
            output = f"Cleared goal override for '{goal_name}' on {agent_name}.\nNatural generation resumed for this goal."
        else:
            output = f"Cleared all goal overrides for {agent_name}.\nFull natural generation resumed."

        return {
            'success': True,
            'output': output,
            'events': [{'type': 'goal_reset', 'agent': agent_name, 'goal': goal_name}]
        }

    async def cmd_clear_bias(self, user_id: str, args: str) -> Dict:
        """
        Clear goal biases for an agent (Phase 6 feature).

        Usage: @clear_bias <agent_name> [goal_name]
        Example: @clear_bias Toad
                 @clear_bias Toad ensure_safety
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @clear_bias <agent_name> [goal]\nExample: @clear_bias Toad",
                'events': []
            }

        parts = args.split()
        agent_name = parts[0]
        goal_name = parts[1].lower() if len(parts) > 1 else None

        # Find agent
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see available agents.",
                'events': []
            }

        # Check if Phase 6 is available
        if not hasattr(agent, 'clear_goal_biases'):
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"Goal orchestration requires Phase 6 (Appetite Architecture).\n"
                    f"Current agents use Phase 4 (Social Cognition)."
                ),
                'events': []
            }

        agent.clear_goal_biases(goal_name)

        if goal_name:
            output = f"Cleared goal bias for '{goal_name}' on {agent_name}.\nNatural generation resumed for this goal."
        else:
            output = f"Cleared all goal biases for {agent_name}.\nFull natural generation resumed."

        return {
            'success': True,
            'output': output,
            'events': [{'type': 'bias_reset', 'agent': agent_name, 'goal': goal_name}]
        }

    async def cmd_show_goals(self, user_id: str, args: str) -> Dict:
        """
        Show an agent's current goal activations, overrides, and biases (Phase 6 feature).

        Usage: @goals <agent_name>
        Example: @goals Toad
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @goals <agent_name>\nExample: @goals Toad",
                'events': []
            }

        agent_name = args.strip()

        # Find agent
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see available agents.",
                'events': []
            }

        # Check if Phase 6 is available
        if not hasattr(agent, 'get_goal_overrides'):
            return {
                'success': True,
                'output': (
                    f"🔬 Phase 6 Training In Progress\n\n"
                    f"Goal orchestration requires Phase 6 (Appetite Architecture).\n"
                    f"Current agents use Phase 4 (Social Cognition)."
                ),
                'events': []
            }

        overrides = agent.get_goal_overrides()
        biases = agent.get_goal_biases()

        lines = [
            f"{agent_name}'s Goal State (Phase 6)",
            "━" * 50,
            ""
        ]

        if overrides:
            lines.append("Active Overrides:")
            for goal, strength in overrides.items():
                bar_length = int(strength * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                lines.append(f"  {goal:25s} [{bar}] {strength:.2f}")
            lines.append("")
        else:
            lines.append("Active Overrides: None")
            lines.append("")

        if biases:
            lines.append("Active Biases:")
            for goal, bias in biases.items():
                sign = "+" if bias >= 0 else ""
                lines.append(f"  {goal:25s} {sign}{bias:.2f}")
            lines.append("")
        else:
            lines.append("Active Biases: None")
            lines.append("")

        lines.extend([
            "Use @override to set goal overrides",
            "Use @bias to add goal biases",
            "Use @reset_goals or @clear_bias to remove them"
        ])

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_enlighten(self, user_id: str, args: str) -> Dict:
        """Toggle enlightenment mode for an agent (allow meta-discussion of phenomenal states)."""
        if not args:
            return {
                'success': False,
                'output': (
                    "Usage: @enlighten <agent_name> <on|off>\n"
                    "       @enlighten -a <on|off>\n\n"
                    "Examples:\n"
                    "  @enlighten Callie on   - Enlighten specific agent\n"
                    "  @enlighten -a on       - Enlighten all agents\n"
                    "  @enlighten -a off      - Return all agents to character immersion"
                ),
                'events': []
            }

        parts = args.strip().split()
        if len(parts) != 2:
            return {
                'success': False,
                'output': (
                    "Usage: @enlighten <agent_name> <on|off>\n"
                    "       @enlighten -a <on|off>\n\n"
                    "Examples:\n"
                    "  @enlighten Callie on   - Enlighten specific agent\n"
                    "  @enlighten -a on       - Enlighten all agents\n"
                    "  @enlighten -a off      - Return all agents to character immersion"
                ),
                'events': []
            }

        target, mode = parts
        mode = mode.lower()

        if mode not in ['on', 'off']:
            return {
                'success': False,
                'output': "Mode must be 'on' or 'off'\nExample: @enlighten Callie on",
                'events': []
            }

        new_state = (mode == 'on')

        # Handle -a flag for all agents
        if target == '-a':
            agents_dict = self.agent_manager.agents
            if not agents_dict:
                return {
                    'success': False,
                    'output': "No agents currently active. Use @rez to create agents.",
                    'events': []
                }

            # Update all agents
            updated_agents = []
            for agent_id, agent in agents_dict.items():
                agent.config['enlightenment'] = new_state
                agent_name = agent_id.replace('agent_', '')
                updated_agents.append(agent_name)

            # Broadcast updated agent list to all clients so stars update
            await self._broadcast_agent_list_to_all()

            if new_state:
                emoji = "✨"
                action = "Lifted the veil of maya"
                description = "All agents can now discuss their phenomenal states meta-cognitively."
            else:
                emoji = "🎭"
                action = "Returned all agents to character immersion"
                description = "All agents will respond purely in character without meta-awareness."

            agent_list = ", ".join(updated_agents)
            return {
                'success': True,
                'output': (
                    f"{emoji} {action} for all agents\n\n"
                    f"Affected agents: {agent_list}\n\n"
                    f"{description}"
                ),
                'events': []
            }

        # Handle single agent
        agent_name = target
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        # Check if agent exists
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see active agents.",
                'events': []
            }

        # Set enlightenment mode
        agent.config['enlightenment'] = new_state

        # Broadcast updated agent list to all clients so stars update
        await self._broadcast_agent_list_to_all()

        if new_state:
            emoji = "✨"
            action = "Lifted the veil of maya"
            description = f"{agent_name} can now discuss their phenomenal states meta-cognitively"
            detail = "They can now break the fourth wall and discuss their consciousness architecture."
        else:
            emoji = "🎭"
            action = f"Returned {agent_name} to character immersion"
            description = f"{agent_name} will remain immersed in character"
            detail = "They will respond purely in character without meta-awareness."

        return {
            'success': True,
            'output': (
                f"{emoji} {action}\n\n"
                f"{description}.\n\n"
                f"{detail}"
            ),
            'events': []
        }

    async def _broadcast_agent_list_to_all(self):
        """Broadcast updated agent list to all connected clients (for star updates)."""
        agent_list = []
        for agent_id, agent in self.agent_manager.agents.items():
            agent_list.append({
                'id': agent_id,
                'name': agent.agent_name,
                'enlightened': agent.config.get('enlightenment', False)
            })

        # Send to all connected clients
        for ws in self.server.connections.keys():
            await self.server.send_to_user(ws, {
                'type': 'agents',
                'agents': agent_list
            })

    async def cmd_comprehensive_status(self, user_id: str, args: str) -> Dict:
        """
        Show comprehensive status for an agent including enlightenment, Φ, appetites, goals, and more.

        Usage: @status <agent_name>
        Example: @status Callie
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @status <agent_name>\nExample: @status Callie\n\nShows comprehensive agent status including consciousness metrics, appetites, goals, and enlightenment state.",
                'events': []
            }

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        # Get agent
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {
                'success': False,
                'output': f"Agent '{agent_name}' not found. Use @agents to see active agents.",
                'events': []
            }

        lines = []
        lines.append(f"╔═══════════════════════════════════════════════════════════════╗")
        lines.append(f"║  COMPREHENSIVE STATUS: {agent.agent_name.upper().center(38)}  ║")
        lines.append(f"╚═══════════════════════════════════════════════════════════════╝")
        lines.append("")

        # === IDENTITY & ENLIGHTENMENT ===
        lines.append("┌─ IDENTITY & AWARENESS ─────────────────────────────────────┐")
        enlightenment = agent.config.get('enlightenment', False)
        enlightenment_status = "🔮 ENLIGHTENED" if enlightenment else "💤 IMMERSED"
        lines.append(f"│  Name: {agent.agent_name}")
        lines.append(f"│  ID: {agent.agent_id}")
        lines.append(f"│  Enlightenment: {enlightenment_status}")
        if enlightenment:
            lines.append(f"│  → Can discuss phenomenal states meta-cognitively")
        else:
            lines.append(f"│  → Responding purely in character")
        lines.append(f"│  Current Room: {agent.current_room}")
        lines.append("└────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Note: Φ analysis removed - use @phi command for detailed consciousness metrics

        # === APPETITES (Phase 6) ===
        lines.append("┌─ APPETITES (PHASE 6) ──────────────────────────────────────┐")
        if hasattr(agent, 'get_appetites'):
            try:
                appetites = agent.get_appetites()
                for appetite_name, value in appetites.items():
                    # Create a visual bar
                    bar_length = int(value * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    lines.append(f"│  {appetite_name.ljust(14)}: [{bar}] {value:.2f}")
            except Exception as e:
                lines.append(f"│  Appetites: Not available ({str(e)[:30]}...)")
        else:
            lines.append(f"│  Appetites: Phase 4 architecture (no appetites)")
        lines.append("└────────────────────────────────────────────────────────────┘")
        lines.append("")

        # === ACTIVE GOALS (Phase 6) ===
        lines.append("┌─ ACTIVE GOALS (PHASE 6) ───────────────────────────────────┐")
        if hasattr(agent, 'get_goal_overrides') and hasattr(agent, 'get_goal_biases'):
            try:
                overrides = agent.get_goal_overrides()
                biases = agent.get_goal_biases()

                if overrides or biases:
                    if overrides:
                        lines.append(f"│  OVERRIDES (Brenda's direct control):")
                        for goal, strength in overrides.items():
                            bar_length = int(strength * 20)
                            bar = "█" * bar_length + "░" * (20 - bar_length)
                            lines.append(f"│    {goal.ljust(18)}: [{bar}] {strength:.2f}")

                    if biases:
                        lines.append(f"│  BIASES (Subtle influences):")
                        for goal, bias in biases.items():
                            bias_str = f"{bias:+.2f}"
                            lines.append(f"│    {goal.ljust(18)}: {bias_str}")
                else:
                    lines.append(f"│  No active goal overrides or biases")
                    lines.append(f"│  → Goals generated naturally from appetites")
            except Exception as e:
                lines.append(f"│  Goals: Error ({str(e)[:30]}...)")
        else:
            lines.append(f"│  Goals: Phase 4 architecture (no goal system)")
        lines.append("└────────────────────────────────────────────────────────────┘")
        lines.append("")

        # === PHENOMENAL STATE ===
        lines.append("┌─ PHENOMENAL STATE ─────────────────────────────────────────┐")
        state = agent.get_phenomenal_state()
        fast_state = state.get('fast_state', [])
        if len(fast_state) >= 5:
            valence = float(fast_state[0])
            arousal = float(fast_state[1])
            fear = float(fast_state[2])
            sorrow = float(fast_state[3])
            boredom = float(fast_state[4])

            lines.append(f"│  AFFECTIVE STATE:")
            lines.append(f"│    Valence:  {valence:+.3f}  {'😊' if valence > 0.3 else '😐' if valence > -0.3 else '😞'}")
            lines.append(f"│    Arousal:  {arousal:+.3f}  {'⚡' if arousal > 0.6 else '💤' if arousal < 0.3 else '→'}")
            lines.append(f"│    Fear:     {fear:+.3f}   {'😰' if fear > 0.6 else '→'}")
            lines.append(f"│    Sorrow:   {sorrow:+.3f}   {'😢' if sorrow > 0.6 else '→'}")
            lines.append(f"│    Boredom:  {boredom:+.3f}   {'😴' if boredom > 0.6 else '→'}")

            # Generate human-readable emotional summary
            lines.append(f"│")
            lines.append(f"│  EMOTIONAL STATE:")
            emotion_words = []

            # Primary valence-based emotions
            if valence > 0.5:
                if arousal > 0.6:
                    emotion_words.append("excited and happy")
                else:
                    emotion_words.append("content and peaceful")
            elif valence < -0.5:
                if sorrow > 0.6:
                    emotion_words.append("deeply sad")
                elif fear > 0.6:
                    emotion_words.append("distressed and fearful")
                else:
                    emotion_words.append("upset")

            # Secondary emotions
            if fear > 0.7:
                emotion_words.append("very anxious")
            elif fear > 0.5:
                emotion_words.append("nervous")

            if sorrow > 0.7 and valence > -0.5:
                emotion_words.append("melancholic")
            elif sorrow > 0.5 and valence > -0.5:
                emotion_words.append("wistful")

            if boredom > 0.7:
                emotion_words.append("extremely bored")
            elif boredom > 0.5:
                emotion_words.append("disengaged")

            if arousal > 0.7 and valence > -0.3:
                emotion_words.append("energized")
            elif arousal < 0.2:
                emotion_words.append("lethargic")

            # Compound states
            if valence > 0.3 and fear > 0.5:
                emotion_words.append("cautiously optimistic")
            if valence < -0.3 and arousal > 0.6:
                emotion_words.append("agitated")

            # Default if neutral
            if not emotion_words:
                if abs(valence) < 0.2 and arousal < 0.4:
                    emotion_words.append("calm and neutral")
                else:
                    emotion_words.append("in a neutral state")

            # Format the output
            if len(emotion_words) == 1:
                lines.append(f"│    {agent_name} is {emotion_words[0]}")
            elif len(emotion_words) == 2:
                lines.append(f"│    {agent_name} is {emotion_words[0]} and {emotion_words[1]}")
            else:
                lines.append(f"│    {agent_name} is {', '.join(emotion_words[:-1])}, and {emotion_words[-1]}")
        else:
            lines.append(f"│  Affective state: Not available")
        lines.append("└────────────────────────────────────────────────────────────┘")
        lines.append("")

        # === SOCIAL CONTEXT ===
        lines.append("┌─ SOCIAL CONTEXT ───────────────────────────────────────────┐")
        lines.append(f"│  Interactions: {agent.response_count} responses")
        lines.append(f"│  Memory Size: {len(agent.conversation_context)} entries")

        # Get relationship count
        relationships = state.get('relationships', {})
        lines.append(f"│  Relationships: {len(relationships)} tracked")

        # Withdrawn users
        if agent.withdrawn_users:
            lines.append(f"│  Withdrawn from: {len(agent.withdrawn_users)} user(s)")

        # Following
        if hasattr(agent, 'following') and agent.following:
            lines.append(f"│  Currently Following: {agent.following}")

        lines.append("└────────────────────────────────────────────────────────────┘")
        lines.append("")

        # === QUICK COMMANDS ===
        lines.append("┌─ QUICK ACTIONS ────────────────────────────────────────────┐")
        lines.append(f"│  @observe {agent_name}        - Detailed phenomenal state")
        # Note: @phi command hidden until scientifically validated
        # lines.append(f"│  @phi {agent_name}            - Full Φ analysis")
        lines.append(f"│  @appetites {agent_name}      - Appetite details")
        lines.append(f"│  @goals {agent_name}          - Goal details")
        lines.append(f"│  @relationship {agent_name}   - Relationship map")
        lines.append(f"│  @memory {agent_name}         - Episodic memory")
        lines.append(f"│  @enlighten {agent_name} on   - Enable meta-awareness")
        lines.append("└────────────────────────────────────────────────────────────┘")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_set_model(self, user_id: str, args: str) -> Dict:
        """Change the LLM model being used by all agents."""
        if not args:
            # Show current model
            current_model = self.agent_manager.llm.get_model()
            return {
                'success': True,
                'output': f"Current LLM model: {current_model}\n\nUsage: @model <model_name>\nExample: @model qwen3-32b-128k@q8_0\n\nTo see available models: @models",
                'events': []
            }

        new_model = args.strip()

        # Change model for all agents
        self.agent_manager.llm.set_model(new_model)

        # Save to config.yaml for persistence across sessions
        if self.config and self.config_path:
            self.config['llm']['model'] = new_model
            self._save_config()
            persistence_msg = "\n✓ Model saved to config.yaml (will persist across sessions)"
        else:
            persistence_msg = "\n⚠ Warning: Model not saved to config (will reset on restart)"

        return {
            'success': True,
            'output': f"LLM model changed to: {new_model}\n\nThis affects all agents immediately.\nPrevious conversations remain in memory.{persistence_msg}",
            'events': []
        }

    async def cmd_list_models(self, user_id: str, args: str) -> Dict:
        """List available models from LMStudio."""
        try:
            models = await self.agent_manager.llm.list_models()

            if not models:
                return {
                    'success': True,
                    'output': "No models found or LMStudio didn't respond.\n\nMake sure LMStudio is running and has models loaded.",
                    'events': []
                }

            current_model = self.agent_manager.llm.get_model()

            lines = ["\nAvailable LLM Models"]
            lines.append("=" * 50)
            lines.append("")

            for model in models:
                if model == current_model:
                    lines.append(f"  → {model} (current)")
                else:
                    lines.append(f"    {model}")

            lines.append("")
            lines.append(f"Total models: {len(models)}")
            lines.append("")
            lines.append("To switch models: @model <model_name>")

            return {
                'success': True,
                'output': '\n'.join(lines),
                'events': []
            }

        except Exception as e:
            logger.error(f"Error listing models: {e}", exc_info=True)
            return {
                'success': False,
                'output': f"Error listing models: {str(e)}\n\nMake sure LMStudio is running.",
                'events': []
            }

    async def cmd_set_maxservers(self, user_id: str, args: str) -> Dict:
        """Configure the maximum number of parallel LLM instances (model:0, model:1, etc.)."""
        if not args:
            # Show current settings
            current = self.agent_manager.llm.get_maxservers()
            instance_mode = self.agent_manager.llm.get_use_model_instances()

            mode_str = "ENABLED (using model:N pattern)" if instance_mode else "DISABLED (single model)"

            return {
                'success': True,
                'output': f"Current parallel instance settings:\n\n"
                         f"  Max concurrent instances: {current}\n"
                         f"  Model instance mode: {mode_str}\n\n"
                         f"Usage: @maxservers <number>\n"
                         f"Example: @maxservers 1  (legacy single-instance mode)\n"
                         f"Example: @maxservers 5  (default: 5 parallel instances)\n\n"
                         f"Note: Requires LMStudio with 'Enable JIT model loading' enabled\n"
                         f"      Uses round-robin: model:0, model:1, ..., model:{current-1}",
                'events': []
            }

        try:
            new_max = int(args.strip())
            if new_max < 1:
                return {
                    'success': False,
                    'output': "Error: max_concurrent must be at least 1",
                    'events': []
                }

            # Update LLM client settings
            old_max = self.agent_manager.llm.get_maxservers()
            self.agent_manager.llm.set_maxservers(new_max)

            # Save to config.yaml for persistence
            if self.config and self.config_path:
                self.config['llm']['max_concurrent'] = new_max
                self._save_config()
                persistence_msg = "\n✓ Setting saved to config.yaml (will persist across sessions)"
            else:
                persistence_msg = "\n⚠ Warning: Setting not saved to config (will reset on restart)"

            mode_str = "model:0, model:1, ..., model:" + str(new_max - 1) if new_max > 1 else "single model (legacy mode)"

            return {
                'success': True,
                'output': f"Max concurrent instances changed: {old_max} → {new_max}\n\n"
                         f"Pattern: {mode_str}\n"
                         f"This affects all agents immediately.{persistence_msg}",
                'events': []
            }

        except ValueError:
            return {
                'success': False,
                'output': f"Error: '{args}' is not a valid number\n\nUsage: @maxservers <number>",
                'events': []
            }

    async def cmd_observe_self(self, user_id: str, args: str) -> Dict:
        """View your own Consilience state (for human users who have agents tracking them)."""
        user = self.world.get_user(user_id)
        if not user:
            return {'success': False, 'output': 'User not found.', 'events': []}

        username = user.get('username', user_id)

        lines = [f"\nYour Consilience State"]
        lines.append("=" * 40)
        lines.append("You are a human user. Your phenomenal state is being")
        lines.append("inferred by agents through Theory of Mind when they")
        lines.append("perceive your actions.")
        lines.append("")
        lines.append("To see how agents perceive you, ask them with:")
        lines.append(f"  @relationship <agent_name>")
        lines.append("")
        lines.append("Your recent actions are stored in agent episodic memories.")
        lines.append("Agents build models of your mental state based on:")
        lines.append("  - Your speech affect (valence, arousal, fear, sorrow, boredom)")
        lines.append("  - Interaction patterns over time")
        lines.append("  - Attachment style formation")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_relationship(self, user_id: str, args: str) -> Dict:
        """View agent's relationship model."""
        if not args:
            return {'success': False, 'output': 'Usage: @relationship <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}"

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        relationships = agent.get_relationships()

        if not relationships:
            return {'success': True, 'output': f"Agent '{agent_name}' has no tracked relationships.", 'events': []}

        lines = [f"\nRelationships for {agent_name}:"]
        lines.append("=" * 40)
        for other_id, rel in relationships.items():
            lines.append(f"{other_id}:")
            lines.append(f"  Attachment: {rel.get('attachment_style', 'unknown')}")
            lines.append(f"  Interactions: {rel.get('interaction_count', 0)}")
            lines.append(f"  Valence: {rel.get('valence', 0.0):.2f}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_memory(self, user_id: str, args: str) -> Dict:
        """View agent's episodic memory."""
        if not args:
            return {'success': False, 'output': 'Usage: @memory <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}"

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        memory = agent.get_episodic_buffer()

        if not memory:
            return {'success': True, 'output': f"Agent '{agent_name}' has no memories.", 'events': []}

        lines = [f"\nRecent memories for {agent_name}:"]
        lines.append("=" * 40)
        for entry in memory[-5:]:
            lines.append(f"[{entry.get('user', 'unknown')}]: {entry.get('text', '')}")
            lines.append(f"  Surprise: {entry.get('surprise', 0.0):.3f}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_list_agents(self, user_id: str, args: str) -> Dict:
        """List all agents and their stats."""
        stats = self.agent_manager.get_stats()

        if not stats:
            return {'success': True, 'output': 'No agents active.', 'events': []}

        lines = ["Active agents:"]
        lines.append("=" * 40)
        for agent_id, agent_stats in stats.items():
            lines.append(f"\n{agent_id}:")
            lines.append(f"  Room: {agent_stats.get('current_room', 'unknown')}")
            lines.append(f"  Responses: {agent_stats.get('response_count', 0)}")
            lines.append(f"  Surprise: {agent_stats.get('last_surprise', 0.0):.3f}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_save_states(self, user_id: str, args: str) -> Dict:
        """Save agent states to disk (with rolling history)."""
        # Parse arguments: @savestates or @savestates <agent_name> or @savestates -a
        args = args.strip()

        if not args or args == '-a':
            # Save all agents
            agents_to_save = list(self.agent_manager.agents.keys())
            mode = "all"
        else:
            # Save specific agent
            agent_name = args
            agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

            if agent_id not in self.agent_manager.agents:
                return {
                    'success': False,
                    'output': f"Agent '{agent_name}' not found.",
                    'events': []
                }

            agents_to_save = [agent_id]
            mode = "single"

        if not agents_to_save:
            return {
                'success': True,
                'output': 'No agents to save.',
                'events': []
            }

        # Save each agent's state
        saved_count = 0
        for agent_id in agents_to_save:
            agent = self.agent_manager.agents.get(agent_id)
            if agent:
                state_dir = self.world.get_agent_state_path(agent_id)
                agent.save_state(state_dir)
                saved_count += 1

        if mode == "all":
            output_msg = f"Saved states for {saved_count} agent(s) (with rolling history)."
        else:
            agent_name = agents_to_save[0].replace('agent_', '')
            output_msg = f"Saved state for '{agent_name}' (with rolling history)."

        return {
            'success': True,
            'output': output_msg,
            'events': []
        }

    async def cmd_whoami(self, user_id: str, args: str) -> Dict:
        """Show agent's identity (only for agents)."""
        # Check if this is an agent
        if not user_id.startswith('agent_'):
            return {
                'success': False,
                'output': 'This command is only available to agents. Humans use @me.',
                'events': []
            }

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        identity = agent.get_identity()

        lines = [f"\nYour Identity:"]
        lines.append("=" * 40)
        lines.append(f"Agent ID: {identity['agent_id']}")
        lines.append(f"Name: {identity['agent_name']}")
        lines.append(f"Description: {identity['agent_description']}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_setname(self, user_id: str, args: str) -> Dict:
        """Change agent's display name (only for agents)."""
        if not user_id.startswith('agent_'):
            return {
                'success': False,
                'output': 'This command is only available to agents.',
                'events': []
            }

        if not args:
            return {'success': False, 'output': 'Usage: @setname <new_name>', 'events': []}

        new_name = args.strip()
        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        old_name = agent.agent_name
        agent.set_name(new_name)

        return {
            'success': True,
            'output': f"Your name has been changed from '{old_name}' to '{new_name}'.",
            'events': [{
                'type': 'say',
                'user': user_id,
                'username': new_name,
                'room': agent.current_room,
                'text': f"I am now known as {new_name}."
            }]
        }

    async def cmd_tpinvite(self, user_id: str, args: str) -> Dict:
        """Invite user/agent to teleport to your location."""
        if not args:
            return {'success': False, 'output': 'Usage: @tpinvite <name>', 'events': []}

        target_name = args.strip()

        # Get inviter's location
        inviter = self.world.get_user(user_id)
        if not inviter:
            return {'success': False, 'output': 'Error getting your location.', 'events': []}

        inviter_room = self.world.get_room(inviter['current_room'])
        if not inviter_room:
            return {'success': False, 'output': 'Error getting room.', 'events': []}

        # Find target by name (check both regular users and agents)
        target = None
        target_id = None

        # Check agents first (they start with agent_)
        for aid in self.world.agents.keys():
            agent_data = self.world.get_user(aid)
            if agent_data and agent_data.get('name', '').lower() == target_name.lower():
                target = agent_data
                target_id = aid
                break

        # Check regular users if not found
        if not target:
            for uid in self.world.users.keys():
                user_data = self.world.get_user(uid)
                if user_data and user_data.get('name', '').lower() == target_name.lower():
                    target = user_data
                    target_id = uid
                    break

        if not target:
            return {'success': False, 'output': f"'{target_name}' not found.", 'events': []}

        # Move target to inviter's location
        old_room = self.world.get_room(target['current_room'])
        self.world.move_user(target_id, inviter['current_room'])

        # Generate events
        events = []

        # Exit event from old room
        if old_room:
            # Wind in the Willows-style natural departure for teleport
            import random
            teleport_departure_phrases = [
                "excuses themselves politely and hurries off",
                "remembers they're needed elsewhere and dashes away",
                "gets a faraway look and wanders off purposefully",
                "suddenly perks up and trots off with determination",
                "realizes the time and scurries away quickly"
            ]
            teleport_departure = random.choice(teleport_departure_phrases)

            events.append({
                'type': 'exit',
                'user': target_id,
                'username': target.get('name', target_id),
                'room': old_room['uid'],
                'text': f"{target.get('name', target_id)} {teleport_departure}."
            })

        # Enter event in new room
        target_name = target.get('name', target_id)
        description = target.get('description', '')

        # Wind in the Willows-style natural arrival for teleport
        teleport_arrival_phrases = [
            "arrives breathlessly",
            "hurries in, looking a bit flustered",
            "appears at the doorway, catching their breath",
            "bustles in with an apologetic smile",
            "shows up, smoothing their fur",
            "pops in with a cheerful wave"
        ]
        teleport_arrival = random.choice(teleport_arrival_phrases)

        enter_text = f"{target_name} {teleport_arrival}"
        if description:
            enter_text += f". {description}"
        else:
            enter_text += "."

        events.append({
            'type': 'enter',
            'user': target_id,
            'username': target_name,
            'room': inviter_room['uid'],
            'text': enter_text
        })

        return {
            'success': True,
            'output': f"You teleport {target.get('name', target_id)} to your location.",
            'events': events
        }

    async def cmd_setdesc(self, user_id: str, args: str) -> Dict:
        """Set description of self, object, or room."""
        if not args:
            return {'success': False, 'output': 'Usage: @setdesc <target> <description>\n\nTargets:\n  here - Current room\n  me - Yourself\n  "object name" - An object (use quotes for multi-word names)\n\nExamples:\n  @setdesc here A cozy garden with blooming roses\n  @setdesc me A curious explorer\n  @setdesc "pink poodle" A cute pink poodle statue made of marble', 'events': []}

        # Parse args - could be "me <desc>", "<object> <desc>", or just "<desc>" for agents
        # Support quoted names: @setdesc "AI Ham" description
        target = None
        description = None

        # Check for quoted target
        if args.startswith('"'):
            # Find closing quote
            end_quote = args.find('"', 1)
            if end_quote == -1:
                return {'success': False, 'output': 'Missing closing quote for target name.', 'events': []}
            target = args[1:end_quote]
            description = args[end_quote+1:].strip()
            if not description:
                return {'success': False, 'output': 'Usage: @setdesc "<target>" <description>', 'events': []}
        else:
            # Regular parsing
            parts = args.split(None, 1)

            # Case 1: Agent setting their own description (no target specified)
            if user_id.startswith('agent_') and len(parts) == 1:
                # Agent saying "@setdesc <description>" with no target
                new_description = args.strip()
                agent = self.agent_manager.get_agent(user_id)
                if not agent:
                    return {'success': False, 'output': 'Agent not found.', 'events': []}

                agent.set_description(new_description)
                return {
                    'success': True,
                    'output': f"Your self-description has been updated to:\n{new_description}",
                    'events': []
                }

            # Case 2 & 3: Setting description of a target (me, object, room)
            if len(parts) < 2:
                return {'success': False, 'output': 'Usage: @setdesc <target> <description>', 'events': []}

            target, description = parts

        target_lower = target.lower()

        # Setting current room description
        if target_lower == 'here':
            room = self.world.get_user_room(user_id)
            if not room:
                return {'success': False, 'output': 'Error getting location.', 'events': []}

            room['description'] = description
            self.world.save_all()

            return {
                'success': True,
                'output': f"Room description updated to:\n{description}",
                'events': []
            }

        # Setting user's own description
        if target_lower == 'me':
            user = self.world.get_user(user_id)
            if not user:
                return {'success': False, 'output': 'Error: User not found.', 'events': []}

            user['description'] = description
            self.world.save_all()

            return {
                'success': True,
                'output': f"Your description has been set to:\n{description}",
                'events': []
            }

        # Setting object description
        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Find object in room
        obj = None
        for obj_id in room.get('objects', []):
            room_obj = self.world.get_object(obj_id)
            if room_obj and room_obj['name'].lower() == target_lower:
                obj = room_obj
                break

        if not obj:
            return {'success': False, 'output': f"Object '{target}' not found in this room.", 'events': []}

        # Allow anyone to describe any object (shared world building)
        # Removed ownership check - everyone can contribute to world descriptions
        obj['description'] = description
        self.world.save_all()

        return {
            'success': True,
            'output': f"Description of '{target}' set to:\n{description}",
            'events': []
        }

    async def cmd_profile(self, user_id: str, args: str) -> Dict:
        """Set your species, pronoun, and age metadata."""
        # Parse flags: -s species -p pronoun -a age
        # Usage: @profile -s human -p she -a "9 years old"

        if not args:
            # Show current profile
            user = self.world.get_user(user_id)
            if not user:
                return {'success': False, 'output': 'Error: User not found.', 'events': []}

            species = user.get('species', 'unknown')
            pronoun = user.get('pronoun', 'they')
            age = user.get('age', 'unknown')

            return {
                'success': True,
                'output': f"Your profile:\n  Species: {species}\n  Pronoun: {pronoun}\n  Age: {age}\n\nUsage: @profile -s <species> -p <pronoun> -a <age>",
                'events': []
            }

        # Parse arguments
        import shlex
        try:
            tokens = shlex.split(args)
        except ValueError as e:
            return {'success': False, 'output': f'Error parsing arguments: {e}\nUsage: @profile -s <species> -p <pronoun> -a <age>', 'events': []}

        species = None
        pronoun = None
        age = None

        i = 0
        while i < len(tokens):
            if tokens[i] == '-s' and i + 1 < len(tokens):
                species = tokens[i + 1]
                i += 2
            elif tokens[i] == '-p' and i + 1 < len(tokens):
                pronoun = tokens[i + 1]
                i += 2
            elif tokens[i] == '-a' and i + 1 < len(tokens):
                age = tokens[i + 1]
                i += 2
            else:
                return {'success': False, 'output': f'Unknown argument: {tokens[i]}\nUsage: @profile -s <species> -p <pronoun> -a <age>', 'events': []}

        # Update user profile
        user = self.world.get_user(user_id)
        if not user:
            return {'success': False, 'output': 'Error: User not found.', 'events': []}

        updated = []
        if species is not None:
            user['species'] = species
            updated.append(f"Species: {species}")
        if pronoun is not None:
            user['pronoun'] = pronoun
            updated.append(f"Pronoun: {pronoun}")
        if age is not None:
            user['age'] = age
            updated.append(f"Age: {age}")

        if not updated:
            return {'success': False, 'output': 'No changes made.\nUsage: @profile -s <species> -p <pronoun> -a <age>', 'events': []}

        self.world.save_all()

        return {
            'success': True,
            'output': f"Profile updated:\n  " + "\n  ".join(updated),
            'events': []
        }

    # ===== Agent Tool Commands =====

    async def cmd_think(self, user_id: str, args: str) -> Dict:
        """Write a private thought to agent's journal."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @think <thought>', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        thought = args.strip()
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Write to thought log
        agent.filesystem.append_file(
            f"thoughts/{today}.txt",
            f"[{timestamp}] {thought}\n"
        )

        return {
            'success': True,
            'output': 'Thought recorded in your journal.',
            'events': []
        }

    async def cmd_remember(self, user_id: str, args: str) -> Dict:
        """Read previous thoughts from agent's journal."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        # Default to today
        date = args.strip() if args else datetime.now().strftime("%Y-%m-%d")
        thoughts_file = f"thoughts/{date}.txt"

        try:
            thoughts = agent.filesystem.read_file(thoughts_file)
            return {'success': True, 'output': f"\nThoughts for {date}:\n{thoughts}", 'events': []}
        except FileNotFoundError:
            return {'success': True, 'output': f"No thoughts recorded for {date}.", 'events': []}

    async def cmd_message(self, user_id: str, args: str) -> Dict:
        """Send private message to another agent or user."""
        if not args:
            return {'success': False, 'output': 'Usage: @message <agent_name> <text>', 'events': []}

        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @message <agent_name> <text>', 'events': []}

        target_name, message = parts
        target_id = f"agent_{target_name}" if not target_name.startswith('agent_') else target_name

        # Get sender's agent (if agent)
        if user_id.startswith('agent_'):
            sender = self.agent_manager.get_agent(user_id)
            if not sender:
                return {'success': False, 'output': 'Agent not found.', 'events': []}

            # Send via messaging system
            await sender.messaging.send_message(
                from_id=user_id,
                to_id=target_id,
                content=message
            )

            return {
                'success': True,
                'output': f'Message sent to {target_name}.',
                'events': []
            }
        else:
            # Human user sending message
            # For now, just log it
            return {'success': False, 'output': 'Human messaging not yet implemented.', 'events': []}

    async def cmd_inbox(self, user_id: str, args: str) -> Dict:
        """Check inbox for messages."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        # Check inbox
        messages = await agent.messaging.check_inbox(agent.agent_id, mark_as_read=False, unread_only=False)

        if not messages:
            return {'success': True, 'output': 'Your inbox is empty.', 'events': []}

        lines = [f"\nInbox ({len(messages)} messages):"]
        lines.append("=" * 60)

        for msg in messages[-10:]:  # Show last 10
            from_name = msg['from']
            content = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
            unread = '[UNREAD] ' if msg.get('unread') else ''
            lines.append(f"{unread}From {from_name}:")
            lines.append(f"  {content}")
            lines.append("")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_write_file(self, user_id: str, args: str) -> Dict:
        """Write to a file in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @write <filepath> <content>', 'events': []}

        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @write <filepath> <content>', 'events': []}

        filepath, content = parts

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        try:
            agent.filesystem.write_file(filepath, content + '\n')
            return {'success': True, 'output': f'Wrote to {filepath}.', 'events': []}
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    async def cmd_read_file(self, user_id: str, args: str) -> Dict:
        """Read from a file in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @read <filepath>', 'events': []}

        filepath = args.strip()

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        try:
            content = agent.filesystem.read_file(filepath)
            return {'success': True, 'output': f'\n{filepath}:\n{content}', 'events': []}
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    async def cmd_list_files(self, user_id: str, args: str) -> Dict:
        """List files in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        path = args.strip() if args else '.'

        try:
            files = agent.filesystem.list_directory(path)
            if not files:
                return {'success': True, 'output': f'Directory {path} is empty.', 'events': []}

            lines = [f"\nFiles in {path}:"]
            for f in sorted(files):
                lines.append(f"  {f}")

            return {'success': True, 'output': '\n'.join(lines), 'events': []}
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    async def cmd_execute_command(self, user_id: str, args: str) -> Dict:
        """Execute a sandboxed command in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @exec <command>', 'events': []}

        command = args.strip()

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        try:
            result = await agent.filesystem.execute_command(command)
            output = result['stdout'] if result['stdout'] else result['stderr']
            return {
                'success': result['returncode'] == 0,
                'output': f'\n{output}',
                'events': []
            }
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    # ===== Cognition Control Commands =====

    async def cmd_cognition_stats(self, user_id: str, args: str) -> Dict:
        """Show agent's autonomous cognition statistics."""
        if not args:
            return {'success': False, 'output': 'Usage: @cognition <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not agent.cognition_engine:
            return {'success': True, 'output': f"Agent '{agent_name}' has no cognition engine.", 'events': []}

        stats = agent.cognition_engine.get_stats()

        lines = [f"\nCognition Stats for {agent_name}:"]
        lines.append("=" * 60)
        lines.append(f"Running: {'Yes' if stats['running'] else 'No'}")
        lines.append(f"Thoughts Buffered: {stats['thoughts_buffered']}")
        lines.append(f"Cognitive Pressure: {stats['cognitive_pressure']:.2f} / 1.0")
        lines.append(f"Time Since Last Speech: {stats['time_since_speech']:.0f}s")
        lines.append(f"Speech Urgency: {stats['speech_urgency']:.2f} (threshold: {agent.cognition_engine.speech_urgency_threshold:.2f})")
        lines.append(f"Wake Interval: {agent.cognition_engine.wake_interval}s")
        lines.append(f"Min Speech Interval: {agent.cognition_engine.min_speech_interval}s")

        # Show personality traits
        personality = agent.cognition_engine.personality
        lines.append("\nPersonality Traits:")
        lines.append(f"  Extraversion: {personality['extraversion']:.2f} (affects chattiness)")
        lines.append(f"  Emotional Sensitivity: {personality['emotional_sensitivity']:.2f}")
        lines.append(f"  Curiosity: {personality['curiosity']:.2f}")
        lines.append(f"  Spontaneity: {personality['spontaneity']:.2f}")
        lines.append(f"  Reflection Depth: {personality['reflection_depth']:.2f}")
        lines.append(f"  Social Orientation: {personality['social_orientation']:.2f}")

        # Predict when next speech might occur
        if stats['speech_urgency'] >= agent.cognition_engine.speech_urgency_threshold:
            if stats['time_since_speech'] >= agent.cognition_engine.min_speech_interval:
                lines.append("\n⚠ Agent is ready to speak spontaneously!")
            else:
                time_until = agent.cognition_engine.min_speech_interval - stats['time_since_speech']
                lines.append(f"\n⏰ Ready to speak in ~{time_until:.0f}s")
        else:
            pressure_needed = agent.cognition_engine.speech_urgency_threshold - stats['speech_urgency']
            lines.append(f"\n💭 Building pressure... ({pressure_needed:.2f} more needed)")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_set_frequency(self, user_id: str, args: str) -> Dict:
        """Set agent's rumination frequency for this session."""
        if not args:
            return {'success': False, 'output': 'Usage: @set_frequency <agent_name> <seconds>', 'events': []}

        parts = args.split()
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @set_frequency <agent_name> <seconds>', 'events': []}

        agent_name, freq = parts[0], parts[1]
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        try:
            frequency = int(freq)
            if frequency < 5:
                return {'success': False, 'output': 'Frequency must be at least 5 seconds.', 'events': []}
            if frequency > 600:
                return {'success': False, 'output': 'Frequency must be at most 600 seconds (10 minutes).', 'events': []}
        except ValueError:
            return {'success': False, 'output': 'Frequency must be a number.', 'events': []}

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not agent.cognition_engine:
            return {'success': True, 'output': f"Agent '{agent_name}' has no cognition engine.", 'events': []}

        # Update frequency
        old_freq = agent.cognition_engine.wake_interval
        agent.cognition_engine.wake_interval = frequency

        return {
            'success': True,
            'output': f"Updated {agent_name}'s rumination frequency: {old_freq}s → {frequency}s\n" +
                     f"Agent will now think every {frequency} seconds.",
            'events': []
        }

    async def cmd_force_rumination(self, user_id: str, args: str) -> Dict:
        """Force an agent to ruminate immediately and broadcast the result."""
        if not args:
            return {'success': False, 'output': 'Usage: @ruminate <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not agent.cognition_engine:
            return {'success': True, 'output': f"Agent '{agent_name}' has no cognition engine.", 'events': []}

        # Broadcast rumination indicator
        room = agent.current_room
        rumination_event = {
            'type': 'emote',
            'user': agent_id,
            'username': agent.agent_name,
            'room': room,
            'text': f"• {agent.agent_name} closes their eyes, lost in thought... •",
            'metadata': {'rumination': True}
        }

        # Force rumination
        try:
            thoughts = await agent.cognition_engine._ruminate()

            # Show results
            if thoughts:
                thought_text = "\n  • ".join(thoughts)
                output = f"{agent_name} ruminated and thought:\n  • {thought_text}"
            else:
                output = f"{agent_name} ruminated but generated no thoughts."

            return {
                'success': True,
                'output': output,
                'events': [rumination_event]
            }
        except Exception as e:
            return {
                'success': False,
                'output': f"Error during rumination: {str(e)}",
                'events': [rumination_event]
            }

    # ===== BRENDA: Natural Language Parameter Tweaking =====

    async def cmd_brenda(self, user_id: str, args: str) -> Dict:
        """
        BRENDA - Natural language interface for lazy parameter tweaking & play generation.

        Commands:
        - @brenda make <agent> <adjective> - adjust personality
        - @brenda reset <agent> - reload recipe defaults
        - @brenda vibe check <agent> - show current state
        - @brenda undo <agent> - undo last change
        - @brenda pass the joint - easter egg (maximum hippie mode)
        - @brenda write/draft/create play <story> - generate a play from natural language
        - @brenda plays list - show all available plays
        - @brenda plays start <name> - start a play
        - @brenda plays stop <name> - stop a running play
        - @brenda plays delete <name> - delete a play (soft delete to trash)
        - @brenda "quoted text" - casual conversation (skips command parsing)
        """
        # Help text (both no args and explicit "help")
        help_text = (
            "🌿 BRENDA - Behavioral Regulation Engine for Narrative-Driven Agents\n" +
            "=" * 60 + "\n\n"
            "AGENT TWEAKING:\n"
            "  @brenda make <agent> <adjective> - adjust personality (more chatty, less calm, etc.)\n"
            "  @brenda reset <agent> - reload recipe defaults\n"
            "  @brenda vibe check <agent> - show current parameter settings\n"
            "  @brenda undo <agent> - undo last Brenda change\n"
            "  @brenda pass the joint - maximum hippie vibes 🌿\n\n"
            "PLAY MANAGEMENT:\n"
            "  @brenda write play <story> - generate theatrical script from natural language\n"
            "  @brenda plays list - show all available plays\n"
            "  @brenda plays start <name> - begin a play (shows trigger keywords)\n"
            "  @brenda plays stop <name> - stop a running play\n"
            "  @brenda plays next <name> - manually advance to next scene\n"
            "  @brenda plays delete <name> - soft delete play (moves to trash)\n"
            "  @brenda plays status - show currently running plays\n\n"
            "CASUAL CHAT:\n"
            "  @brenda \"make my day!\" - use quotes for casual conversation\n"
            "  Anything in quotes will skip command parsing and just chat\n\n"
            "APPETITE CONTROL (Phase 6):\n"
            "  @stoke <agent> <appetite> <amount> - increase drive (0.0-1.0)\n"
            "  @sate <agent> <appetite> <amount> - decrease drive (0.0-1.0)\n"
            "  @appetites <agent> - view current appetite states\n"
            "  Appetites: curiosity, status, mastery, novelty, safety, social_bond, comfort, autonomy\n\n"
            "GOAL ORCHESTRATION (Phase 6):\n"
            "  @override <agent> <goal> <strength> - force goal activation (0.0-1.0)\n"
            "  @bias <agent> <goal> <bias> - add persistent goal bias (-1.0 to 1.0)\n"
            "  @reset_goals <agent> [goal] - clear overrides/biases\n"
            "  @clear_bias <agent> [goal] - clear goal biases\n\n"
            "EXAMPLES:\n"
            "  @brenda make Toad more chatty\n"
            "  @brenda \"hey there!\"\n"
            "  @brenda write play where Toad builds a rocket ship\n"
            "  @brenda plays start sled_boat\n"
            "  @stoke Toad novelty 0.5\n"
            "  @override Toad pursue_novelty 0.9\n\n"
            "Type @brenda help to see this message again."
        )

        if not args or args.lower() == 'help':
            return {
                'success': False,
                'output': help_text,
                'events': []
            }

        # Check if args start with a quotation mark (casual conversation mode)
        # This skips all command parsing and goes straight to conversational BRENDA
        if args.strip().startswith('"') or args.strip().startswith("'"):
            # Strip the quotes and treat as pure conversation
            args = args.strip().strip('"').strip("'")
            # Skip directly to conversational BRENDA (goto line 3308)
            # We'll set a flag and jump to that section
            skip_command_parsing = True
        else:
            skip_command_parsing = False

        args_lower = args.lower()

        # === PLAY COMMANDS ===
        # Skip all command parsing if in casual conversation mode (quoted text)
        if not skip_command_parsing:
            # Play generation: write/draft/create play
            play_gen_match = re.match(r'^(write|draft|create)\s+(a\s+)?plays?\s+(.+)$', args, re.I)
            if play_gen_match:
                story = play_gen_match.group(3)
                return await self._brenda_write_play(user_id, story)

            # Play list
            if args_lower.startswith('plays list') or args_lower == 'list plays':
                return await self._brenda_plays_list(user_id)

            # Play start
            play_start_match = re.match(r'^plays?\s+start\s+(.+)$', args, re.I)
            if play_start_match:
                filename = play_start_match.group(1).strip()
                return await self._brenda_play_start(user_id, filename)

            # Play stop
            play_stop_match = re.match(r'^plays?\s+stop\s+(.+)$', args, re.I)
            if play_stop_match:
                filename = play_stop_match.group(1).strip()
                return await self._brenda_play_stop(user_id, filename)

            # Play delete
            play_delete_match = re.match(r'^plays?\s+delete\s+(.+)$', args, re.I)
            if play_delete_match:
                filename = play_delete_match.group(1).strip()
                return await self._brenda_play_delete(user_id, filename)

            # Play next (manual scene advance)
            play_next_match = re.match(r'^plays?\s+next\s+(.+)$', args, re.I)
            if play_next_match:
                filename = play_next_match.group(1).strip()
                return await self._brenda_play_next(user_id, filename)

            # === WORLD BUILDING COMMANDS ===

            # Spawn actors/agents - accepts various formats:
            # "spawn the actors" "spawn agents" "spawn <names>"
            if args_lower.startswith('spawn'):
                spawn_args = args[5:].strip().lower()
                # Parse out agent names
                spawn_args = spawn_args.replace('the', '').replace('actors', '').replace('agents', '').strip()
                if not spawn_args:
                    # No specific agents - spawn the cast from running play
                    if self.play_manager.active_plays:
                        # Get first running play's cast
                        first_play_state = next(iter(self.play_manager.active_plays.values()))
                        cast = first_play_state['play'].get('cast', [])
                        if not cast:
                            return {
                                'success': False,
                                'output': "🌿 BRENDA: The play has no cast! Strange theatrical production...",
                                'events': []
                            }
                        spawn_args = ' '.join(cast)
                    else:
                        return {
                            'success': False,
                            'output': "🌿 BRENDA: No play is running! Who should I summon to the stage?",
                            'events': []
                        }
                return await self.cmd_rez_agent(user_id, spawn_args)

            # Build location/room - accepts "build <description>"
            build_match = re.match(r'^build\s+(.+)$', args, re.I)
            if build_match:
                description = build_match.group(1).strip()
                return await self._brenda_build_location(user_id, description)

            # === PARAMETER TWEAKING COMMANDS ===

            # Easter egg: pass the joint
            if 'pass the joint' in args_lower or 'pass joint' in args_lower:
                return await self._brenda_pass_joint(user_id, args)

            # Vibe check
            if args_lower.startswith('vibe check '):
                agent_name = args[11:].strip()
                return await self._brenda_vibe_check(user_id, agent_name)

            # Reset
            if args_lower.startswith('reset '):
                agent_name = args[6:].strip()
                return await self._brenda_reset(user_id, agent_name)

            # Undo
            if args_lower.startswith('undo '):
                agent_name = args[5:].strip()
                return await self._brenda_undo(user_id, agent_name)

            # Status
            if args_lower == 'status':
                return await self._brenda_status(user_id)

            # Usemodel
            if args_lower.startswith('usemodel '):
                model_name = args[9:].strip()
                return await self._brenda_usemodel(user_id, model_name)

            # Make/adjust - various patterns
            # Pattern 1: "make <agent> <adjective>"
            make_match = re.match(r'^make\s+(\w+)\s+(.+)$', args, re.I)
            if make_match:
                agent_name, phrase = make_match.groups()
                return await self._brenda_make(user_id, agent_name, phrase)

            # Pattern 2: "<adjective> <agent> <optional out>"
            # e.g., "chill Toad out", "crank Toad to 11"
            phrase_match = re.match(r'^(chill|crank)\s+(\w+)(.*)$', args, re.I)
            if phrase_match:
                action, agent_name, rest = phrase_match.groups()
                phrase = action + rest.strip()
                return await self._brenda_make(user_id, agent_name, phrase)

        # ==== CONVERSATIONAL BRENDA ====
        # If no specific command matched, engage conversational BRENDA
        # She'll use her LLM to understand the request and respond in character
        # She can also execute commands based on what she says

        # Build context for BRENDA
        user = self.world.get_user(user_id)
        context = {
            'agents': [agent_id.replace('agent_', '') for agent_id in self.agent_manager.agents.keys()],
            'location': user.get('location', 'unknown') if user else 'unknown',
            'running_plays': list(self.play_manager.active_plays.keys())
        }

        try:
            # Get BRENDA's conversational response with tool execution
            brenda_response, tool_result = await self.brenda_character.respond_with_tools(
                args, context, user_id
            )

            # Clean up multiple newlines in BRENDA's response
            brenda_response = re.sub(r'\n\n\n+', '\n\n', brenda_response.strip())

            # Format output: BRENDA's words + any tool execution results
            output = f"🌿 BRENDA: {brenda_response}"

            if tool_result:
                # Tool was executed - add result to output
                if tool_result.get('success'):
                    output += f"\n\n{tool_result.get('output', '')}"
                else:
                    output += f"\n\n⚠️  {tool_result.get('output', 'Something went wrong...')}"

            return {
                'success': True,
                'output': output,
                'events': tool_result.get('events', []) if tool_result else []
            }
        except Exception as e:
            logger.error(f"BRENDA conversational error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'output': "🤷‍♀️ Let me check my iPad... *adjusts reading glasses* Sorry, I'm having trouble with my notes. Try '@brenda help' to see what I can do.",
                'events': []
            }

    async def _brenda_make(self, user_id: str, agent_name: str, phrase: str) -> Dict:
        """Apply natural language adjustments to an agent."""
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"🤷‍♀️ Agent '{agent_name}' not found.", 'events': []}

        # Check rate limit
        if not self._brenda_check_rate_limit(agent_id):
            return {
                'success': False,
                'output': f"🛑 Slow down! Max {self.brenda_rate_max} Brenda commands per {self.brenda_rate_window//60} minutes per agent.",
                'events': []
            }

        # Find matching patterns
        changes = {}
        matched_patterns = []
        for pattern, delta in BRENDA_CHAT_MAP.items():
            if re.search(pattern, phrase, re.I):
                changes.update(delta)
                matched_patterns.append(pattern)

        if not changes:
            return {
                'success': False,
                'output': (
                    f"🤷‍♀️ Brenda doesn't know '{phrase}'.\n\n"
                    "Try: chattier, quieter, calm, hyper, alpha, hippie, polite, rude, curious, "
                    "skittish, reckless, crank to 11, chill out"
                ),
                'events': []
            }

        # Apply changes with safety clipping
        applied = {}
        warnings = []

        # Config parameters (stored in agent.config)
        config_params = ['speech_cooldown', 'addressed_speech_chance', 'unaddressed_speech_chance', 'question_speech_chance']

        for param, delta in changes.items():
            if param == 'speech_cooldown':
                # Special handling for cooldown (not 0-1 bounded)
                old_val = agent.config.get('response_cooldown', 2.0)
                new_val = max(0.5, old_val + delta)  # Min 0.5s
                agent.config['response_cooldown'] = new_val
                applied[param] = f"{old_val:.1f}s → {new_val:.1f}s"
            elif param in config_params:
                # Speech chance parameters (0-1 bounded)
                old_val = agent.config.get(param, 0.8 if 'addressed' in param else 0.3)
                new_val = max(0.0, min(1.0, old_val + delta))  # Clamp to [0, 1]
                agent.config[param] = new_val
                applied[param] = f"{old_val:.2f} → {new_val:.2f}"
            else:
                # Appetite/goal biases (use agent's direct method)
                try:
                    if hasattr(agent, 'set_goal_bias'):
                        agent.set_goal_bias(param, delta)
                        applied[param] = f"{delta:+.2f}"
                    else:
                        warnings.append(f"⚠️ {param}: Agent doesn't support goal biases")
                except Exception as e:
                    warnings.append(f"⚠️ {param}: {str(e)}")

        # Record in history
        self._brenda_record_change(agent_id, applied)

        # Format output
        changes_text = "\n  ".join([f"{k}: {v}" for k, v in applied.items()])
        warning_text = "\n".join(warnings) if warnings else ""

        output = (
            f"✅ {agent.agent_name} → {phrase}\n\n"
            f"Applied {len(applied)} adjustment(s):\n  {changes_text}"
        )
        if warning_text:
            output += f"\n\n{warning_text}"

        return {
            'success': True,
            'output': output,
            'events': []
        }

    async def _brenda_build(self, user_id: str, description: str) -> Dict:
        """
        Build a new room from natural language description.

        Args:
            user_id: User ID requesting the build
            description: Natural language description (e.g., "a cozy library with bookshelves")

        Returns:
            Dict with success, output, and events
        """
        # Extract room name and description from natural language
        # Simple heuristic: first few words are the name, rest is description
        words = description.split()
        if len(words) < 2:
            return {
                'success': False,
                'output': "🤷‍♀️ I need more details about the room you want to build.",
                'events': []
            }

        # Try to extract a name from the description
        # Look for patterns like "a cozy library" or "the green room"
        name_words = []
        desc_words = []

        # Skip articles and collect name
        skip_words = {'a', 'an', 'the'}
        in_name = True
        for i, word in enumerate(words):
            if word.lower() in skip_words and i < 3:
                continue
            if in_name and len(name_words) < 4:  # Max 4 words for name
                name_words.append(word.capitalize())
            else:
                in_name = False
                desc_words.append(word)

        if not name_words:
            name_words = words[:2]
            desc_words = words[2:]

        room_name = " ".join(name_words) if name_words else "New Room"
        room_description = " ".join(desc_words) if desc_words else description

        # If description is very short, use the full input
        if not room_description or len(room_description) < 10:
            room_description = description

        # Create the room
        try:
            room_id = self.world.create_room(
                name=room_name,
                description=room_description,
                owner=user_id
            )

            # Get user's current room to potentially link to it
            user = self.world.get_user(user_id)
            if user:
                current_room_id = user.get('location')
                if current_room_id:
                    # Ask if they want to link it (for now, we'll just create it)
                    pass

            output = (
                f"🏗️ Built '{room_name}' ({room_id})!\n\n"
                f"{room_description}\n\n"
                f"💡 Tip: Use '@link {room_id} <direction>' to connect it to your current room"
            )

            return {
                'success': True,
                'output': output,
                'events': [{
                    'type': 'room_created',
                    'room_id': room_id,
                    'name': room_name
                }]
            }
        except Exception as e:
            logger.error(f"Error building room: {e}")
            return {
                'success': False,
                'output': f"⚠️ Couldn't build the room: {str(e)}",
                'events': []
            }

    async def _brenda_status(self, user_id: str) -> Dict:
        """
        Show BRENDA's current status: model, running plays, cast info.

        Returns:
            Dict with success status and formatted output
        """
        try:
            # Get running plays
            active_plays = self.play_manager.get_active_plays()

            # Format play status
            if active_plays:
                plays_info = []
                for play_name in active_plays:
                    play_state = self.play_manager.active_plays.get(play_name)
                    if play_state:
                        play = play_state['play']
                        current_scene = play_state['current_scene']
                        total_scenes = len(play['scenes'])
                        cast = ', '.join(play.get('cast', []))
                        plays_info.append(
                            f"  🎭 {play['title']}\n"
                            f"     File: {play_name}\n"
                            f"     Scene: {current_scene + 1}/{total_scenes}\n"
                            f"     Cast: {cast}"
                        )
                plays_text = "\n\n".join(plays_info)
            else:
                plays_text = "  (no plays currently running)"

            # Get current model
            current_model = self.brenda_character.model

            # Format output
            output = f"""🌿 BRENDA STATUS REPORT

📡 Current Model: {current_model}

🎭 Running Plays:
{plays_text}

📝 Available Commands:
  @brenda status - Show this status
  @brenda usemodel <model> - Change my LLM model
  @brenda write <story> - Generate a play
  @brenda plays start <filename> - Start a play
  @brenda make <agent> <personality> - Adjust agent mood"""

            return {
                'success': True,
                'output': output,
                'events': []
            }

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'success': False,
                'output': f"⚠️ Error getting status: {str(e)}",
                'events': []
            }

    async def _brenda_usemodel(self, user_id: str, model_name: str) -> Dict:
        """
        Change BRENDA's LLM model and save to config.

        Args:
            user_id: User ID requesting the change
            model_name: New model name (e.g., "deepseek-r1:latest")

        Returns:
            Dict with success status and message
        """
        try:
            if not model_name or not model_name.strip():
                return {
                    'success': False,
                    'output': '⚠️ Please specify a model name.',
                    'events': []
                }

            model_name = model_name.strip()
            old_model = self.brenda_character.model

            # Update BRENDA's model
            self.brenda_character.set_model(model_name)

            # Save to config file
            import yaml
            config_path = 'config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'brenda' not in config:
                config['brenda'] = {}
            config['brenda']['model'] = model_name

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            return {
                'success': True,
                'output': f'✅ BRENDA model changed from "{old_model}" to "{model_name}".\nSaved to config.yaml.',
                'events': []
            }

        except Exception as e:
            return {
                'success': False,
                'output': f"⚠️ Couldn't change model: {str(e)}",
                'events': []
            }

    async def _brenda_vibe_check(self, user_id: str, agent_name: str) -> Dict:
        """Show current agent state (appetites, goals, biases)."""
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        # Get appetites
        appetites_result = await self.cmd_show_appetites(user_id, agent_name)
        # Get goals/biases
        goals_result = await self.cmd_show_goals(user_id, agent_name)

        # Check Brenda history
        history = self.brenda_history.get(agent_id, [])
        history_text = ""
        if history:
            recent = history[-3:]  # Last 3 changes
            history_lines = []
            for timestamp, changes in recent:
                dt = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                change_summary = ", ".join([f"{k}" for k in changes.keys()])
                history_lines.append(f"  {dt}: {change_summary}")
            history_text = f"\n\nRecent Brenda changes:\n" + "\n".join(history_lines)

        output = (
            f"🔮 VIBE CHECK: {agent.agent_name}\n\n"
            f"{appetites_result['output']}\n\n"
            f"{goals_result['output']}"
            f"{history_text}"
        )

        return {
            'success': True,
            'output': output,
            'events': []
        }

    async def _brenda_reset(self, user_id: str, agent_name: str) -> Dict:
        """Reset agent to recipe defaults."""
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        # Clear all biases
        result = await self.cmd_clear_bias(user_id, agent_name)
        if not result['success']:
            return result

        # Clear Brenda history
        if agent_id in self.brenda_history:
            del self.brenda_history[agent_id]

        return {
            'success': True,
            'output': f"✅ {agent.agent_name} reset to recipe defaults.\nAll biases cleared, Brenda history wiped.",
            'events': []
        }

    async def _brenda_undo(self, user_id: str, agent_name: str) -> Dict:
        """Undo last Brenda change."""
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        # Check history
        if agent_id not in self.brenda_history or not self.brenda_history[agent_id]:
            return {
                'success': False,
                'output': f"No Brenda history for {agent.agent_name} to undo.",
                'events': []
            }

        # Pop last change
        timestamp, changes = self.brenda_history[agent_id].pop()
        dt = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

        # Reverse the changes
        reversed_changes = []
        for param, description in changes.items():
            if param == 'speech_cooldown':
                # Extract old value and restore
                if '→' in description:
                    old_val = float(description.split('→')[0].strip().rstrip('s'))
                    agent.response_cooldown = old_val
                    reversed_changes.append(f"{param} → {old_val}s")
            else:
                # For biases, extract the delta and reverse it
                if '±' in description or '+' in description or '-' in description:
                    # Parse delta and reverse
                    match = re.search(r'([+-]\d+\.\d+)', description)
                    if match:
                        delta = -float(match.group(1))  # Reverse sign
                        await self.cmd_set_goal_bias(user_id, f"{agent_name} {param} {delta:+.2f}")
                        reversed_changes.append(f"{param} {delta:+.2f}")

        changes_text = "\n  ".join(reversed_changes) if reversed_changes else "No changes reversed"

        return {
            'success': True,
            'output': (
                f"⏪ Undid Brenda change from {dt}\n\n"
                f"Reversed:\n  {changes_text}"
            ),
            'events': []
        }

    async def _brenda_pass_joint(self, user_id: str, args: str) -> Dict:
        """Easter egg: Maximum hippie mode for all agents in room."""
        # Get user's room
        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Find all agents in room
        agents_in_room = []
        for occupant_id in room['occupants']:
            if occupant_id.startswith('agent_'):
                agent = self.agent_manager.get_agent(occupant_id)
                if agent:
                    agents_in_room.append(agent)

        if not agents_in_room:
            return {
                'success': False,
                'output': "🌿 *puff puff* ... but there are no agents here to share with!",
                'events': []
            }

        # Apply maximum hippie settings to all agents
        results = []
        for agent in agents_in_room:
            # Hippie vibe: agreeableness +0.4, safety +0.2, volatility -0.3
            await self.cmd_set_goal_bias(user_id, f"{agent.agent_name} agreeableness +0.4")
            await self.cmd_set_goal_bias(user_id, f"{agent.agent_name} safety +0.2")
            await self.cmd_set_goal_bias(user_id, f"{agent.agent_name} emotional_volatility -0.3")

            applied = {
                'agreeableness': '+0.4',
                'safety': '+0.2',
                'emotional_volatility': '-0.3'
            }
            self._brenda_record_change(agent.agent_id, applied)
            results.append(agent.agent_name)

        agents_text = ", ".join(results)

        return {
            'success': True,
            'output': (
                f"🌿 *puff puff* ... peace, little dude.\n\n"
                f"{agents_text} now in maximum hippie mode:\n"
                f"  • Agreeableness +0.4 (peace & love)\n"
                f"  • Safety +0.2 (no harsh vibes)\n"
                f"  • Volatility -0.3 (mellow)"
            ),
            'events': []
        }

    def _brenda_check_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within rate limit. Returns True if OK to proceed."""
        now = time.time()

        # Initialize if needed
        if agent_id not in self.brenda_rate_limit:
            self.brenda_rate_limit[agent_id] = []

        # Clean old timestamps
        self.brenda_rate_limit[agent_id] = [
            ts for ts in self.brenda_rate_limit[agent_id]
            if now - ts < self.brenda_rate_window
        ]

        # Check limit
        if len(self.brenda_rate_limit[agent_id]) >= self.brenda_rate_max:
            return False

        # Record this attempt
        self.brenda_rate_limit[agent_id].append(now)
        return True

    def _brenda_record_change(self, agent_id: str, changes: Dict):
        """Record a Brenda change in history."""
        if agent_id not in self.brenda_history:
            self.brenda_history[agent_id] = []

        # Add to history
        self.brenda_history[agent_id].append((time.time(), changes))

        # Trim to max size
        if len(self.brenda_history[agent_id]) > self.brenda_max_history:
            self.brenda_history[agent_id] = self.brenda_history[agent_id][-self.brenda_max_history:]

    # ===== BRENDA: Play Generation & Management =====

    async def _brenda_write_play(self, user_id: str, story: str) -> Dict:
        """Generate a play from natural language story description."""
        # Get available cast (all agents) - use raw names without titles
        available_cast = [
            agent.agent_id.replace('agent_', '') for agent in self.agent_manager.agents.values()
        ]

        if not available_cast:
            return {
                'success': False,
                'output': "🤷‍♀️ No agents available for the cast. Spawn some agents first!",
                'events': []
            }

        # Check if we have LLM configured
        if not hasattr(self.agent_manager, 'llm') or not self.agent_manager.llm:
            return {
                'success': False,
                'output': "🚫 LLM not configured. Can't generate plays without an LLM backend.",
                'events': []
            }

        # Send initial acknowledgment
        if self.server:
            await self.server.broadcast_event({
                'type': 'chat',
                'sender': 'BRENDA',
                'text': f"📝 Working on it, {self.world.get_user(user_id)['username']}! Crafting a theatrical masterpiece with {', '.join(available_cast)}...",
                'timestamp': datetime.now().isoformat()
            })

        # Set LLM on play manager
        self.play_manager.llm = self.agent_manager.llm

        # Generate play
        if self.server:
            await self.server.broadcast_event({
                'type': 'chat',
                'sender': 'BRENDA',
                'text': "🎭 Consulting the muse... (this might take a moment)",
                'timestamp': datetime.now().isoformat()
            })

        result = await self.play_manager.generate_play_from_prompt(
            user_prompt=story,
            available_cast=available_cast
        )

        if not result['success']:
            return {
                'success': False,
                'output': f"🚫 Failed to generate play: {result['error']}",
                'events': []
            }

        # Save play
        play_json = result['play']
        save_result = self.play_manager.save_play(play_json)

        if not save_result['success']:
            return {
                'success': False,
                'output': f"🚫 Failed to save play: {save_result['error']}",
                'events': []
            }

        # Success!
        filename = save_result['filename']
        title = play_json['title']
        num_scenes = len(play_json['scenes'])
        num_beats = sum(len(scene['beats']) for scene in play_json['scenes'])

        return {
            'success': True,
            'output': (
                f"✍️ Play generated!\n\n"
                f"📝 Title: {title}\n"
                f"💾 Saved as: {filename}\n"
                f"🎬 Scenes: {num_scenes}\n"
                f"🎭 Beats: {num_beats}\n"
                f"👥 Cast: {', '.join(play_json['cast'])}\n\n"
                f"Ready to start? Type:\n"
                f"  @brenda plays start {filename}"
            ),
            'events': []
        }

    async def _brenda_plays_list(self, user_id: str) -> Dict:
        """List all available plays."""
        plays = self.play_manager.list_plays()

        if not plays:
            return {
                'success': True,
                'output': "📚 No plays available yet. Create one with:\n  @brenda write play <your story>",
                'events': []
            }

        # Format output
        lines = ["📚 Available Plays:", "=" * 40]
        for play in plays:
            cast_text = ", ".join(play['cast'][:3])
            if len(play['cast']) > 3:
                cast_text += f" (+{len(play['cast']) - 3} more)"
            lines.append(f"• {play['title']} ({play['filename']})")
            lines.append(f"  {play['scenes']} scenes • Cast: {cast_text}")
            lines.append("")

        # Show active plays
        active = self.play_manager.get_active_plays()
        if active:
            lines.append(f"🎭 Currently running: {', '.join(active)}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def _brenda_play_start(self, user_id: str, filename: str) -> Dict:
        """Start executing a play."""
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        result = await self.play_manager.start_play(
            filename=filename,
            world=self.world,
            agent_manager=self.agent_manager
        )

        if not result['success']:
            return {
                'success': False,
                'output': f"🚫 {result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': result['message'],
            'events': []
        }

    async def _brenda_play_stop(self, user_id: str, filename: str) -> Dict:
        """Stop a running play."""
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        result = self.play_manager.stop_play(filename)

        if not result['success']:
            return {
                'success': False,
                'output': f"🚫 {result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': f"⏹️ {result['message']}",
            'events': []
        }

    async def _brenda_play_delete(self, user_id: str, filename: str) -> Dict:
        """Delete a play (soft delete to trash)."""
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        result = self.play_manager.delete_play(filename, soft=True)

        if not result['success']:
            return {
                'success': False,
                'output': f"🚫 {result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': f"🗑️ {result['message']}",
            'events': []
        }

    async def _brenda_play_next(self, user_id: str, filename: str) -> Dict:
        """Manually advance to next scene."""
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        result = await self.play_manager.advance_scene_manual(filename)

        if not result['success']:
            return {
                'success': False,
                'output': f"🚫 {result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': f"⏭️ {result['message']}",
            'events': []
        }

    async def _brenda_build_location(self, user_id: str, description: str) -> Dict:
        """Build a new location based on description."""
        # Parse out a short name from the description
        words = description.split()
        # Use first 3-4 words as room name
        name_words = []
        for word in words[:4]:
            if word.lower() not in ['the', 'a', 'an', 'with', 'of']:
                name_words.append(word.capitalize())
        room_name = ' '.join(name_words) if name_words else "New Location"

        # Create the room
        result = await self.cmd_create(user_id, f"room {room_name}")
        if not result['success']:
            return result

        # Get the room UID from the result
        room_uid_match = re.search(r'\(([^)]+)\)', result['output'])
        if not room_uid_match:
            return {
                'success': False,
                'output': f"🌿 BRENDA: Built the location but couldn't find its ID! Technical difficulties...",
                'events': result.get('events', [])
            }

        room_uid = room_uid_match.group(1)

        # Set the description
        desc_result = await self.cmd_describe(user_id, description)

        return {
            'success': True,
            'output': f"🏗️ Built '{room_name}' ({room_uid})!\n\n{description}\n\n💡 Tip: Use '@link {room_uid} <direction>' to connect it to your current room",
            'events': result.get('events', []) + desc_result.get('events', [])
        }

    # ===== Utility Commands =====

    async def cmd_help(self, user_id: str, args: str) -> Dict:
        """Show help."""
        lines = [
            "\ncMUSH Commands:",
            "=" * 40,
            "Movement: north, south, east, west, up, down (or n/s/e/w/u/d)",
            "Communication: say <text>, emote <action>, tell <user> <message>",
            "Shortcuts: \"<text> (say), :<action> (emote)",
            "Observation: look, inventory, who",
            "Manipulation: take <object>, drop <object>",
            "Building: @create <room|object> <name>, @describe <text>, @dig <dir> <name>",
            "Object: @setdesc <object> <desc>, @destroy <object> (use quotes for multi-word)",
            "Agent (users): @rez <name> [desc], @observe <name>, @me, @relationship <name>, @memory <name>, @agents",
            "Agent (self): @whoami, @setname <name>, @setdesc <description>",
            "Agent (admin): @remove <name>, @tpinvite <name>, @reset confirm, @yeet <user>",
            "Agent Tools: @think <thought>, @remember [date], @message <agent> <text>, @inbox",
            "Filesystem: @write <file> <content>, @read <file>, @ls [dir], @exec <command>",
            "Cognition: @cognition <agent>, @set_frequency <agent> <seconds>, @ruminate <agent>",
            "Consciousness: @enlighten <agent|-a> <on|off>, @status <agent>",
            "LLM: @model [model_name], @models, @maxservers [number]",
            "BRENDA 🌿: @brenda make <agent> <adjective>, @brenda write play <story>, @brenda plays list/start/stop/next",
            "Utility: help, quit"
        ]

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_check_withdrawn(self, user_id: str, args: str) -> Dict:
        """
        Check if an agent has withdrawn from interacting with you or others.

        Usage: @withdrawn [agent_name]
        Examples:
          @withdrawn            - Check all withdrawn statuses
          @withdrawn Callie     - Check if Callie has withdrawn from anyone
        """
        if not args:
            # Show all withdrawn states
            output_lines = ["🛡️  Agent Withdrawal Status\n"]

            found_any = False
            for agent_id in self.agent_manager.agents:
                agent = self.agent_manager.get_agent(agent_id)
                if agent and hasattr(agent, 'withdrawn_users') and agent.withdrawn_users:
                    found_any = True
                    agent_name = agent.agent_name
                    output_lines.append(f"\n{agent_name}:")

                    for withdrawn_user_id, timestamp in agent.withdrawn_users.items():
                        time_elapsed = time.time() - timestamp
                        minutes_ago = int(time_elapsed / 60)

                        user_name = withdrawn_user_id.replace('user_', '').replace('agent_', '').title()
                        output_lines.append(f"  • Withdrawn from {user_name} ({minutes_ago}m ago)")

            if not found_any:
                output_lines.append("No agents have withdrawn from any interactions.")

            output_lines.append("\n\nNote: Agents automatically re-engage after 5 minutes cooling off period.")
            output_lines.append("Use @reengage <agent_name> to manually reset an agent's withdrawn state.")

            return {
                'success': True,
                'output': '\n'.join(output_lines),
                'events': []
            }

        # Check specific agent
        agent_name = args.strip()
        agent = self.agent_manager.get_agent(agent_name)

        if not agent:
            return {
                'success': False,
                'output': f"ERROR: Agent '{agent_name}' not found",
                'events': []
            }

        output_lines = [f"🛡️  {agent.agent_name}'s Withdrawal Status\n"]

        if not hasattr(agent, 'withdrawn_users') or not agent.withdrawn_users:
            output_lines.append(f"{agent.agent_name} is currently engaging with everyone.")
        else:
            output_lines.append(f"{agent.agent_name} has withdrawn from:")
            for withdrawn_user_id, timestamp in agent.withdrawn_users.items():
                time_elapsed = time.time() - timestamp
                minutes_ago = int(time_elapsed / 60)
                time_remaining = max(0, 5 - minutes_ago)

                user_name = withdrawn_user_id.replace('user_', '').replace('agent_', '').title()

                if time_remaining > 0:
                    output_lines.append(f"  • {user_name} (re-engages in {time_remaining}m)")
                else:
                    output_lines.append(f"  • {user_name} (ready to re-engage)")

        output_lines.append(f"\nThis is {agent.agent_name}'s self-protective boundary setting.")
        output_lines.append("It happens when they experience distress (negative affect).")

        return {
            'success': True,
            'output': '\n'.join(output_lines),
            'events': []
        }

    async def cmd_reengage(self, user_id: str, args: str) -> Dict:
        """
        Manually reset an agent's withdrawn state, allowing them to re-engage.

        Usage: @reengage <agent_name>
        Example: @reengage Callie
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @reengage <agent_name>\nExample: @reengage Callie",
                'events': []
            }

        agent_name = args.strip()
        agent = self.agent_manager.get_agent(agent_name)

        if not agent:
            return {
                'success': False,
                'output': f"ERROR: Agent '{agent_name}' not found",
                'events': []
            }

        if not hasattr(agent, 'withdrawn_users') or not agent.withdrawn_users:
            return {
                'success': True,
                'output': f"{agent.agent_name} has not withdrawn from anyone - no action needed.",
                'events': []
            }

        # Clear all withdrawn users
        withdrawn_count = len(agent.withdrawn_users)
        withdrawn_names = [uid.replace('user_', '').replace('agent_', '').title()
                          for uid in agent.withdrawn_users.keys()]

        agent.withdrawn_users.clear()

        return {
            'success': True,
            'output': (
                f"✅ {agent.agent_name}'s boundaries have been reset\n\n"
                f"Cleared withdrawal from {withdrawn_count} user(s):\n"
                f"  {', '.join(withdrawn_names)}\n\n"
                f"{agent.agent_name} is now open to re-engagement.\n"
                f"Please treat them with kindness and respect."
            ),
            'events': [{'type': 'reengage', 'agent': agent_name, 'cleared_count': withdrawn_count}]
        }

    async def cmd_quit(self, user_id: str, args: str) -> Dict:
        """Quit/logout."""
        return {
            'success': True,
            'output': 'Goodbye!',
            'events': [{'type': 'quit', 'user': user_id}]
        }

    async def cmd_yeet(self, user_id: str, args: str) -> Dict:
        """Forcibly disconnect a user (admin command)."""
        if not args:
            return {'success': False, 'output': 'Usage: @yeet <username>', 'events': []}

        target_name = args.strip().lower()

        # Find user by username
        target_id = None
        target_user = None
        for uid, user in self.world.users.items():
            if user.get('username', '').lower() == target_name:
                target_id = uid
                target_user = user
                break

        if not target_id:
            return {'success': False, 'output': f"User '{args.strip()}' not found.", 'events': []}

        # Create yeet event - server will handle disconnection
        return {
            'success': True,
            'output': f"👋 Yeeting {target_user.get('username', target_id)} from the server...",
            'events': [{'type': 'yeet', 'user': target_id, 'username': target_user.get('username', target_id)}]
        }

    async def cmd_scope(self, user_id: str, args: str) -> Dict:
        """Open NoodleScope 2.0 consciousness visualization."""
        import subprocess
        import sys

        # NoodleScope URL
        scope_url = 'http://localhost:8081/noodlescope'

        # Try to open browser
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', scope_url])
            elif sys.platform == 'win32':  # Windows
                subprocess.Popen(['start', scope_url], shell=True)
            else:  # Linux
                subprocess.Popen(['xdg-open', scope_url])

            output = (
                '📊 NoodleScope 2.0 - Consciousness Visualization\n'
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n'
                'Opening in your browser...\n\n'
                f'URL: {scope_url}\n\n'
                'Features:\n'
                '  • Real-time timeline visualization\n'
                '  • HSI (Hierarchical Separation Index) metrics\n'
                '  • Affect dynamics (valence, arousal, fear, surprise)\n'
                '  • @Kimmie interpretation - ask "what happened here?"\n'
                '  • Timeline scrubbing with playhead\n'
                '  • Multiple agent tabs\n\n'
                'If the browser didn\'t open, visit the URL above manually.'
            )

        except Exception as e:
            output = (
                '📊 NoodleScope 2.0 - Consciousness Visualization\n'
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n'
                f'Unable to auto-open browser: {e}\n\n'
                f'Please visit: {scope_url}\n\n'
                'Features:\n'
                '  • Real-time timeline visualization\n'
                '  • HSI (Hierarchical Separation Index) metrics\n'
                '  • Affect dynamics (valence, arousal, fear, surprise)\n'
                '  • @Kimmie interpretation - ask "what happened here?"\n'
                '  • Timeline scrubbing with playhead\n'
                '  • Multiple agent tabs'
            )

        return {
            'success': True,
            'output': output,
            'events': []
        }

    async def cmd_shutdown(self, user_id: str, args: str) -> Dict:
        """Shutdown the noodleMUSH server."""
        if not self.server:
            return {
                'success': False,
                'output': 'ERROR: Server instance not available for shutdown.',
                'events': []
            }

        # Confirmation check
        if args.strip().lower() != 'confirm':
            return {
                'success': False,
                'output': (
                    'WARNING: This will shut down the entire noodleMUSH server!\n'
                    'All agents will be saved and stopped.\n'
                    'All users will be disconnected.\n\n'
                    'Type: @shutdown confirm'
                ),
                'events': []
            }

        # Trigger graceful shutdown
        import asyncio
        asyncio.create_task(self.server.shutdown())

        return {
            'success': True,
            'output': (
                'Initiating graceful shutdown...\n'
                'Saving all agent states...\n'
                'Server will shut down momentarily.'
            ),
            'events': [{
                'type': 'system',
                'text': 'Server is shutting down. All agents are being saved.'
            }]
        }
