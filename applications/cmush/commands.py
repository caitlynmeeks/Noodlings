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
#   Command Parser
#
#   When you type something like "say Hello!" or "@rez chester"
#   in noodleMUSH, this module figures out what you're trying to
#   do and makes it happen.
#
#   Commands are organized into categories:
#   - Moving around (go north, enter cafe)
#   - Talking (say, whisper, emote)
#   - Looking at things (look, examine)
#   - Working with objects (take, drop, give)
#   - Building the world (create rooms, link exits)
#   - Managing Noodlings (rez, reset, configure)
#
#   Each category lives in its own "mixin" file so this main
#   file doesn't get too overwhelming.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands
# PURPOSE:  Parse user input and dispatch to command handlers
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CommandParser    Routes commands to appropriate handlers
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Command Parser for cMUSH

Handles all user commands organized into mixins:
- World: Movement, Communication, Observation, Manipulation
- Building: Room/object creation
- Agent: Agent lifecycle (rez, remove, reset)
- Orchestration: Phase 6 appetites and goals
- Consciousness: Enlightenment, status
- LLM: Model control
- User: Profile commands
- Tools: Agent filesystem/messaging
- Cognition: Cognition control
- Utility: Help, quit, shutdown

Author: Caitlyn + Claude
Date: October 2025 (Refactored December 2025)
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
import time
import re
import json

from recipe_loader import RecipeLoader
from fuzzy_match import find_best_matches, disambiguate_matches, format_disambiguation_prompt
from brenda_commands import BrendaCommandsMixin

# Import command mixins
from commands_world import WorldCommandsMixin
from commands_building import BuildingCommandsMixin
from commands_agent import AgentCommandsMixin
from commands_orchestration import OrchestrationCommandsMixin
from commands_consciousness import ConsciousnessCommandsMixin
from commands_llm import LLMCommandsMixin
from commands_user import UserCommandsMixin
from commands_tools import ToolsCommandsMixin
from commands_cognition import CognitionCommandsMixin
from commands_utility import UtilityCommandsMixin

logger = logging.getLogger(__name__)


class CommandParser(
    WorldCommandsMixin,
    BuildingCommandsMixin,
    AgentCommandsMixin,
    OrchestrationCommandsMixin,
    ConsciousnessCommandsMixin,
    LLMCommandsMixin,
    UserCommandsMixin,
    ToolsCommandsMixin,
    CognitionCommandsMixin,
    UtilityCommandsMixin,
    BrendaCommandsMixin
):
    """
    Parse and execute cMUSH commands.

    Commands are parsed from user input and executed against
    the world state, returning formatted output for the user.
    """

    def __init__(self, world, agent_manager, server=None, config=None, config_path=None, script_manager=None):
        """
        Initialize command parser.

        Args:
            world: World state manager
            agent_manager: Agent manager instance
            server: Server instance (for shutdown command)
            config: Server config dict (for saving changes)
            config_path: Path to config.yaml (for persistence)
            script_manager: Script manager instance (for scripting system)
        """
        self.world = world
        self.agent_manager = agent_manager
        self.server = server
        self.config = config
        self.config_path = config_path
        self.script_manager = script_manager
        self.recipe_loader = RecipeLoader("recipes")

        # Initialize BRENDA subsystem (from BrendaCommandsMixin)
        self._init_brenda(config)

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
            '@derez': self.cmd_remove,
            '@reset': self.cmd_reset,
            '@tpinvite': self.cmd_tpinvite,

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

            # Consciousness status
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

            # Lab system: Double-blind affect testing
            '@lab': self.cmd_lab,

            # Utility
            'help': self.cmd_help,
            'quit': self.cmd_quit,
            'logout': self.cmd_quit,
            '@yeet': self.cmd_yeet,
            '@shutdown': self.cmd_shutdown
        }

    def _resolve_entity(
        self,
        query: str,
        room_id: str,
        include_objects: bool = True,
        include_agents: bool = True,
        include_users: bool = True
    ) -> Tuple[Optional[str], Optional[str], Optional[List[Tuple[str, str]]]]:
        """
        Resolve entity name using fuzzy matching.

        Args:
            query: User's search term (e.g., "red", "_fire_", "anklebiter")
            room_id: Room to search in
            include_objects: Search objects
            include_agents: Search agents
            include_users: Search users

        Returns:
            (entity_id, entity_type, ambiguous_matches)
            - If clear match: (id, type, None)
            - If ambiguous: (None, None, [(id, name, score), ...])
            - If no match: (None, None, None)
        """
        room = self.world.rooms.get(room_id)
        if not room:
            return (None, None, None)

        candidates = []

        # Collect occupants (agents + users)
        if include_agents or include_users:
            for occ_id in room['occupants']:
                occ = self.world.get_user(occ_id)
                if occ:
                    is_agent = occ_id.startswith('agent_')
                    if (is_agent and include_agents) or (not is_agent and include_users):
                        name = occ.get('username', occ.get('name', occ_id))
                        candidates.append((occ_id, name))

        # Collect objects
        if include_objects:
            for obj_id in room.get('objects', []):
                obj = self.world.objects.get(obj_id)
                if obj:
                    candidates.append((obj_id, obj['name']))

        # Fuzzy match
        matches = find_best_matches(query, candidates, threshold=0.3)

        if not matches:
            return (None, None, None)

        # Check if we need disambiguation
        entity_id = disambiguate_matches(matches)

        if entity_id:
            # Clear match
            entity_type = 'agent' if entity_id.startswith('agent_') else 'object' if entity_id.startswith('obj_') else 'user'
            return (entity_id, entity_type, None)
        else:
            # Ambiguous - return all matches for user to choose
            return (None, None, matches)

    async def parse_and_execute(
        self,
        user_id: str,
        command_text: str
    ) -> Dict:
        """
        Parse and execute a command (or compound commands separated by semicolons).

        Args:
            user_id: User executing command
            command_text: Command string (supports ; delimiter for multiple commands)

        Returns:
            Response dict with:
                - success: bool
                - output: str (formatted text for user)
                - events: list (events to broadcast)
        """
        if not command_text.strip():
            return {'success': False, 'output': '', 'events': []}

        # Check for compound commands (semicolon-separated)
        if ';' in command_text:
            commands = [cmd.strip() for cmd in command_text.split(';') if cmd.strip()]

            # Execute each command sequentially
            all_outputs = []
            all_events = []
            all_success = True

            for cmd in commands:
                result = await self._execute_single_command(user_id, cmd)
                if result['output']:
                    all_outputs.append(result['output'])
                all_events.extend(result.get('events', []))
                if not result.get('success', True):
                    all_success = False

            return {
                'success': all_success,
                'output': '\n'.join(all_outputs),
                'events': all_events
            }

        # Single command - execute directly
        return await self._execute_single_command(user_id, command_text)

    async def _execute_single_command(
        self,
        user_id: str,
        command_text: str
    ) -> Dict:
        """
        Execute a single command.

        Args:
            user_id: User executing command
            command_text: Single command string

        Returns:
            Response dict with success, output, events
        """

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
