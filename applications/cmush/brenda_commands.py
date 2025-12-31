"""
BRENDA Commands - Natural Language Parameter Tweaking & Play Generation

BRENDA: Behavioral Regulation Engine for Narrative-Driven Agents

This module contains all BRENDA-related commands and helpers,
extracted from commands.py for maintainability.

This is a mixin class - CommandParser inherits from it.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
import time
import re

logger = logging.getLogger(__name__)

# ===== BRENDA: Natural Language Parameter Tweaking =====
# Lazy stoner-friendly mappings: phrase -> goal & config adjustments
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


class BrendaCommandsMixin:
    """
    Mixin class providing BRENDA commands for CommandParser.

    BRENDA: Behavioral Regulation Engine for Narrative-Driven Agents

    This mixin expects the following attributes on self:
    - world: World state manager
    - agent_manager: Agent manager instance
    - server: Server instance (optional)
    - config: Server config dict
    - brenda_character: BrendaCharacter instance
    - play_manager: PlayManager instance
    - brenda_history: Dict tracking changes per agent
    - brenda_rate_limit: Dict tracking rate limits per agent
    - brenda_max_history: Max history entries per agent
    - brenda_rate_window: Rate limit window in seconds
    - brenda_rate_max: Max commands per window

    And the following methods:
    - cmd_show_appetites, cmd_show_goals, cmd_clear_bias, cmd_set_goal_bias
    - cmd_create, cmd_describe, cmd_rez_agent
    """

    def _init_brenda(self, config: Dict):
        """
        Initialize BRENDA subsystem. Call this from CommandParser.__init__.

        Args:
            config: Server config dict
        """
        from brenda_character import BrendaCharacter
        from play_manager import PlayManager

        # BRENDA state tracking (lazy parameter tweaking)
        self.brenda_history = {}  # agent_id -> list of (timestamp, changes_dict)
        self.brenda_rate_limit = {}  # agent_id -> list of timestamps
        self.brenda_max_history = 10  # per agent
        self.brenda_rate_window = 300  # 5 minutes
        self.brenda_rate_max = 999  # max commands per window (dev: unlimited)

        # BRENDA character (conversational stage manager)
        llm_config = config.get('llm', {}) if config else {}
        brenda_config = config.get('brenda', {}) if config else {}

        # Use Brenda-specific model if configured, otherwise fall back to general LLM model
        brenda_model = brenda_config.get('model', llm_config.get('model', 'SMALL'))

        self.brenda_character = BrendaCharacter(
            api_base=llm_config.get('api_base', 'http://localhost:11434/v1'),
            api_key=llm_config.get('api_key', 'not-needed'),
            model=brenda_model,
            timeout=llm_config.get('timeout', 60)
        )

        # BRENDA play manager (drama generation)
        self.play_manager = PlayManager(
            plays_dir="plays",
            server=self.server,
            brenda_character=self.brenda_character
        )

        # Register tools that BRENDA can use
        self._register_brenda_tools()

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
                return {'success': False, 'output': '  Need both agent name and description.', 'events': []}
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

    # ===== BRENDA: Main Command =====

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
            "BRENDA - Behavioral Regulation Engine for Narrative-Driven Agents\n" +
            "=" * 60 + "\n\n"
            "AGENT TWEAKING:\n"
            "  @brenda make <agent> <adjective> - adjust personality (more chatty, less calm, etc.)\n"
            "  @brenda reset <agent> - reload recipe defaults\n"
            "  @brenda vibe check <agent> - show current parameter settings\n"
            "  @brenda undo <agent> - undo last Brenda change\n"
            "  @brenda pass the joint - maximum hippie vibes\n\n"
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
                                'output': "BRENDA: The play has no cast! Strange theatrical production...",
                                'events': []
                            }
                        spawn_args = ' '.join(cast)
                    else:
                        return {
                            'success': False,
                            'output': "BRENDA: No play is running! Who should I summon to the stage?",
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
            output = f"BRENDA: {brenda_response}"

            if tool_result:
                # Tool was executed - add result to output
                if tool_result.get('success'):
                    output += f"\n\n{tool_result.get('output', '')}"
                else:
                    output += f"\n\n  {tool_result.get('output', 'Something went wrong...')}"

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
                'output': "Let me check my iPad... *adjusts reading glasses* Sorry, I'm having trouble with my notes. Try '@brenda help' to see what I can do.",
                'events': []
            }

    # ===== BRENDA: Helper Methods =====

    async def _brenda_make(self, user_id: str, agent_name: str, phrase: str) -> Dict:
        """Apply natural language adjustments to an agent."""
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        # Check rate limit
        if not self._brenda_check_rate_limit(agent_id):
            return {
                'success': False,
                'output': f"Slow down! Max {self.brenda_rate_max} Brenda commands per {self.brenda_rate_window//60} minutes per agent.",
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
                    f"Brenda doesn't know '{phrase}'.\n\n"
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
                applied[param] = f"{old_val:.1f}s -> {new_val:.1f}s"
            elif param in config_params:
                # Speech chance parameters (0-1 bounded)
                old_val = agent.config.get(param, 0.8 if 'addressed' in param else 0.3)
                new_val = max(0.0, min(1.0, old_val + delta))  # Clamp to [0, 1]
                agent.config[param] = new_val
                applied[param] = f"{old_val:.2f} -> {new_val:.2f}"
            else:
                # Appetite/goal biases (use agent's direct method)
                try:
                    if hasattr(agent, 'set_goal_bias'):
                        agent.set_goal_bias(param, delta)
                        applied[param] = f"{delta:+.2f}"
                    else:
                        warnings.append(f" {param}: Agent doesn't support goal biases")
                except Exception as e:
                    warnings.append(f" {param}: {str(e)}")

        # Record in history
        self._brenda_record_change(agent_id, applied)

        # Format output
        changes_text = "\n  ".join([f"{k}: {v}" for k, v in applied.items()])
        warning_text = "\n".join(warnings) if warnings else ""

        output = (
            f" {agent.agent_name} -> {phrase}\n\n"
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
                'output': "I need more details about the room you want to build.",
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
                f"Built '{room_name}' ({room_id})!\n\n"
                f"{room_description}\n\n"
                f"Tip: Use '@link {room_id} <direction>' to connect it to your current room"
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
                'output': f" Couldn't build the room: {str(e)}",
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
                            f"   {play['title']}\n"
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
            output = f"""BRENDA STATUS REPORT

Current Model: {current_model}

Running Plays:
{plays_text}

Available Commands:
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
                'output': f" Error getting status: {str(e)}",
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
                    'output': ' Please specify a model name.',
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
                'output': f' BRENDA model changed from "{old_model}" to "{model_name}".\nSaved to config.yaml.',
                'events': []
            }

        except Exception as e:
            return {
                'success': False,
                'output': f" Couldn't change model: {str(e)}",
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
            f"VIBE CHECK: {agent.agent_name}\n\n"
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
            'output': f" {agent.agent_name} reset to recipe defaults.\nAll biases cleared, Brenda history wiped.",
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
                if '->' in description:
                    old_val = float(description.split('->')[0].strip().rstrip('s'))
                    agent.response_cooldown = old_val
                    reversed_changes.append(f"{param} -> {old_val}s")
            else:
                # For biases, extract the delta and reverse it
                if '+' in description or '-' in description:
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
                f"Undid Brenda change from {dt}\n\n"
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
                'output': "*puff puff* ... but there are no agents here to share with!",
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
                f"*puff puff* ... peace, little dude.\n\n"
                f"{agents_text} now in maximum hippie mode:\n"
                f"  Agreeableness +0.4 (peace & love)\n"
                f"  Safety +0.2 (no harsh vibes)\n"
                f"  Volatility -0.3 (mellow)"
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
                'output': "No agents available for the cast. Spawn some agents first!",
                'events': []
            }

        # Check if we have LLM configured
        if not hasattr(self.agent_manager, 'llm') or not self.agent_manager.llm:
            return {
                'success': False,
                'output': "LLM not configured. Can't generate plays without an LLM backend.",
                'events': []
            }

        # Send initial acknowledgment
        if self.server:
            await self.server.broadcast_event({
                'type': 'chat',
                'sender': 'BRENDA',
                'text': f"Working on it, {self.world.get_user(user_id)['username']}! Crafting a theatrical masterpiece with {', '.join(available_cast)}...",
                'timestamp': datetime.now().isoformat()
            })

        # Set LLM on play manager
        self.play_manager.llm = self.agent_manager.llm

        # Generate play
        if self.server:
            await self.server.broadcast_event({
                'type': 'chat',
                'sender': 'BRENDA',
                'text': " Consulting the muse... (this might take a moment)",
                'timestamp': datetime.now().isoformat()
            })

        result = await self.play_manager.generate_play_from_prompt(
            user_prompt=story,
            available_cast=available_cast
        )

        if not result['success']:
            return {
                'success': False,
                'output': f"Failed to generate play: {result['error']}",
                'events': []
            }

        # Save play
        play_json = result['play']
        save_result = self.play_manager.save_play(play_json)

        if not save_result['success']:
            return {
                'success': False,
                'output': f"Failed to save play: {save_result['error']}",
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
                f"Play generated!\n\n"
                f"Title: {title}\n"
                f"Saved as: {filename}\n"
                f"Scenes: {num_scenes}\n"
                f"Beats: {num_beats}\n"
                f"Cast: {', '.join(play_json['cast'])}\n\n"
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
                'output': "No plays available yet. Create one with:\n  @brenda write play <your story>",
                'events': []
            }

        # Format output
        lines = ["Available Plays:", "=" * 40]
        for play in plays:
            cast_text = ", ".join(play['cast'][:3])
            if len(play['cast']) > 3:
                cast_text += f" (+{len(play['cast']) - 3} more)"
            lines.append(f"  {play['title']} ({play['filename']})")
            lines.append(f"  {play['scenes']} scenes - Cast: {cast_text}")
            lines.append("")

        # Show active plays
        active = self.play_manager.get_active_plays()
        if active:
            lines.append(f" Currently running: {', '.join(active)}")

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
                'output': f"{result['error']}",
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
                'output': f"{result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': f"{result['message']}",
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
                'output': f"{result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': f"{result['message']}",
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
                'output': f"{result['error']}",
                'events': []
            }

        return {
            'success': True,
            'output': f"{result['message']}",
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
                'output': f"BRENDA: Built the location but couldn't find its ID! Technical difficulties...",
                'events': result.get('events', [])
            }

        room_uid = room_uid_match.group(1)

        # Set the description
        desc_result = await self.cmd_describe(user_id, description)

        return {
            'success': True,
            'output': f"Built '{room_name}' ({room_uid})!\n\n{description}\n\nTip: Use '@link {room_uid} <direction>' to connect it to your current room",
            'events': result.get('events', []) + desc_result.get('events', [])
        }
