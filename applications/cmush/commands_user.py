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
#   User Commands
#
#   Commands for managing your own identity and viewing how
#   Noodlings perceive you. Both humans and Noodlings can
#   use these to customize their presence in the world.
#
#   Identity:
#     @setname "Chester"      -> Change your display name
#     @setdesc "A curious..." -> Set your description
#     @whoami                 -> See your identity (agents)
#     @me                     -> View self (humans)
#
#   Relationships:
#     @relationship chester   -> How does Chester see you?
#     @memory chester         -> What does Chester remember?
#
#   Movement:
#     @tpinvite bob           -> Invite Bob to your location
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands_user
# PURPOSE:  User profile and relationship commands
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   UserCommandsMixin    Identity, relationships, teleport
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
User Commands Mixin for cMUSH

Contains commands for user profile and interaction:
- @me: View self observation
- @whoami: Show agent identity (agents only)
- @setname: Change name
- @setdesc: Set description
- @profile: View profile
- @relationship: View agent relationships
- @memory: View agent memories
- @tpinvite: Teleport invitation

Author: Caitlyn + Claude
Date: December 2025
"""

from typing import Dict
from fuzzy_match import format_disambiguation_prompt


class UserCommandsMixin:
    """Mixin providing user profile commands for CommandParser."""

    async def cmd_observe_self(self, user_id: str, args: str) -> Dict:
        """View your own Consilience state (for human users)."""
        user = self.world.get_user(user_id)
        if not user:
            return {'success': False, 'output': 'User not found.', 'events': []}

        lines = ["\nYour Consilience State"]
        lines.append("=" * 40)
        lines.append("You are a human user. Your phenomenal state is being")
        lines.append("inferred by agents through Theory of Mind when they")
        lines.append("perceive your actions.")
        lines.append("")
        lines.append("To see how agents perceive you, ask them with:")
        lines.append("  @relationship <agent_name>")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}

    async def cmd_whoami(self, user_id: str, args: str) -> Dict:
        """Show agent's identity (only for agents)."""
        if not user_id.startswith('agent_'):
            return {
                'success': False,
                'output': 'This command is only available to agents. Humans use @me.',
                'events': []
            }

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        lines = [f"\nYou are {agent.agent_name}"]
        lines.append("=" * 40)
        if hasattr(agent, 'agent_description') and agent.agent_description:
            lines.append(agent.agent_description)
        lines.append(f"\nYour ID: {user_id}")
        lines.append(f"Current room: {agent.current_room}")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}

    async def cmd_setname(self, user_id: str, args: str) -> Dict:
        """Change your display name."""
        if not args:
            return {'success': False, 'output': 'Usage: @setname <new_name>', 'events': []}

        new_name = args.strip()

        if user_id.startswith('agent_'):
            agent = self.agent_manager.get_agent(user_id)
            if agent:
                agent.agent_name = new_name
            agent_data = self.world.get_user(user_id)
            if agent_data:
                agent_data['name'] = new_name
                agent_data['username'] = new_name
        else:
            user = self.world.get_user(user_id)
            if user:
                user['username'] = new_name

        self.world.save_all()

        return {
            'success': True,
            'output': f"Name changed to: {new_name}",
            'events': []
        }

    async def cmd_setdesc(self, user_id: str, args: str) -> Dict:
        """Set description for yourself, current room, or an object."""
        if not args:
            return {
                'success': False,
                'output': (
                    "Usage:\n"
                    "  @setdesc me <description>      - Set your description\n"
                    "  @setdesc here <description>    - Set room description\n"
                    '  @setdesc "object" <description> - Set object description'
                ),
                'events': []
            }

        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': "Usage: @setdesc <target> <description>", 'events': []}

        target, description = parts

        # Handle "me" - set user/agent description
        if target.lower() == 'me':
            if user_id.startswith('agent_'):
                agent = self.agent_manager.get_agent(user_id)
                if agent:
                    agent.agent_description = description
                agent_data = self.world.get_user(user_id)
                if agent_data:
                    agent_data['description'] = description
            else:
                user = self.world.get_user(user_id)
                if user:
                    user['description'] = description

            self.world.save_all()
            return {'success': True, 'output': f"Your description set to: {description}", 'events': []}

        # Handle "here" - set room description
        if target.lower() == 'here':
            room = self.world.get_user_room(user_id)
            if not room:
                return {'success': False, 'output': 'Error getting location.', 'events': []}

            room['description'] = description
            self.world.save_all()
            return {'success': True, 'output': f"Room description set to: {description}", 'events': []}

        # Handle object description
        object_name = target.strip('"')
        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        for obj_id in room.get('objects', []):
            obj = self.world.get_object(obj_id)
            if obj and obj['name'].lower() == object_name.lower():
                obj['description'] = description
                self.world.save_all()
                return {
                    'success': True,
                    'output': f"Description for '{obj['name']}' set to: {description}",
                    'events': []
                }

        return {'success': False, 'output': f"Object '{object_name}' not found.", 'events': []}

    async def cmd_profile(self, user_id: str, args: str) -> Dict:
        """View your profile or another user's profile."""
        if args:
            target_name = args.strip()
            room = self.world.get_user_room(user_id)
            if not room:
                return {'success': False, 'output': 'You are nowhere.', 'events': []}

            entity_id, entity_type, ambiguous = self._resolve_entity(
                target_name, room['uid'], include_objects=False
            )

            if ambiguous:
                return {
                    'success': False,
                    'output': format_disambiguation_prompt(target_name, ambiguous),
                    'events': []
                }

            if not entity_id:
                return {'success': False, 'output': f"User '{target_name}' not found.", 'events': []}

            target_id = entity_id
        else:
            target_id = user_id

        user = self.world.get_user(target_id)
        if not user:
            return {'success': False, 'output': 'User not found.', 'events': []}

        is_agent = target_id.startswith('agent_')
        name = user.get('username', user.get('name', target_id))

        lines = [f"\nProfile: {name}"]
        lines.append("=" * 40)
        lines.append(f"Type: {'Noodling' if is_agent else 'Noodler'}")

        desc = user.get('description', '')
        if is_agent:
            agent = self.agent_manager.get_agent(target_id)
            if agent and hasattr(agent, 'agent_description') and agent.agent_description:
                desc = agent.agent_description

        lines.append(f"Description: {desc if desc else 'Not set'}")

        room = self.world.get_room(user['current_room'])
        lines.append(f"Location: {room['name'] if room else 'Unknown'}")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}

    async def cmd_relationship(self, user_id: str, args: str) -> Dict:
        """View agent's relationship model."""
        if not args:
            return {'success': False, 'output': 'Usage: @relationship <agent_name>', 'events': []}

        query = args.strip()

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'You are nowhere.', 'events': []}

        entity_id, entity_type, ambiguous = self._resolve_entity(
            query, room['uid'], include_objects=False, include_users=False
        )

        if ambiguous:
            return {
                'success': False,
                'output': format_disambiguation_prompt(query, ambiguous),
                'events': []
            }

        if not entity_id or entity_type != 'agent':
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent = self.agent_manager.get_agent(entity_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent_name = agent.agent_name

        relationships = agent.get_relationships() if hasattr(agent, 'get_relationships') else {}

        if not relationships:
            return {'success': True, 'output': f"{agent_name} has no tracked relationships.", 'events': []}

        lines = [f"\nRelationships for {agent_name}:"]
        lines.append("=" * 40)
        for other_id, rel in relationships.items():
            lines.append(f"{other_id}:")
            lines.append(f"  Attachment: {rel.get('attachment_style', 'unknown')}")
            lines.append(f"  Interactions: {rel.get('interaction_count', 0)}")
            lines.append(f"  Valence: {rel.get('valence', 0.0):.2f}")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}

    async def cmd_memory(self, user_id: str, args: str) -> Dict:
        """View agent's memory system."""
        if not args:
            return {
                'success': False,
                'output': 'Usage: @memory <agent_name> [--stats|--working|--episodic|--search <query>]',
                'events': []
            }

        parts = args.strip().split()
        query = parts[0]
        flags = parts[1:] if len(parts) > 1 else []

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'You are nowhere.', 'events': []}

        entity_id, entity_type, ambiguous = self._resolve_entity(
            query, room['uid'], include_objects=False, include_users=False
        )

        if ambiguous:
            return {
                'success': False,
                'output': format_disambiguation_prompt(query, ambiguous),
                'events': []
            }

        if not entity_id or entity_type != 'agent':
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent = self.agent_manager.get_agent(entity_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent_name = agent.agent_name

        # Handle flags
        if '--stats' in flags:
            stats = agent.get_memory_stats() if hasattr(agent, 'get_memory_stats') else {}
            lines = [f"\nMemory Statistics for {agent_name}"]
            lines.append("=" * 60)
            lines.append(f"Working Memory: {stats.get('working_count', 0)}/{stats.get('working_capacity', 0)}")
            lines.append(f"Episodic Memory: {stats.get('episodic_count', 0)}/{stats.get('episodic_capacity', 0)}")
            return {'success': True, 'output': '\n'.join(lines), 'events': []}

        elif '--working' in flags:
            working = agent.get_working_memory() if hasattr(agent, 'get_working_memory') else []
            lines = [f"\nWorking Memory for {agent_name} ({len(working)} entries)"]
            lines.append("=" * 60)
            for i, entry in enumerate(working[-10:], 1):
                text = entry.get('text', '')[:80]
                lines.append(f"{i}. [{entry.get('user', '?')}] {text}...")
            return {'success': True, 'output': '\n'.join(lines), 'events': []}

        elif '--episodic' in flags:
            episodic = agent.get_episodic_memory(limit=15) if hasattr(agent, 'get_episodic_memory') else []
            lines = [f"\nEpisodic Memory for {agent_name} ({len(episodic)} entries)"]
            lines.append("=" * 60)
            for i, entry in enumerate(episodic, 1):
                text = entry.get('text', '')[:80]
                lines.append(f"{i}. [{entry.get('user', '?')}] {text}...")
            return {'success': True, 'output': '\n'.join(lines), 'events': []}

        else:
            # Default: show recent memories
            memory = agent.get_episodic_buffer() if hasattr(agent, 'get_episodic_buffer') else []
            if not memory:
                return {'success': True, 'output': f"{agent_name} has no memories.", 'events': []}

            lines = [f"\nRecent memories for {agent_name}:"]
            lines.append("=" * 40)
            for entry in memory[-5:]:
                lines.append(f"[{entry.get('user', 'unknown')}]: {entry.get('text', '')[:100]}")

            return {'success': True, 'output': '\n'.join(lines), 'events': []}

    async def cmd_tpinvite(self, user_id: str, args: str) -> Dict:
        """Teleport an agent to your location."""
        if not args:
            return {'success': False, 'output': 'Usage: @tpinvite <agent_name>', 'events': []}

        target_name = args.strip()
        agent_id = f"agent_{target_name}" if not target_name.startswith('agent_') else target_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{target_name}' not found.", 'events': []}

        user_room = self.world.get_user_room(user_id)
        if not user_room:
            return {'success': False, 'output': 'Error getting your location.', 'events': []}

        # Move agent
        old_room = agent.current_room
        agent.current_room = user_room['uid']

        # Update world state
        agent_data = self.world.get_user(agent_id)
        if agent_data:
            agent_data['current_room'] = user_room['uid']

        # Update room occupants
        old_room_data = self.world.get_room(old_room)
        if old_room_data and agent_id in old_room_data.get('occupants', []):
            old_room_data['occupants'].remove(agent_id)

        if agent_id not in user_room.get('occupants', []):
            user_room.setdefault('occupants', []).append(agent_id)

        self.world.save_all()

        return {
            'success': True,
            'output': f"Teleported {agent.agent_name} to your location.",
            'events': [{
                'type': 'enter',
                'user': agent_id,
                'room': user_room['uid'],
                'text': f"{agent.agent_name} appears in a flash of light."
            }]
        }

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
