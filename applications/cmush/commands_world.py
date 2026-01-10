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
#   World Commands
#
#   These are the everyday commands you use to exist in the
#   virtual world - moving between rooms, talking to people,
#   looking at things, and picking stuff up.
#
#   Movement:
#     north, south, east, west, up, down (or n/s/e/w/u/d)
#
#   Communication:
#     say Hello!     -> You say, "Hello!"
#     emote waves    -> Chester waves.
#     tell bob Hi    -> Whisper to Bob
#
#   Observation:
#     look           -> See the room and who's in it
#     inventory      -> What are you carrying?
#     who            -> Who's online?
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands_world
# PURPOSE:  Basic world interaction commands
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   WorldCommandsMixin    Movement, communication, observation
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
World Commands Mixin for cMUSH

Contains basic world interaction commands:
- Movement: north, south, east, west, up, down (cmd_move)
- Communication: say, emote, tell
- Observation: look, inventory, who
- Manipulation: take, drop

Author: Caitlyn + Claude
Date: December 2025
"""

from typing import Dict
from fuzzy_match import find_best_matches, disambiguate_matches, format_disambiguation_prompt


class WorldCommandsMixin:
    """Mixin providing world interaction commands for CommandParser."""

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

        # Format username in all caps for consistency with other messages
        display_name = username.upper()
        output = f'{display_name} say, "{args}"'

        event = {
            'type': 'say',
            'user': user_id,
            'username': username,
            'room': room['uid'],
            'text': args
        }

        # Check for play chat triggers
        await self.play_manager.check_chat_trigger(args, room['uid'])

        # Trigger OnHear for scripted prims in room
        if self.script_manager:
            self.script_manager.broadcast_hear_to_room(room['uid'], user_id, args)

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
            query = args.strip()

            # Fuzzy match entity in room
            entity_id, entity_type, ambiguous = self._resolve_entity(query, room['uid'])

            if ambiguous:
                # Multiple matches - ask user to be more specific
                return {
                    'success': False,
                    'output': format_disambiguation_prompt(query, ambiguous),
                    'events': []
                }

            if not entity_id:
                return {
                    'success': False,
                    'output': f"You don't see '{query}' here.",
                    'events': []
                }

            # Handle looking at entity (occupant or object)
            if entity_type in ['agent', 'user']:
                occ = self.world.get_user(entity_id)
                if not occ:
                    return {'success': False, 'output': 'Error: Entity not found.', 'events': []}

                lines = []
                display_name = occ.get('username', occ.get('name', entity_id))
                lines.append(f"\n{display_name} [{entity_type}]")
                lines.append("=" * (len(display_name) + len(entity_type) + 3))

                # Get description
                if entity_type == 'agent':
                    agent = self.agent_manager.get_agent(entity_id)
                    if agent and hasattr(agent, 'agent_description') and agent.agent_description:
                        lines.append(agent.agent_description)
                    else:
                        lines.append(f"{display_name} hasn't set a description yet.")
                else:
                    desc = occ.get('description', '')
                    lines.append(desc if desc else f"{display_name} hasn't set a description yet.")

                return {
                    'success': True,
                    'output': '\n'.join(lines),
                    'events': []
                }

            # Handle object lookup
            if entity_type == 'object':
                obj = self.world.get_object(entity_id)
                if not obj:
                    return {'success': False, 'output': 'Error: Object not found.', 'events': []}

                lines = []
                lines.append(f"\n{obj['name']}")
                lines.append("=" * len(obj['name']))
                lines.append(obj.get('description', 'Nothing special.'))
                return {
                    'success': True,
                    'output': '\n'.join(lines),
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
                    name = occ.get('username', occ.get('name', occ.get('uid', 'Unknown')))
                    is_agent = occ.get('uid', '').startswith('agent_')

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

        query = args.strip()

        # Find object in room
        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        # Fuzzy match object (objects only)
        entity_id, entity_type, ambiguous = self._resolve_entity(
            query, room['uid'], include_agents=False, include_users=False
        )

        if ambiguous:
            return {
                'success': False,
                'output': format_disambiguation_prompt(query, ambiguous),
                'events': []
            }

        if not entity_id or entity_type != 'object':
            return {'success': False, 'output': f"You don't see '{query}' here.", 'events': []}

        obj_id = entity_id

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

        query = args.strip()

        user = self.world.get_user(user_id)
        inventory = user.get('inventory', [])

        # Build candidates from inventory
        candidates = []
        for oid in inventory:
            obj = self.world.get_object(oid)
            if obj:
                candidates.append((oid, obj['name']))

        # Fuzzy match
        matches = find_best_matches(query, candidates, threshold=0.3)

        if not matches:
            return {'success': False, 'output': f"You don't have '{query}'.", 'events': []}

        obj_id = disambiguate_matches(matches)

        if not obj_id:
            # Ambiguous
            return {
                'success': False,
                'output': format_disambiguation_prompt(query, matches),
                'events': []
            }

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
