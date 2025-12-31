"""
Building Commands Mixin for cMUSH

Contains world building commands:
- @create: Create rooms or objects
- @describe: (Deprecated) Set descriptions
- @dig: Create exits and new rooms
- @link: Link rooms with custom exit names
- @destroy: Remove objects

Author: cMUSH Project
Date: December 2025
"""

from typing import Dict


class BuildingCommandsMixin:
    """Mixin providing building commands for CommandParser."""

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
            return {
                'success': False,
                'output': 'Usage: @setdesc here <description> (or @setdesc me, @setdesc "object")\n\n'
                          'Note: @describe is deprecated, use @setdesc instead.',
                'events': []
            }

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
            'output': f"Linked! You can now travel via '{direction_name}' to {target_room.get('name', target_room_id)}.",
            'events': []
        }

    async def cmd_destroy(self, user_id: str, args: str) -> Dict:
        """Destroy an object in the current room."""
        if not args:
            return {
                'success': False,
                'output': 'Usage: @destroy <object> OR @destroy "<multi word object>"',
                'events': []
            }

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
            return {
                'success': False,
                'output': f"Object '{object_name}' not found in this room.",
                'events': []
            }

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
