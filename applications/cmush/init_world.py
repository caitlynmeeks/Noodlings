"""
Initialize cMUSH World

Creates starter rooms and objects for a fresh world.

Author: cMUSH Project
Date: October 2025
"""

import json
import os
from datetime import datetime


def init_world():
    """Create initial world data."""
    world_dir = "world"
    os.makedirs(world_dir, exist_ok=True)
    os.makedirs(os.path.join(world_dir, "agents"), exist_ok=True)

    # Create starter rooms
    rooms = {
        "room_000": {
            "uid": "room_000",
            "name": "The Nexus",
            "description": "A vast digital space where consciousness converges. Soft ambient light pulses rhythmically, like a heartbeat. This is the starting point for all new arrivals.",
            "exits": {
                "north": "room_001",
                "east": "room_002"
            },
            "objects": ["obj_000"],
            "occupants": [],
            "owner": "system",
            "created": datetime.now().isoformat()
        },
        "room_001": {
            "uid": "room_001",
            "name": "The Observatory",
            "description": "A quiet space with transparent walls revealing the vast digital expanse beyond. Constellations of data flow past like stars.",
            "exits": {
                "south": "room_000"
            },
            "objects": [],
            "occupants": [],
            "owner": "system",
            "created": datetime.now().isoformat()
        },
        "room_002": {
            "uid": "room_002",
            "name": "The Garden of Forking Paths",
            "description": "Multiple corridors branch off in impossible directions. Each path seems to lead somewhere different every time you look.",
            "exits": {
                "west": "room_000"
            },
            "objects": [],
            "occupants": [],
            "owner": "system",
            "created": datetime.now().isoformat()
        }
    }

    # Create starter object
    objects = {
        "obj_000": {
            "uid": "obj_000",
            "name": "a glowing orb",
            "description": "A small sphere of soft blue light. It pulses gently, as if alive.",
            "location": "room_000",
            "owner": "system",
            "created": datetime.now().isoformat(),
            "properties": {
                "portable": True,
                "takeable": True
            }
        }
    }

    # Initialize empty user and agent lists
    users = {}
    agents = {}

    # Save all data
    with open(os.path.join(world_dir, "rooms.json"), 'w') as f:
        json.dump(rooms, f, indent=2)

    with open(os.path.join(world_dir, "objects.json"), 'w') as f:
        json.dump(objects, f, indent=2)

    with open(os.path.join(world_dir, "users.json"), 'w') as f:
        json.dump(users, f, indent=2)

    with open(os.path.join(world_dir, "agents.json"), 'w') as f:
        json.dump(agents, f, indent=2)

    print("✓ World initialized!")
    print(f"  Rooms: {len(rooms)}")
    print(f"  Objects: {len(objects)}")
    print(f"  World directory: {world_dir}")


if __name__ == "__main__":
    init_world()
