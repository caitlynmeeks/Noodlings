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
#   Legacy World Converter
#
#   Converts older noodleMUSH world files (JSON format) into the
#   newer PROJECT_SPEC format with proper Stages, Zones, Noodlings,
#   and Instances. This is a one-time migration tool for upgrading
#   existing worlds to the current project architecture.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.convert_legacy_world
# PURPOSE:  Migrate old world data to PROJECT_SPEC.md format
# LAYER:    Backend / Migration Tool
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   convert_legacy_world()    Main conversion entry point
#   convert_room_to_zone()    Transform room JSON to zone YAML
#   convert_agent_to_noodling() Create noodling from agent
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Legacy World to Project Format Converter

Converts legacy cmush world data (rooms.json, agents.json) into the PROJECT_SPEC.md
compliant format with proper Stages, Zones, Noodlings, and Instances.

Usage:
    python convert_legacy_world.py [project_path]

If project_path is not specified, reads from PROJECT_PATH env var or
~/.noodlestudio/current_project.json

Author: Caitlyn + Claude
"""

import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp as ISO string."""
    return datetime.now().isoformat()


def sanitize_name(name: str) -> str:
    """Convert name to filesystem-safe string."""
    safe = name.lower().replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")
    return safe or "unnamed"


def write_yaml(path: str, data: dict):
    """Write data to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_legacy_data(world_dir: str) -> tuple:
    """Load legacy world data from JSON files."""
    rooms = {}
    agents = {}
    users = {}

    rooms_path = os.path.join(world_dir, "rooms.json")
    if os.path.exists(rooms_path):
        with open(rooms_path, 'r') as f:
            rooms = json.load(f)

    agents_path = os.path.join(world_dir, "agents.json")
    if os.path.exists(agents_path):
        with open(agents_path, 'r') as f:
            agents = json.load(f)

    users_path = os.path.join(world_dir, "users.json")
    if os.path.exists(users_path):
        with open(users_path, 'r') as f:
            users = json.load(f)

    return rooms, agents, users


def convert_room_to_zone(room_id: str, room_data: dict) -> dict:
    """Convert a legacy room to a zone definition."""
    return {
        "name": room_data.get("name", room_id),
        "id": room_id,
        "spatial": {
            "center": [0, 0, 0],
            "radius": 50.0,
            "falloff": 20.0,
            "shape": "sphere"
        },
        "text": {
            "description": room_data.get("description", ""),
            "features": room_data.get("objects", []),
            "exits": room_data.get("exits", {})
        },
        "perception": {
            "visibility": 30.0,
            "audibility": 50.0,
            "lighting": "ambient"
        },
        "ambient": {
            "sounds": [],
            "mood": "neutral",
            "temperature": "comfortable"
        }
    }


def convert_agent_to_noodling(agent_id: str, agent_data: dict) -> tuple:
    """
    Convert a legacy agent to noodling manifest, recipe, and assembly.

    Returns: (manifest, recipe, assembly)
    """
    name = agent_data.get("name", agent_id)
    safe_name = sanitize_name(name)
    now = get_timestamp()

    config = agent_data.get("config", {})

    # Manifest (noodling.yaml)
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": agent_data.get("description", ""),
        "author": "",
        "created": now,
        "modified": now,
        "tags": [agent_data.get("species", "noodling")],
        "recipe": "recipe.yaml",
        "assembly": "assembly.yaml",
        "charm_weights": None,
        "neural_graphs": {},
        "scripts": [],
        "assets": {
            "portrait": None,
            "voice_reference": None,
            "expressions": {},
            "vision_memories": []
        },
        "processors": [],
        "preview": {
            "personality": agent_data.get("description", "")[:100],
            "species": agent_data.get("species", "noodling"),
            "complexity": "minimal",
            "facet_count": 3,
            "llm_facets": 1,
            "has_trained_weights": False,
            "has_voice": False
        }
    }

    # Recipe (recipe.yaml)
    recipe = {
        "name": name,
        "species": agent_data.get("species", "noodling"),
        "description": agent_data.get("description", "A character."),
        "identity_prompt": f"You are {name}.\n\n{agent_data.get('description', '')}",
        "language_mode": "verbal",
        "pronouns": agent_data.get("pronouns", "they/them"),
        "constraints": {
            "max_tokens": config.get("max_tokens", 100),
            "temperature": config.get("temperature", 0.8),
            "response_cooldown": config.get("response_cooldown", 2.0)
        },
        "llm": {
            "provider": "local",
            "model": "SMALL"
        },
        "personality": {
            "extraversion": 0.5,
            "impulsivity": 0.3,
            "curiosity": 0.7,
            "emotional_volatility": 0.4,
            "vanity": 0.2
        },
        "appetites": {
            "curiosity": 0.7,
            "status": 0.3,
            "mastery": 0.5,
            "novelty": 0.6,
            "safety": 0.5,
            "social_bond": 0.6,
            "comfort": 0.5,
            "autonomy": 0.5
        },
        "facet_assembly": f"Noodlings/{safe_name}",
        "spawn_message": "appears"
    }

    # Assembly (assembly.yaml) - reference the existing facet_assembly if available
    facet_ref = agent_data.get("facet_assembly", "")
    if facet_ref:
        # Keep reference to existing assembly
        assembly = {
            "name": f"{name} Assembly",
            "version": "1.0.0",
            "description": f"Facet topology for {name}",
            "extends": facet_ref,  # Reference to existing assembly
            "facets": {},
            "connections": []
        }
    else:
        # Minimal assembly
        assembly = {
            "name": f"{name} Assembly",
            "version": "1.0.0",
            "description": f"Facet topology for {name}",
            "facets": {
                "INCOMING": {
                    "type": "INCOMING",
                    "position": [100, 200]
                },
                "main_response": {
                    "type": "LLMFacet",
                    "prompt": f"Respond as {name}.",
                    "model": "SMALL",
                    "position": [400, 200]
                },
                "OUTGOING": {
                    "type": "OUTGOING",
                    "position": [700, 200]
                }
            },
            "connections": [
                {"from": "INCOMING", "to": "main_response"},
                {"from": "main_response", "to": "OUTGOING"}
            ]
        }

    return manifest, recipe, assembly


def create_instance(agent_id: str, agent_data: dict, noodling_path: str, instance_path: str) -> dict:
    """Create an instance.yaml for an agent in a stage."""
    now = get_timestamp()

    # Calculate relative path from instance to noodling
    rel_noodling = os.path.relpath(noodling_path, instance_path)

    return {
        "noodling": rel_noodling,
        "overrides": {
            "name": agent_data.get("name", agent_id),
            "position": [0, 0, 0],
            "rotation": [0, 0, 0],
            "zone": agent_data.get("current_room", "room_000")
        },
        "created": now,
        "last_active": now
    }


def convert_legacy_world(project_path: str, world_dir: str = None):
    """
    Convert legacy world data to project format.

    Args:
        project_path: Path to the target project
        world_dir: Path to legacy world directory (default: applications/cmush/world)
    """
    if world_dir is None:
        # Default to cmush/world relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        world_dir = os.path.join(script_dir, "world")

    print(f"Converting legacy world: {world_dir}")
    print(f"Target project: {project_path}")

    # Verify project exists
    manifest_path = os.path.join(project_path, "project.noodleproj")
    if not os.path.exists(manifest_path):
        print(f"Error: Not a valid project: {project_path}")
        return False

    # Load legacy data
    rooms, agents, users = load_legacy_data(world_dir)
    print(f"Found {len(rooms)} rooms, {len(agents)} agents, {len(users)} users")

    if not rooms:
        print("No rooms found in legacy world")
        return False

    # Create Stages directory
    stages_dir = os.path.join(project_path, "Stages")
    os.makedirs(stages_dir, exist_ok=True)

    # Create Noodlings directory
    noodlings_dir = os.path.join(project_path, "Noodlings")
    os.makedirs(noodlings_dir, exist_ok=True)

    # Find the main room (room_000 or first room)
    main_room_id = "room_000" if "room_000" in rooms else list(rooms.keys())[0]
    main_room = rooms[main_room_id]
    stage_name = sanitize_name(main_room.get("name", "the_nexus"))

    # Create stage
    stage_path = os.path.join(stages_dir, stage_name)
    if os.path.exists(stage_path):
        print(f"Removing existing stage: {stage_path}")
        shutil.rmtree(stage_path)

    os.makedirs(stage_path)
    os.makedirs(os.path.join(stage_path, "Zones"))
    os.makedirs(os.path.join(stage_path, "Instances"))
    os.makedirs(os.path.join(stage_path, "Props"))

    now = get_timestamp()

    # Create stage.yaml
    zone_refs = []
    for room_id in rooms:
        zone_refs.append(f"Zones/{room_id}.zone.yaml")

    stage_def = {
        "name": main_room.get("name", "The Nexus"),
        "description": main_room.get("description", ""),
        "created": now,
        "modified": now,
        "geometry": None,
        "world": {
            "bounds": {
                "min": [-100, 0, -100],
                "max": [100, 50, 100]
            },
            "ambient": {
                "time_of_day": "day",
                "weather": "clear",
                "soundscape": None
            }
        },
        "spawn": {
            "position": [0, 0, 0],
            "zone": main_room_id
        },
        "zones": zone_refs,
        "instances": [],
        "props": []
    }

    write_yaml(os.path.join(stage_path, "stage.yaml"), stage_def)
    print(f"Created stage: {stage_name}")

    # Create zones from rooms
    for room_id, room_data in rooms.items():
        zone_data = convert_room_to_zone(room_id, room_data)
        zone_path = os.path.join(stage_path, "Zones", f"{room_id}.zone.yaml")
        write_yaml(zone_path, zone_data)
        print(f"  Created zone: {room_id} ({room_data.get('name', 'unnamed')})")

    # Create noodlings and instances from agents
    instance_refs = []
    for agent_id, agent_data in agents.items():
        name = agent_data.get("name", agent_id)
        safe_name = sanitize_name(name)

        # Create noodling folder
        noodling_path = os.path.join(noodlings_dir, safe_name)
        if os.path.exists(noodling_path):
            print(f"  Noodling exists, skipping: {safe_name}")
        else:
            os.makedirs(noodling_path)
            os.makedirs(os.path.join(noodling_path, "Scripts"))
            os.makedirs(os.path.join(noodling_path, "NeuralGraphs"))
            os.makedirs(os.path.join(noodling_path, "Assets", "expressions"))
            os.makedirs(os.path.join(noodling_path, "Assets", "memories"))
            os.makedirs(os.path.join(noodling_path, "Processors"))

            manifest, recipe, assembly = convert_agent_to_noodling(agent_id, agent_data)

            write_yaml(os.path.join(noodling_path, "noodling.yaml"), manifest)
            write_yaml(os.path.join(noodling_path, "recipe.yaml"), recipe)
            write_yaml(os.path.join(noodling_path, "assembly.yaml"), assembly)

            print(f"  Created noodling: {safe_name}")

        # Create instance in stage
        instance_name = safe_name
        instance_path = os.path.join(stage_path, "Instances", instance_name)

        if os.path.exists(instance_path):
            print(f"  Instance exists, skipping: {instance_name}")
        else:
            os.makedirs(instance_path)

            instance_data = create_instance(agent_id, agent_data, noodling_path, instance_path)
            write_yaml(os.path.join(instance_path, "instance.yaml"), instance_data)

            # Create initial state
            state = {
                "instance_id": instance_name,
                "timestamp": now,
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "zone": agent_data.get("current_room", main_room_id),
                "affect": {
                    "valence": 0.0,
                    "arousal": 0.3,
                    "dominance": 0.5,
                    "boredom": 0.0,
                    "sorrow": 0.0
                },
                "charm_state": None,
                "memories": {
                    "short_term": [],
                    "episodic": []
                },
                "script_storage": {}
            }

            with open(os.path.join(instance_path, "state.json"), 'w') as f:
                json.dump(state, f, indent=2)

            instance_refs.append(f"Instances/{instance_name}")
            print(f"  Created instance: {instance_name}")

    # Update stage.yaml with instance refs
    stage_def["instances"] = instance_refs
    write_yaml(os.path.join(stage_path, "stage.yaml"), stage_def)

    # Update project manifest with default stage
    with open(manifest_path, 'r') as f:
        project_manifest = json.load(f)

    project_manifest["default_stage"] = f"Stages/{stage_name}"
    project_manifest["modified"] = now

    with open(manifest_path, 'w') as f:
        json.dump(project_manifest, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Stage: {stage_name}")
    print(f"  Zones: {len(rooms)}")
    print(f"  Noodlings: {len(agents)}")
    print(f"  Instances: {len(instance_refs)}")
    print(f"  Default stage set to: Stages/{stage_name}")

    return True


def get_project_path_from_env() -> str:
    """Get project path from environment or settings."""
    # Check environment variable
    project_path = os.environ.get("PROJECT_PATH")
    if project_path and os.path.exists(project_path):
        return project_path

    # Check settings file
    settings_file = Path.home() / ".noodlestudio" / "current_project.json"
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                path = settings.get("project_path")
                if path and os.path.exists(path):
                    return path
        except:
            pass

    return None


if __name__ == "__main__":
    # Get project path from arg or environment
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = get_project_path_from_env()

    if not project_path:
        print("Usage: python convert_legacy_world.py [project_path]")
        print("Or set PROJECT_PATH environment variable")
        sys.exit(1)

    success = convert_legacy_world(project_path)
    sys.exit(0 if success else 1)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
