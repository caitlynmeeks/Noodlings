"""
Project Bridge - Adapts new PROJECT_SPEC.md format to legacy World interface.

This module allows the server to load data from either:
1. Legacy world/ directory (JSON files)
2. New project format (PROJECT_SPEC.md compliant)

When PROJECT_PATH environment variable is set, it loads from the project.
Otherwise, falls back to legacy format.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ProjectBridge:
    """
    Bridge between new project format and legacy World class.

    Translates:
    - Stages/xxx/Zones/*.zone.yaml → rooms.json format
    - Stages/xxx/Instances/ → agents.json format
    - Stages/xxx/Props/ → objects.json format
    - Noodlings/ → recipes/
    """

    def __init__(self, project_path: str):
        """
        Initialize bridge for a project.

        Args:
            project_path: Path to project folder containing project.noodleproj
        """
        self.project_path = project_path
        self.project_name = os.path.basename(project_path)

        # Verify it's a valid project
        manifest_path = os.path.join(project_path, "project.noodleproj")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Not a valid project: {project_path}")

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        logger.info(f"ProjectBridge initialized for: {self.project_name}")

    def get_default_stage_path(self) -> Optional[str]:
        """Get path to the default stage."""
        default_stage = self.manifest.get("default_stage")
        if default_stage:
            return os.path.join(self.project_path, default_stage)

        # Fall back to first stage found
        stages_dir = os.path.join(self.project_path, "Stages")
        if os.path.exists(stages_dir):
            for name in os.listdir(stages_dir):
                stage_path = os.path.join(stages_dir, name)
                if os.path.isdir(stage_path) and os.path.exists(
                    os.path.join(stage_path, "stage.yaml")
                ):
                    return stage_path
        return None

    def load_rooms_from_zones(self, stage_path: str) -> Dict:
        """
        Convert stage zones to legacy rooms.json format.

        Args:
            stage_path: Path to stage folder

        Returns:
            Dict in rooms.json format
        """
        rooms = {}
        zones_dir = os.path.join(stage_path, "Zones")

        if not os.path.exists(zones_dir):
            logger.warning(f"No Zones directory in {stage_path}")
            return rooms

        for filename in os.listdir(zones_dir):
            if not filename.endswith(".zone.yaml"):
                continue

            zone_path = os.path.join(zones_dir, filename)
            try:
                with open(zone_path, 'r') as f:
                    zone = yaml.safe_load(f)

                zone_id = zone.get("id", filename.replace(".zone.yaml", ""))

                # Convert zone to room format
                room = {
                    "uid": zone_id,
                    "name": zone.get("name", zone_id),
                    "description": zone.get("text", {}).get("description", ""),
                    "exits": zone.get("text", {}).get("exits", {}),
                    "objects": zone.get("text", {}).get("features", []),
                    "occupants": [],  # Will be populated by agents
                    "owner": "system",
                    "created": "2025-01-01T00:00:00"
                }

                rooms[zone_id] = room
                logger.debug(f"Loaded zone as room: {zone_id}")

            except Exception as e:
                logger.error(f"Error loading zone {filename}: {e}")

        return rooms

    def load_agents_from_instances(self, stage_path: str) -> Dict:
        """
        Convert stage instances to legacy agents.json format.

        Args:
            stage_path: Path to stage folder

        Returns:
            Dict in agents.json format
        """
        agents = {}
        instances_dir = os.path.join(stage_path, "Instances")

        if not os.path.exists(instances_dir):
            logger.warning(f"No Instances directory in {stage_path}")
            return agents

        for instance_name in os.listdir(instances_dir):
            instance_path = os.path.join(instances_dir, instance_name)
            if not os.path.isdir(instance_path):
                continue

            instance_yaml = os.path.join(instance_path, "instance.yaml")
            if not os.path.exists(instance_yaml):
                continue

            try:
                with open(instance_yaml, 'r') as f:
                    instance = yaml.safe_load(f)

                # Resolve noodling path
                noodling_ref = instance.get("noodling", "")
                noodling_path = os.path.normpath(
                    os.path.join(instance_path, noodling_ref)
                )

                # Load noodling recipe to get details
                recipe_data = self._load_noodling_recipe(noodling_path)

                overrides = instance.get("overrides", {})
                agent_id = f"agent_{instance_name}"

                # Build agent entry in legacy format
                agent = {
                    "name": overrides.get("name", instance_name),
                    "species": recipe_data.get("species", "noodling"),
                    "pronouns": recipe_data.get("pronouns", "they/them"),
                    "location": overrides.get("zone", "default"),
                    "description": recipe_data.get("description", ""),
                    "facet_assembly": self._get_assembly_ref(noodling_path),
                    "checkpoint": recipe_data.get("checkpoint",
                        "../../models/checkpoints/best_checkpoint.npz"),
                    "config": {
                        "max_tokens": recipe_data.get("constraints", {}).get("max_tokens", 100),
                        "temperature": recipe_data.get("constraints", {}).get("temperature", 0.8),
                        "response_cooldown": recipe_data.get("constraints", {}).get("response_cooldown", 2.0),
                        "facet_assembly": {
                            "ref": self._get_assembly_ref(noodling_path)
                        }
                    },
                    "checkpoint_path": os.path.join(instance_path, "checkpoint.npz"),
                    "current_room": overrides.get("zone", "room_000")
                }

                agents[agent_id] = agent
                logger.debug(f"Loaded instance as agent: {agent_id}")

            except Exception as e:
                logger.error(f"Error loading instance {instance_name}: {e}")
                import traceback
                traceback.print_exc()

        return agents

    def load_objects_from_props(self, stage_path: str) -> Dict:
        """
        Convert stage props to legacy objects.json format.

        Args:
            stage_path: Path to stage folder

        Returns:
            Dict in objects.json format (currently minimal)
        """
        objects = {}
        props_dir = os.path.join(stage_path, "Props")

        if not os.path.exists(props_dir):
            return objects

        for prop_name in os.listdir(props_dir):
            prop_path = os.path.join(props_dir, prop_name)
            if not os.path.isdir(prop_path):
                continue

            prop_yaml = os.path.join(prop_path, "prop.yaml")
            if not os.path.exists(prop_yaml):
                continue

            try:
                with open(prop_yaml, 'r') as f:
                    prop = yaml.safe_load(f)

                # Load prim template for details
                prim_ref = prop.get("prim", "")
                prim_path = os.path.normpath(os.path.join(prop_path, prim_ref))
                prim_data = self._load_prim(prim_path)

                obj_id = f"obj_{prop_name}"
                objects[obj_id] = {
                    "name": prop.get("name", prop_name),
                    "description": prim_data.get("display", {}).get("text_description", ""),
                    "location": prop.get("zone", "default"),
                    "verbs": prim_data.get("verbs", {}),
                    "state": prop.get("state", {})
                }

            except Exception as e:
                logger.error(f"Error loading prop {prop_name}: {e}")

        return objects

    def _load_noodling_recipe(self, noodling_path: str) -> Dict:
        """Load recipe.yaml from a noodling folder."""
        recipe_path = os.path.join(noodling_path, "recipe.yaml")
        if os.path.exists(recipe_path):
            try:
                with open(recipe_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except:
                pass
        return {}

    def _load_prim(self, prim_path: str) -> Dict:
        """Load prim.yaml from a prim folder."""
        prim_yaml = os.path.join(prim_path, "prim.yaml")
        if os.path.exists(prim_yaml):
            try:
                with open(prim_yaml, 'r') as f:
                    return yaml.safe_load(f) or {}
            except:
                pass
        return {}

    def _get_assembly_ref(self, noodling_path: str) -> str:
        """Get facet assembly reference for a noodling."""
        # Check for assembly.yaml in noodling folder
        assembly_path = os.path.join(noodling_path, "assembly.yaml")
        if os.path.exists(assembly_path):
            # Return relative path from project root
            return os.path.relpath(noodling_path, self.project_path)

        # Fall back to library reference
        noodling_name = os.path.basename(noodling_path)
        return f"library/{noodling_name}"

    def create_world_dir(self, target_dir: str) -> str:
        """
        Create a legacy-compatible world directory from project data.

        This creates JSON files in the target directory that the World class
        can load directly.

        Args:
            target_dir: Directory to create world files in

        Returns:
            Path to created world directory
        """
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, "agents"), exist_ok=True)

        stage_path = self.get_default_stage_path()
        if not stage_path:
            logger.warning("No default stage found, creating empty world")
            # Create minimal files
            self._write_json(target_dir, "rooms.json", {})
            self._write_json(target_dir, "objects.json", {})
            self._write_json(target_dir, "users.json", {})
            self._write_json(target_dir, "agents.json", {})
            self._write_json(target_dir, "stages.json", {})
            return target_dir

        # Load and convert data
        rooms = self.load_rooms_from_zones(stage_path)
        agents = self.load_agents_from_instances(stage_path)
        objects = self.load_objects_from_props(stage_path)

        # Write legacy format files
        self._write_json(target_dir, "rooms.json", rooms)
        self._write_json(target_dir, "agents.json", agents)
        self._write_json(target_dir, "objects.json", objects)
        self._write_json(target_dir, "users.json", {})  # Users managed separately

        # Create stages.json from stage.yaml
        stages = self._load_stages_json(stage_path)
        self._write_json(target_dir, "stages.json", stages)

        logger.info(f"Created world directory: {target_dir}")
        logger.info(f"  Rooms: {len(rooms)}, Agents: {len(agents)}, Objects: {len(objects)}")

        return target_dir

    def _load_stages_json(self, stage_path: str) -> Dict:
        """Load stage and convert to stages.json format."""
        stage_yaml = os.path.join(stage_path, "stage.yaml")
        if not os.path.exists(stage_yaml):
            return {}

        try:
            with open(stage_yaml, 'r') as f:
                stage = yaml.safe_load(f)

            stage_id = os.path.basename(stage_path)

            # Build entities dict from instances
            entities = {}
            instances_dir = os.path.join(stage_path, "Instances")
            if os.path.exists(instances_dir):
                for inst_name in os.listdir(instances_dir):
                    inst_path = os.path.join(instances_dir, inst_name)
                    if os.path.isdir(inst_path):
                        inst_yaml = os.path.join(inst_path, "instance.yaml")
                        if os.path.exists(inst_yaml):
                            with open(inst_yaml, 'r') as f:
                                inst = yaml.safe_load(f)
                            overrides = inst.get("overrides", {})
                            entity_id = f"agent_{inst_name}"
                            entities[entity_id] = {
                                "entity_id": entity_id,
                                "entity_type": "agent",
                                "name": overrides.get("name", inst_name),
                                "zone": overrides.get("zone", "default"),
                                "position": {"x": 0, "y": 0, "z": 0},
                                "rotation": {"x": 0, "y": 0, "z": 0}
                            }

            return {
                "room_000": {  # Map to room_000 for compatibility
                    "stage_id": stage_id,
                    "name": stage.get("name", stage_id),
                    "description": stage.get("description", ""),
                    "entities": entities,
                    "zones": {"main": list(entities.keys())},
                    "zone_graph": {"main": []}
                }
            }

        except Exception as e:
            logger.error(f"Error loading stage: {e}")
            return {}

    def _write_json(self, directory: str, filename: str, data: Dict):
        """Write JSON file."""
        path = os.path.join(directory, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def get_project_path() -> Optional[str]:
    """
    Get project path from environment or settings.

    Checks:
    1. PROJECT_PATH environment variable
    2. ~/.noodlestudio/current_project.json

    Returns:
        Project path or None if not set
    """
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


def setup_world_from_project(project_path: str, temp_world_dir: str = None) -> str:
    """
    Set up a legacy-compatible world directory from a project.

    Args:
        project_path: Path to project folder
        temp_world_dir: Optional temp directory (default: project/Library/world_cache)

    Returns:
        Path to world directory for World class
    """
    bridge = ProjectBridge(project_path)

    # Check if project has actual stage content
    stage_path = bridge.get_default_stage_path()
    if not stage_path:
        # No stages in project - fall back to legacy world
        logger.warning("Project has no stages, falling back to legacy world")
        legacy_world_dir = os.path.join(os.path.dirname(__file__), "world")
        if os.path.exists(legacy_world_dir):
            logger.info(f"Using legacy world: {legacy_world_dir}")
            return legacy_world_dir
        else:
            logger.warning("No legacy world found either, creating empty world")

    if temp_world_dir is None:
        temp_world_dir = os.path.join(project_path, "Library", "world_cache")

    return bridge.create_world_dir(temp_world_dir)
