# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Project Migrator - Converts legacy data to PROJECT_SPEC.md format.
#
#   Migrates from: - cmush/world/agents.json → Stages/*/Insta...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.project_migrator
# PURPOSE:  Project Migrator
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ProjectMigrator, migrate_to_project()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml


class ProjectMigrator:
    """
    Migrates legacy NoodleStudio/noodleMUSH data to the new project format.
    """

    def __init__(self, source_root: str, target_project_path: str):
        """
        Args:
            source_root: Root of the noodlings_clean repository
            target_project_path: Path to the new project folder to create
        """
        self.source_root = source_root
        self.target_path = target_project_path
        self.log: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def migrate(self, dry_run: bool = False) -> bool:
        """
        Perform the migration.

        Args:
            dry_run: If True, only log what would be done without making changes

        Returns:
            True if successful (or dry run completed), False on critical error
        """
        self._log(f"Starting migration from {self.source_root}")
        self._log(f"Target project: {self.target_path}")
        self._log(f"Dry run: {dry_run}")

        try:
            # Step 1: Create project structure
            if not dry_run:
                self._create_project_structure()

            # Step 2: Migrate recipes to Noodlings
            self._migrate_recipes(dry_run)

            # Step 3: Migrate facet assemblies
            self._migrate_assemblies(dry_run)

            # Step 4: Migrate library noodlings
            self._migrate_library_noodlings(dry_run)

            # Step 5: Migrate neural canvas files
            self._migrate_neural_canvas(dry_run)

            # Step 6: Migrate stages/rooms to Stages with Zones
            self._migrate_stages(dry_run)

            # Step 7: Migrate agents to Instances
            self._migrate_agents(dry_run)

            # Step 8: Migrate generations
            self._migrate_generations(dry_run)

            # Step 9: Create project manifest
            if not dry_run:
                self._create_project_manifest()

            self._log("Migration completed successfully")
            return True

        except Exception as e:
            self._error(f"Migration failed: {e}")
            import traceback
            self._error(traceback.format_exc())
            return False

    def get_report(self) -> str:
        """Get a formatted migration report."""
        lines = ["=" * 60, "MIGRATION REPORT", "=" * 60, ""]

        lines.append("LOG:")
        for entry in self.log:
            lines.append(f"  {entry}")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for entry in self.warnings:
                lines.append(f"  {entry}")

        if self.errors:
            lines.append("")
            lines.append("ERRORS:")
            for entry in self.errors:
                lines.append(f"  {entry}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Migration steps
    # -------------------------------------------------------------------------

    def _create_project_structure(self):
        """Create the new project folder structure."""
        self._log("Creating project structure...")

        os.makedirs(self.target_path, exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Noodlings"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Prims"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Stages"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Generations", "Images"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Generations", "Audio"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "SharedAssets"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Library", "StateHistory"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "Library", "ConversationLogs"), exist_ok=True)

    def _migrate_recipes(self, dry_run: bool):
        """Migrate recipe YAML files from cmush/recipes/."""
        recipes_path = os.path.join(self.source_root, "applications", "cmush", "recipes")

        if not os.path.exists(recipes_path):
            self._warn(f"Recipes path not found: {recipes_path}")
            return

        self._log(f"Migrating recipes from {recipes_path}")

        for filename in os.listdir(recipes_path):
            if not filename.endswith(".yaml"):
                continue

            recipe_path = os.path.join(recipes_path, filename)
            self._log(f"  Processing recipe: {filename}")

            try:
                with open(recipe_path, 'r') as f:
                    recipe = yaml.safe_load(f)

                if not recipe:
                    self._warn(f"  Empty recipe: {filename}")
                    continue

                # Determine noodling name
                name = recipe.get("name", filename.replace(".yaml", ""))
                safe_name = self._sanitize_name(name)

                if not dry_run:
                    noodling_path = os.path.join(self.target_path, "Noodlings", safe_name)
                    os.makedirs(noodling_path, exist_ok=True)
                    os.makedirs(os.path.join(noodling_path, "Scripts"), exist_ok=True)
                    os.makedirs(os.path.join(noodling_path, "NeuralGraphs"), exist_ok=True)
                    os.makedirs(os.path.join(noodling_path, "Assets"), exist_ok=True)
                    os.makedirs(os.path.join(noodling_path, "Processors"), exist_ok=True)

                    # Copy recipe
                    target_recipe = os.path.join(noodling_path, "recipe.yaml")
                    shutil.copy2(recipe_path, target_recipe)

                    # Create noodling.yaml manifest
                    manifest = self._create_noodling_manifest(recipe, safe_name)
                    manifest_path = os.path.join(noodling_path, "noodling.yaml")
                    self._write_yaml(manifest_path, manifest)

                self._log(f"  Migrated: {name} -> Noodlings/{safe_name}/")

            except Exception as e:
                self._error(f"  Failed to migrate {filename}: {e}")

    def _migrate_assemblies(self, dry_run: bool):
        """Migrate facet assembly YAML files."""
        assemblies_path = os.path.join(self.source_root, "applications", "noodlestudio",
                                        "facet_assemblies")

        if not os.path.exists(assemblies_path):
            self._warn(f"Assemblies path not found: {assemblies_path}")
            return

        self._log(f"Migrating assemblies from {assemblies_path}")

        for filename in os.listdir(assemblies_path):
            if not filename.endswith(".yaml"):
                continue

            assembly_path = os.path.join(assemblies_path, filename)
            self._log(f"  Processing assembly: {filename}")

            try:
                with open(assembly_path, 'r') as f:
                    assembly = yaml.safe_load(f)

                if not assembly:
                    continue

                # Try to find matching noodling
                name = assembly.get("name", filename.replace(".yaml", "").replace("_default", ""))

                # Map common assembly names to noodlings
                name_map = {
                    "anklebiter_default": "anklebiter",
                    "red_fire_anklebiter": "red_fire_anklebiter",
                    "red_fire_anklebiter_minimal": "red_fire_anklebiter",
                    "callie_default": "callie",
                    "mr_toad": "mr_toad",
                    "empty_noodling_default": "empty_noodling",
                    "simple_test": "simple_test"
                }

                base_name = filename.replace(".yaml", "")
                safe_name = self._sanitize_name(name_map.get(base_name, base_name))

                if not dry_run:
                    noodling_path = os.path.join(self.target_path, "Noodlings", safe_name)
                    os.makedirs(noodling_path, exist_ok=True)

                    target_assembly = os.path.join(noodling_path, "assembly.yaml")
                    shutil.copy2(assembly_path, target_assembly)

                self._log(f"  Migrated: {filename} -> Noodlings/{safe_name}/assembly.yaml")

            except Exception as e:
                self._error(f"  Failed to migrate {filename}: {e}")

    def _migrate_library_noodlings(self, dry_run: bool):
        """Migrate noodlings from library folder."""
        library_path = os.path.join(self.source_root, "applications", "noodlestudio",
                                    "library", "noodlings")

        if not os.path.exists(library_path):
            self._warn(f"Library noodlings path not found: {library_path}")
            return

        self._log(f"Migrating library noodlings from {library_path}")

        for noodling_name in os.listdir(library_path):
            noodling_dir = os.path.join(library_path, noodling_name)
            if not os.path.isdir(noodling_dir):
                continue

            self._log(f"  Processing: {noodling_name}")

            try:
                if not dry_run:
                    target_dir = os.path.join(self.target_path, "Noodlings", noodling_name)

                    # Copy entire folder
                    if os.path.exists(target_dir):
                        # Merge - don't overwrite existing
                        for item in os.listdir(noodling_dir):
                            src = os.path.join(noodling_dir, item)
                            dst = os.path.join(target_dir, item)
                            if not os.path.exists(dst):
                                if os.path.isdir(src):
                                    shutil.copytree(src, dst)
                                else:
                                    shutil.copy2(src, dst)
                    else:
                        shutil.copytree(noodling_dir, target_dir)

                    # Ensure required subdirs exist
                    os.makedirs(os.path.join(target_dir, "Scripts"), exist_ok=True)
                    os.makedirs(os.path.join(target_dir, "NeuralGraphs"), exist_ok=True)
                    os.makedirs(os.path.join(target_dir, "Assets"), exist_ok=True)
                    os.makedirs(os.path.join(target_dir, "Processors"), exist_ok=True)

                self._log(f"  Migrated: {noodling_name}")

            except Exception as e:
                self._error(f"  Failed to migrate {noodling_name}: {e}")

    def _migrate_neural_canvas(self, dry_run: bool):
        """Migrate .nncanvas files."""
        canvas_path = os.path.join(self.source_root, "facet_assemblies", "charm_networks")

        if not os.path.exists(canvas_path):
            self._log("No neural canvas files to migrate")
            return

        self._log(f"Migrating neural canvas files from {canvas_path}")

        for filename in os.listdir(canvas_path):
            if not filename.endswith(".nncanvas"):
                continue

            src_path = os.path.join(canvas_path, filename)
            self._log(f"  Processing: {filename}")

            # Copy to SharedAssets for now (can be linked to specific noodlings later)
            if not dry_run:
                shared_path = os.path.join(self.target_path, "SharedAssets", "NeuralGraphs")
                os.makedirs(shared_path, exist_ok=True)
                shutil.copy2(src_path, os.path.join(shared_path, filename))

            self._log(f"  Migrated: {filename}")

    def _migrate_stages(self, dry_run: bool):
        """Migrate stages and rooms to new Stage format with Zones."""
        world_path = os.path.join(self.source_root, "applications", "cmush", "world")

        # Load rooms.json
        rooms_path = os.path.join(world_path, "rooms.json")
        rooms = {}
        if os.path.exists(rooms_path):
            with open(rooms_path, 'r') as f:
                rooms = json.load(f)
            self._log(f"Loaded {len(rooms)} rooms from rooms.json")

        # Load stages.json
        stages_path = os.path.join(world_path, "stages.json")
        stages = {}
        if os.path.exists(stages_path):
            with open(stages_path, 'r') as f:
                stages = json.load(f)
            self._log(f"Loaded {len(stages)} stages from stages.json")

        # Create a single stage with rooms as zones
        # (In the old model, rooms were discrete; in new model, they're overlapping zones)
        if not dry_run:
            stage_name = "the_nexus"
            stage_path = os.path.join(self.target_path, "Stages", stage_name)
            os.makedirs(stage_path, exist_ok=True)
            os.makedirs(os.path.join(stage_path, "Zones"), exist_ok=True)
            os.makedirs(os.path.join(stage_path, "Instances"), exist_ok=True)
            os.makedirs(os.path.join(stage_path, "Props"), exist_ok=True)

            # Create stage.yaml
            stage_def = {
                "name": "The Nexus",
                "description": "The main world - migrated from noodleMUSH",
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "geometry": None,
                "world": {
                    "bounds": {
                        "min": [-500, 0, -500],
                        "max": [500, 100, 500]
                    },
                    "ambient": {
                        "time_of_day": "night",
                        "weather": "clear",
                        "soundscape": "forest_night"
                    }
                },
                "spawn": {
                    "position": [0, 0, 0],
                    "zone": "room_000"
                },
                "zones": [],
                "instances": [],
                "props": []
            }

            # Convert rooms to zones
            zone_positions = self._calculate_zone_positions(rooms)

            for room_id, room in rooms.items():
                zone_file = f"{room_id}.zone.yaml"
                stage_def["zones"].append(f"Zones/{zone_file}")

                pos = zone_positions.get(room_id, [0, 0, 0])

                zone_def = {
                    "name": room.get("name", room_id),
                    "id": room_id,
                    "spatial": {
                        "center": pos,
                        "radius": 30.0,
                        "falloff": 15.0,
                        "shape": "sphere"
                    },
                    "text": {
                        "description": room.get("description", ""),
                        "features": room.get("objects", []),
                        "exits": room.get("exits", {})
                    },
                    "perception": {
                        "visibility": 25.0,
                        "audibility": 40.0,
                        "lighting": "ambient"
                    },
                    "ambient": {
                        "sounds": [],
                        "mood": "neutral",
                        "temperature": "comfortable"
                    }
                }

                zone_path = os.path.join(stage_path, "Zones", zone_file)
                self._write_yaml(zone_path, zone_def)
                self._log(f"  Created zone: {room_id} ({room.get('name', 'Unnamed')})")

            stage_yaml_path = os.path.join(stage_path, "stage.yaml")
            self._write_yaml(stage_yaml_path, stage_def)
            self._log(f"Created stage: {stage_name} with {len(rooms)} zones")

    def _migrate_agents(self, dry_run: bool):
        """Migrate agents to Instances in stages."""
        agents_path = os.path.join(self.source_root, "applications", "cmush",
                                   "world", "agents.json")

        if not os.path.exists(agents_path):
            self._warn("agents.json not found")
            return

        with open(agents_path, 'r') as f:
            agents = json.load(f)

        self._log(f"Migrating {len(agents)} agents")

        stage_path = os.path.join(self.target_path, "Stages", "the_nexus")
        instances_path = os.path.join(stage_path, "Instances")

        for agent_id, agent in agents.items():
            self._log(f"  Processing agent: {agent_id}")

            try:
                # Find corresponding noodling
                name = agent.get("name", agent_id)
                safe_name = self._sanitize_name(name)

                # Check if noodling exists
                noodling_path = os.path.join(self.target_path, "Noodlings", safe_name)
                if not os.path.exists(noodling_path):
                    # Try to match by facet_assembly reference
                    assembly_ref = agent.get("facet_assembly", "")
                    if "empty_noodling" in assembly_ref:
                        noodling_ref = "empty_noodling"
                    elif "red_fire" in assembly_ref:
                        noodling_ref = "red_fire_anklebiter"
                    else:
                        noodling_ref = "empty_noodling"
                        self._warn(f"  No matching noodling for {agent_id}, using empty_noodling")
                else:
                    noodling_ref = safe_name

                if not dry_run:
                    instance_path = os.path.join(instances_path, safe_name)
                    os.makedirs(instance_path, exist_ok=True)

                    # Create instance.yaml
                    rel_noodling = os.path.relpath(
                        os.path.join(self.target_path, "Noodlings", noodling_ref),
                        instance_path
                    )

                    instance_def = {
                        "noodling": rel_noodling,
                        "overrides": {
                            "name": name,
                            "position": [0, 0, 0],
                            "rotation": [0, 0, 0],
                            "zone": agent.get("current_room", "room_000")
                        },
                        "created": datetime.now().isoformat(),
                        "last_active": datetime.now().isoformat()
                    }

                    instance_yaml = os.path.join(instance_path, "instance.yaml")
                    self._write_yaml(instance_yaml, instance_def)

                    # Migrate agent state if exists
                    old_state_path = os.path.join(
                        self.source_root, "applications", "cmush", "world",
                        "agents", agent_id, "agent_state.json"
                    )
                    if os.path.exists(old_state_path):
                        with open(old_state_path, 'r') as f:
                            old_state = json.load(f)

                        new_state = {
                            "instance_id": safe_name,
                            "timestamp": datetime.now().isoformat(),
                            "position": [0, 0, 0],
                            "rotation": [0, 0, 0],
                            "zone": agent.get("current_room", "room_000"),
                            "affect": old_state.get("affect", {
                                "valence": 0.0,
                                "arousal": 0.3,
                                "dominance": 0.5,
                                "boredom": 0.0,
                                "sorrow": 0.0
                            }),
                            "charm_state": old_state.get("charm_state"),
                            "memories": old_state.get("memories", {
                                "short_term": [],
                                "episodic": []
                            }),
                            "script_storage": {}
                        }

                        state_path = os.path.join(instance_path, "state.json")
                        with open(state_path, 'w') as f:
                            json.dump(new_state, f, indent=2)

                self._log(f"  Migrated: {agent_id} -> Instances/{safe_name}/")

            except Exception as e:
                self._error(f"  Failed to migrate {agent_id}: {e}")

    def _migrate_generations(self, dry_run: bool):
        """Migrate AI-generated content."""
        gen_path = os.path.join(self.source_root, "applications", "noodlestudio",
                               "library", "Generations")

        if not os.path.exists(gen_path):
            self._log("No generations to migrate")
            return

        self._log(f"Migrating generations from {gen_path}")

        if not dry_run:
            target_gen = os.path.join(self.target_path, "Generations")
            for subdir in os.listdir(gen_path):
                src = os.path.join(gen_path, subdir)
                dst = os.path.join(target_gen, subdir)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        # Merge
                        for item in os.listdir(src):
                            item_src = os.path.join(src, item)
                            item_dst = os.path.join(dst, item)
                            if not os.path.exists(item_dst):
                                if os.path.isdir(item_src):
                                    shutil.copytree(item_src, item_dst)
                                else:
                                    shutil.copy2(item_src, item_dst)
                    else:
                        shutil.copytree(src, dst)

        self._log("Generations migrated")

    def _create_project_manifest(self):
        """Create project.noodleproj manifest."""
        manifest = {
            "name": "Migrated Project",
            "version": "1.0.0",
            "spec_version": "1.0.0",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "noodlestudio_version": "0.2.0",
            "description": "Project migrated from legacy noodleMUSH data",
            "author": "",
            "tags": ["migrated"],
            "default_stage": "Stages/the_nexus",
            "cloud": {
                "project_id": None,
                "last_sync": None,
                "sync_enabled": False
            }
        }

        manifest_path = os.path.join(self.target_path, "project.noodleproj")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Create .gitignore
        gitignore = """# NoodleStudio Project
Library/
*.tmp
*.bak
.DS_Store
__pycache__/
"""
        gitignore_path = os.path.join(self.target_path, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write(gitignore)

        self._log("Project manifest created")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _create_noodling_manifest(self, recipe: Dict, safe_name: str) -> Dict:
        """Create a noodling.yaml manifest from a recipe."""
        return {
            "name": recipe.get("name", safe_name),
            "version": "1.0.0",
            "description": recipe.get("description", ""),
            "author": "",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "tags": [],
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
                "personality": "",
                "species": recipe.get("species", "noodling"),
                "complexity": "minimal",
                "facet_count": 0,
                "llm_facets": 0,
                "has_trained_weights": False,
                "has_voice": False
            }
        }

    def _calculate_zone_positions(self, rooms: Dict) -> Dict[str, List[float]]:
        """
        Calculate 3D positions for zones based on room exit relationships.

        Places rooms in a grid based on their connections.
        """
        positions = {}
        visited = set()
        spacing = 60.0  # Distance between zone centers

        def place_room(room_id: str, x: float, z: float):
            if room_id in visited:
                return
            visited.add(room_id)
            positions[room_id] = [x, 0, z]

            room = rooms.get(room_id, {})
            exits = room.get("exits", {})

            # Place connected rooms
            if "north" in exits and exits["north"] not in visited:
                place_room(exits["north"], x, z - spacing)
            if "south" in exits and exits["south"] not in visited:
                place_room(exits["south"], x, z + spacing)
            if "east" in exits and exits["east"] not in visited:
                place_room(exits["east"], x + spacing, z)
            if "west" in exits and exits["west"] not in visited:
                place_room(exits["west"], x - spacing, z)

        # Start with room_000 at origin
        if "room_000" in rooms:
            place_room("room_000", 0, 0)

        # Place any remaining unconnected rooms
        offset = len(positions) * spacing
        for room_id in rooms:
            if room_id not in positions:
                positions[room_id] = [offset, 0, 0]
                offset += spacing

        return positions

    def _sanitize_name(self, name: str) -> str:
        """Convert name to filesystem-safe string."""
        safe = name.lower().replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")
        return safe or "unnamed"

    def _write_yaml(self, path: str, data: Dict):
        """Write YAML file."""
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                     allow_unicode=True)

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"[MIGRATE] {msg}")

    def _warn(self, msg: str):
        self.warnings.append(msg)
        print(f"[MIGRATE WARNING] {msg}")

    def _error(self, msg: str):
        self.errors.append(msg)
        print(f"[MIGRATE ERROR] {msg}")


def migrate_to_project(source_root: str, target_path: str, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Convenience function to migrate legacy data to new project format.

    Args:
        source_root: Root of noodlings_clean repository
        target_path: Path for new project
        dry_run: If True, only report what would be done

    Returns:
        (success, report)
    """
    migrator = ProjectMigrator(source_root, target_path)
    success = migrator.migrate(dry_run=dry_run)
    return success, migrator.get_report()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python project_migrator.py <source_root> <target_project> [--dry-run]")
        print("")
        print("Example:")
        print("  python project_migrator.py /path/to/noodlings_clean /path/to/MyProject")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]
    dry_run = "--dry-run" in sys.argv

    success, report = migrate_to_project(source, target, dry_run)
    print(report)
    sys.exit(0 if success else 1)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
