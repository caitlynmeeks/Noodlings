"""
Project Manager - Handles project creation, loading, and structure.

Implements the PROJECT_SPEC.md specification:
- Projects are self-contained, portable folders
- Noodlings are reusable character prefabs
- Stages are continuous 3D spaces with soft zones
- Prims are scriptable prop templates
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal


class ProjectManager(QObject):
    """
    Manages NoodleStudio projects according to PROJECT_SPEC.md.

    Signals:
        projectOpened: Emitted when a project is opened (path: str)
        projectClosed: Emitted when a project is closed
        projectModified: Emitted when project metadata changes
    """

    projectOpened = pyqtSignal(str)
    projectClosed = pyqtSignal()
    projectModified = pyqtSignal()

    # Current spec version
    SPEC_VERSION = "1.0.0"

    def __init__(self):
        super().__init__()
        self.current_project_path: Optional[str] = None
        self.current_project_name: Optional[str] = None
        self._metadata: Optional[Dict] = None

    def create_project(self, parent_dir: str, project_name: str,
                       description: str = "", author: str = "") -> bool:
        """
        Create a new NoodleStudio project with full folder structure.

        Args:
            parent_dir: Parent directory where project folder will be created
            project_name: Name of the project
            description: Optional project description
            author: Optional author name

        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = os.path.join(parent_dir, project_name)
            if os.path.exists(project_path):
                return False

            # Create main project directory
            os.makedirs(project_path)

            # Create Noodlings folder (reusable character prefabs)
            os.makedirs(os.path.join(project_path, "Noodlings"))

            # Create Prims folder (reusable prop templates)
            os.makedirs(os.path.join(project_path, "Prims"))

            # Create Stages folder (scenes/worlds)
            os.makedirs(os.path.join(project_path, "Stages"))

            # Create Generations folder (AI-generated content)
            os.makedirs(os.path.join(project_path, "Generations", "Images"))
            os.makedirs(os.path.join(project_path, "Generations", "Audio"))

            # Create SharedAssets folder (project-wide resources)
            os.makedirs(os.path.join(project_path, "SharedAssets", "Skyboxes"))
            os.makedirs(os.path.join(project_path, "SharedAssets", "Music"))
            os.makedirs(os.path.join(project_path, "SharedAssets", "SoundEffects"))

            # Create Library folder (local cache - never synced)
            os.makedirs(os.path.join(project_path, "Library", "StateHistory"))
            os.makedirs(os.path.join(project_path, "Library", "ConversationLogs"))
            os.makedirs(os.path.join(project_path, "Library", "ThumbnailCache"))

            # Create project manifest
            now = self._get_timestamp()
            metadata = {
                "name": project_name,
                "version": "1.0.0",
                "spec_version": self.SPEC_VERSION,
                "created": now,
                "modified": now,
                "noodlestudio_version": "0.2.0",
                "description": description,
                "author": author,
                "tags": [],
                "default_stage": None,
                "cloud": {
                    "project_id": None,
                    "last_sync": None,
                    "sync_enabled": False
                }
            }

            metadata_path = os.path.join(project_path, "project.noodleproj")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create .gitignore
            gitignore_content = """# NoodleStudio Project
# Local cache - never sync or commit
Library/

# Temporary files
*.tmp
*.bak
*~

# OS files
.DS_Store
Thumbs.db

# Python cache
__pycache__/
*.pyc

# Optional: Generations can be large
# Generations/
"""
            gitignore_path = os.path.join(project_path, ".gitignore")
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)

            # Open the new project
            self.open_project(project_path)

            return True

        except Exception as e:
            print(f"Error creating project: {e}")
            return False

    def open_project(self, project_path: str) -> bool:
        """
        Open an existing project.

        Args:
            project_path: Path to project directory

        Returns:
            True if successful, False otherwise
        """
        try:
            metadata_path = os.path.join(project_path, "project.noodleproj")
            if not os.path.exists(metadata_path):
                return False

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Close current project if any
            if self.current_project_path:
                self.close_project()

            self.current_project_path = project_path
            self.current_project_name = metadata.get("name", os.path.basename(project_path))
            self._metadata = metadata

            # Update last opened timestamp
            metadata["last_opened"] = self._get_timestamp()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.projectOpened.emit(project_path)

            return True

        except Exception as e:
            print(f"Error opening project: {e}")
            return False

    def close_project(self):
        """Close the current project and shutdown the server."""
        if self.current_project_path:
            # Trigger graceful server shutdown
            try:
                import requests
                requests.post('http://localhost:8081/api/shutdown',
                            json={'delay': 1}, timeout=2)
            except Exception as e:
                print(f"Warning: Could not shutdown server: {e}")

            self.current_project_path = None
            self.current_project_name = None
            self._metadata = None
            self.projectClosed.emit()

    def save_project(self) -> bool:
        """Save project metadata."""
        if not self.current_project_path or not self._metadata:
            return False

        try:
            self._metadata["modified"] = self._get_timestamp()
            metadata_path = os.path.join(self.current_project_path, "project.noodleproj")
            with open(metadata_path, 'w') as f:
                json.dump(self._metadata, f, indent=2)
            self.projectModified.emit()
            return True
        except Exception as e:
            print(f"Error saving project: {e}")
            return False

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def get_noodlings_path(self) -> Optional[str]:
        """Get path to Noodlings folder."""
        if not self.current_project_path:
            return None
        return os.path.join(self.current_project_path, "Noodlings")

    def get_prims_path(self) -> Optional[str]:
        """Get path to Prims folder."""
        if not self.current_project_path:
            return None
        return os.path.join(self.current_project_path, "Prims")

    def get_stages_path(self) -> Optional[str]:
        """Get path to Stages folder."""
        if not self.current_project_path:
            return None
        return os.path.join(self.current_project_path, "Stages")

    def get_generations_path(self, subtype: str = "") -> Optional[str]:
        """Get path to Generations folder or subfolder."""
        if not self.current_project_path:
            return None
        base = os.path.join(self.current_project_path, "Generations")
        if subtype:
            return os.path.join(base, subtype)
        return base

    def get_shared_assets_path(self, subtype: str = "") -> Optional[str]:
        """Get path to SharedAssets folder or subfolder."""
        if not self.current_project_path:
            return None
        base = os.path.join(self.current_project_path, "SharedAssets")
        if subtype:
            return os.path.join(base, subtype)
        return base

    def get_library_path(self, subtype: str = "") -> Optional[str]:
        """Get path to Library folder or subfolder (local cache)."""
        if not self.current_project_path:
            return None
        base = os.path.join(self.current_project_path, "Library")
        if subtype:
            return os.path.join(base, subtype)
        return base

    # -------------------------------------------------------------------------
    # Noodling operations
    # -------------------------------------------------------------------------

    def list_noodlings(self) -> List[str]:
        """List all noodling names in the project."""
        noodlings_path = self.get_noodlings_path()
        if not noodlings_path or not os.path.exists(noodlings_path):
            return []

        return [d for d in os.listdir(noodlings_path)
                if os.path.isdir(os.path.join(noodlings_path, d))
                and os.path.exists(os.path.join(noodlings_path, d, "noodling.yaml"))]

    def get_noodling_path(self, noodling_name: str) -> Optional[str]:
        """Get path to a specific noodling folder."""
        noodlings_path = self.get_noodlings_path()
        if not noodlings_path:
            return None
        return os.path.join(noodlings_path, noodling_name)

    def create_noodling(self, name: str, description: str = "",
                        author: str = "", tags: List[str] = None) -> Optional[str]:
        """
        Create a new noodling with standard folder structure.

        Returns the path to the new noodling folder, or None on failure.
        """
        noodlings_path = self.get_noodlings_path()
        if not noodlings_path:
            return None

        # Sanitize name for filesystem
        safe_name = self._sanitize_name(name)
        noodling_path = os.path.join(noodlings_path, safe_name)

        if os.path.exists(noodling_path):
            return None

        try:
            os.makedirs(noodling_path)
            os.makedirs(os.path.join(noodling_path, "Scripts"))
            os.makedirs(os.path.join(noodling_path, "NeuralGraphs"))
            os.makedirs(os.path.join(noodling_path, "Assets", "expressions"))
            os.makedirs(os.path.join(noodling_path, "Assets", "memories"))
            os.makedirs(os.path.join(noodling_path, "Processors"))

            now = self._get_timestamp()

            # Create noodling.yaml manifest
            manifest = {
                "name": name,
                "version": "1.0.0",
                "description": description,
                "author": author or self._metadata.get("author", ""),
                "created": now,
                "modified": now,
                "tags": tags or [],
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
                    "species": "noodling",
                    "complexity": "minimal",
                    "facet_count": 0,
                    "llm_facets": 0,
                    "has_trained_weights": False,
                    "has_voice": False
                }
            }

            manifest_path = os.path.join(noodling_path, "noodling.yaml")
            self._write_yaml(manifest_path, manifest)

            # Create empty recipe.yaml
            recipe = {
                "name": name,
                "species": "noodling",
                "description": description or "A newly-formed Noodling.",
                "identity_prompt": f"You are {name}, a Noodling.\n\nDescribe your personality here.",
                "language_mode": "verbal",
                "pronouns": "they/them",
                "constraints": {
                    "max_tokens": 100,
                    "temperature": 0.8,
                    "response_cooldown": 2.0
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
                "spawn_message": "appears in a shimmer of light"
            }

            recipe_path = os.path.join(noodling_path, "recipe.yaml")
            self._write_yaml(recipe_path, recipe)

            # Create empty assembly.yaml
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
                        "prompt": "Respond as {name}.",
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

            assembly_path = os.path.join(noodling_path, "assembly.yaml")
            self._write_yaml(assembly_path, assembly)

            return noodling_path

        except Exception as e:
            print(f"Error creating noodling: {e}")
            if os.path.exists(noodling_path):
                shutil.rmtree(noodling_path)
            return None

    # -------------------------------------------------------------------------
    # Stage operations
    # -------------------------------------------------------------------------

    def list_stages(self) -> List[str]:
        """List all stage names in the project."""
        stages_path = self.get_stages_path()
        if not stages_path or not os.path.exists(stages_path):
            return []

        return [d for d in os.listdir(stages_path)
                if os.path.isdir(os.path.join(stages_path, d))
                and os.path.exists(os.path.join(stages_path, d, "stage.yaml"))]

    def get_stage_path(self, stage_name: str) -> Optional[str]:
        """Get path to a specific stage folder."""
        stages_path = self.get_stages_path()
        if not stages_path:
            return None
        return os.path.join(stages_path, stage_name)

    def create_stage(self, name: str, description: str = "") -> Optional[str]:
        """
        Create a new stage with standard folder structure.

        Returns the path to the new stage folder, or None on failure.
        """
        stages_path = self.get_stages_path()
        if not stages_path:
            return None

        safe_name = self._sanitize_name(name)
        stage_path = os.path.join(stages_path, safe_name)

        if os.path.exists(stage_path):
            return None

        try:
            os.makedirs(stage_path)
            os.makedirs(os.path.join(stage_path, "Zones"))
            os.makedirs(os.path.join(stage_path, "Instances"))
            os.makedirs(os.path.join(stage_path, "Props"))

            now = self._get_timestamp()

            # Create stage.yaml
            stage_def = {
                "name": name,
                "description": description or "A new stage",
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
                    "zone": None  # No default zone - user creates zones as needed
                },
                "zones": [],  # Empty - no auto-created zones
                "instances": [],
                "props": []
            }

            stage_yaml_path = os.path.join(stage_path, "stage.yaml")
            self._write_yaml(stage_yaml_path, stage_def)

            # NOTE: Default Zone removed - stages start empty, user creates zones via context menu

            # Set as default stage if none set
            if self._metadata and not self._metadata.get("default_stage"):
                self._metadata["default_stage"] = f"Stages/{safe_name}"
                self.save_project()

            return stage_path

        except Exception as e:
            print(f"Error creating stage: {e}")
            if os.path.exists(stage_path):
                shutil.rmtree(stage_path)
            return None

    # -------------------------------------------------------------------------
    # Prim operations
    # -------------------------------------------------------------------------

    def list_prims(self) -> List[str]:
        """List all prim template names in the project."""
        prims_path = self.get_prims_path()
        if not prims_path or not os.path.exists(prims_path):
            return []

        return [d for d in os.listdir(prims_path)
                if os.path.isdir(os.path.join(prims_path, d))
                and os.path.exists(os.path.join(prims_path, d, "prim.yaml"))]

    def get_prim_path(self, prim_name: str) -> Optional[str]:
        """Get path to a specific prim template folder."""
        prims_path = self.get_prims_path()
        if not prims_path:
            return None
        return os.path.join(prims_path, prim_name)

    def create_prim(self, name: str, description: str = "",
                    text_description: str = "") -> Optional[str]:
        """
        Create a new prim template with standard folder structure.

        Returns the path to the new prim folder, or None on failure.
        """
        prims_path = self.get_prims_path()
        if not prims_path:
            return None

        safe_name = self._sanitize_name(name)
        prim_path = os.path.join(prims_path, safe_name)

        if os.path.exists(prim_path):
            return None

        try:
            os.makedirs(prim_path)
            os.makedirs(os.path.join(prim_path, "Scripts"))
            os.makedirs(os.path.join(prim_path, "Assets"))

            # Create prim.yaml
            prim_def = {
                "name": name,
                "version": "1.0.0",
                "description": description,
                "author": self._metadata.get("author", "") if self._metadata else "",
                "tags": [],
                "display": {
                    "icon": None,
                    "model": None,
                    "text_description": text_description or f"a {name.lower()}"
                },
                "verbs": {
                    "look": {
                        "response": f"You see {text_description or f'a {name.lower()}'}."
                    }
                },
                "scripts": [],
                "events": [],
                "physics": {
                    "movable": True,
                    "container": False,
                    "size": [0.5, 0.5, 0.5]
                },
                "default_state": {}
            }

            prim_yaml_path = os.path.join(prim_path, "prim.yaml")
            self._write_yaml(prim_yaml_path, prim_def)

            return prim_path

        except Exception as e:
            print(f"Error creating prim: {e}")
            if os.path.exists(prim_path):
                shutil.rmtree(prim_path)
            return None

    # -------------------------------------------------------------------------
    # Instance operations (agents in stages)
    # -------------------------------------------------------------------------

    def create_instance(self, stage_name: str, noodling_name: str,
                        instance_name: str = None,
                        position: List[float] = None,
                        zone: str = "default") -> Optional[str]:
        """
        Create a new agent instance in a stage.

        Args:
            stage_name: Name of the stage
            noodling_name: Name of the noodling template to instantiate
            instance_name: Optional instance name (defaults to noodling name)
            position: Optional [x, y, z] position
            zone: Zone ID where the instance spawns

        Returns the path to the new instance folder, or None on failure.
        """
        stage_path = self.get_stage_path(stage_name)
        noodling_path = self.get_noodling_path(noodling_name)

        if not stage_path or not noodling_path:
            return None

        if not os.path.exists(stage_path) or not os.path.exists(noodling_path):
            return None

        safe_name = self._sanitize_name(instance_name or noodling_name)
        instance_path = os.path.join(stage_path, "Instances", safe_name)

        # Handle duplicate names
        counter = 1
        original_path = instance_path
        while os.path.exists(instance_path):
            instance_path = f"{original_path}_{counter}"
            safe_name = f"{self._sanitize_name(instance_name or noodling_name)}_{counter}"
            counter += 1

        try:
            os.makedirs(instance_path)

            now = self._get_timestamp()

            # Calculate relative path to noodling
            rel_noodling = os.path.relpath(noodling_path, instance_path)

            # Create instance.yaml
            instance_def = {
                "noodling": rel_noodling,
                "overrides": {
                    "name": instance_name or noodling_name,
                    "position": position or [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "zone": zone
                },
                "created": now,
                "last_active": now
            }

            instance_yaml_path = os.path.join(instance_path, "instance.yaml")
            self._write_yaml(instance_yaml_path, instance_def)

            # Create initial state.json
            state = {
                "instance_id": safe_name,
                "timestamp": now,
                "position": position or [0, 0, 0],
                "rotation": [0, 0, 0],
                "zone": zone,
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

            state_path = os.path.join(instance_path, "state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            return instance_path

        except Exception as e:
            print(f"Error creating instance: {e}")
            if os.path.exists(instance_path):
                shutil.rmtree(instance_path)
            return None

    # -------------------------------------------------------------------------
    # Prop operations (prims in stages)
    # -------------------------------------------------------------------------

    def create_prop(self, stage_name: str, prim_name: str,
                    prop_name: str = None,
                    position: List[float] = None,
                    zone: str = "default") -> Optional[str]:
        """
        Create a new prop instance in a stage.

        Args:
            stage_name: Name of the stage
            prim_name: Name of the prim template to instantiate
            prop_name: Optional prop name (defaults to prim name)
            position: Optional [x, y, z] position
            zone: Zone ID where the prop spawns

        Returns the path to the new prop folder, or None on failure.
        """
        stage_path = self.get_stage_path(stage_name)
        prim_path = self.get_prim_path(prim_name)

        if not stage_path or not prim_path:
            return None

        if not os.path.exists(stage_path) or not os.path.exists(prim_path):
            return None

        safe_name = self._sanitize_name(prop_name or prim_name)
        prop_path = os.path.join(stage_path, "Props", safe_name)

        # Handle duplicate names
        counter = 1
        original_path = prop_path
        while os.path.exists(prop_path):
            prop_path = f"{original_path}_{counter}"
            safe_name = f"{self._sanitize_name(prop_name or prim_name)}_{counter}"
            counter += 1

        try:
            os.makedirs(prop_path)

            now = self._get_timestamp()

            # Calculate relative path to prim
            rel_prim = os.path.relpath(prim_path, prop_path)

            # Create prop.yaml
            prop_def = {
                "prim": rel_prim,
                "name": prop_name or prim_name,
                "position": position or [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": 1.0,
                "zone": zone,
                "parent": None,
                "state": {},
                "created": now
            }

            prop_yaml_path = os.path.join(prop_path, "prop.yaml")
            self._write_yaml(prop_yaml_path, prop_def)

            # Create initial state.json
            state = {
                "prop_id": safe_name,
                "timestamp": now,
                "position": position or [0, 0, 0],
                "rotation": [0, 0, 0],
                "zone": zone,
                "state": {},
                "script_storage": {}
            }

            state_path = os.path.join(prop_path, "state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            return prop_path

        except Exception as e:
            print(f"Error creating prop: {e}")
            if os.path.exists(prop_path):
                shutil.rmtree(prop_path)
            return None

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def is_project_open(self) -> bool:
        """Check if a project is currently open."""
        return self.current_project_path is not None

    def get_metadata(self) -> Optional[Dict]:
        """Get current project metadata."""
        return self._metadata.copy() if self._metadata else None

    def update_metadata(self, updates: Dict) -> bool:
        """Update project metadata fields."""
        if not self._metadata:
            return False

        self._metadata.update(updates)
        return self.save_project()

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.now().isoformat()

    def _sanitize_name(self, name: str) -> str:
        """Convert name to filesystem-safe string."""
        # Replace spaces with underscores, remove special chars
        safe = name.lower().replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")
        return safe or "unnamed"

    def _write_yaml(self, path: str, data: Dict):
        """Write data to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                     allow_unicode=True)

    # -------------------------------------------------------------------------
    # Legacy compatibility (deprecated)
    # -------------------------------------------------------------------------

    def get_assets_path(self, asset_type: str = "") -> Optional[str]:
        """
        DEPRECATED: Use get_noodlings_path, get_prims_path, etc.

        Maintained for backward compatibility.
        """
        if not self.current_project_path:
            return None

        # Map old asset types to new locations
        type_map = {
            "Noodlings": "Noodlings",
            "Ensembles": "Noodlings",  # Ensembles are now part of Noodlings
            "Prims": "Prims",
            "Scripts": "SharedAssets",
            "Stages": "Stages"
        }

        if asset_type in type_map:
            return os.path.join(self.current_project_path, type_map[asset_type])

        return os.path.join(self.current_project_path, "SharedAssets")

    def import_ensemble(self, source_path: str) -> bool:
        """DEPRECATED: Ensembles are now part of Noodling folders."""
        print("Warning: import_ensemble is deprecated. Use create_noodling instead.")
        return False

    def import_noodling(self, source_path: str) -> bool:
        """DEPRECATED: Use create_noodling or copy noodling folder directly."""
        print("Warning: import_noodling is deprecated.")
        return False
