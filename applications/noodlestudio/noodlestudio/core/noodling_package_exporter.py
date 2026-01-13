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
#   Noodling Package Exporter - Export noodlings to Unity-compatible packages
#
#   Creates .noodling folders that Unity's NoodlingBehaviour can load.
#   Target: Christina's ToMars? VR project and future Unity integrations.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.noodling_package_exporter
# PURPOSE:  Unity package export
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NoodlingPackageExporter, ExportManifest, ExportOptions
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExportManifest:
    """Package manifest for .noodling folder."""
    name: str
    version: str = "1.0.0"
    noodlestudio_version: str = "0.9.0"
    description: str = ""
    author: str = ""
    created: str = ""
    exports: Dict[str, str] = None

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if self.exports is None:
            self.exports = {
                "character": "character.json",
                "assembly": "assembly.json",
                "expressions": "expressions.json"
            }


@dataclass
class ExportOptions:
    """Export configuration options."""
    include_plays: bool = True
    bake_prompts: bool = False  # Inline prompt templates vs reference
    expression_preset: str = "vrm"  # vrm, vrm1, custom


class NoodlingPackageExporter:
    """
    Export noodlings to .noodling packages for Unity.

    Creates a folder structure compatible with Unity's NoodlingBehaviour:
        aria.noodling/
        ├── manifest.json       # Package metadata
        ├── character.json      # Personality, motivation, initial PAD
        ├── assembly.json       # Facet configuration
        ├── expressions.json    # PAD → FACS → VRM blendshape mapping
        └── plays/              # Optional narrative beats
    """

    def __init__(self, project_manager):
        """
        Initialize exporter with project manager.

        Args:
            project_manager: The ProjectManager instance
        """
        self.project_manager = project_manager

    def export(
        self,
        noodling_name: str,
        output_path: Path,
        options: Optional[ExportOptions] = None
    ) -> Path:
        """
        Export a noodling to a .noodling package.

        Args:
            noodling_name: Name of the noodling folder in Noodlings/
            output_path: Directory to create the package in
            options: Export options

        Returns:
            Path to the created .noodling folder

        Raises:
            ValueError: If noodling not found or missing required files
        """
        options = options or ExportOptions()

        # Get noodling folder path
        noodling_path = self.project_manager.get_noodling_path(noodling_name)
        if not noodling_path or not os.path.exists(noodling_path):
            raise ValueError(f"Noodling not found: {noodling_name}")

        # Load noodling manifest
        manifest_path = os.path.join(noodling_path, "noodling.yaml")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Missing noodling.yaml in {noodling_name}")

        with open(manifest_path, 'r') as f:
            noodling_data = yaml.safe_load(f)

        # Load recipe if referenced
        recipe_data = {}
        recipe_ref = noodling_data.get('recipe')
        if recipe_ref:
            recipe_path = os.path.join(noodling_path, recipe_ref)
            if os.path.exists(recipe_path):
                with open(recipe_path, 'r') as f:
                    recipe_data = yaml.safe_load(f)

        # Load assembly if referenced
        assembly_data = {}
        assembly_ref = noodling_data.get('assembly')
        if assembly_ref:
            assembly_path = os.path.join(noodling_path, assembly_ref)
            if os.path.exists(assembly_path):
                with open(assembly_path, 'r') as f:
                    assembly_data = yaml.safe_load(f)

        # Create package folder
        safe_name = noodling_name.lower().replace(' ', '_').replace('-', '_')
        package_name = f"{safe_name}.noodling"
        package_path = Path(output_path) / package_name
        package_path.mkdir(parents=True, exist_ok=True)

        # Export each component
        self._export_manifest(noodling_data, recipe_data, package_path, options)
        self._export_character(noodling_data, recipe_data, package_path)
        self._export_assembly(assembly_data, package_path, options)
        self._export_expressions(package_path, options)

        if options.include_plays:
            self._export_plays(noodling_path, noodling_name, package_path)

        logger.info(f"Exported {noodling_name} to {package_path}")
        return package_path

    def _export_manifest(
        self,
        noodling_data: Dict,
        recipe_data: Dict,
        package_path: Path,
        options: ExportOptions
    ):
        """Write manifest.json."""
        name = recipe_data.get('name') or noodling_data.get('name', 'Unknown')

        manifest = ExportManifest(
            name=name,
            version=noodling_data.get('version', '1.0.0'),
            description=noodling_data.get('description', ''),
            author=noodling_data.get('author', '')
        )

        if options.include_plays:
            manifest.exports["plays"] = "plays/"

        self._write_json(package_path / "manifest.json", asdict(manifest))

    def _export_character(
        self,
        noodling_data: Dict,
        recipe_data: Dict,
        package_path: Path
    ):
        """Write character.json with personality and PAD state."""
        name = recipe_data.get('name') or noodling_data.get('name', 'Unknown')

        # Extract initial PAD from affect_baseline or spawn_defaults
        affect = recipe_data.get('affect_baseline') or {}
        spawn_affect = noodling_data.get('spawn_defaults', {}).get('affect', {})

        # Merge with spawn_defaults as fallback
        if not affect:
            affect = spawn_affect

        # Map 5D internal (valence) to 3D Unity (pleasure)
        initial_pad = {
            "pleasure": affect.get('valence', 0.0),
            "arousal": affect.get('arousal', 0.5),
            "dominance": affect.get('dominance', 0.5)
        }

        # Extract personality traits from OCEAN model or tags
        personality = recipe_data.get('personality', {})
        personality_traits = []
        if personality:
            # Convert OCEAN scores to trait descriptions
            if personality.get('openness', 0) > 0.7:
                personality_traits.append('curious')
            if personality.get('conscientiousness', 0) > 0.7:
                personality_traits.append('disciplined')
            if personality.get('extraversion', 0) > 0.6:
                personality_traits.append('outgoing')
            elif personality.get('extraversion', 0) < 0.4:
                personality_traits.append('introspective')
            if personality.get('agreeableness', 0) > 0.7:
                personality_traits.append('kind')
            if personality.get('neuroticism', 0) < 0.3:
                personality_traits.append('calm')
            elif personality.get('neuroticism', 0) > 0.7:
                personality_traits.append('sensitive')

        # Add tags as traits if available
        tags = noodling_data.get('tags', [])
        personality_traits.extend(tags[:5])  # Limit to avoid bloat

        # Extract voice info
        voice = {}
        if 'identity_prompt' in recipe_data:
            # Try to extract speech pattern hints
            voice['tone'] = 'as defined in identity prompt'
        if 'vocalizations' in recipe_data:
            voice['vocalizations'] = [v.get('sound') for v in recipe_data['vocalizations']]

        # Build motivation from identity_prompt or description
        motivation = recipe_data.get('description', '') or noodling_data.get('description', '')

        character = {
            "id": noodling_data.get('id', name.lower().replace(' ', '_')),
            "name": name,
            "full_name": name,
            "role": recipe_data.get('species', ''),
            "initial_pad": initial_pad,
            "motivation": motivation[:500] if motivation else '',  # Truncate if too long
            "personality_traits": personality_traits,
            "voice": voice,
            "backstory": recipe_data.get('description', '')[:1000] if recipe_data.get('description') else ''
        }

        self._write_json(package_path / "character.json", character)

    def _export_assembly(
        self,
        assembly_data: Dict,
        package_path: Path,
        options: ExportOptions
    ):
        """Write assembly.json with facet definitions."""
        if not assembly_data:
            # Create minimal assembly
            assembly_export = {
                "id": "default_cognition",
                "name": "Default Cognition Assembly",
                "version": "1.0.0",
                "facets": [],
                "connections": []
            }
        else:
            facets = self._serialize_facets(assembly_data.get('facets', []), options)
            connections = self._serialize_connections(assembly_data.get('connections', []))

            assembly_export = {
                "id": assembly_data.get('name', 'assembly').lower().replace(' ', '_'),
                "name": assembly_data.get('name', 'Cognition Assembly'),
                "version": assembly_data.get('version', '1.0.0'),
                "facets": facets,
                "connections": connections
            }

            if options.bake_prompts:
                assembly_export["prompt_templates"] = self._bake_prompts(
                    assembly_data.get('facets', [])
                )

        self._write_json(package_path / "assembly.json", assembly_export)

    def _serialize_facets(self, facets: List[Dict], options: ExportOptions) -> List[Dict]:
        """Serialize facets to Unity-compatible format."""
        serialized = []
        for facet in facets:
            facet_data = {
                "id": facet.get('id', ''),
                "name": facet.get('name', facet.get('id', '')),
                "type": facet.get('type', 'Unknown'),
                "description": ""
            }

            # Handle LLM facets specially
            if facet.get('type') == 'LLM':
                config = facet.get('config', {})
                facet_data["prompt_template"] = config.get('prompt', '')
                facet_data["model"] = config.get('model_label', 'SMALL')
                # Extract input/output from connections (simplified)
                facet_data["inputs"] = ["input"]
                facet_data["outputs"] = ["output"]

            serialized.append(facet_data)
        return serialized

    def _serialize_connections(self, connections: List[Dict]) -> List[Dict]:
        """Serialize connections to Unity format."""
        serialized = []
        for conn in connections:
            from_str = f"{conn.get('from', '')}.{conn.get('from_pad', 'output')}"
            to_str = f"{conn.get('to', '')}.{conn.get('to_pad', 'input')}"
            serialized.append({
                "from": from_str,
                "to": to_str
            })
        return serialized

    def _bake_prompts(self, facets: List[Dict]) -> Dict[str, str]:
        """Inline all prompt templates from LLM facets."""
        templates = {}
        for facet in facets:
            if facet.get('type') == 'LLM':
                config = facet.get('config', {})
                prompt = config.get('prompt', '')
                if prompt:
                    template_name = f"{facet.get('id', 'unknown')}.prompt"
                    templates[template_name] = prompt
        return templates

    def _export_expressions(self, package_path: Path, options: ExportOptions):
        """
        Write expressions.json with standard PAD → FACS → VRM mapping.

        Uses established Mehrabian/Russell PAD-to-emotion weights.
        """
        expressions = {
            "mapping_version": "1.0.0",
            "avatar_type": options.expression_preset.upper(),

            # PAD → Emotion weights (Mehrabian/Russell model)
            "pad_to_emotion_weights": {
                "joy": {"pleasure": 0.8, "arousal": 0.3, "dominance": 0.2},
                "sadness": {"pleasure": -0.7, "arousal": -0.3, "dominance": -0.3},
                "anger": {"pleasure": -0.5, "arousal": 0.7, "dominance": 0.5},
                "fear": {"pleasure": -0.6, "arousal": 0.7, "dominance": -0.6},
                "surprise": {"pleasure": 0.0, "arousal": 0.8, "dominance": 0.0},
                "disgust": {"pleasure": -0.6, "arousal": 0.2, "dominance": 0.3},
                "contempt": {"pleasure": -0.3, "arousal": 0.1, "dominance": 0.6},
                "concentration": {"pleasure": 0.0, "arousal": 0.4, "dominance": 0.4}
            },

            # Emotion → FACS Action Units
            "emotion_to_aus": {
                "joy": {"AU6": 0.8, "AU12": 0.9},
                "sadness": {"AU1": 0.7, "AU4": 0.5, "AU15": 0.6},
                "anger": {"AU4": 0.8, "AU5": 0.5, "AU7": 0.6, "AU23": 0.7},
                "fear": {"AU1": 0.8, "AU2": 0.7, "AU4": 0.5, "AU5": 0.9, "AU20": 0.6},
                "surprise": {"AU1": 0.9, "AU2": 0.9, "AU5": 0.8, "AU26": 0.7},
                "disgust": {"AU9": 0.7, "AU15": 0.5, "AU16": 0.4},
                "contempt": {"AU12": 0.3, "AU14": 0.6},
                "concentration": {"AU4": 0.4, "AU7": 0.3}
            },

            # FACS AU → VRM Blendshapes
            "au_to_vrm_blendshapes": {
                "AU1": [{"blendshape": "Brow_InnerUp", "weight": 1.0}],
                "AU2": [{"blendshape": "Brow_OuterUp", "weight": 1.0}],
                "AU4": [{"blendshape": "Brow_Down", "weight": 1.0}],
                "AU5": [{"blendshape": "Eye_Wide", "weight": 1.0}],
                "AU6": [{"blendshape": "Cheek_Raise", "weight": 1.0}],
                "AU7": [{"blendshape": "Eye_Squint", "weight": 1.0}],
                "AU9": [{"blendshape": "Nose_Wrinkle", "weight": 1.0}],
                "AU12": [{"blendshape": "Mouth_Smile", "weight": 1.0}],
                "AU14": [{"blendshape": "Mouth_Dimple", "weight": 1.0}],
                "AU15": [{"blendshape": "Mouth_Frown", "weight": 1.0}],
                "AU16": [{"blendshape": "Mouth_LowerDown", "weight": 1.0}],
                "AU20": [{"blendshape": "Mouth_Stretch", "weight": 1.0}],
                "AU23": [{"blendshape": "Mouth_Tight", "weight": 1.0}],
                "AU26": [{"blendshape": "Jaw_Open", "weight": 0.5}]
            },

            "transition_settings": {
                "blend_duration_ms": 200,
                "idle_variation": True,
                "blink_rate_per_minute": 15
            }
        }

        self._write_json(package_path / "expressions.json", expressions)

    def _export_plays(self, noodling_path: str, noodling_name: str, package_path: Path):
        """Export plays from noodling's plays/ subfolder."""
        plays_source = Path(noodling_path) / "plays"
        plays_dest = package_path / "plays"

        if not plays_source.exists():
            # No plays to export
            return

        plays_dest.mkdir(exist_ok=True)

        # Export each .play.yaml or .play.json
        for play_file in plays_source.glob("*.play.*"):
            try:
                if play_file.suffix in ['.yaml', '.yml']:
                    with open(play_file, 'r') as f:
                        play_data = yaml.safe_load(f)
                elif play_file.suffix == '.json':
                    with open(play_file, 'r') as f:
                        play_data = json.load(f)
                else:
                    continue

                # Convert to Unity format
                unity_play = self._serialize_play(play_data, noodling_name)

                # Write as .play.json
                out_name = play_file.stem.replace('.play', '') + '.play.json'
                self._write_json(plays_dest / out_name, unity_play)

            except Exception as e:
                logger.warning(f"Failed to export play {play_file}: {e}")

    def _serialize_play(self, play_data: Dict, noodling_name: str) -> Dict:
        """Serialize a play to Unity JSON format."""
        # Handle different play formats
        characters = play_data.get('characters', {})
        beats = play_data.get('beats', [])

        # Convert characters
        unity_characters = {}
        for char_id, char_data in characters.items():
            if isinstance(char_data, dict):
                unity_characters[char_id] = {
                    "voice": char_data.get('voice', ''),
                    "initial_pad": char_data.get('initial_pad', {
                        "pleasure": 0.0,
                        "arousal": 0.5,
                        "dominance": 0.5
                    })
                }

        # Convert beats
        unity_beats = []
        for beat in beats:
            unity_beat = {
                "id": beat.get('id', ''),
                "character": beat.get('character', noodling_name),
                "speaks": beat.get('speaks', beat.get('line', '')),
                "pad_drift": beat.get('pad_drift', {}),
                "computer_use": beat.get('computer_use', None),
                "wait_after": beat.get('wait_after', 0)
            }
            unity_beats.append(unity_beat)

        return {
            "name": play_data.get('name', 'Untitled Play'),
            "version": play_data.get('version', '1.0'),
            "characters": unity_characters,
            "beats": unity_beats
        }

    def _write_json(self, path: Path, data: Dict):
        """Write JSON with pretty formatting."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = [
    'NoodlingPackageExporter',
    'ExportManifest',
    'ExportOptions',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
