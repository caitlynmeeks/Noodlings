# ──────────────────────────────────────────────────────────────
#   Tests for Noodling Package Exporter
#
#   Tests for Unity package export functionality.
#   Ensures .noodling packages are correctly formatted for Unity.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaml


# =============================================================================
# ExportManifest Tests
# =============================================================================

class TestExportManifest:
    """Tests for ExportManifest dataclass."""

    def test_default_values(self):
        """Default manifest has expected values."""
        from noodlestudio.core.noodling_package_exporter import ExportManifest
        manifest = ExportManifest(name="TestNoodling")

        assert manifest.name == "TestNoodling"
        assert manifest.version == "1.0.0"
        assert manifest.noodlestudio_version == "0.9.0"
        assert manifest.description == ""
        assert manifest.author == ""
        assert manifest.created != ""  # Auto-set
        assert "character" in manifest.exports
        assert "assembly" in manifest.exports
        assert "expressions" in manifest.exports

    def test_custom_values(self):
        """Manifest accepts custom values."""
        from noodlestudio.core.noodling_package_exporter import ExportManifest
        manifest = ExportManifest(
            name="ARIA",
            version="2.0.0",
            description="AI pilot for ToMars?",
            author="Christina"
        )

        assert manifest.name == "ARIA"
        assert manifest.version == "2.0.0"
        assert manifest.description == "AI pilot for ToMars?"
        assert manifest.author == "Christina"


class TestExportOptions:
    """Tests for ExportOptions dataclass."""

    def test_default_values(self):
        """Default options have expected values."""
        from noodlestudio.core.noodling_package_exporter import ExportOptions
        options = ExportOptions()

        assert options.include_plays is True
        assert options.bake_prompts is False
        assert options.expression_preset == "vrm"

    def test_custom_values(self):
        """Options accept custom values."""
        from noodlestudio.core.noodling_package_exporter import ExportOptions
        options = ExportOptions(
            include_plays=False,
            bake_prompts=True,
            expression_preset="vrm1"
        )

        assert options.include_plays is False
        assert options.bake_prompts is True
        assert options.expression_preset == "vrm1"


# =============================================================================
# NoodlingPackageExporter Tests
# =============================================================================

class TestNoodlingPackageExporter:
    """Tests for NoodlingPackageExporter class."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure with a test noodling."""
        # Create project structure
        noodlings_path = tmp_path / "Noodlings"
        noodlings_path.mkdir()

        # Create a test noodling
        test_noodling = noodlings_path / "test_character"
        test_noodling.mkdir()

        # Create noodling.yaml
        noodling_yaml = {
            "id": "com.test.character",
            "name": "Test Character",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "A test character for export.",
            "tags": ["test", "demo"],
            "recipe": "recipe.yaml",
            "assembly": "assembly.yaml",
            "spawn_defaults": {
                "affect": {
                    "valence": 0.5,
                    "arousal": 0.4,
                    "dominance": 0.6,
                    "boredom": 0.1,
                    "sorrow": 0.0
                }
            }
        }
        with open(test_noodling / "noodling.yaml", 'w') as f:
            yaml.dump(noodling_yaml, f)

        # Create recipe.yaml
        recipe_yaml = {
            "name": "Test Character",
            "description": "A friendly test character.",
            "affect_baseline": {
                "valence": 0.5,
                "arousal": 0.4,
                "dominance": 0.6
            },
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.5,
                "agreeableness": 0.8,
                "neuroticism": 0.2
            }
        }
        with open(test_noodling / "recipe.yaml", 'w') as f:
            yaml.dump(recipe_yaml, f)

        # Create assembly.yaml
        assembly_yaml = {
            "name": "Test Assembly",
            "version": "1.0.0",
            "facets": [
                {
                    "id": "incoming",
                    "type": "INCOMING",
                    "position": [100, 300]
                },
                {
                    "id": "response",
                    "type": "LLM",
                    "name": "Response Generator",
                    "config": {
                        "model_label": "SMALL",
                        "prompt": "You are a helpful assistant."
                    }
                },
                {
                    "id": "outgoing",
                    "type": "OUTGOING",
                    "position": [500, 300]
                }
            ],
            "connections": [
                {"from": "incoming", "from_pad": "output", "to": "response", "to_pad": "input"},
                {"from": "response", "from_pad": "output", "to": "outgoing", "to_pad": "input"}
            ]
        }
        with open(test_noodling / "assembly.yaml", 'w') as f:
            yaml.dump(assembly_yaml, f)

        # Create mock project manager
        project_manager = MagicMock()
        project_manager.get_noodling_path.return_value = str(test_noodling)
        project_manager.list_noodlings.return_value = ["test_character"]

        return {
            "project_path": tmp_path,
            "noodling_path": test_noodling,
            "project_manager": project_manager
        }

    def test_export_creates_package_folder(self, temp_project, tmp_path):
        """Export creates .noodling folder."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        package_path = exporter.export("test_character", output_dir)

        assert package_path.exists()
        assert package_path.name == "test_character.noodling"
        assert package_path.is_dir()

    def test_manifest_json_valid(self, temp_project, tmp_path):
        """manifest.json has required fields."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        package_path = exporter.export("test_character", output_dir)

        manifest_path = package_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        assert "name" in manifest
        assert "version" in manifest
        assert "noodlestudio_version" in manifest
        assert "exports" in manifest
        assert manifest["exports"]["character"] == "character.json"
        assert manifest["exports"]["assembly"] == "assembly.json"
        assert manifest["exports"]["expressions"] == "expressions.json"

    def test_character_json_has_pad(self, temp_project, tmp_path):
        """character.json includes initial PAD state."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        package_path = exporter.export("test_character", output_dir)

        character_path = package_path / "character.json"
        assert character_path.exists()

        with open(character_path, 'r') as f:
            character = json.load(f)

        assert "initial_pad" in character
        assert "pleasure" in character["initial_pad"]
        assert "arousal" in character["initial_pad"]
        assert "dominance" in character["initial_pad"]

        # Check valence was mapped to pleasure
        assert character["initial_pad"]["pleasure"] == 0.5
        assert character["initial_pad"]["arousal"] == 0.4
        assert character["initial_pad"]["dominance"] == 0.6

    def test_assembly_json_has_facets(self, temp_project, tmp_path):
        """assembly.json includes facet definitions."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        package_path = exporter.export("test_character", output_dir)

        assembly_path = package_path / "assembly.json"
        assert assembly_path.exists()

        with open(assembly_path, 'r') as f:
            assembly = json.load(f)

        assert "facets" in assembly
        assert "connections" in assembly
        assert len(assembly["facets"]) == 3
        assert len(assembly["connections"]) == 2

    def test_expressions_json_has_mappings(self, temp_project, tmp_path):
        """expressions.json includes PAD -> FACS -> VRM chain."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        package_path = exporter.export("test_character", output_dir)

        expressions_path = package_path / "expressions.json"
        assert expressions_path.exists()

        with open(expressions_path, 'r') as f:
            expressions = json.load(f)

        assert "pad_to_emotion_weights" in expressions
        assert "emotion_to_aus" in expressions
        assert "au_to_vrm_blendshapes" in expressions
        assert "transition_settings" in expressions

        # Check some specific mappings
        assert "joy" in expressions["pad_to_emotion_weights"]
        assert "AU12" in expressions["au_to_vrm_blendshapes"]

    def test_bake_prompts_inlines_content(self, temp_project, tmp_path):
        """bake_prompts option inlines prompt templates."""
        from noodlestudio.core.noodling_package_exporter import (
            NoodlingPackageExporter, ExportOptions
        )

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        options = ExportOptions(bake_prompts=True)
        package_path = exporter.export("test_character", output_dir, options)

        assembly_path = package_path / "assembly.json"
        with open(assembly_path, 'r') as f:
            assembly = json.load(f)

        assert "prompt_templates" in assembly
        assert len(assembly["prompt_templates"]) > 0

    def test_export_without_assembly(self, temp_project, tmp_path):
        """Export works for noodlings without assemblies."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        # Remove assembly reference
        noodling_yaml_path = temp_project["noodling_path"] / "noodling.yaml"
        with open(noodling_yaml_path, 'r') as f:
            noodling_data = yaml.safe_load(f)
        del noodling_data["assembly"]
        with open(noodling_yaml_path, 'w') as f:
            yaml.dump(noodling_data, f)

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        package_path = exporter.export("test_character", output_dir)

        # Should still create assembly.json with minimal structure
        assembly_path = package_path / "assembly.json"
        assert assembly_path.exists()

        with open(assembly_path, 'r') as f:
            assembly = json.load(f)

        assert assembly["facets"] == []
        assert assembly["connections"] == []

    def test_export_noodling_not_found(self, temp_project, tmp_path):
        """Export raises ValueError for missing noodling."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        temp_project["project_manager"].get_noodling_path.return_value = None

        exporter = NoodlingPackageExporter(temp_project["project_manager"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="Noodling not found"):
            exporter.export("nonexistent", output_dir)


# =============================================================================
# Expression Mapping Tests
# =============================================================================

class TestExportExpressions:
    """Tests for expression mapping export."""

    def test_default_emotion_weights(self, tmp_path):
        """Default PAD -> emotion weights are correct."""
        from noodlestudio.core.noodling_package_exporter import (
            NoodlingPackageExporter, ExportOptions
        )

        # Create minimal mock
        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)

        # Export expressions directly
        exporter._export_expressions(tmp_path, ExportOptions())

        expressions_path = tmp_path / "expressions.json"
        with open(expressions_path, 'r') as f:
            expressions = json.load(f)

        # Check Mehrabian/Russell model weights
        joy = expressions["pad_to_emotion_weights"]["joy"]
        assert joy["pleasure"] > 0.5  # Joy has high pleasure
        assert joy["arousal"] > 0  # Joy has positive arousal

        sadness = expressions["pad_to_emotion_weights"]["sadness"]
        assert sadness["pleasure"] < 0  # Sadness has negative pleasure

        anger = expressions["pad_to_emotion_weights"]["anger"]
        assert anger["arousal"] > 0.5  # Anger has high arousal
        assert anger["dominance"] > 0  # Anger has positive dominance

    def test_default_facs_mappings(self, tmp_path):
        """Default emotion -> AU mappings are correct."""
        from noodlestudio.core.noodling_package_exporter import (
            NoodlingPackageExporter, ExportOptions
        )

        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)
        exporter._export_expressions(tmp_path, ExportOptions())

        expressions_path = tmp_path / "expressions.json"
        with open(expressions_path, 'r') as f:
            expressions = json.load(f)

        # Joy should activate AU6 (cheek raise) and AU12 (lip corner puller)
        joy_aus = expressions["emotion_to_aus"]["joy"]
        assert "AU6" in joy_aus
        assert "AU12" in joy_aus

        # Surprise should activate AU1, AU2 (brow raise), AU5 (upper lid raise)
        surprise_aus = expressions["emotion_to_aus"]["surprise"]
        assert "AU1" in surprise_aus
        assert "AU2" in surprise_aus
        assert "AU5" in surprise_aus

    def test_vrm_blendshape_mappings(self, tmp_path):
        """AU -> VRM blendshape mappings are correct."""
        from noodlestudio.core.noodling_package_exporter import (
            NoodlingPackageExporter, ExportOptions
        )

        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)
        exporter._export_expressions(tmp_path, ExportOptions())

        expressions_path = tmp_path / "expressions.json"
        with open(expressions_path, 'r') as f:
            expressions = json.load(f)

        au_mappings = expressions["au_to_vrm_blendshapes"]

        # AU12 (smile) should map to Mouth_Smile
        assert any(
            bs["blendshape"] == "Mouth_Smile"
            for bs in au_mappings["AU12"]
        )

        # AU4 (brow lowerer) should map to Brow_Down
        assert any(
            bs["blendshape"] == "Brow_Down"
            for bs in au_mappings["AU4"]
        )


# =============================================================================
# Character Export Tests
# =============================================================================

class TestExportCharacter:
    """Tests for character.json export."""

    def test_valence_maps_to_pleasure(self, tmp_path):
        """Internal valence maps to Unity's pleasure."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)

        noodling_data = {"name": "Test"}
        recipe_data = {
            "name": "Test",
            "affect_baseline": {
                "valence": 0.7,
                "arousal": 0.3,
                "dominance": 0.5
            }
        }

        exporter._export_character(noodling_data, recipe_data, tmp_path)

        character_path = tmp_path / "character.json"
        with open(character_path, 'r') as f:
            character = json.load(f)

        # valence should be exported as pleasure
        assert character["initial_pad"]["pleasure"] == 0.7

    def test_personality_traits_from_ocean(self, tmp_path):
        """Personality traits are derived from OCEAN model."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)

        noodling_data = {"name": "Test", "tags": []}
        recipe_data = {
            "name": "Test",
            "personality": {
                "openness": 0.9,       # High -> curious
                "conscientiousness": 0.8,  # High -> disciplined
                "extraversion": 0.3,   # Low -> introspective
                "agreeableness": 0.8,  # High -> kind
                "neuroticism": 0.2     # Low -> calm
            }
        }

        exporter._export_character(noodling_data, recipe_data, tmp_path)

        character_path = tmp_path / "character.json"
        with open(character_path, 'r') as f:
            character = json.load(f)

        traits = character["personality_traits"]
        assert "curious" in traits
        assert "disciplined" in traits
        assert "introspective" in traits
        assert "kind" in traits
        assert "calm" in traits


# =============================================================================
# Play Export Tests
# =============================================================================

class TestExportPlays:
    """Tests for plays/ export."""

    @pytest.fixture
    def noodling_with_plays(self, tmp_path):
        """Create a noodling with plays subfolder."""
        noodling_path = tmp_path / "noodling"
        noodling_path.mkdir()

        plays_path = noodling_path / "plays"
        plays_path.mkdir()

        # Create a test play
        play_data = {
            "name": "Test Play",
            "version": "1.0",
            "characters": {
                "test_char": {
                    "voice": "calm",
                    "initial_pad": {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.5}
                }
            },
            "beats": [
                {
                    "id": "beat1",
                    "character": "test_char",
                    "speaks": "Hello there.",
                    "pad_drift": {"pleasure": 0.1}
                }
            ]
        }
        with open(plays_path / "intro.play.yaml", 'w') as f:
            yaml.dump(play_data, f)

        return noodling_path

    def test_plays_exported_to_subfolder(self, noodling_with_plays, tmp_path):
        """Plays are exported to plays/ subfolder."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)

        output_path = tmp_path / "output"
        output_path.mkdir()

        exporter._export_plays(str(noodling_with_plays), "test", output_path)

        plays_output = output_path / "plays"
        assert plays_output.exists()

        # Check play was converted to JSON
        play_files = list(plays_output.glob("*.play.json"))
        assert len(play_files) == 1

    def test_play_converted_to_json_format(self, noodling_with_plays, tmp_path):
        """Plays are converted to Unity JSON format."""
        from noodlestudio.core.noodling_package_exporter import NoodlingPackageExporter

        project_manager = MagicMock()
        exporter = NoodlingPackageExporter(project_manager)

        output_path = tmp_path / "output"
        output_path.mkdir()

        exporter._export_plays(str(noodling_with_plays), "test", output_path)

        play_file = output_path / "plays" / "intro.play.json"
        with open(play_file, 'r') as f:
            play = json.load(f)

        assert "name" in play
        assert "characters" in play
        assert "beats" in play
        assert play["name"] == "Test Play"
        assert len(play["beats"]) == 1


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
