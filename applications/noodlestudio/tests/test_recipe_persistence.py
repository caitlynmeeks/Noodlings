# ------------------------------------------------------------------
#   Recipe Persistence Tests (B.1)
#
#   Verifies: inspector noodling edits write to YAML on disk,
#   not to a dead server endpoint.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_recipe_persistence
# PURPOSE:  Recipe Persistence Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def _create_temp_project(tmp_path, noodling_ref='test_noodling'):
    """Create a minimal project on disk with one noodling template and instance.

    Returns (project_root, instance_path, recipe_path, noodling_yaml_path).
    """
    project_root = tmp_path / 'TestProject'
    project_root.mkdir()

    # Template: Library/Noodlings/{ref}/
    template_dir = project_root / 'Library' / 'Noodlings' / noodling_ref
    template_dir.mkdir(parents=True)

    recipe_path = template_dir / 'recipe.yaml'
    recipe_path.write_text(yaml.dump({
        'name': 'guide',
        'display_name': 'Original Name',
        'personality': {
            'openness': 0.85,
            'conscientiousness': 0.5,
        },
        'affect_baseline': {
            'valence': 0.4,
            'arousal': 0.5,
            'dominance': 0.5,
        },
        'assembly': 'assembly.yaml',
    }, default_flow_style=False))

    noodling_yaml_path = template_dir / 'noodling.yaml'
    noodling_yaml_path.write_text(yaml.dump({
        'name': 'Original Name',
        'description': 'Original description',
        'author': 'test',
        'version': '1.0.0',
    }, default_flow_style=False))

    # Stage with instance
    stage_dir = project_root / 'Stages' / 'test_stage'
    instances_dir = stage_dir / 'Instances' / 'inst_001'
    instances_dir.mkdir(parents=True)

    instance_path = instances_dir / 'instance.yaml'
    instance_path.write_text(yaml.dump({
        'noodling': '../../../../Library/Noodlings/' + noodling_ref,
        'overrides': {
            'name': 'Original Name',
            'position': [0, 0, 0],
            'zone': 'main',
        },
        'created': '2026-02-15T00:00:00Z',
    }, default_flow_style=False))

    return (
        str(project_root),
        str(instances_dir),
        str(recipe_path),
        str(noodling_yaml_path),
    )


def _make_entity_data(instance_path, noodling_ref=None):
    """Build entity_data dict matching what the hierarchy provides.

    noodling_ref is read from instance.yaml's 'noodling' field (the
    relative path), matching the real hierarchy code.
    """
    inst_yaml = os.path.join(instance_path, 'instance.yaml')
    with open(inst_yaml) as f:
        inst_data = yaml.safe_load(f)

    ref = noodling_ref or inst_data.get('noodling', '')

    return {
        'type': 'noodling',
        'id': f'agent_{os.path.basename(instance_path)}',
        'name': inst_data.get('overrides', {}).get('name', 'Unknown'),
        'path': instance_path,
        'noodling_ref': ref,
        'zone': inst_data.get('overrides', {}).get('zone', 'default'),
        'data': inst_data,
        'node_id': 'test_node_001',
    }


class TestRecipePersistence:
    """B.1: Inspector noodling edits must persist to YAML files on disk."""

    def test_name_change_persists_to_instance_yaml(self, qapp, tmp_path):
        """Changing name in inspector writes overrides.name to instance.yaml."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        from PyQt6.QtCore import pyqtSignal, QObject

        project_root, instance_path, recipe_path, noodling_yaml_path = \
            _create_temp_project(tmp_path)
        entity_data = _make_entity_data(instance_path)

        # Build minimal host with the mixin
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        class FakeSignalHost(QObject):
            nameChanged = pyqtSignal(str, str, str)

        host = FakeSignalHost()

        class InspectorHost(EntityInspectorMixin):
            pass

        inspector = InspectorHost()
        inspector.nameChanged = host.nameChanged
        inspector.current_entity = ('noodling', entity_data)
        inspector.api_base = "http://localhost:8081/api"
        inspector.is_saving = False

        # Simulate property fields as the real inspector would
        name_field = QLineEdit("New Noodling Name")
        desc_field = QTextEdit("Original description")
        inspector.property_fields = {
            'name': name_field,
            'description': desc_field,
        }

        # Call save
        inspector._save_noodling_changes(entity_data)

        # Verify instance.yaml has updated name
        with open(os.path.join(instance_path, 'instance.yaml')) as f:
            saved = yaml.safe_load(f)
        assert saved['overrides']['name'] == 'New Noodling Name', \
            "Name must persist to instance.yaml overrides"

    def test_description_change_persists_to_recipe_yaml(self, qapp, tmp_path):
        """Changing description in inspector writes to recipe.yaml."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        from PyQt6.QtCore import pyqtSignal, QObject

        project_root, instance_path, recipe_path, noodling_yaml_path = \
            _create_temp_project(tmp_path)
        entity_data = _make_entity_data(instance_path)

        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        class FakeSignalHost(QObject):
            nameChanged = pyqtSignal(str, str, str)

        host = FakeSignalHost()

        class InspectorHost(EntityInspectorMixin):
            pass

        inspector = InspectorHost()
        inspector.nameChanged = host.nameChanged
        inspector.current_entity = ('noodling', entity_data)
        inspector.api_base = "http://localhost:8081/api"
        inspector.is_saving = False

        name_field = QLineEdit("Original Name")
        desc_field = QTextEdit("A completely new description for this noodling")
        inspector.property_fields = {
            'name': name_field,
            'description': desc_field,
        }

        inspector._save_noodling_changes(entity_data)

        # Verify recipe.yaml has updated description
        with open(recipe_path) as f:
            saved = yaml.safe_load(f)
        assert saved.get('description') == 'A completely new description for this noodling', \
            "Description must persist to recipe.yaml"

    def test_save_does_not_call_server(self, qapp, tmp_path):
        """Save must not attempt any HTTP requests."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        from PyQt6.QtCore import pyqtSignal, QObject
        from unittest.mock import patch

        project_root, instance_path, recipe_path, noodling_yaml_path = \
            _create_temp_project(tmp_path)
        entity_data = _make_entity_data(instance_path)

        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        class FakeSignalHost(QObject):
            nameChanged = pyqtSignal(str, str, str)

        host = FakeSignalHost()

        class InspectorHost(EntityInspectorMixin):
            pass

        inspector = InspectorHost()
        inspector.nameChanged = host.nameChanged
        inspector.current_entity = ('noodling', entity_data)
        inspector.api_base = "http://localhost:8081/api"
        inspector.is_saving = False

        name_field = QLineEdit("Test Name")
        desc_field = QTextEdit("Test description")
        inspector.property_fields = {
            'name': name_field,
            'description': desc_field,
        }

        # Patch requests.post to detect any server calls
        with patch('noodlestudio.panels.inspector_entity.requests.post') as mock_post:
            inspector._save_noodling_changes(entity_data)
            mock_post.assert_not_called()

    def test_persistence_survives_reopen(self, qapp, tmp_path):
        """Edit, save, then reload entity_data from disk -- values match."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        from PyQt6.QtCore import pyqtSignal, QObject

        project_root, instance_path, recipe_path, noodling_yaml_path = \
            _create_temp_project(tmp_path)
        entity_data = _make_entity_data(instance_path)

        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        class FakeSignalHost(QObject):
            nameChanged = pyqtSignal(str, str, str)

        host = FakeSignalHost()

        class InspectorHost(EntityInspectorMixin):
            pass

        inspector = InspectorHost()
        inspector.nameChanged = host.nameChanged
        inspector.current_entity = ('noodling', entity_data)
        inspector.api_base = "http://localhost:8081/api"
        inspector.is_saving = False

        # Simulate edits
        name_field = QLineEdit("Renamed Noodling")
        desc_field = QTextEdit("Entirely new personality description")
        inspector.property_fields = {
            'name': name_field,
            'description': desc_field,
        }

        inspector._save_noodling_changes(entity_data)

        # Now "reopen": read from disk and build entity_data again
        reopened_entity = _make_entity_data(instance_path)
        assert reopened_entity['name'] == 'Renamed Noodling', \
            "Instance override name must survive reopen"

        # Recipe description must also survive
        with open(recipe_path) as f:
            recipe = yaml.safe_load(f)
        assert recipe['description'] == 'Entirely new personality description'

    def test_save_preserves_existing_recipe_fields(self, qapp, tmp_path):
        """Saving description must not clobber personality or affect fields."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        from PyQt6.QtCore import pyqtSignal, QObject

        project_root, instance_path, recipe_path, noodling_yaml_path = \
            _create_temp_project(tmp_path)
        entity_data = _make_entity_data(instance_path)

        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        class FakeSignalHost(QObject):
            nameChanged = pyqtSignal(str, str, str)

        host = FakeSignalHost()

        class InspectorHost(EntityInspectorMixin):
            pass

        inspector = InspectorHost()
        inspector.nameChanged = host.nameChanged
        inspector.current_entity = ('noodling', entity_data)
        inspector.api_base = "http://localhost:8081/api"
        inspector.is_saving = False

        name_field = QLineEdit("Original Name")
        desc_field = QTextEdit("Updated description")
        inspector.property_fields = {
            'name': name_field,
            'description': desc_field,
        }

        inspector._save_noodling_changes(entity_data)

        with open(recipe_path) as f:
            recipe = yaml.safe_load(f)

        # Personality and affect must still be present
        assert recipe['personality']['openness'] == 0.85, \
            "Saving description must not clobber personality"
        assert recipe['affect_baseline']['valence'] == 0.4, \
            "Saving description must not clobber affect_baseline"
        assert recipe['assembly'] == 'assembly.yaml', \
            "Saving description must not clobber assembly ref"

    def test_save_preserves_existing_instance_fields(self, qapp, tmp_path):
        """Saving name must not clobber position or zone in instance.yaml."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        from PyQt6.QtCore import pyqtSignal, QObject

        project_root, instance_path, recipe_path, noodling_yaml_path = \
            _create_temp_project(tmp_path)
        entity_data = _make_entity_data(instance_path)

        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        class FakeSignalHost(QObject):
            nameChanged = pyqtSignal(str, str, str)

        host = FakeSignalHost()

        class InspectorHost(EntityInspectorMixin):
            pass

        inspector = InspectorHost()
        inspector.nameChanged = host.nameChanged
        inspector.current_entity = ('noodling', entity_data)
        inspector.api_base = "http://localhost:8081/api"
        inspector.is_saving = False

        name_field = QLineEdit("New Name")
        desc_field = QTextEdit("desc")
        inspector.property_fields = {
            'name': name_field,
            'description': desc_field,
        }

        inspector._save_noodling_changes(entity_data)

        with open(os.path.join(instance_path, 'instance.yaml')) as f:
            inst = yaml.safe_load(f)

        assert inst['overrides']['position'] == [0, 0, 0], \
            "Saving name must not clobber position"
        assert inst['overrides']['zone'] == 'main', \
            "Saving name must not clobber zone"
        assert 'noodling' in inst, \
            "Saving name must not clobber noodling ref"
