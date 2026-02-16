# ------------------------------------------------------------------
#   B.10 Smoke Walk Fixes -- Steps 3-5
#
#   Tests for three bugs found during manual smoke walk:
#   B.10.1: Facet rename in inspector updates node header
#   B.10.2: Description edit persists on re-selection
#   B.10.3: Neural canvas bridge (dead nodes, method name, zombie)
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_b10_smoke_walk_fixes
# PURPOSE:  B.10 Bug Fix Verification
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# =====================================================================
# B.10.1: Facet rename updates in-memory model and node header
# =====================================================================

class TestFacetRenameUpdatesModel:
    """B.10.1a: create_bound_lineedit must setattr on model before pushing undo."""

    def test_lineedit_sets_model_value(self, qapp):
        """Editing a bound lineedit must update the object attribute."""
        from noodlestudio.core.facet_system import Facet
        from noodlestudio.core.property_binding import PropertyMeta
        from noodlestudio.panels.inspector_base import InspectorBaseMixin
        from PyQt6.QtWidgets import QFormLayout

        facet = Facet(
            id="test_facet_rename",
            name="Original Name",
            facet_type="IntuitionFacet",
            prompt="test"
        )

        # Minimal inspector-like object using the real mixin
        class TestInspector(InspectorBaseMixin):
            def __init__(self):
                self._bound_widgets = {}
                self.is_loading = False
                self._commands = []

            def _push_generic_property_command(self, obj, prop_name, old_val, new_val, **kw):
                self._commands.append((prop_name, old_val, new_val))

        inspector = TestInspector()
        meta = PropertyMeta(name='name', prop_type=str, display_name='Name')
        layout = QFormLayout()
        line_edit = inspector.create_bound_lineedit(facet, meta, layout)

        # Simulate user editing the name
        line_edit.setText("New Name")
        line_edit.editingFinished.emit()

        # Model must be updated BEFORE command is pushed
        assert facet.name == "New Name", "setattr must update model object"
        assert len(inspector._commands) == 1
        assert inspector._commands[0] == ('name', 'Original Name', 'New Name')

    def test_spinbox_sets_model_value(self, qapp):
        """Editing a bound spinbox must update the object attribute."""
        from noodlestudio.core.facet_system import Facet
        from noodlestudio.core.property_binding import PropertyMeta
        from noodlestudio.panels.inspector_base import InspectorBaseMixin
        from PyQt6.QtWidgets import QFormLayout

        facet = Facet(
            id="test_facet_spin",
            name="SpinTest",
            facet_type="IntuitionFacet",
            prompt="test",
            max_tokens=150
        )

        class TestInspector(InspectorBaseMixin):
            def __init__(self):
                self._bound_widgets = {}
                self.is_loading = False
                self._commands = []

            def _push_generic_property_command(self, obj, prop_name, old_val, new_val, **kw):
                self._commands.append((prop_name, old_val, new_val))

        inspector = TestInspector()
        meta = PropertyMeta(name='max_tokens', prop_type=int, display_name='Max Tokens')
        layout = QFormLayout()
        spin = inspector.create_bound_spinbox(facet, meta, layout)

        spin.setValue(200)

        assert facet.max_tokens == 200, "spinbox must setattr on model"

    def test_combobox_sets_model_value(self, qapp):
        """Editing a bound combobox must update the object attribute."""
        from noodlestudio.core.facet_system import Facet
        from noodlestudio.core.property_binding import PropertyMeta
        from noodlestudio.panels.inspector_base import InspectorBaseMixin
        from PyQt6.QtWidgets import QFormLayout

        facet = Facet(
            id="test_facet_combo",
            name="ComboTest",
            facet_type="IntuitionFacet",
            prompt="test",
            model="SMALL"
        )

        class TestInspector(InspectorBaseMixin):
            def __init__(self):
                self._bound_widgets = {}
                self.is_loading = False
                self._commands = []

            def _push_generic_property_command(self, obj, prop_name, old_val, new_val, **kw):
                self._commands.append((prop_name, old_val, new_val))

        inspector = TestInspector()
        meta = PropertyMeta(
            name='model', prop_type=str, display_name='Model',
            choices=['SMALL', 'MEDIUM', 'LARGE']
        )
        layout = QFormLayout()
        combo = inspector.create_bound_combobox(facet, meta, layout)

        combo.setCurrentText("LARGE")

        assert facet.model == "LARGE", "combobox must setattr on model"


class TestFacetsEditorRefreshNode:
    """B.10.1b: FacetsEditorPanel must have refresh_node_for_facet method."""

    def test_refresh_method_exists(self, qapp):
        """FacetsEditorPanel must expose refresh_node_for_facet."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        assert hasattr(editor, 'refresh_node_for_facet')

    def test_refresh_unknown_id_no_crash(self, qapp):
        """Calling refresh with unknown facet ID must not raise."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.refresh_node_for_facet("nonexistent_id")


# =====================================================================
# B.10.2: Description edit persists on re-selection
# =====================================================================

class TestLoadEntityGuardOrdering:
    """B.10.2: is_saving guard must not corrupt current_entity."""

    def test_is_saving_blocks_current_entity_assignment(self, qapp):
        """If is_saving is True, load_entity must not set current_entity."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        panel = InspectorPanel(None)
        panel.is_saving = True
        panel.current_entity = None

        entity_data = {'id': 'test_001', 'name': 'TestNoodling'}
        panel.load_entity('noodling', entity_data)

        assert panel.current_entity is None, (
            "is_saving guard must prevent current_entity assignment"
        )

    def test_reselection_after_save_window_expires(self, qapp):
        """After is_saving expires, re-selection must work normally."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        panel = InspectorPanel(None)
        entity_data = {'id': 'test_002', 'name': 'TestNoodling2'}

        # Simulate save in progress
        panel.is_saving = True
        panel.current_entity = None
        panel.load_entity('noodling', entity_data)
        assert panel.current_entity is None

        # Simulate save timer expiry
        panel.is_saving = False
        panel.load_entity('noodling', entity_data)
        assert panel.current_entity is not None
        assert panel.current_entity[0] == 'noodling'


# =====================================================================
# B.10.2 (path fix): Recipe load uses same resolution as save
# =====================================================================

def _create_project_tree(tmp_path):
    """Helper: create a minimal project directory tree for path resolution tests.

    Returns (project_root, instance_path, noodling_ref, noodling_dir).
    """
    project_root = tmp_path / "MyProject"
    noodling_dir = project_root / "Noodlings" / "ajo_majo"
    noodling_dir.mkdir(parents=True)
    stage_dir = project_root / "Stages" / "the_nexus" / "Instances" / "ajo"
    stage_dir.mkdir(parents=True)

    # recipe.yaml -- this is what the inspector reads/writes
    recipe = {
        'name': 'guide',
        'description': 'A cute axolotl',
        'personality': {'openness': 0.85}
    }
    with open(noodling_dir / "recipe.yaml", 'w') as f:
        yaml.dump(recipe, f)

    # assembly.yaml -- this is what facets editor / inspector loads
    assembly_yaml = (
        "name: Ajo Assembly\n"
        "facets:\n"
        "  - id: f1\n"
        "    name: Intuition\n"
        "    type: IntuitionFacet\n"
        "    prompt: Be wise\n"
        "connections: []\n"
    )
    with open(noodling_dir / "assembly.yaml", 'w') as f:
        f.write(assembly_yaml)

    # instance.yaml
    noodling_ref = os.path.relpath(str(noodling_dir), str(stage_dir))
    inst = {'noodling': noodling_ref, 'overrides': {'name': 'Ajo Majo'}}
    with open(stage_dir / "instance.yaml", 'w') as f:
        yaml.dump(inst, f)

    return str(project_root), str(stage_dir), noodling_ref, str(noodling_dir)


class TestRecipeLoadMatchesSave:
    """B.10.2: _load_noodling_recipe must use _resolve_recipe_path (same as save)."""

    def test_load_finds_same_file_as_save(self, qapp, tmp_path):
        """Load and save must resolve to the same recipe.yaml file."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo Majo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': {},
        }

        panel = InspectorPanel(None)

        # Load must find recipe
        recipe_data = panel._load_noodling_recipe(entity_data, {})
        assert recipe_data.get('description') == 'A cute axolotl', (
            f"Load must read recipe.yaml from noodling dir; got: {recipe_data}"
        )

        # Save path must resolve to the same file
        save_path = panel._resolve_recipe_path(entity_data)
        expected = os.path.join(noodling_dir, 'recipe.yaml')
        assert os.path.normpath(save_path) == os.path.normpath(expected)

    def test_load_returns_empty_when_no_recipe(self, qapp, tmp_path):
        """If no recipe.yaml exists, _load_noodling_recipe returns empty dict."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        entity_data = {
            'id': 'agent_missing',
            'name': 'Ghost',
            'path': str(tmp_path / "nonexistent" / "Instances" / "ghost"),
            'noodling_ref': '../../../../Noodlings/ghost',
            'data': {},
        }

        panel = InspectorPanel(None)
        recipe_data = panel._load_noodling_recipe(entity_data, {})
        assert recipe_data == {}

    def test_overrides_applied_on_top_of_recipe(self, qapp, tmp_path):
        """Instance overrides merge on top of recipe data."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        agent = {'overrides': {'description': 'Override description'}}
        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': agent,
        }

        panel = InspectorPanel(None)
        recipe_data = panel._load_noodling_recipe(entity_data, agent)
        assert recipe_data['description'] == 'Override description'


# =====================================================================
# B.10.3a: NeuralCanvasFacet creation
# =====================================================================

class TestNeuralCanvasFacetCreation:
    """B.10.3a: Adding NeuralCanvasFacet must create .nncanvas file."""

    def test_create_blank_nncanvas_creates_file(self, qapp, tmp_path):
        """_create_blank_nncanvas produces a valid .nncanvas JSON file."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        noodling_dir = tmp_path / "Noodlings" / "ajo"
        noodling_dir.mkdir(parents=True)
        assembly_path = str(noodling_dir / "assembly.yaml")

        with open(assembly_path, 'w') as f:
            f.write("name: test\nfacets: []\n")

        editor = FacetsEditorPanel()
        editor.current_assembly_path = assembly_path

        result = editor._create_blank_nncanvas("abcdef1234567890", "My Charm Network")
        assert result is not None
        assert result.endswith('.nncanvas')

        # The file must exist at the absolute path (assembly_dir / filename)
        abs_path = os.path.join(str(noodling_dir), os.path.basename(result))
        assert os.path.exists(abs_path), f".nncanvas file must exist: {abs_path}"

        with open(abs_path) as f:
            data = json.load(f)

        assert data['version'] == '1.0'
        assert data['name'] == 'My Charm Network'
        assert data['nodes'] == []
        assert data['connections'] == []

    def test_create_blank_nncanvas_returns_none_without_path(self, qapp):
        """Without assembly path, _create_blank_nncanvas returns None."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.current_assembly_path = None

        result = editor._create_blank_nncanvas("abc123", "Test")
        assert result is None

    def test_create_blank_nncanvas_clean_filename(self, qapp, tmp_path):
        """Filenames must not contain parentheses or special characters."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        assembly_path = str(tmp_path / "assembly.yaml")
        with open(assembly_path, 'w') as f:
            f.write("name: test\nfacets: []\n")

        editor = FacetsEditorPanel()
        editor.current_assembly_path = assembly_path

        result = editor._create_blank_nncanvas("abcdef12", "Neural Canvas (NNCanvas)")
        assert result is not None
        basename = os.path.basename(result)
        assert '(' not in basename, f"Filename must not contain parens: {basename}"
        assert ')' not in basename, f"Filename must not contain parens: {basename}"

    def test_facet_with_nncanvas_path_is_not_dead(self):
        """Facet with nncanvas_path set allows double-click bridge."""
        from noodlestudio.core.facet_system import Facet

        facet = Facet(
            id="nc_test",
            name="Neural Net",
            facet_type="NeuralCanvasFacet",
            prompt="",
            nncanvas_path="networks/test.nncanvas"
        )
        assert facet.nncanvas_path is not None
        assert facet.nncanvas_path == "networks/test.nncanvas"


# =====================================================================
# B.10.3b: Inspector asset method name
# =====================================================================

class TestInspectorAssetNeuralCanvasMethod:
    """B.10.3b: NeuralCanvasPanel must have _load_from_file, not load_canvas."""

    def test_neural_canvas_has_load_from_file(self, qapp):
        """NeuralCanvasPanel._load_from_file must exist."""
        from noodlestudio.panels.neural_canvas.neural_canvas_panel import NeuralCanvasPanel

        panel = NeuralCanvasPanel()
        assert hasattr(panel, '_load_from_file')

    def test_neural_canvas_no_load_canvas_method(self, qapp):
        """NeuralCanvasPanel must NOT have load_canvas (it never existed)."""
        from noodlestudio.panels.neural_canvas.neural_canvas_panel import NeuralCanvasPanel

        panel = NeuralCanvasPanel()
        assert not hasattr(panel, 'load_canvas'), (
            "load_canvas does not exist -- inspector_asset must use _load_from_file"
        )


# =====================================================================
# B.10.3c: Neural canvas starts empty
# =====================================================================

class TestNeuralCanvasStartsEmpty:
    """B.10.3c: Neural canvas panel must not load zombie data on startup."""

    def test_panel_starts_with_empty_graph(self, qapp):
        """On construction, neural canvas panel has zero nodes."""
        from noodlestudio.panels.neural_canvas.neural_canvas_panel import NeuralCanvasPanel

        panel = NeuralCanvasPanel()
        assert len(panel.graph.nodes) == 0, (
            "Panel must start empty, not load default.nncanvas"
        )

    def test_panel_starts_with_empty_name(self, qapp):
        """On construction, graph name indicates no file loaded."""
        from noodlestudio.panels.neural_canvas.neural_canvas_panel import NeuralCanvasPanel

        panel = NeuralCanvasPanel()
        assert "no neural canvas" in panel.graph.name.lower()


# =====================================================================
# B.10.4: Inspector facet dropdown loads assembly from entity_data path
# =====================================================================

class TestInspectorAssemblyPathResolution:
    """B.10.4: _get_agent_assembly must resolve from entity_data path, not __file__."""

    def test_assembly_loads_from_entity_data_path(self, qapp, tmp_path):
        """_get_agent_assembly finds assembly.yaml via instance path + noodling_ref."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo Majo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': {},
        }

        panel = InspectorPanel(None)
        assembly = panel._get_agent_assembly('agent_ajo', entity_data)

        assert assembly is not None, "Assembly must be found via entity_data path"
        assert assembly.name == 'Ajo Assembly'
        assert len(assembly.facets) == 1

    def test_assembly_returns_none_for_missing_path(self, qapp, tmp_path):
        """_get_agent_assembly returns None when noodling directory doesn't exist."""
        from noodlestudio.panels.inspector_panel import InspectorPanel

        entity_data = {
            'id': 'agent_ghost',
            'name': 'Ghost',
            'path': str(tmp_path / "nonexistent" / "Instances" / "ghost"),
            'noodling_ref': '../../../../Noodlings/ghost',
            'data': {},
        }

        panel = InspectorPanel(None)
        assembly = panel._get_agent_assembly('agent_ghost', entity_data)
        assert assembly is None

    def test_assembly_matches_facets_editor_resolution(self, qapp, tmp_path):
        """Inspector and facets editor must resolve to the same assembly file."""
        import os

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo Majo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': {},
        }

        # Inspector path resolution
        noodling_dir_resolved = os.path.normpath(os.path.join(instance_path, noodling_ref))
        expected_assembly = os.path.join(noodling_dir_resolved, 'assembly.yaml')

        # This is the same pattern _load_facet_assembly_for_noodling uses
        assert os.path.exists(expected_assembly), "Assembly must exist at resolved path"

        # Inspector must load from the same path
        from noodlestudio.panels.inspector_panel import InspectorPanel
        panel = InspectorPanel(None)
        assembly = panel._get_agent_assembly('agent_ajo', entity_data)
        assert assembly is not None


# =====================================================================
# B.10.5: Dedicated facet inspector view
# =====================================================================

class TestDedicatedFacetInspector:
    """B.10.5: Clicking facet shows dedicated view, not dropdown append."""

    def test_load_facet_sets_state(self, qapp):
        """load_facet must set current_facet and clear current_entity."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_llm",
            name="Ajo's Mind",
            facet_type="LLMFacet",
            prompt="You are a wise axolotl.",
            model="MEDIUM",
            temperature=0.7,
            max_tokens=150
        )

        panel.load_facet(facet)

        assert panel.current_facet == facet
        assert panel.current_entity is None, (
            "current_entity must be None so noodling reload works"
        )

    def test_load_facet_builds_properties(self, qapp):
        """load_facet must populate the properties layout."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_llm_build",
            name="Ajo's Mind",
            facet_type="LLMFacet",
            prompt="You are a wise axolotl.",
            model="MEDIUM",
            temperature=0.7,
            max_tokens=150
        )

        panel.load_facet(facet)

        # Properties layout must have content (Basic, LLM Config, I/O, stretch)
        assert panel.properties_layout.count() >= 3, (
            f"Expected at least 3 sections, got {panel.properties_layout.count()}"
        )

    def test_load_facet_shows_llm_widgets(self, qapp):
        """LLMFacet must show Model, Temperature, Max Tokens, Prompt fields."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QLabel, QDoubleSpinBox, QSpinBox, QComboBox

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_llm_widgets",
            name="Ajo's Mind",
            facet_type="LLMFacet",
            prompt="You are a wise axolotl.",
            model="MEDIUM",
            temperature=0.7,
            max_tokens=150
        )

        panel.load_facet(facet)

        # Check for type-specific widgets
        combos = panel.properties_widget.findChildren(QComboBox)
        spinboxes = panel.properties_widget.findChildren(QSpinBox)
        double_spinboxes = panel.properties_widget.findChildren(QDoubleSpinBox)

        # Must have at least one QComboBox (model)
        assert len(combos) >= 1, "Must have Model dropdown"
        # Must have at least one QDoubleSpinBox (temperature)
        assert len(double_spinboxes) >= 1, "Must have Temperature spinner"
        # Must have at least one QSpinBox (max tokens)
        assert len(spinboxes) >= 1, "Must have Max Tokens spinner"

    def test_load_facet_shows_name_field(self, qapp):
        """Facet inspector must show editable Name field."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QLineEdit

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_name_field",
            name="Ham Sandwich",
            facet_type="LLMFacet",
            prompt="test"
        )

        panel.load_facet(facet)

        # Find a QLineEdit containing the facet name
        line_edits = panel.properties_widget.findChildren(QLineEdit)
        name_found = any(le.text() == "Ham Sandwich" for le in line_edits)
        assert name_found, "Must show editable Name field with facet name"

    def test_load_facet_none_clears_state(self, qapp):
        """load_facet(None) clears current_facet."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_deselect",
            name="Test Facet",
            facet_type="LLMFacet",
            prompt="test"
        )

        panel.load_facet(facet)
        assert panel.current_facet == facet

        panel.load_facet(None)
        assert panel.current_facet is None

    def test_entity_load_clears_facet_mode(self, qapp):
        """load_entity must clear current_facet."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_clear",
            name="Test Facet",
            facet_type="LLMFacet",
            prompt="test"
        )

        panel.load_facet(facet)
        assert panel.current_facet == facet

        # Load an entity -- must clear facet mode
        entity_data = {'id': 'agent_test', 'name': 'TestNoodling', 'data': {}}
        panel.load_entity('noodling', entity_data)
        assert panel.current_facet is None

    def test_neural_canvas_facet_section(self, qapp):
        """NeuralCanvasFacet must show NNCanvas Path field."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QLineEdit

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_nc",
            name="Neural Net",
            facet_type="NeuralCanvasFacet",
            prompt="",
            nncanvas_path="networks/test.nncanvas"
        )

        panel.load_facet(facet)

        line_edits = panel.properties_widget.findChildren(QLineEdit)
        path_found = any(le.text() == "networks/test.nncanvas" for le in line_edits)
        assert path_found, "Must show NNCanvas path field"

    def test_charm_network_facet_section(self, qapp):
        """CharmNetworkFacet must show info label."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QLabel

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_charm",
            name="Charm Network",
            facet_type="CharmNetworkFacet",
            prompt=""
        )

        panel.load_facet(facet)

        labels = panel.properties_widget.findChildren(QLabel)
        info_found = any("affect model" in lbl.text().lower() for lbl in labels)
        assert info_found, "Must show charm network info label"


# =====================================================================
# B.10.6: No Affect Baseline section, no facet dropdown in noodling inspector
# =====================================================================

class TestNoAffectBaselineInNoodlingInspector:
    """B.10.6: Noodling inspector must not show Affect Baseline or facet dropdown."""

    def test_no_affect_baseline_section(self, qapp, tmp_path):
        """Noodling inspector must not display Affect Baseline section."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.widgets.collapsible_section import CollapsibleSection
        from PyQt6.QtWidgets import QLabel

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        # Add affect_baseline to recipe so we can verify it's NOT displayed
        recipe_path = os.path.join(noodling_dir, 'recipe.yaml')
        with open(recipe_path, 'r') as f:
            recipe_data = yaml.safe_load(f) or {}
        recipe_data['affect_baseline'] = {
            'valence': 0.4, 'arousal': 0.35, 'dominance': 0.7,
            'boredom': 0.1, 'sorrow': 0.2
        }
        with open(recipe_path, 'w') as f:
            yaml.dump(recipe_data, f)

        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo Majo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': {},
        }

        panel = InspectorPanel(None)
        panel.load_entity('noodling', entity_data)

        # Find all CollapsibleSection titles
        sections = panel.properties_widget.findChildren(CollapsibleSection)
        section_titles = [s.title_text for s in sections]

        assert "Affect Baseline" not in section_titles, (
            f"Affect Baseline must not appear in noodling inspector. "
            f"Found sections: {section_titles}"
        )

        # Also verify no "Valence" / "Arousal" labels in the noodling view
        labels = panel.properties_widget.findChildren(QLabel)
        label_texts = [lbl.text() for lbl in labels]
        assert not any("Valence" in t for t in label_texts), (
            "Valence label must not appear in noodling inspector"
        )

    def test_no_facet_dropdown_in_noodling_inspector(self, qapp, tmp_path):
        """Noodling inspector must not contain a facet dropdown."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from PyQt6.QtWidgets import QComboBox

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo Majo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': {},
        }

        panel = InspectorPanel(None)
        panel.load_entity('noodling', entity_data)

        # No QComboBox should exist in the noodling inspector
        combos = panel.properties_widget.findChildren(QComboBox)
        assert len(combos) == 0, (
            f"Noodling inspector must not contain a facet dropdown. "
            f"Found {len(combos)} QComboBox(es)"
        )

    def test_noodling_inspector_still_shows_basics(self, qapp, tmp_path):
        """Noodling inspector must still show Name, UUID, Description."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.widgets.collapsible_section import CollapsibleSection
        from PyQt6.QtWidgets import QLineEdit, QTextEdit

        project_root, instance_path, noodling_ref, noodling_dir = _create_project_tree(tmp_path)

        entity_data = {
            'id': 'agent_ajo',
            'name': 'Ajo Majo',
            'path': instance_path,
            'noodling_ref': noodling_ref,
            'data': {},
        }

        panel = InspectorPanel(None)
        panel.load_entity('noodling', entity_data)

        # Must still have the "Noodling" basics section
        sections = panel.properties_widget.findChildren(CollapsibleSection)
        section_titles = [s.title_text for s in sections]
        assert "Noodling" in section_titles, (
            f"Noodling basics section must still exist. Found: {section_titles}"
        )

        # Must have Name field (QLineEdit)
        line_edits = panel.properties_widget.findChildren(QLineEdit)
        assert len(line_edits) >= 1, "Must have at least the Name field"

        # Must have Description field (QTextEdit)
        text_edits = panel.properties_widget.findChildren(QTextEdit)
        assert len(text_edits) >= 1, "Must have Description text area"


# =====================================================================
# B.10.5 addendum: LLM type matching for real assemblies
# =====================================================================

class TestLLMTypeMatchingForRealAssemblies:
    """B.10.5: Inspector must show LLM config for type='LLM' (not just 'LLMFacet')."""

    def test_type_llm_shows_llm_widgets(self, qapp):
        """Facet with facet_type='LLM' (from YAML) must show LLM config."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QComboBox, QSpinBox, QDoubleSpinBox

        panel = InspectorPanel(None)
        facet = Facet(
            id="test_llm_yaml",
            name="Ajo's Mind",
            facet_type="LLM",
            prompt="You are a wise axolotl.",
            model="MEDIUM",
            temperature=0.7,
            max_tokens=150
        )

        panel.load_facet(facet)

        combos = panel.properties_widget.findChildren(QComboBox)
        spinboxes = panel.properties_widget.findChildren(QSpinBox)
        double_spinboxes = panel.properties_widget.findChildren(QDoubleSpinBox)

        assert len(combos) >= 1, "type='LLM' must show Model dropdown"
        assert len(double_spinboxes) >= 1, "type='LLM' must show Temperature spinner"
        assert len(spinboxes) >= 1, "type='LLM' must show Max Tokens spinner"

    def test_cognitive_facet_types_show_llm_widgets(self, qapp):
        """Cognitive facets (IntuitionFacet, EmotionFacet) must show LLM config."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QComboBox

        for facet_type in ("IntuitionFacet", "EmotionFacet", "SocialFacet",
                           "MemoryFacet", "PlanningFacet"):
            panel = InspectorPanel(None)
            facet = Facet(
                id=f"test_{facet_type.lower()}",
                name=f"Test {facet_type}",
                facet_type=facet_type,
                prompt="Test prompt",
                model="SMALL",
                temperature=0.7,
                max_tokens=150
            )

            panel.load_facet(facet)

            combos = panel.properties_widget.findChildren(QComboBox)
            assert len(combos) >= 1, (
                f"facet_type='{facet_type}' must show Model dropdown"
            )

    def test_incoming_outgoing_no_llm_config(self, qapp):
        """INCOMING/OUTGOING terminal nodes must NOT show LLM config."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        from noodlestudio.core.facet_system import Facet
        from PyQt6.QtWidgets import QComboBox

        for facet_type in ("INCOMING", "OUTGOING"):
            panel = InspectorPanel(None)
            facet = Facet(
                id=f"test_{facet_type.lower()}",
                name=facet_type,
                facet_type=facet_type,
                prompt=""
            )

            panel.load_facet(facet)

            combos = panel.properties_widget.findChildren(QComboBox)
            assert len(combos) == 0, (
                f"facet_type='{facet_type}' must NOT show Model dropdown"
            )


# =====================================================================
# B.10.1 addendum: Undo refreshes inspector widget via _bound_widgets
# =====================================================================

class TestUndoRefreshesBoundWidgets:
    """B.10.1 addendum: GenericPropertyCommand._refresh_widget() must update
    widgets registered in _bound_widgets, not just _binding_manager."""

    def test_undo_updates_lineedit_via_bound_widgets(self, qapp):
        """After undo, the inspector QLineEdit must show the reverted value."""
        from noodlestudio.core.facet_system import Facet
        from noodlestudio.core.commands.facet_commands import GenericPropertyCommand
        from PyQt6.QtWidgets import QLineEdit, QWidget

        facet = Facet(
            id="test_undo_refresh",
            name="Original Name",
            facet_type="LLM",
            prompt="test"
        )

        # Minimal inspector mock with _bound_widgets
        class FakeInspector(QWidget):
            def __init__(self):
                super().__init__()
                self._bound_widgets = {}
                self._binding_manager = type('FakeBM', (), {
                    'get_bindings_for_object': lambda self, obj: []
                })()

            def _auto_save_facet_assembly(self):
                pass

        inspector = FakeInspector()
        name_widget = QLineEdit("Original Name")
        inspector._bound_widgets['name'] = name_widget

        # Simulate: user renamed to "New Name", then undo
        facet.name = "New Name"
        name_widget.setText("New Name")

        cmd = GenericPropertyCommand(
            inspector=inspector,
            obj=facet,
            property_name='name',
            old_value='Original Name',
            new_value='New Name',
            display_name='Name'
        )

        # Simulate undo (bypassing first_redo since we set it manually)
        cmd._first_redo = False
        cmd._undo()

        assert facet.name == "Original Name", "Undo must revert facet.name"
        assert name_widget.text() == "Original Name", (
            "Undo must update the QLineEdit via _bound_widgets"
        )
