# ------------------------------------------------------------------
#   Assembly Persistence Tests (B.2)
#
#   Verifies: facet prompt edits trigger disk writes,
#   auto-save uses correct path, inspector auto-save works.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_assembly_persistence
# PURPOSE:  Assembly Persistence Tests
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


def _create_assembly_on_disk(tmp_path):
    """Create a minimal assembly YAML file and return (assembly, path)."""
    from noodlestudio.core.facet_system import FacetAssembly, Facet, FacetConnection

    assembly = FacetAssembly(name="test_assembly")
    assembly.facets = [
        Facet(
            id="incoming_1",
            facet_type="INCOMING",
            name="Input",
            prompt="",
            position={'x': 100, 'y': 200},
        ),
        Facet(
            id="llm_1",
            facet_type="LLMFacet",
            name="Process",
            prompt="Original prompt text",
            position={'x': 300, 'y': 200},
        ),
        Facet(
            id="outgoing_1",
            facet_type="OUTGOING",
            name="Output",
            prompt="",
            position={'x': 500, 'y': 200},
        ),
    ]
    assembly.connections = [
        FacetConnection(from_facet="incoming_1", from_pad="output",
                        to_facet="llm_1", to_pad="input"),
        FacetConnection(from_facet="llm_1", from_pad="output",
                        to_facet="outgoing_1", to_pad="input"),
    ]

    assembly_path = str(tmp_path / "assembly.yaml")
    assembly.save_yaml(assembly_path)
    return assembly, assembly_path


class TestPromptEditPersistence:
    """Prompt edits in FloatingTextEditor must trigger disk writes."""

    def test_prompt_edit_triggers_save(self, qapp, tmp_path):
        """After on_applied updates facet.prompt, assembly is saved to disk."""
        assembly, assembly_path = _create_assembly_on_disk(tmp_path)

        # Simulate what the facets editor does: set up current_assembly and path
        from noodlestudio.panels.facets_editor_assembly_mixin import FacetsEditorAssemblyMixin

        class FakeEditor(FacetsEditorAssemblyMixin):
            def __init__(self):
                self.current_assembly = assembly
                self.current_assembly_name = assembly.name
                self.current_assembly_path = assembly_path

        editor = FakeEditor()

        # Simulate prompt edit (what on_applied does)
        llm_facet = assembly.get_facet("llm_1")
        llm_facet.prompt = "Updated prompt from floating editor"

        # After edit, save should persist to disk
        editor._save_assembly_to_disk()

        # Verify disk has the new prompt
        from noodlestudio.core.facet_system import FacetAssembly
        reloaded = FacetAssembly.load_yaml(assembly_path)
        saved_facet = reloaded.get_facet("llm_1")
        assert saved_facet.prompt == "Updated prompt from floating editor", \
            "Prompt edit must persist to disk"


class TestAutoSaveOnSwitch:
    """Auto-save before switching assemblies must use correct path."""

    def test_auto_save_uses_current_path_not_hardcoded(self, qapp, tmp_path):
        """Auto-save on assembly switch must use current_assembly_path,
        not scan a hardcoded facet_assemblies/ directory."""
        assembly, assembly_path = _create_assembly_on_disk(tmp_path)

        from noodlestudio.core.facet_system import FacetAssembly, Facet

        # Set up editor with an assembly loaded from a project path
        from noodlestudio.panels.facets_editor_assembly_mixin import FacetsEditorAssemblyMixin

        class FakeEditor(FacetsEditorAssemblyMixin):
            def __init__(self):
                self.current_assembly = assembly
                self.current_assembly_name = assembly.name
                self.current_assembly_path = assembly_path
                self.scene_transition_lock = False
                self.node_graphics = {}
                self.wire_graphics = []
                self.grid_lines = []
                self.grid_visible = False

            def hide_empty_state(self):
                pass

        editor = FakeEditor()

        # Modify the in-memory assembly
        llm_facet = assembly.get_facet("llm_1")
        llm_facet.prompt = "Modified before switch"

        # Create a new assembly to switch to
        new_assembly = FacetAssembly(name="new_assembly")
        new_assembly.facets = [
            Facet(id="in_1", facet_type="INCOMING", name="In", prompt="",
                  position={'x': 0, 'y': 0}),
        ]

        # Fake out the scene/view
        class FakeScene:
            def clear(self): pass
            def update(self): pass
            def addItem(self, item): pass
        class FakeView:
            def centerOn(self, x, y): pass
        class FakeLabel:
            def setText(self, t): pass

        editor.scene = FakeScene()
        editor.view = FakeView()
        editor.assembly_label = FakeLabel()

        # Switch assembly -- this should auto-save the old one
        editor.load_assembly_from_data(new_assembly, source_path=str(tmp_path / "new.yaml"))

        # Verify the old assembly was saved with the modified prompt
        reloaded = FacetAssembly.load_yaml(assembly_path)
        saved_facet = reloaded.get_facet("llm_1")
        assert saved_facet.prompt == "Modified before switch", \
            "Auto-save on switch must persist to the correct project path"


class TestInspectorAutoSave:
    """Inspector auto-save must delegate to the facets editor's save method."""

    def test_inspector_auto_save_uses_editor_save(self, qapp, tmp_path):
        """_auto_save_facet_assembly() must call _save_assembly_to_disk()
        on the facets editor, not check for a non-existent assembly.filepath."""
        assembly, assembly_path = _create_assembly_on_disk(tmp_path)

        # Modify assembly in memory
        llm_facet = assembly.get_facet("llm_1")
        llm_facet.prompt = "Inspector triggered save"

        # Build a stub facets editor
        class StubEditor:
            def __init__(self):
                self.current_assembly = assembly
                self.current_assembly_path = assembly_path
                self.save_called = False

            def _save_assembly_to_disk(self):
                from noodlestudio.panels.facets_editor_assembly_mixin import FacetsEditorAssemblyMixin
                FacetsEditorAssemblyMixin._save_assembly_to_disk(self)
                self.save_called = True

        stub_editor = StubEditor()

        # Build a stub main window
        class StubMainWindow:
            def __init__(self):
                self.facets_editor = stub_editor

        # Build the inspector and call auto-save
        from noodlestudio.panels.inspector_panel import InspectorPanel
        inspector = InspectorPanel()

        # Override window() to return our stub
        inspector.window = lambda: StubMainWindow()

        inspector._auto_save_facet_assembly()

        # Verify saved to disk
        from noodlestudio.core.facet_system import FacetAssembly
        reloaded = FacetAssembly.load_yaml(assembly_path)
        saved_facet = reloaded.get_facet("llm_1")
        assert saved_facet.prompt == "Inspector triggered save", \
            "Inspector auto-save must persist to disk via facets editor"
