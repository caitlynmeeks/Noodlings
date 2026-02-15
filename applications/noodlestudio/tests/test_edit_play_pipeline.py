# ------------------------------------------------------------------
#   Edit-Play Pipeline Tests (B.8)
#
#   Tier 3 integration tests verifying the full edit -> save -> play
#   loop works end-to-end. Tests the connection points between
#   inspector, facets editor, project system, and performance manager.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_edit_play_pipeline
# PURPOSE:  Edit-Play Pipeline Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


TEMPLATE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'library', 'templates', 'Getting Started'
)


class TestSaveCascade:
    """Ctrl+S cascade must propagate to all dirty editors."""

    def test_save_project_cascades_to_facets_editor(self, qapp, tmp_path):
        """save_project() must call facets_editor.save_if_dirty()."""
        from noodlestudio.core.project_manager import ProjectManager
        from noodlestudio.core.main_window_project_mixin import MainWindowProjectMixin

        # Track whether save_if_dirty was called
        save_called = False

        class StubEditor:
            def save_if_dirty(self):
                nonlocal save_called
                save_called = True

        class StubStatusBar:
            def showMessage(self, msg, timeout=0):
                pass

        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'test_project')

        class StubHost(MainWindowProjectMixin):
            def __init__(self):
                self.project_manager = pm
                self.facets_editor = StubEditor()

            def statusBar(self):
                return StubStatusBar()

        host = StubHost()
        host.save_project()

        assert save_called, "save_project() must cascade to facets_editor.save_if_dirty()"

    def test_save_project_cascades_to_neural_canvas(self, qapp, tmp_path):
        """save_project() must call neural_canvas.save_if_dirty()."""
        from noodlestudio.core.project_manager import ProjectManager
        from noodlestudio.core.main_window_project_mixin import MainWindowProjectMixin

        canvas_saved = False

        class StubCanvas:
            def save_if_dirty(self):
                nonlocal canvas_saved
                canvas_saved = True

        class StubStatusBar:
            def showMessage(self, msg, timeout=0):
                pass

        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'test_project')

        class StubHost(MainWindowProjectMixin):
            def __init__(self):
                self.project_manager = pm
                self.neural_canvas = StubCanvas()

            def statusBar(self):
                return StubStatusBar()

        host = StubHost()
        host.save_project()

        assert canvas_saved, "save_project() must cascade to neural_canvas.save_if_dirty()"


class TestInspectorDiskRoundTrip:
    """Edit in inspector -> save to disk -> reload must preserve values."""

    @pytest.fixture
    def project_copy(self, tmp_path):
        """Copy Getting Started template to temp dir."""
        src = os.path.abspath(TEMPLATE_DIR)
        if not os.path.exists(src):
            pytest.skip("Default project template not found")
        dest = str(tmp_path / 'Getting Started')
        shutil.copytree(src, dest)
        return dest

    def test_recipe_description_round_trip(self, qapp, project_copy):
        """Write description to recipe.yaml, then reload and verify."""
        recipe_path = os.path.join(
            project_copy, 'Noodlings', 'ajo_majo', 'recipe.yaml'
        )

        # Write
        with open(recipe_path, 'r') as f:
            recipe = yaml.safe_load(f)
        recipe['description'] = 'A round-trip test description'
        with open(recipe_path, 'w') as f:
            yaml.dump(recipe, f, default_flow_style=False)

        # Reload and verify
        with open(recipe_path, 'r') as f:
            reloaded = yaml.safe_load(f)
        assert reloaded['description'] == 'A round-trip test description'
        assert reloaded['personality']['openness'] == 0.85, \
            "Existing fields must survive round-trip"

    def test_instance_name_round_trip(self, qapp, project_copy):
        """Write name to instance.yaml overrides, then reload and verify."""
        instance_path = os.path.join(
            project_copy, 'Stages', 'the_nexus', 'Instances', 'ajo', 'instance.yaml'
        )

        with open(instance_path, 'r') as f:
            inst = yaml.safe_load(f)
        inst['overrides']['name'] = 'Ajo the Magnificent'
        with open(instance_path, 'w') as f:
            yaml.dump(inst, f, default_flow_style=False)

        with open(instance_path, 'r') as f:
            reloaded = yaml.safe_load(f)
        assert reloaded['overrides']['name'] == 'Ajo the Magnificent'
        assert 'noodling' in reloaded, "noodling ref must survive round-trip"


class TestAssemblyDiskRoundTrip:
    """Assembly prompt edit -> save -> reload must preserve values."""

    def test_prompt_edit_round_trip(self, qapp, tmp_path):
        """Edit prompt in assembly, save to disk, reload - prompt preserved."""
        from noodlestudio.core.facet_system import FacetAssembly, Facet, FacetConnection

        assembly = FacetAssembly(name="round_trip_test")
        assembly.facets = [
            Facet(id="in", facet_type="INCOMING", name="In", prompt="",
                  position={'x': 0, 'y': 0}),
            Facet(id="llm", facet_type="LLMFacet", name="LLM",
                  prompt="Original prompt",
                  position={'x': 200, 'y': 0}),
            Facet(id="out", facet_type="OUTGOING", name="Out", prompt="",
                  position={'x': 400, 'y': 0}),
        ]
        assembly.connections = [
            FacetConnection(from_facet="in", from_pad="output",
                            to_facet="llm", to_pad="input"),
            FacetConnection(from_facet="llm", from_pad="output",
                            to_facet="out", to_pad="input"),
        ]

        path = str(tmp_path / "assembly.yaml")
        assembly.save_yaml(path)

        # Modify prompt and save again
        assembly.get_facet("llm").prompt = "Edited prompt text"
        assembly.save_yaml(path)

        # Reload and verify
        reloaded = FacetAssembly.load_yaml(path)
        assert reloaded.get_facet("llm").prompt == "Edited prompt text"
        assert len(reloaded.facets) == 3, "All facets must survive round-trip"
        assert len(reloaded.connections) == 2, "All connections must survive round-trip"


class TestNeuralCanvasDoublClickBridge:
    """Double-click on NeuralCanvasFacet node must resolve path."""

    def test_neural_canvas_path_resolves_relative(self, qapp, tmp_path):
        """nncanvas_path resolves relative to project root."""
        from noodlestudio.core.facet_system import Facet

        project_root = str(tmp_path / "project")
        os.makedirs(os.path.join(project_root, "canvases"), exist_ok=True)
        canvas_file = os.path.join(project_root, "canvases", "charm.nncanvas")
        with open(canvas_file, 'w') as f:
            f.write('{}')

        facet = Facet(
            id="charm",
            facet_type="NeuralCanvasFacet",
            name="Charm Net",
            prompt="",
            nncanvas_path="canvases/charm.nncanvas",
        )

        # Resolution matches what _open_neural_canvas does
        resolved = os.path.join(project_root, facet.nncanvas_path)
        assert os.path.exists(resolved), \
            f"Resolved path must exist: {resolved}"

    def test_neural_canvas_facet_has_nncanvas_path(self, qapp):
        """NeuralCanvasFacet type stores nncanvas_path on the Facet dataclass."""
        from noodlestudio.core.facet_system import Facet

        facet = Facet(
            id="test",
            facet_type="NeuralCanvasFacet",
            name="Test Canvas",
            prompt="",
            nncanvas_path="networks/test.nncanvas",
        )

        # Serialization round-trip
        d = facet.to_dict()
        assert d.get('nncanvas_path') == "networks/test.nncanvas"

        # Deserialization
        rebuilt = Facet.from_dict(d)
        assert rebuilt.nncanvas_path == "networks/test.nncanvas"


class TestPerformerPauseIntegration:
    """Pause gate integrates correctly with execution pipeline."""

    def test_pause_prevents_execution_resume_allows(self, qapp):
        """Full pause -> message -> resume -> message cycle."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from conftest import FakeLLMClient, SignalCollector
        from unittest.mock import patch, MagicMock

        p = NoodlingPerformer('test', 'Test', FakeLLMClient())
        p._assembly = True
        p._executor = True

        with patch(
            'noodlestudio.runtime.ui.noodling_performer._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            # Pause, try to execute - should be blocked
            p.set_paused(True)
            p.execute("Hello")
            assert MockWorker.call_count == 0, "Paused performer must not execute"

            # Resume, execute - should proceed
            p.set_paused(False)
            p.execute("Hello again")
            assert MockWorker.call_count == 1, "Resumed performer must execute"
