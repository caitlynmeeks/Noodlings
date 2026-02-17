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
#   NoodleStudio Test Fixtures
#
#   Shared fixtures for all NoodleStudio tests. Run tests wit...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.conftest
# PURPOSE:  NoodleStudio Test Fixtures
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   qapp(), main_window(), find_test_radiance()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import sys
import os
from pathlib import Path
import numpy as np

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# Qt Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def qapp():
    """Create QApplication once for all tests."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qapp, qtbot):
    """Create a MainWindow instance for testing."""
    from noodlestudio.core.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    # Don't show the window (headless testing)
    yield window

    window.close()


# ============================================================================
# Radiance/Gaussian Fixtures
# ============================================================================

def find_test_radiance():
    """Find a test .radiance file."""
    candidates = [
        Path(__file__).parent.parent.parent.parent / "external/vrm_samples/alicia_densified_tuned.radiance",
        Path(__file__).parent.parent.parent.parent / "external/vrm_samples/alicia_textured.radiance",
        Path(__file__).parent.parent.parent / "cmush/world/radiances/alicia.radiance",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


@pytest.fixture
def synthetic_radiance_asset():
    """Create a synthetic RadianceAsset for testing without file I/O."""
    from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

    asset = RadianceAsset()
    n = 1000
    asset.positions = np.random.randn(n, 3).astype(np.float32) * 0.5
    asset.positions[:, 1] += 1.0  # Center around y=1
    asset.scales = np.ones((n, 3), dtype=np.float32) * 0.02
    asset.rotations = np.zeros((n, 4), dtype=np.float32)
    asset.rotations[:, 3] = 1.0  # Identity quaternion
    asset.opacities = np.ones(n, dtype=np.float32)
    asset.sh_dc = np.random.rand(n, 3).astype(np.float32)

    return asset


@pytest.fixture
def radiance_component(synthetic_radiance_asset):
    """Create a RadianceComponent with synthetic data."""
    from noodlestudio.core.radiance_component import RadianceComponent

    component = RadianceComponent(entity_id="test_component")
    component._asset = synthetic_radiance_asset
    component._asset_path = "synthetic"

    return component


@pytest.fixture
def loaded_radiance_component():
    """Create a RadianceComponent with real asset if available, else synthetic."""
    from noodlestudio.core.radiance_component import RadianceComponent

    component = RadianceComponent(entity_id="test_loaded")

    radiance_path = find_test_radiance()
    if radiance_path:
        component.load_asset(radiance_path)
    else:
        # Fall back to synthetic
        from noodlestudio.core.semantic_world.radiance_format import RadianceAsset
        asset = RadianceAsset()
        n = 1000
        asset.positions = np.random.randn(n, 3).astype(np.float32) * 0.5
        asset.positions[:, 1] += 1.0
        asset.scales = np.ones((n, 3), dtype=np.float32) * 0.02
        asset.rotations = np.zeros((n, 4), dtype=np.float32)
        asset.rotations[:, 3] = 1.0
        asset.opacities = np.ones(n, dtype=np.float32)
        asset.sh_dc = np.random.rand(n, 3).astype(np.float32)
        component._asset = asset
        component._asset_path = "synthetic"

    return component


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_noodling_data():
    """Mock noodling entity data for testing."""
    return {
        'id': 'test_noodling_001',
        'name': 'TestNoodling',
        'noodling_ref': 'empty_noodling',
        'path': '/test/path/instance.yaml',
        'data': {
            'noodling': 'empty_noodling',
            'facet_assembly': {
                'ref': 'library/empty_noodling'
            }
        }
    }


@pytest.fixture
def mock_prop_data():
    """Mock prop entity data for testing."""
    return {
        'id': 'test_prop_001',
        'name': 'TestProp',
        'prim_ref': 'cube',
        'position': [0, 0, 0]
    }


@pytest.fixture
def mock_zone_data():
    """Mock zone entity data for testing."""
    return {
        'id': 'test_zone_001',
        'name': 'TestZone',
        'bounds': {'min': [-5, 0, -5], 'max': [5, 10, 5]}
    }


# ============================================================================
# Facet System Fixtures
# ============================================================================

@pytest.fixture
def empty_facet_assembly():
    """Create an empty FacetAssembly for testing."""
    from noodlestudio.core.facet_system import FacetAssembly

    return FacetAssembly(name="test_assembly")


@pytest.fixture
def simple_facet_assembly():
    """Create a simple FacetAssembly with INCOMING -> LLM -> OUTGOING."""
    from noodlestudio.core.facet_system import (
        FacetAssembly, Facet, FacetType, Connection
    )

    assembly = FacetAssembly(name="simple_test")

    # Add facets
    incoming = Facet(
        id="incoming_1",
        type=FacetType.INCOMING,
        name="Input",
        position=(100, 200)
    )
    llm = Facet(
        id="llm_1",
        type=FacetType.LLM,
        name="Process",
        position=(300, 200),
        config={'model': 'test-model'}
    )
    outgoing = Facet(
        id="outgoing_1",
        type=FacetType.OUTGOING,
        name="Output",
        position=(500, 200)
    )

    assembly.facets = [incoming, llm, outgoing]
    assembly.connections = [
        Connection(from_facet="incoming_1", to_facet="llm_1"),
        Connection(from_facet="llm_1", to_facet="outgoing_1"),
    ]

    return assembly


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create standard subdirectories
    (project_dir / "Noodlings").mkdir()
    (project_dir / "Stages").mkdir()
    (project_dir / "Prims").mkdir()
    (project_dir / "Library").mkdir()

    return project_dir


@pytest.fixture
def temp_stage_dir(temp_project_dir):
    """Create a temporary stage directory."""
    stage_dir = temp_project_dir / "Stages" / "test_stage"
    stage_dir.mkdir(parents=True)

    # Create minimal stage.yaml
    stage_yaml = stage_dir / "stage.yaml"
    stage_yaml.write_text("name: Test Stage\nzones: []\n")

    return stage_dir


# ============================================================================
# Fake LLM Client (for tests that create NoodlingPerformer)
# ============================================================================

class FakeLLMClient:
    """Lightweight stand-in for HeadlessLLMClient.

    Returns canned responses for testing without real LLM calls.
    Call real ``__init__`` / real interface -- no MagicMock.
    """

    async def close(self):
        pass


class SignalCollector:
    """Collects signal emissions for test assertions.

    Usage:
        collector = SignalCollector()
        performer.responseReady.connect(collector)
        # ... trigger signal ...
        assert collector.values == ["expected text"]
    """

    def __init__(self):
        self.values: list = []

    def __call__(self, *args):
        if len(args) == 1:
            self.values.append(args[0])
        elif len(args) == 0:
            self.values.append(None)
        else:
            self.values.append(args)


# ============================================================================
# Guide Performance Manager Fixtures
# ============================================================================

class StubFacetsEditor:
    """Lightweight stand-in for UnifiedEditorPanel.

    Records execution events for test assertions without any Qt
    widget overhead.  Every test that needs to inspect live-viz
    events reads ``editor.events`` instead of digging through
    MagicMock call_args.
    """

    def __init__(self):
        self.events: list[dict] = []
        self._selected_noodling_id = None
        self._ensemble_noodlings: list[dict] = []
        self._loaded_assemblies: list[tuple] = []  # (assembly, path)

    def _handle_execution_event(self, event: dict):
        self.events.append(event)

    def load_assembly_from_data(self, assembly, *, force_reload=False,
                                source_path=None):
        self._loaded_assemblies.append((assembly, source_path))

    def set_ensemble_noodlings(self, noodlings: list):
        self._ensemble_noodlings = noodlings
        if noodlings:
            self._selected_noodling_id = noodlings[0]['id']

    def select_noodling(self, noodling_id: str):
        for entry in self._ensemble_noodlings:
            if entry['id'] == noodling_id:
                self._selected_noodling_id = noodling_id
                return

    def clear_ensemble_noodlings(self):
        self._ensemble_noodlings = []
        self._selected_noodling_id = None


_SENTINEL = object()


class StubMainWindow:
    """Minimal stand-in for MainWindow.

    Provides the attributes GuidePerformanceManager actually reads
    from main_window: ``unified_editor``, ``center_tabs``, and acts
    as a valid parent reference.

    Pass ``unified_editor=None`` explicitly to simulate a main window
    with no editor.  Omit the argument to get a default
    StubFacetsEditor automatically.
    """

    def __init__(self, facets_editor=_SENTINEL, unified_editor=_SENTINEL):
        # Support both old and new kwarg for backward compat in tests
        if unified_editor is not _SENTINEL:
            self.unified_editor = unified_editor
        elif facets_editor is not _SENTINEL:
            self.unified_editor = facets_editor
        else:
            self.unified_editor = StubFacetsEditor()
        self.center_tabs = None  # Only used in start_performance tab switch


class StubWindow:
    """Lightweight stand-in for GuidePerformanceWindow.

    Records method calls so tests can assert on them without
    creating real Qt widgets. Supports both single and ensemble APIs.
    """

    def __init__(self):
        self.texts: list[str] = []
        self.errors: list[str] = []
        self.busy_states: list = []   # (busy,) or (busy, name)
        self.blend_shapes_calls: list = []  # (shapes,) or (shapes, nid)
        self._vrm_viewport = None
        self._typed_chars: list[str] = []
        self._text_blocks_begun = 0
        self._text_blocks_ended = 0
        self._noodling_texts: list[tuple] = []  # (nid, name, text)
        self._speaking_mode_calls: list[tuple] = []

    def append_guide_text(self, text):
        self.texts.append(text)

    def _show_error(self, msg):
        self.errors.append(msg)

    def set_busy(self, busy, name=None):
        self.busy_states.append((busy, name))

    def set_blend_shapes(self, shapes, noodling_id='default'):
        self.blend_shapes_calls.append((shapes, noodling_id))

    def begin_guide_text(self):
        self._text_blocks_begun += 1

    def begin_noodling_text(self, noodling_id, name):
        self._text_blocks_begun += 1

    def append_character(self, char):
        self._typed_chars.append(char)

    def end_guide_text(self):
        self._text_blocks_ended += 1

    def end_noodling_text(self):
        self._text_blocks_ended += 1

    def append_noodling_text(self, noodling_id, name, text):
        self._noodling_texts.append((noodling_id, name, text))

    def set_speaking_mode(self, active, intensity=0.7, noodling_id='default'):
        self._speaking_mode_calls.append((active, intensity, noodling_id))

    def set_performer_name(self, noodling_id, name):
        pass

    def set_active_speaker(self, noodling_id=None):
        pass

    def show_play_header(self, title):
        pass

    def close(self):
        pass

    def isVisible(self):
        return True


@pytest.fixture
def guide_manager():
    """Real GuidePerformanceManager with lightweight stub dependencies.

    Calls real ``__init__``, so all attributes are properly set.
    No MagicMock -- uses StubMainWindow, StubFacetsEditor, and
    StubWindow for testable behaviour verification.
    """
    from noodlestudio.runtime.ui.guide_performance_manager import (
        GuidePerformanceManager,
    )

    stub_editor = StubFacetsEditor()
    stub_main = StubMainWindow(unified_editor=stub_editor)

    manager = GuidePerformanceManager(stub_main)
    manager._assembly_editor = stub_editor  # Pre-cache to skip lookup
    manager._window = StubWindow()

    return manager


@pytest.fixture
def performer(qapp):
    """Real NoodlingPerformer with lightweight fake dependencies.

    Calls real ``__init__``, so all attributes are properly set.
    Uses FakeLLMClient -- no MagicMock, no ``__new__`` bypass.
    """
    from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

    p = NoodlingPerformer(
        noodling_id='ajo',
        name='Ajo',
        llm_client=FakeLLMClient()
    )
    return p


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
