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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
