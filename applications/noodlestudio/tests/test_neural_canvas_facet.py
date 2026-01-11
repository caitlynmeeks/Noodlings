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
#   Tests for NeuralCanvasFacet - Execute .nncanvas files as facets
#
#   Tests the bridge between:
#   - Facets system (cognitive architecture)
#   - NeuralCanvas (visual neural network editor)
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_neural_canvas_facet
# PURPOSE:  Tests for NeuralCanvasFacet
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import asyncio
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mark all async tests for anyio
pytestmark = pytest.mark.anyio


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_nncanvas_json():
    """Create a simple NeuralGraph JSON for testing."""
    # Uses valid NodeType values from neural_node.py:
    # INPUT, OUTPUT, LINEAR, LSTM, GRU, TANH, RELU, etc.
    # Parameters go inside "params" dict per NeuralNode.from_dict()
    return {
        "name": "Test Affect Network",
        "version": "1.0.0",
        "nodes": [
            {
                "id": "input_1",
                "name": "Affect Input",
                "type": "INPUT",
                "params": {"input_size": 5},
                "position": [100, 100]
            },
            {
                "id": "linear_1",
                "name": "Hidden Layer",
                "type": "LINEAR",
                "params": {"in_features": 5, "out_features": 8},
                "position": [300, 100]
            },
            {
                "id": "relu_1",
                "name": "Activation",
                "type": "RELU",
                "params": {},
                "position": [400, 100]
            },
            {
                "id": "output_1",
                "name": "Output",
                "type": "OUTPUT",
                "params": {"output_size": 8},
                "position": [500, 100]
            }
        ],
        "connections": [
            {"from_node": "input_1", "from_port": 0, "to_node": "linear_1", "to_port": 0},
            {"from_node": "linear_1", "from_port": 0, "to_node": "relu_1", "to_port": 0},
            {"from_node": "relu_1", "from_port": 0, "to_node": "output_1", "to_port": 0}
        ]
    }


@pytest.fixture
def temp_nncanvas_file(sample_nncanvas_json, tmp_path):
    """Create a temporary .nncanvas file."""
    nncanvas_path = tmp_path / "test_network.nncanvas"
    with open(nncanvas_path, 'w') as f:
        json.dump(sample_nncanvas_json, f)
    return str(nncanvas_path)


# ==============================================================================
# Test NeuralCanvasFacet Creation
# ==============================================================================

class TestNeuralCanvasFacetCreation:
    """Tests for NeuralCanvasFacet instantiation."""

    def test_create_with_valid_path(self, temp_nncanvas_file):
        """Can create NeuralCanvasFacet with valid .nncanvas file."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="test_neural_1",
            name="Test Neural",
            nncanvas_path=temp_nncanvas_file
        )

        assert facet.id == "test_neural_1"
        assert facet.name == "Test Neural"
        assert facet.nncanvas_path == temp_nncanvas_file
        assert facet.graph is not None

    def test_create_with_missing_file(self, tmp_path):
        """Can create NeuralCanvasFacet with missing file (loads later)."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="test_neural_2",
            name="Test Neural",
            nncanvas_path=str(tmp_path / "nonexistent.nncanvas")
        )

        assert facet.graph is None  # Not loaded yet
        assert facet._initialized is False

    def test_create_with_empty_path(self):
        """Can create NeuralCanvasFacet with empty path."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="test_neural_3",
            name="Test Neural",
            nncanvas_path=""
        )

        assert facet.graph is None

    def test_create_with_relative_path(self, sample_nncanvas_json, tmp_path):
        """Can create NeuralCanvasFacet with relative path."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        # Create file in project root
        nncanvas_path = tmp_path / "networks" / "test.nncanvas"
        nncanvas_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nncanvas_path, 'w') as f:
            json.dump(sample_nncanvas_json, f)

        facet = NeuralCanvasFacet(
            facet_id="test_neural_4",
            name="Test Neural",
            nncanvas_path="networks/test.nncanvas",
            project_root=str(tmp_path)
        )

        assert facet.graph is not None


class TestNeuralCanvasFacetGraphLoading:
    """Tests for graph loading."""

    def test_graph_loaded_correctly(self, temp_nncanvas_file):
        """Graph is loaded and parsed correctly."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="test_1",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        assert facet.graph is not None
        assert facet.graph.name == "Test Affect Network"
        assert len(facet.graph.nodes) == 4  # INPUT, LINEAR, RELU, OUTPUT
        assert len(facet.graph.connections) == 3

    def test_executor_created(self, temp_nncanvas_file):
        """Executor is created for loaded graph."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="test_2",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        assert facet.executor is not None

    def test_reload_graph(self, temp_nncanvas_file, sample_nncanvas_json, tmp_path):
        """Can reload graph after file changes."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="test_3",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        original_name = facet.graph.name

        # Modify file
        sample_nncanvas_json["name"] = "Updated Network"
        with open(temp_nncanvas_file, 'w') as f:
            json.dump(sample_nncanvas_json, f)

        # Reload
        facet.reload_graph()

        assert facet.graph.name == "Updated Network"


# ==============================================================================
# Test NeuralCanvasFacet Execution
# ==============================================================================

class TestNeuralCanvasFacetExecution:
    """Tests for facet execution."""

    async def test_execute_with_affect_input(self, temp_nncanvas_file):
        """Can execute with affect vector input."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="exec_1",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        result = await facet.execute({
            'affect': [0.5, 0.6, 0.4, 0.1, 0.2]
        })

        assert 'error' not in result or result.get('error') is None

    async def test_execute_with_input_key(self, temp_nncanvas_file):
        """Can execute with 'input' key."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="exec_2",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        result = await facet.execute({
            'input': [0.5, 0.6, 0.4, 0.1, 0.2]
        })

        assert 'error' not in result or result.get('error') is None

    async def test_execute_returns_error_without_graph(self):
        """Returns error when no graph loaded."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="exec_3",
            name="Test",
            nncanvas_path=""
        )

        result = await facet.execute({'affect': [0.5, 0.6, 0.4, 0.1, 0.2]})

        assert 'error' in result

    async def test_execute_tracks_stats(self, temp_nncanvas_file):
        """Execution tracks statistics."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="exec_4",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        await facet.execute({'affect': [0.5, 0.6, 0.4, 0.1, 0.2]})
        await facet.execute({'affect': [0.3, 0.4, 0.5, 0.2, 0.1]})

        stats = facet.get_execution_stats()
        assert stats['execution_count'] == 2
        assert stats['total_time'] > 0

    def test_sync_execution(self, temp_nncanvas_file):
        """Can execute synchronously."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="exec_5",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        result = facet.execute_sync({'affect': [0.5, 0.6, 0.4, 0.1, 0.2]})

        assert isinstance(result, dict)


# ==============================================================================
# Test NeuralCanvasFacet Serialization
# ==============================================================================

class TestNeuralCanvasFacetSerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict(self, temp_nncanvas_file):
        """Can serialize to dict."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="serial_1",
            name="Test Neural",
            nncanvas_path=temp_nncanvas_file
        )

        data = facet.to_dict()

        assert data['id'] == "serial_1"
        assert data['name'] == "Test Neural"
        assert data['type'] == "NeuralCanvasFacet"
        assert data['nncanvas_path'] == temp_nncanvas_file

    def test_from_dict(self, temp_nncanvas_file):
        """Can deserialize from dict."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        data = {
            'id': 'serial_2',
            'name': 'Test Neural 2',
            'type': 'NeuralCanvasFacet',
            'nncanvas_path': temp_nncanvas_file
        }

        facet = NeuralCanvasFacet.from_dict(data)

        assert facet.id == "serial_2"
        assert facet.name == "Test Neural 2"
        assert facet.nncanvas_path == temp_nncanvas_file


# ==============================================================================
# Test Facet System Integration
# ==============================================================================

class TestFacetSystemIntegration:
    """Tests for integration with Facet dataclass."""

    def test_facet_dataclass_nncanvas_path(self):
        """Facet dataclass has nncanvas_path field."""
        from noodlestudio.core.facet_system import Facet

        facet = Facet(
            id="integration_1",
            name="Neural Test",
            facet_type="NeuralCanvasFacet",
            prompt="",
            nncanvas_path="path/to/network.nncanvas"
        )

        assert facet.nncanvas_path == "path/to/network.nncanvas"

    def test_facet_to_dict_includes_nncanvas_path(self):
        """Facet.to_dict() includes nncanvas_path."""
        from noodlestudio.core.facet_system import Facet

        facet = Facet(
            id="integration_2",
            name="Neural Test",
            facet_type="NeuralCanvasFacet",
            prompt="",
            nncanvas_path="networks/charm.nncanvas"
        )

        data = facet.to_dict()

        assert 'nncanvas_path' in data
        assert data['nncanvas_path'] == "networks/charm.nncanvas"

    def test_facet_from_dict_loads_nncanvas_path(self):
        """Facet.from_dict() loads nncanvas_path."""
        from noodlestudio.core.facet_system import Facet

        data = {
            'id': 'integration_3',
            'name': 'Neural Test',
            'type': 'NeuralCanvasFacet',
            'prompt': '',
            'nncanvas_path': 'networks/test.nncanvas'
        }

        facet = Facet.from_dict(data)

        assert facet.nncanvas_path == "networks/test.nncanvas"


# ==============================================================================
# Test Validation
# ==============================================================================

class TestNeuralCanvasFacetValidation:
    """Tests for graph validation."""

    def test_validate_loaded_graph(self, temp_nncanvas_file):
        """Can validate loaded graph."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="valid_1",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        result = facet.validate_graph()

        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result

    def test_validate_without_graph(self):
        """Validation fails without graph."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="valid_2",
            name="Test",
            nncanvas_path=""
        )

        result = facet.validate_graph()

        assert result['valid'] is False
        assert len(result['errors']) > 0


# ==============================================================================
# Test Input/Output Pads
# ==============================================================================

class TestNeuralCanvasFacetPads:
    """Tests for input/output pad discovery."""

    def test_get_input_pads(self, temp_nncanvas_file):
        """Can get input pad names from graph."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="pads_1",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        input_pads = facet.get_input_pads()

        assert isinstance(input_pads, list)
        assert len(input_pads) > 0

    def test_get_output_pads(self, temp_nncanvas_file):
        """Can get output pad names from graph."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="pads_2",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        output_pads = facet.get_output_pads()

        assert isinstance(output_pads, list)
        assert len(output_pads) > 0

    def test_default_pads_without_graph(self):
        """Returns default pads without graph."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="pads_3",
            name="Test",
            nncanvas_path=""
        )

        assert facet.get_input_pads() == ["input"]
        assert facet.get_output_pads() == ["output"]


# ==============================================================================
# Test Token Usage (Always Zero for Neural)
# ==============================================================================

class TestNeuralCanvasFacetTokens:
    """Tests for token usage (always zero for neural computation)."""

    async def test_token_usage_is_zero(self, temp_nncanvas_file):
        """Token usage is always zero."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="tokens_1",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        await facet.execute({'affect': [0.5, 0.6, 0.4, 0.1, 0.2]})

        usage = facet.get_token_usage()

        assert usage['last_tokens'] == 0
        assert usage['total_tokens'] == 0
        assert usage['avg_tokens'] == 0


# ==============================================================================
# Test repr
# ==============================================================================

class TestNeuralCanvasFacetRepr:
    """Tests for string representation."""

    def test_repr_with_graph(self, temp_nncanvas_file):
        """Repr shows loaded status."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="repr_1",
            name="Test",
            nncanvas_path=temp_nncanvas_file
        )

        repr_str = repr(facet)

        assert "Test" in repr_str
        assert "loaded" in repr_str

    def test_repr_without_graph(self):
        """Repr shows not loaded status."""
        from noodlestudio.core.neural_canvas_facet import NeuralCanvasFacet

        facet = NeuralCanvasFacet(
            facet_id="repr_2",
            name="Test",
            nncanvas_path=""
        )

        repr_str = repr(facet)

        assert "not loaded" in repr_str


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
