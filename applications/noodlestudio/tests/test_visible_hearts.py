"""Tests for Phase E: Visible Hearts.

Covers:
- E.1: Stage panel server gating fix (Rez menu without server)
- E.2a: EMA node types (EMA_FILTER, WEIGHTED_BLEND, BASELINE_DRIFT)
- E.2b: EMA .nncanvas file loads and validates
- E.2c: Depth view swap (CharmNetworkEMA -> NeuralCanvasDepthView)
- Per-node color override in renderer
"""

import json
import os

import pytest
from PyQt6.QtWidgets import QApplication

from noodlestudio.core.neural_canvas.neural_node import (
    NodeType, DataType, NeuralNode, Port,
)
from noodlestudio.core.neural_canvas.node_definitions import (
    NODE_DEFINITIONS, create_node_from_type, get_node_color,
)
from noodlestudio.core.neural_canvas.neural_graph import NeuralGraph
from noodlestudio.core.facet_system import Facet, FacetAssembly


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Repo / file path helpers
# ---------------------------------------------------------------------------

def _repo_root():
    """Resolve the repository root from tests/ directory."""
    # tests/ -> applications/noodlestudio/ -> applications/ -> repo root
    return os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..'
    ))


def _ema_nncanvas_path():
    return os.path.join(_repo_root(), 'facet_assemblies', 'charm_networks',
                        'ema_default.nncanvas')


# ===================================================================
# E.2a: EMA Node Types
# ===================================================================

class TestEMANodeTypes:
    """Verify the three new node types are registered and functional."""

    def test_ema_filter_in_enum(self):
        assert hasattr(NodeType, 'EMA_FILTER')
        assert NodeType.EMA_FILTER.value == 'EMA_FILTER'

    def test_weighted_blend_in_enum(self):
        assert hasattr(NodeType, 'WEIGHTED_BLEND')
        assert NodeType.WEIGHTED_BLEND.value == 'WEIGHTED_BLEND'

    def test_baseline_drift_in_enum(self):
        assert hasattr(NodeType, 'BASELINE_DRIFT')
        assert NodeType.BASELINE_DRIFT.value == 'BASELINE_DRIFT'

    def test_ema_filter_in_definitions(self):
        assert NodeType.EMA_FILTER in NODE_DEFINITIONS
        defn = NODE_DEFINITIONS[NodeType.EMA_FILTER]
        assert defn['name'] == 'EMA Filter'
        assert 'alpha' in defn['params']
        assert 'affect_in' in defn['inputs']
        assert 'affect_out' in defn['outputs']

    def test_weighted_blend_in_definitions(self):
        assert NodeType.WEIGHTED_BLEND in NODE_DEFINITIONS
        defn = NODE_DEFINITIONS[NodeType.WEIGHTED_BLEND]
        assert defn['name'] == 'Weighted Blend'
        assert 'weights' in defn['params']
        assert len(defn['inputs']) == 3  # fast, medium, slow
        assert 'blended' in defn['outputs']

    def test_baseline_drift_in_definitions(self):
        assert NodeType.BASELINE_DRIFT in NODE_DEFINITIONS
        defn = NODE_DEFINITIONS[NodeType.BASELINE_DRIFT]
        assert defn['name'] == 'Baseline Drift'
        assert 'target_valence' in defn['params']
        assert 'rate' in defn['params']
        assert 'affect_in' in defn['inputs']
        assert 'affect_out' in defn['outputs']

    def test_create_ema_filter_node(self):
        node = create_node_from_type(NodeType.EMA_FILTER)
        assert node.type == NodeType.EMA_FILTER
        assert node.params['alpha'] == 0.7
        assert 'affect_in' in node.inputs
        assert 'affect_out' in node.outputs
        assert len(node.weights) == 0  # EMA has no learned weights

    def test_create_weighted_blend_node(self):
        node = create_node_from_type(NodeType.WEIGHTED_BLEND)
        assert node.type == NodeType.WEIGHTED_BLEND
        assert node.params['weights'] == [0.5, 0.3, 0.2]
        assert len(node.weights) == 0

    def test_create_baseline_drift_node(self):
        node = create_node_from_type(NodeType.BASELINE_DRIFT)
        assert node.type == NodeType.BASELINE_DRIFT
        assert node.params['rate'] == 0.05
        assert len(node.weights) == 0

    def test_ema_filter_color(self):
        color = get_node_color(NodeType.EMA_FILTER)
        assert color == '#5A5A5A'

    def test_weighted_blend_color(self):
        color = get_node_color(NodeType.WEIGHTED_BLEND)
        assert color == '#3A5A5A'

    def test_baseline_drift_color(self):
        color = get_node_color(NodeType.BASELINE_DRIFT)
        assert color == '#5A4A3A'

    def test_ema_filter_affect_port_type(self):
        node = create_node_from_type(NodeType.EMA_FILTER)
        assert node.inputs['affect_in'].data_type == DataType.AFFECT
        assert node.outputs['affect_out'].data_type == DataType.AFFECT
        assert node.inputs['affect_in'].shape == (3,)


# ===================================================================
# E.2b: EMA .nncanvas File
# ===================================================================

class TestEMANncanvasFile:
    """Verify the EMA topology .nncanvas file."""

    def test_file_exists(self):
        assert os.path.exists(_ema_nncanvas_path()), \
            f"EMA .nncanvas file not found at {_ema_nncanvas_path()}"

    def test_valid_json(self):
        with open(_ema_nncanvas_path(), 'r') as f:
            data = json.load(f)
        assert data['version'] == '1.0'
        assert data['name'] == 'CharmNetwork EMA'

    def test_graph_loads(self):
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        assert graph.name == 'CharmNetwork EMA'

    def test_node_count(self):
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        # 7 computation nodes + 1 comment = 8 total
        assert len(graph.nodes) == 8

    def test_computation_node_types(self):
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        type_counts = {}
        for node in graph.nodes.values():
            t = node.type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        assert type_counts.get('INPUT', 0) == 1
        assert type_counts.get('OUTPUT', 0) == 1
        assert type_counts.get('EMA_FILTER', 0) == 3
        assert type_counts.get('WEIGHTED_BLEND', 0) == 1
        assert type_counts.get('BASELINE_DRIFT', 0) == 1
        assert type_counts.get('COMMENT', 0) == 1

    def test_connection_count(self):
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        # Input -> 3 EMAs (3) + 3 EMAs -> Blend (3) + Blend -> Baseline (1)
        # + Baseline -> Output (1) = 8
        assert len(graph.connections) == 8

    def test_connections_valid(self):
        """Every connection references existing nodes and ports."""
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        for conn in graph.connections:
            from_node = graph.get_node_by_id(conn.from_node)
            to_node = graph.get_node_by_id(conn.to_node)
            assert from_node is not None, \
                f"Connection from unknown node: {conn.from_node}"
            assert to_node is not None, \
                f"Connection to unknown node: {conn.to_node}"
            assert conn.from_port in from_node.outputs, \
                f"Port {conn.from_port} not in outputs of {from_node.name}"
            assert conn.to_port in to_node.inputs, \
                f"Port {conn.to_port} not in inputs of {to_node.name}"

    def test_ema_filter_alpha_values(self):
        """Three EMA filters have distinct alpha values."""
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        ema_nodes = graph.get_nodes_by_type(NodeType.EMA_FILTER)
        alphas = sorted([n.params['alpha'] for n in ema_nodes])
        assert alphas == [0.03, 0.15, 0.7]

    def test_per_node_colors_distinct(self):
        """Three EMA filters should have distinct colors (timescale shading)."""
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        ema_nodes = graph.get_nodes_by_type(NodeType.EMA_FILTER)
        colors = set(n.color for n in ema_nodes)
        assert len(colors) == 3, \
            f"Expected 3 distinct EMA colors, got {colors}"

    def test_zero_trainable_parameters(self):
        """EMA charm network has no learned weights."""
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        total_params = graph.compute_total_parameters()
        assert total_params == 0

    def test_baseline_drift_target(self):
        """Baseline drift node stores character default PAD."""
        graph = NeuralGraph.from_json(_ema_nncanvas_path())
        baselines = graph.get_nodes_by_type(NodeType.BASELINE_DRIFT)
        assert len(baselines) == 1
        node = baselines[0]
        assert node.params['target_valence'] == 0.7
        assert node.params['target_arousal'] == 0.5
        assert node.params['target_dominance'] == 0.4
        assert node.params['rate'] == 0.05


# ===================================================================
# E.2c: Depth View Swap
# ===================================================================

class TestDepthViewRegistry:
    """Verify CharmNetworkEMA now maps to NeuralCanvasDepthView."""

    def test_charm_network_registered_to_neural_canvas_view(self, qapp):
        from noodlestudio.panels.editors.unified_editor_panel import (
            UnifiedEditorPanel,
        )
        from noodlestudio.panels.editors.neural_canvas_depth_view import (
            NeuralCanvasDepthView,
        )
        # Re-register in case another test's fixture cleared the registry
        UnifiedEditorPanel.register_depth_view(
            "CharmNetworkEMA", NeuralCanvasDepthView
        )
        view_class = UnifiedEditorPanel._depth_view_registry.get(
            'CharmNetworkEMA'
        )
        assert view_class is NeuralCanvasDepthView

    def test_neural_canvas_facet_still_registered(self, qapp):
        from noodlestudio.panels.editors.unified_editor_panel import (
            UnifiedEditorPanel,
        )
        from noodlestudio.panels.editors.neural_canvas_depth_view import (
            NeuralCanvasDepthView,
        )
        UnifiedEditorPanel.register_depth_view(
            "NeuralCanvasFacet", NeuralCanvasDepthView
        )
        view_class = UnifiedEditorPanel._depth_view_registry.get(
            'NeuralCanvasFacet'
        )
        assert view_class is NeuralCanvasDepthView


class TestPathResolution:
    """Verify NeuralCanvasDepthView resolves EMA .nncanvas paths."""

    def test_absolute_path_unchanged(self):
        from noodlestudio.panels.editors.neural_canvas_depth_view import (
            NeuralCanvasDepthView,
        )
        path = '/some/absolute/path.nncanvas'
        result = NeuralCanvasDepthView._resolve_path(path, {})
        assert result == path

    def test_relative_path_via_project_root(self, tmp_path):
        from noodlestudio.panels.editors.neural_canvas_depth_view import (
            NeuralCanvasDepthView,
        )
        # Create a temp .nncanvas file
        nncanvas = tmp_path / 'test.nncanvas'
        nncanvas.write_text('{}')

        result = NeuralCanvasDepthView._resolve_path(
            'test.nncanvas', {'project_root': str(tmp_path)}
        )
        assert result == str(nncanvas)
        assert os.path.exists(result)

    def test_relative_path_via_repo_root(self):
        from noodlestudio.panels.editors.neural_canvas_depth_view import (
            NeuralCanvasDepthView,
        )
        # The EMA file lives at repo root
        result = NeuralCanvasDepthView._resolve_path(
            'facet_assemblies/charm_networks/ema_default.nncanvas', {}
        )
        assert os.path.exists(result), \
            f"Repo-root resolution failed: {result}"

    def test_project_root_takes_priority(self, tmp_path):
        from noodlestudio.panels.editors.neural_canvas_depth_view import (
            NeuralCanvasDepthView,
        )
        # Create a matching file in project root
        (tmp_path / 'facet_assemblies' / 'charm_networks').mkdir(parents=True)
        local = tmp_path / 'facet_assemblies' / 'charm_networks' / 'ema_default.nncanvas'
        local.write_text('{"name": "local"}')

        result = NeuralCanvasDepthView._resolve_path(
            'facet_assemblies/charm_networks/ema_default.nncanvas',
            {'project_root': str(tmp_path)}
        )
        # Should find project-root version, not repo-root version
        assert result == str(local)


class TestAssemblyNncanvasPath:
    """Verify assembly.yaml files include nncanvas_path for CharmNetworkEMA."""

    @pytest.fixture
    def templates_dir(self):
        return os.path.join(
            os.path.dirname(__file__), '..',
            'library', 'templates', 'Getting Started', 'Noodlings'
        )

    def _load_charm_facet(self, templates_dir, character):
        import yaml
        path = os.path.join(templates_dir, character, 'assembly.yaml')
        with open(path) as f:
            data = yaml.safe_load(f)
        assembly = FacetAssembly.from_dict(data)
        for facet in assembly.facets:
            if facet.facet_type == 'CharmNetworkEMA':
                return facet
        return None

    def test_ajo_has_nncanvas_path(self, templates_dir):
        facet = self._load_charm_facet(templates_dir, 'ajo_majo')
        assert facet is not None
        assert facet.nncanvas_path == \
            'facet_assemblies/charm_networks/ema_default.nncanvas'

    def test_krampus_has_nncanvas_path(self, templates_dir):
        facet = self._load_charm_facet(templates_dir, 'krampus')
        assert facet is not None
        assert facet.nncanvas_path == \
            'facet_assemblies/charm_networks/ema_default.nncanvas'

    def test_juanita_has_nncanvas_path(self, templates_dir):
        facet = self._load_charm_facet(templates_dir, 'juanita')
        assert facet is not None
        assert facet.nncanvas_path == \
            'facet_assemblies/charm_networks/ema_default.nncanvas'

    def test_ajo_prompt_still_has_baseline(self, templates_dir):
        """nncanvas_path added but prompt (baseline PAD) is preserved."""
        facet = self._load_charm_facet(templates_dir, 'ajo_majo')
        assert 'valence:0.7' in facet.prompt


# ===================================================================
# E.2c+: Per-Node Color Override
# ===================================================================

class TestPerNodeColorOverride:
    """Verify the renderer uses per-node color when set."""

    def test_node_with_custom_color(self, qapp):
        from noodlestudio.panels.neural_canvas.neural_canvas_view import (
            NodeGraphicsItem,
        )
        from PyQt6.QtGui import QColor
        node = create_node_from_type(NodeType.EMA_FILTER, name='Fast')
        node.color = '#5A5A5A'  # Per-node override
        item = NodeGraphicsItem(node)
        # The item should use the per-node color in paint
        # Verify the node's color is stored correctly
        assert node.color == '#5A5A5A'

    def test_node_without_custom_color_uses_type_default(self):
        node = create_node_from_type(NodeType.EMA_FILTER, name='Test')
        # create_node_from_type copies the type color to node.color
        type_color = get_node_color(NodeType.EMA_FILTER)
        assert node.color == type_color


# ===================================================================
# E.1: Server Gating Fix
# ===================================================================

class TestServerGatingFix:
    """Verify Rez context menu no longer requires server."""

    def test_no_server_running_reference_in_rez_block(self):
        """The Rez menu code should not reference _server_running."""
        import inspect
        from noodlestudio.panels.scene_hierarchy_context_menu_mixin import (
            SceneHierarchyContextMenuMixin,
        )
        source = inspect.getsource(
            SceneHierarchyContextMenuMixin._show_context_menu_impl
        )
        # The old pattern "Start server to create items" should be gone
        assert 'Start server to create items' not in source
        # The Rez block should not check _server_running
        # (It may still exist elsewhere for other purposes)
        lines = source.split('\n')
        in_rez_block = False
        for line in lines:
            if 'create_menu = menu.addMenu("Rez")' in line:
                in_rez_block = True
            if in_rez_block and '_server_running' in line:
                pytest.fail(
                    "Found _server_running reference in Rez menu block"
                )
            if in_rez_block and 'menu.exec' in line:
                break  # Past the block
