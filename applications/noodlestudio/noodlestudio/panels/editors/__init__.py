# Unified editor infrastructure (Phase C)
# C.1: shared mixins, breadcrumb, depth protocol
# C.3: AssemblyEditorView, graphics items
# C.4: execution visualization, ensemble mode
# C.5: NeuralCanvasDepthView registered for depth navigation
# D.1.5: CharmNetworkDepthView for EMA charm networks

from .unified_editor_panel import UnifiedEditorPanel
from .neural_canvas_depth_view import NeuralCanvasDepthView
from .charm_network_depth_view import CharmNetworkDepthView

UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", NeuralCanvasDepthView)
UnifiedEditorPanel.register_depth_view("CharmNetworkEMA", CharmNetworkDepthView)
