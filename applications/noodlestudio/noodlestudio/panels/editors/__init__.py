# Unified editor infrastructure (Phase C)
# C.1: shared mixins, breadcrumb, depth protocol
# C.3: AssemblyEditorView, graphics items
# C.4: execution visualization, ensemble mode
# C.5: NeuralCanvasDepthView registered for depth navigation
# E.2c: CharmNetworkEMA now uses NeuralCanvasDepthView (visible hearts)

from .unified_editor_panel import UnifiedEditorPanel
from .neural_canvas_depth_view import NeuralCanvasDepthView

UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", NeuralCanvasDepthView)
UnifiedEditorPanel.register_depth_view("CharmNetworkEMA", NeuralCanvasDepthView)
