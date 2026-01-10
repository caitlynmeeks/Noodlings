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
#   Facet Track Widget - Timeline visualization of facet executions
#
#   Renders facet execution blocks as colored bars on the tim...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.widgets.facet_track
# PURPOSE:  facet track facet implementation
# LAYER:    Studio / Widgets
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetBlockItem, FacetTrack, CycleTrack, FacetSwimlanesWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QToolTip
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath

from typing import List, Optional, Dict, Any
import sys
sys.path.insert(0, '../..')

# Import from our own module
try:
    from noodlestudio.core.timeline_recorder import FacetRecord, CycleRecord
except ImportError:
    # Fallback for direct execution
    from dataclasses import dataclass, field

    @dataclass
    class FacetRecord:
        facet_id: str = ""
        facet_name: str = ""
        facet_type: str = ""
        start_time: float = 0.0
        end_time: float = 0.0
        duration_ms: float = 0.0
        token_count: int = 0
        salience: float = 0.5
        inputs: Dict[str, Any] = field(default_factory=dict)
        outputs: Dict[str, Any] = field(default_factory=dict)
        prompt: Optional[str] = None
        execution_id: str = ""
        cycle: int = 0


# =============================================================================
# Color Constants (monochromatic as per Caitlyn's guidelines)
# =============================================================================

FACET_COLORS = {
    # LLM Facets - Purple tones
    'LLMFacet': QColor(156, 39, 176),           # #9C27B0 - Deep purple
    'CharacterLayerFacet': QColor(171, 71, 188), # #AB47BC - Light purple

    # Neural Facets - Green tones
    'CharmNetworkFacet': QColor(76, 175, 80),    # #4CAF50 - Green
    'SubconsciousFacet': QColor(102, 187, 106),  # #66BB6A - Light green

    # Script Facets - Blue tones
    'ScriptedFacet': QColor(33, 150, 243),       # #2196F3 - Blue
    'InsightEmergenceFacet': QColor(66, 165, 245), # #42A5F5 - Light blue

    # Intelligence Facets - Teal tones
    'ContextIntelligenceFacet': QColor(0, 150, 136),  # #009688 - Teal

    # Flow Control - Orange tones
    'TickerGateFacet': QColor(255, 152, 0),      # #FF9800 - Orange
    'ConditionalBranchFacet': QColor(255, 167, 38),  # #FFA726 - Light orange
    'RateLimiterFacet': QColor(255, 183, 77),    # #FFB74D - Amber
    'CacheFacet': QColor(255, 193, 7),           # #FFC107 - Yellow

    # Convergence - Red tones
    'ConvergenceFacet': QColor(244, 67, 54),     # #F44336 - Red
    'SyncGate': QColor(229, 115, 115),           # #E57373 - Light red

    # Special Nodes - Gray tones
    'SpecialNode': QColor(96, 125, 139),         # #607D8B - Blue-gray
    'INCOMING': QColor(96, 125, 139),
    'OUTGOING': QColor(96, 125, 139),

    # Default
    'default': QColor(66, 66, 66),               # #424242 - Dark gray
}

# Track background
TRACK_BG = QColor(21, 21, 21)        # Very dark gray
TRACK_BORDER = QColor(48, 48, 48)    # Slightly lighter
BLOCK_BORDER = QColor(255, 255, 255, 60)  # Subtle white outline


# =============================================================================
# Facet Block Item
# =============================================================================

class FacetBlockItem(QGraphicsItem):
    """
    Single facet execution block on the timeline.

    Clickable, hoverable, shows tooltip with facet details.
    """

    def __init__(
        self,
        facet: FacetRecord,
        x: float,
        width: float,
        height: float,
        parent=None
    ):
        super().__init__(parent)
        self.facet = facet
        self.x_pos = x
        self.block_width = max(width, 4)  # Minimum 4px width for visibility
        self.block_height = height
        self.hovered = False

        # Get color based on facet type
        self.color = FACET_COLORS.get(
            facet.facet_type,
            FACET_COLORS.get('default')
        )

        # Enable hover and selection
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        # Tooltip
        self.setToolTip(self._build_tooltip())

    def _build_tooltip(self) -> str:
        """Build informative tooltip."""
        lines = [
            f"<b>{self.facet.facet_name}</b>",
            f"Type: {self.facet.facet_type}",
            f"Duration: {self.facet.duration_ms:.1f}ms",
        ]
        if self.facet.token_count > 0:
            lines.append(f"Tokens: {self.facet.token_count}")
        return "<br>".join(lines)

    def boundingRect(self) -> QRectF:
        return QRectF(self.x_pos, 0, self.block_width, self.block_height)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.boundingRect()

        # Background color (brighter if hovered/selected)
        color = self.color.lighter(130) if (self.hovered or self.isSelected()) else self.color

        # Draw rounded rect
        painter.setPen(QPen(BLOCK_BORDER, 1))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(rect, 3, 3)

        # Draw facet name if block is wide enough
        if self.block_width > 40:
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            font = QFont("Monaco", 8)
            painter.setFont(font)

            # Truncate name if needed
            name = self.facet.facet_name
            if len(name) > 10 and self.block_width < 100:
                name = name[:8] + ".."

            text_rect = rect.adjusted(4, 2, -4, -2)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Emit click signal through scene
            scene = self.scene()
            if hasattr(scene, 'facetClicked'):
                scene.facetClicked.emit(self.facet)


# =============================================================================
# Facet Track
# =============================================================================

class FacetTrack(QGraphicsItem):
    """
    Single swimlane showing all executions of one facet type.

    Like a track in a video editor - each facet execution appears
    as a clip/block on this track.
    """

    def __init__(
        self,
        facet_name: str,
        facet_type: str,
        facets: List[FacetRecord],
        max_time: float,
        width: float,
        parent=None
    ):
        super().__init__(parent)
        self.facet_name = facet_name
        self.facet_type = facet_type
        self.facets = facets
        self.max_time = max(max_time, 0.1)
        self.width = width
        self.height = 28  # Track height (compact for swimlanes)

        # Get track color
        self.color = FACET_COLORS.get(facet_type, FACET_COLORS['default'])

        # Create block items as children
        self._create_blocks()

    def _create_blocks(self):
        """Create FacetBlockItem for each facet execution."""
        for facet in self.facets:
            # Calculate position and width
            x = (facet.start_time / self.max_time) * self.width
            end_x = (facet.end_time / self.max_time) * self.width if facet.end_time > 0 else x + 10
            block_width = end_x - x

            block = FacetBlockItem(
                facet=facet,
                x=x,
                width=block_width,
                height=self.height - 4,
                parent=self
            )
            block.setPos(0, 2)  # Slight padding from track edge

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw track background
        painter.fillRect(self.boundingRect(), TRACK_BG)

        # Draw track border
        painter.setPen(QPen(TRACK_BORDER, 1))
        painter.drawRect(self.boundingRect())

        # Draw facet name label on left
        painter.setPen(QPen(self.color.lighter(150), 1))
        font = QFont("Monaco", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(5, self.height - 8, self.facet_name)


# =============================================================================
# Cycle Track
# =============================================================================

class CycleTrack(QGraphicsItem):
    """
    Track showing cognition cycle boundaries.

    Cycles appear as spanning bars across the top of the timeline,
    color-coded by type (reactive=blue, autonomous=gray).
    """

    def __init__(
        self,
        cycles: List[CycleRecord],
        max_time: float,
        width: float,
        parent=None
    ):
        super().__init__(parent)
        self.cycles = cycles
        self.max_time = max(max_time, 0.1)
        self.width = width
        self.height = 20  # Thinner than facet tracks

        # Cycle type colors
        self.cycle_colors = {
            'reactive': QColor(33, 150, 243, 150),   # Blue with alpha
            'autonomous': QColor(158, 158, 158, 150), # Gray with alpha
        }

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw track background
        painter.fillRect(self.boundingRect(), QColor(15, 15, 15))

        # Draw track label
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        font = QFont("Monaco", 8)
        painter.setFont(font)
        painter.drawText(5, self.height - 5, "CYCLES")

        # Draw each cycle as a colored span
        for cycle in self.cycles:
            x = (cycle.start_time / self.max_time) * self.width
            end_x = (cycle.end_time / self.max_time) * self.width if cycle.end_time > 0 else x + 20
            w = end_x - x

            color = self.cycle_colors.get(cycle.cycle_type, self.cycle_colors['reactive'])

            # Draw cycle span
            painter.setPen(QPen(color.darker(120), 1))
            painter.setBrush(QBrush(color))
            rect = QRectF(x, 2, w, self.height - 4)
            painter.drawRoundedRect(rect, 2, 2)

            # Draw cycle number
            if w > 20:
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(cycle.cycle_number))


# =============================================================================
# Multi-Facet Timeline (combines multiple facet tracks)
# =============================================================================

class FacetSwimlanesWidget(QGraphicsItem):
    """
    Container widget holding all facet swimlane tracks.

    Layout:
    ┌─────────────────────────────────────────────────────┐
    │  CYCLES   ████████░░░░░░░████████░░░░░░░████████   │
    ├─────────────────────────────────────────────────────┤
    │  INCOMING        █░░░░░░░░░█░░░░░░░░█░░░░░░░░░░░░  │
    │  CharmNetwork    ░█░░░░░░░░░█░░░░░░░░█░░░░░░░░░░░  │
    │  ContextIntel    ░░████░░░░░░░░░░░░░░░████░░░░░░░  │
    │  CharacterLayer  ░░░░░░████░░░░░░░░░░░░░░░████░░░  │
    │  OUTGOING        ░░░░░░░░░█░░░░░░░░░░░░░░░░░░░█░░  │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        cycles: List[CycleRecord],
        max_time: float,
        width: float,
        parent=None
    ):
        super().__init__(parent)
        self.cycles = cycles
        self.max_time = max_time
        self.width = width

        # Group facets by name
        self.facets_by_name: Dict[str, List[FacetRecord]] = {}
        self.facet_types: Dict[str, str] = {}

        for cycle in cycles:
            for facet in cycle.facets:
                if facet.facet_name not in self.facets_by_name:
                    self.facets_by_name[facet.facet_name] = []
                    self.facet_types[facet.facet_name] = facet.facet_type
                self.facets_by_name[facet.facet_name].append(facet)

        # Build tracks
        self._build_tracks()

    def _build_tracks(self):
        """Create all track items."""
        y_offset = 0

        # Add cycle track first
        cycle_track = CycleTrack(self.cycles, self.max_time, self.width, parent=self)
        cycle_track.setPos(0, y_offset)
        y_offset += cycle_track.height + 2

        # Add facet tracks in execution order
        # Sort by first occurrence time
        sorted_facets = sorted(
            self.facets_by_name.items(),
            key=lambda x: min(f.start_time for f in x[1]) if x[1] else 0
        )

        for facet_name, facets in sorted_facets:
            facet_type = self.facet_types.get(facet_name, 'default')
            track = FacetTrack(
                facet_name=facet_name,
                facet_type=facet_type,
                facets=facets,
                max_time=self.max_time,
                width=self.width,
                parent=self
            )
            track.setPos(0, y_offset)
            y_offset += track.height + 1

        self.total_height = y_offset

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, getattr(self, 'total_height', 200))

    def paint(self, painter: QPainter, option, widget=None):
        # Container doesn't draw anything itself
        pass


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    """Visual test of facet track rendering."""
    from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
    import sys

    app = QApplication(sys.argv)

    # Create test data
    test_facets = [
        FacetRecord(
            facet_id="1", facet_name="CharmNetwork", facet_type="CharmNetworkFacet",
            start_time=0.1, end_time=0.3, duration_ms=200
        ),
        FacetRecord(
            facet_id="2", facet_name="ContextIntel", facet_type="ContextIntelligenceFacet",
            start_time=0.35, end_time=0.8, duration_ms=450
        ),
        FacetRecord(
            facet_id="3", facet_name="RoastEngine", facet_type="LLMFacet",
            start_time=0.85, end_time=1.5, duration_ms=650, token_count=340
        ),
    ]

    # Create scene and view
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setWindowTitle("Facet Track Test")
    view.setStyleSheet("background-color: rgb(10, 10, 10);")
    view.resize(1000, 200)

    # Add test track
    track = FacetTrack(
        facet_name="TestFacets",
        facet_type="LLMFacet",
        facets=test_facets,
        max_time=2.0,
        width=900
    )
    scene.addItem(track)

    view.show()
    sys.exit(app.exec())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
