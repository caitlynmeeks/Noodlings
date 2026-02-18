# ------------------------------------------------------------------
#
#   Charm Network Depth View
#
#   Three-layer EMA visualization for the depth-stack navigation.
#   Shows fast/medium/slow timescale PAD bars with current state.
#   Pushed onto the stack when user double-clicks a CharmNetworkEMA
#   facet node in the assembly editor.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.panels.editors.charm_network_depth_view
# PURPOSE:  Depth view for CharmNetworkEMA facet
# LAYER:    Studio / Panels / Editors
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import re

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout,
    QFrame,
)


def _parse_baseline(prompt: str) -> dict:
    """Parse baseline PAD from prompt string like 'valence:0.7,arousal:0.5,dominance:0.4'."""
    baseline = {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}
    if not prompt:
        return baseline
    for key in ('valence', 'arousal', 'dominance'):
        m = re.search(rf'{key}:\s*([-\d.]+)', prompt)
        if m:
            baseline[key] = float(m.group(1))
    return baseline


# ---------------------------------------------------------------------------
# PAD bar widget -- horizontal bar showing a single PAD dimension value
# ---------------------------------------------------------------------------

class _PADBar(QWidget):
    """Horizontal bar for a single PAD dimension.

    Handles two range types:
    - valence: -1.0 to +1.0 (center at zero, fill extends left or right)
    - arousal/dominance: 0.0 to 1.0 (fill from left edge)
    """

    def __init__(self, dimension: str, value: float = 0.0,
                 bar_color: QColor = QColor(180, 180, 180),
                 parent=None):
        super().__init__(parent)
        self._dimension = dimension  # 'valence', 'arousal', or 'dominance'
        self._value = value
        self._bar_color = bar_color
        self._bipolar = (dimension == 'valence')  # -1..1 vs 0..1
        self.setFixedHeight(16)
        self.setMinimumWidth(120)

    def set_value(self, value: float):
        self._value = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background track
        painter.fillRect(0, 0, w, h, QColor(40, 40, 40))
        painter.setPen(QColor(60, 60, 60))
        painter.drawRect(0, 0, w - 1, h - 1)

        if self._bipolar:
            # Valence: -1..1, center line at midpoint
            center_x = w // 2

            # Center line marker
            painter.setPen(QColor(80, 80, 80))
            painter.drawLine(center_x, 0, center_x, h)

            # Fill from center toward value
            # normalized: -1 -> 0, 0 -> center, +1 -> w
            norm = (self._value + 1.0) / 2.0
            fill_x = int(norm * w)
            left = min(center_x, fill_x)
            right = max(center_x, fill_x)
            painter.fillRect(left, 2, right - left, h - 4, self._bar_color)
        else:
            # Arousal/Dominance: 0..1, fill from left
            fill_w = int(max(0.0, min(1.0, self._value)) * w)
            painter.fillRect(0, 2, fill_w, h - 4, self._bar_color)

        # Value text
        painter.setPen(QColor(210, 210, 210))
        painter.setFont(QFont("Monaco", 9))
        text = f"{self._value:+.2f}" if self._bipolar else f"{self._value:.2f}"
        painter.drawText(4, h - 3, text)

        painter.end()


# ---------------------------------------------------------------------------
# Layer group -- three PAD bars for one EMA timescale
# ---------------------------------------------------------------------------

class _LayerGroup(QWidget):
    """One EMA layer: group box with three PAD bars."""

    def __init__(self, layer_name: str, alpha: str,
                 bar_color: QColor, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        group = QGroupBox(f"{layer_name} (alpha {alpha})")
        group.setStyleSheet(
            "QGroupBox { color: #AAA; border: 1px solid #444; "
            "border-radius: 3px; margin-top: 8px; padding-top: 12px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
            "padding: 0 4px; }"
        )
        group_layout = QFormLayout(group)
        group_layout.setContentsMargins(8, 4, 8, 4)

        self.v_bar = _PADBar('valence', bar_color=bar_color)
        self.a_bar = _PADBar('arousal', bar_color=bar_color)
        self.d_bar = _PADBar('dominance', bar_color=bar_color)

        label_style = "color: #888; font-size: 9pt;"
        for label_text, bar in [("V:", self.v_bar), ("A:", self.a_bar), ("D:", self.d_bar)]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet(label_style)
            group_layout.addRow(lbl, bar)

        layout.addWidget(group)

    def set_pad(self, pad: dict):
        self.v_bar.set_value(pad.get('valence', 0.0))
        self.a_bar.set_value(pad.get('arousal', 0.5))
        self.d_bar.set_value(pad.get('dominance', 0.5))


# ---------------------------------------------------------------------------
# Main depth view
# ---------------------------------------------------------------------------

class CharmNetworkDepthView(QWidget):
    """Depth-stack view for CharmNetworkEMA facets.

    Shows three EMA layers (fast, medium, slow) with PAD bars.
    Implements DepthViewProtocol for the UnifiedEditorPanel stack.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._baseline = {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}
        self._context = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Title
        self._title = QLabel("Charm Network")
        self._title.setStyleSheet(
            "color: #D2D2D2; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(self._title)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #555; max-height: 1px;")
        layout.addWidget(sep)

        # Input section (from Mood Reader)
        input_label = QLabel("Input (from Mood Reader)")
        input_label.setStyleSheet("color: #AAA; font-size: 10pt; margin-top: 4px;")
        layout.addWidget(input_label)

        self._input_values = QLabel("V: +0.00   A: 0.50   D: 0.50")
        self._input_values.setStyleSheet(
            "color: #888; font-size: 9pt; "
            "font-family: 'Monaco', 'Consolas', monospace; margin-left: 12px;"
        )
        layout.addWidget(self._input_values)

        # Three EMA layers with distinct gray tones
        self._fast_layer = _LayerGroup("Fast", "0.7", QColor(200, 200, 200))
        self._medium_layer = _LayerGroup("Medium", "0.15", QColor(150, 150, 150))
        self._slow_layer = _LayerGroup("Slow", "0.03", QColor(100, 100, 100))

        layout.addWidget(self._fast_layer)
        layout.addWidget(self._medium_layer)
        layout.addWidget(self._slow_layer)

        # Output section
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #555; max-height: 1px;")
        layout.addWidget(sep2)

        output_label = QLabel("Output (blended 0.5 / 0.3 / 0.2)")
        output_label.setStyleSheet("color: #AAA; font-size: 10pt; margin-top: 4px;")
        layout.addWidget(output_label)

        self._output_values = QLabel("V: +0.00   A: 0.50   D: 0.50")
        self._output_values.setStyleSheet(
            "color: #D2D2D2; font-size: 10pt; font-weight: bold; "
            "font-family: 'Monaco', 'Consolas', monospace; margin-left: 12px;"
        )
        layout.addWidget(self._output_values)

        layout.addStretch()

    def _format_pad(self, pad: dict) -> str:
        v = pad.get('valence', 0.0)
        a = pad.get('arousal', 0.5)
        d = pad.get('dominance', 0.5)
        return f"V: {v:+.2f}   A: {a:.2f}   D: {d:.2f}"

    def _show_baseline(self):
        """Initialize all layers and labels from baseline values."""
        b = self._baseline
        self._input_values.setText(self._format_pad(b))
        self._fast_layer.set_pad(b)
        self._medium_layer.set_pad(b)
        self._slow_layer.set_pad(b)
        self._output_values.setText(self._format_pad(b))

    # ==================== DepthViewProtocol ====================

    def load_data(self, data_path: str, context: dict) -> None:
        """Load baseline from the data_path (prompt string with PAD values).

        Args:
            data_path: The facet's prompt string (e.g. 'valence:0.7,...')
            context: Additional context (noodling_name, etc.)
        """
        self._context = context or {}
        self._baseline = _parse_baseline(data_path)

        noodling_name = self._context.get('noodling_name', '')
        if noodling_name:
            self._title.setText(f"Charm Network -- {noodling_name}")

        self._show_baseline()

    def save_data(self) -> None:
        """No-op -- this is a read-only visualization."""
        pass

    def get_breadcrumb_label(self) -> str:
        return "Charm Network"

    def has_unsaved_changes(self) -> bool:
        return False
