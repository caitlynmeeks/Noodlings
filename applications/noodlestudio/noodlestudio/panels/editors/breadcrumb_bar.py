"""Clickable breadcrumb path bar for depth-stack navigation.

Displays the current depth path as clickable segments separated by '>'.
Clicking a segment emits segmentClicked with the depth index, allowing
the unified editor to pop views back to that level.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QSizePolicy,
)


class BreadcrumbBar(QWidget):
    """Horizontal breadcrumb path with clickable segments."""

    segmentClicked = pyqtSignal(int)  # depth index

    # Maximum segments before truncation
    MAX_VISIBLE_SEGMENTS = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments = []  # list of (label_str, depth_index)
        self._buttons = []
        self._separators = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 2, 8, 2)
        self._layout.setSpacing(2)
        self._layout.addStretch()

        self.setFixedHeight(28)
        self.setStyleSheet("""
            BreadcrumbBar {
                background-color: #2A2A2A;
                border-bottom: 1px solid #3A3A3A;
            }
        """)

    def set_path(self, segments: list):
        """Set the breadcrumb path.

        Args:
            segments: List of label strings, one per depth level.
                      Index 0 is the root (leftmost).
        """
        self._clear_widgets()
        self._segments = [(label, i) for i, label in enumerate(segments)]

        visible = self._segments
        truncated = False

        if len(visible) > self.MAX_VISIBLE_SEGMENTS:
            # Keep first and last (MAX-1) segments, replace middle with "..."
            visible = [self._segments[0]] + self._segments[-(self.MAX_VISIBLE_SEGMENTS - 1):]
            truncated = True

        for i, (label, depth_index) in enumerate(visible):
            if i > 0:
                sep = QLabel(">")
                sep.setStyleSheet("color: #666666; font-size: 11px;")
                sep.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                self._layout.insertWidget(self._layout.count() - 1, sep)
                self._separators.append(sep)

            if truncated and i == 1:
                # Insert ellipsis before second visible segment
                ellipsis = QLabel("...")
                ellipsis.setStyleSheet("color: #666666; font-size: 11px;")
                ellipsis.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                self._layout.insertWidget(self._layout.count() - 1, ellipsis)
                self._separators.append(ellipsis)

                sep2 = QLabel(">")
                sep2.setStyleSheet("color: #666666; font-size: 11px;")
                sep2.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                self._layout.insertWidget(self._layout.count() - 1, sep2)
                self._separators.append(sep2)

            is_last = (i == len(visible) - 1)
            btn = self._make_segment_button(label, depth_index, is_current=is_last)
            self._layout.insertWidget(self._layout.count() - 1, btn)
            self._buttons.append(btn)

    def clear(self):
        """Remove all breadcrumb segments."""
        self._clear_widgets()
        self._segments = []

    def segment_count(self) -> int:
        """Number of segments in the full (non-truncated) path."""
        return len(self._segments)

    def _make_segment_button(self, label: str, depth_index: int,
                             is_current: bool) -> QPushButton:
        btn = QPushButton(label)
        btn.setFlat(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        if is_current:
            # Current (deepest) segment: brighter, not clickable
            btn.setStyleSheet("""
                QPushButton {
                    color: #CCCCCC;
                    font-size: 12px;
                    font-weight: bold;
                    border: none;
                    padding: 2px 6px;
                    background: transparent;
                }
            """)
            btn.setCursor(Qt.CursorShape.ArrowCursor)
            btn.setEnabled(False)
        else:
            # Ancestor segment: clickable, dimmer
            btn.setStyleSheet("""
                QPushButton {
                    color: #888888;
                    font-size: 12px;
                    border: none;
                    padding: 2px 6px;
                    background: transparent;
                }
                QPushButton:hover {
                    color: #CCCCCC;
                    text-decoration: underline;
                }
            """)
            btn.clicked.connect(lambda checked, idx=depth_index: self.segmentClicked.emit(idx))

        return btn

    def _clear_widgets(self):
        for btn in self._buttons:
            self._layout.removeWidget(btn)
            btn.deleteLater()
        for sep in self._separators:
            self._layout.removeWidget(sep)
            sep.deleteLater()
        self._buttons.clear()
        self._separators.clear()
