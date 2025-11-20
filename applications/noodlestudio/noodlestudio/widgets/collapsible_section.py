"""
CollapsibleSection - Custom collapsible widget without QGroupBox signal issues

Provides expand/collapse functionality using direct mouse event handling
instead of Qt's checkable QGroupBox mechanism, which has double-trigger bugs.

Author: Noodlings Project
Date: November 2025
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QMouseEvent


class ClickableHeader(QFrame):
    """Custom QFrame that properly handles mouse clicks without double-triggering."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent):
        """Override mousePressEvent - single entry point for clicks."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Override to consume release events."""
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Override to consume double-click events."""
        event.accept()


class CollapsibleSection(QWidget):
    """
    Custom collapsible section widget.

    Provides clean expand/collapse without QGroupBox.toggled signal issues.
    Uses direct mouse event handling for deterministic behavior.

    Features:
    - Click header to toggle
    - Arrow indicator (▼ expanded, ▶ collapsed)
    - Smooth show/hide without double-trigger
    - Consistent with Scene Hierarchy expansion behavior
    """

    # Signal emitted when expanded/collapsed (for external observers, if needed)
    toggled = pyqtSignal(bool)

    def __init__(self, title: str, parent=None):
        """
        Initialize collapsible section.

        Args:
            title: Section header text
            parent: Parent widget
        """
        super().__init__(parent)
        self.is_expanded = True
        self.title_text = title

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header widget (clickable) - use custom subclass for proper event handling
        self.header = ClickableHeader()
        self.header.setFrameShape(QFrame.Shape.StyledPanel)
        self.header.setFrameShadow(QFrame.Shadow.Raised)
        self.header.setMinimumHeight(28)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)

        # Connect clicked signal to toggle
        self.header.clicked.connect(self.toggle)

        # Style header with subtle background
        header_palette = self.header.palette()
        header_palette.setColor(QPalette.ColorRole.Window, QColor(60, 60, 60))
        self.header.setAutoFillBackground(True)
        self.header.setPalette(header_palette)

        # Header layout
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        # Arrow label
        self.arrow_label = QLabel("▼")
        arrow_font = QFont("Courier", 10)
        self.arrow_label.setFont(arrow_font)
        self.arrow_label.setFixedWidth(16)
        self.arrow_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)  # Prevent event propagation

        # Title label
        self.title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        self.title_label.setFont(title_font)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)  # Prevent event propagation

        header_layout.addWidget(self.arrow_label)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # Content container (no layout set yet - will be set later)
        self.content = QWidget()
        self.content_layout = None  # Will be set via set_content_layout() or add_widget()

        # Add to main layout
        main_layout.addWidget(self.header)
        main_layout.addWidget(self.content)

    def toggle(self):
        """Toggle expanded/collapsed state."""
        # Debug: Track toggle calls to identify double-trigger source
        import traceback
        print(f"[CollapsibleSection] toggle() called on '{self.title_text}', current state: {self.is_expanded}")
        print(f"[CollapsibleSection] Call stack:\n{''.join(traceback.format_stack()[-3:-1])}")
        self.set_expanded(not self.is_expanded)

    def set_expanded(self, expanded: bool):
        """
        Set expansion state explicitly.

        Args:
            expanded: True to expand, False to collapse
        """
        # DIAGNOSTIC: Log EVERY call to set_expanded with full stack trace
        import traceback
        print(f"\n{'='*80}")
        print(f"[DIAGNOSTIC] set_expanded({expanded}) called on '{self.title_text}'")
        print(f"[DIAGNOSTIC] Current state: is_expanded={self.is_expanded}")
        print(f"[DIAGNOSTIC] Will {'SKIP (no change)' if self.is_expanded == expanded else 'PROCEED with state change'}")
        print(f"[DIAGNOSTIC] Full call stack:")
        print(''.join(traceback.format_stack()))
        print(f"{'='*80}\n")

        if self.is_expanded == expanded:
            return  # No change needed

        self.is_expanded = expanded

        if expanded:
            self.arrow_label.setText("▼")
            self.content.show()
        else:
            self.arrow_label.setText("▶")
            self.content.hide()

        # Emit signal for external observers (if needed)
        print(f"[DIAGNOSTIC] About to emit toggled signal: {expanded}")
        self.toggled.emit(expanded)
        print(f"[DIAGNOSTIC] toggled signal emitted successfully")

    def add_widget(self, widget):
        """
        Add widget to content area.

        Args:
            widget: Widget to add
        """
        # Lazily create default VBoxLayout if none exists
        if self.content_layout is None:
            default_layout = QVBoxLayout()
            default_layout.setContentsMargins(12, 8, 12, 8)
            default_layout.setSpacing(6)
            self.set_content_layout(default_layout)

        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """
        Add layout to content area.

        Args:
            layout: Layout to add
        """
        # Lazily create default VBoxLayout if none exists
        if self.content_layout is None:
            default_layout = QVBoxLayout()
            default_layout.setContentsMargins(12, 8, 12, 8)
            default_layout.setSpacing(6)
            self.set_content_layout(default_layout)

        self.content_layout.addLayout(layout)

    def set_content_layout(self, layout):
        """
        Replace content layout entirely.

        Args:
            layout: New layout for content area
        """
        # Only set if content doesn't have a layout yet
        if self.content.layout() is None:
            self.content.setLayout(layout)
            self.content_layout = layout
        else:
            # Layout already exists - cannot replace after widget has layout
            raise RuntimeError("Cannot replace layout after it's been set. Call set_content_layout() before add_widget().")

    def content_form_layout(self):
        """
        Return content layout for form-based sections.

        For API compatibility with QGroupBox patterns, provides access
        to the QFormLayout set via set_content_layout().
        """
        return self.content.layout()
