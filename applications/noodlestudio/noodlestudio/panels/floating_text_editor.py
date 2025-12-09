"""
Floating Text Editor - In-graph text editing panel

Floating semi-transparent panel for editing facet fields without leaving
the node graph view. Appears centered over the selected node.

Features:
- Large Monaco text area for comfortable editing
- Semi-transparent background
- Draggable
- Auto-save on Apply or when switching fields
- ESC to close

Author: Commander Spock + Cadet Caity
Date: November 28, 2025
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QGraphicsProxyWidget, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QSettings
from PyQt6.QtGui import QFont, QColor, QPalette, QMouseEvent
from typing import Optional, Callable


class DoubleClickHeader(QLabel):
    """Header label that detects double-clicks for maximize/restore and supports dragging."""

    doubleClicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drag_position = None

    def mousePressEvent(self, event: QMouseEvent):
        """Start dragging on left-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle window dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.window().move(event.globalPosition().toPoint() - self.drag_position)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Emit signal on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class FloatingTextEditor(QDialog):
    """
    Floating text editor for facet fields.

    Appears as draggable panel over node graph.
    """

    # Signal when text is applied
    textApplied = pyqtSignal(str, str)  # field_key, new_value

    def __init__(self, field_name: str, field_key: str, initial_value: str,
                 read_only: bool = False, parent=None):
        """
        Initialize floating editor.

        Args:
            field_name: Display name (e.g., "Processing Prompt")
            field_key: Field identifier (e.g., "prompt")
            initial_value: Current field value
            read_only: If True, field cannot be edited
            parent: Parent widget
        """
        super().__init__(parent)
        self.field_name = field_name
        self.field_key = field_key
        self.initial_value = initial_value
        self.read_only = read_only

        # Load saved font size preference
        settings = QSettings("NoodleStudio", "FloatingTextEditor")
        self.font_size = settings.value("font_size", 12, type=int)

        # Track maximized state
        self.is_maximized = False
        self.normal_geometry = None

        self.init_ui()

    def init_ui(self):
        """Initialize user interface."""
        # Frameless window so we can use our custom header for double-click
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        # Set initial size (resizable, not fixed!)
        self.resize(600, 500)
        self.setMinimumSize(400, 300)  # Prevent too small

        # For dragging the window
        self.drag_position = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar widget (contains title + close button)
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border-bottom: 1px solid #444;
            }
        """)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        # Draggable title label (double-click to maximize)
        self.header = DoubleClickHeader(f"  ✎  {self.field_name}")
        self.header.setStyleSheet("""
            QLabel {
                color: #CCCCCC;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        self.header.doubleClicked.connect(self.toggle_maximize)
        header_layout.addWidget(self.header, 1)  # Stretch to fill space

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(40, 40)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #999999;
                border: none;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C42B1C;
                color: #FFFFFF;
            }
        """)
        header_layout.addWidget(close_btn)

        layout.addWidget(header_widget)

        # Text edit area
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.initial_value)
        self.text_edit.setReadOnly(self.read_only)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2A2A2A;
                color: #CCCCCC;
                border: none;
                font-family: Monaco, Consolas, monospace;
                padding: 10px;
                selection-background-color: #4A4A4A;
            }
        """)
        # Apply saved font size
        self.text_edit.setFont(QFont("Monaco", self.font_size))

        # Install event filter to catch Cmd+/- before text edit processes them
        self.text_edit.installEventFilter(self)

        layout.addWidget(self.text_edit)

        # Button bar with proper background
        button_bar = QWidget()
        button_bar.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border-top: 1px solid #444;
            }
        """)
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(10, 10, 10, 10)

        # Font size controls (left side) - no label, just buttons
        decrease_btn = QPushButton("A-")
        decrease_btn.setMaximumWidth(40)
        decrease_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 8px 4px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
        """)
        decrease_btn.clicked.connect(self.decrease_font_size)
        button_layout.addWidget(decrease_btn)

        self.font_size_label = QLabel(f"{self.font_size}pt")
        self.font_size_label.setStyleSheet("color: #CCCCCC; font-size: 10pt; min-width: 35px;")
        button_layout.addWidget(self.font_size_label)

        increase_btn = QPushButton("A+")
        increase_btn.setMaximumWidth(40)
        increase_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 8px 4px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
        """)
        increase_btn.clicked.connect(self.increase_font_size)
        button_layout.addWidget(increase_btn)

        button_layout.addSpacing(20)

        # Copy button (always available) - same height as other buttons
        copy_btn = QPushButton("Copy")
        copy_btn.setMaximumWidth(60)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
        """)
        copy_btn.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_btn)

        button_layout.addSpacing(20)

        # Show read-only indicator or edit buttons
        if self.read_only:
            readonly_label = QLabel("Read-only (selectable for copying)")
            readonly_label.setStyleSheet("color: #888888; font-size: 11px;")
            button_layout.addWidget(readonly_label)
        else:
            # Cancel button first (left)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3A3A3A;
                    color: #CCCCCC;
                    border: 1px solid #555;
                    padding: 8px 20px;
                }
                QPushButton:hover {
                    background-color: #4A4A4A;
                }
            """)
            button_layout.addWidget(cancel_btn)

        button_layout.addStretch()

        # Apply button (always rightmost) - monochrome gray, not blue
        apply_btn = QPushButton("Close" if self.read_only else "Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.apply_and_close)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 8px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
        """)
        button_layout.addWidget(apply_btn)

        layout.addWidget(button_bar)

        # Set dialog background with border (frameless window needs visible border)
        self.setStyleSheet("""
            QDialog {
                background-color: #2D2D2D;
                border: 1px solid #555555;
            }
        """)

        # Setup keyboard shortcuts for font scaling
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for font scaling."""
        from PyQt6.QtGui import QShortcut, QKeySequence
        import sys

        # Platform-aware shortcuts
        cmd_or_ctrl = "Cmd" if sys.platform == "darwin" else "Ctrl"

        # Cmd/Ctrl + Plus/Equal - Increase font size (both = and + work)
        zoom_in1 = QShortcut(QKeySequence(f"{cmd_or_ctrl}++"), self)
        zoom_in1.activated.connect(self.increase_font_size)
        zoom_in2 = QShortcut(QKeySequence(f"{cmd_or_ctrl}+="), self)
        zoom_in2.activated.connect(self.increase_font_size)

        # Cmd/Ctrl + Minus - Decrease font size
        zoom_out = QShortcut(QKeySequence(f"{cmd_or_ctrl}+-"), self)
        zoom_out.activated.connect(self.decrease_font_size)

        # Cmd/Ctrl + 0 - Reset to default
        reset_zoom = QShortcut(QKeySequence(f"{cmd_or_ctrl}+0"), self)
        reset_zoom.activated.connect(self.reset_font_size)

    def increase_font_size(self):
        """Increase editor font size (Cmd/Ctrl + or A+ button)."""
        # Use stored font_size (source of truth), not reading back from widget
        # (pointSize() can return -1 if font uses pixel size instead)
        self.font_size = min(self.font_size + 4, 48)  # Max 48pt, +4pt per click
        self.update_font()

    def decrease_font_size(self):
        """Decrease editor font size (Cmd/Ctrl - or A- button)."""
        # Use stored font_size (source of truth), not reading back from widget
        # (pointSize() can return -1 if font uses pixel size instead)
        self.font_size = max(self.font_size - 4, 8)  # Min 8pt, -4pt per click
        self.update_font()

    def reset_font_size(self):
        """Reset font to default size (Cmd/Ctrl 0)."""
        self.font_size = 12
        self.update_font()

    def update_font(self):
        """Update text editor font size and save preference."""
        # Update both QFont AND stylesheet (like console does)
        self.text_edit.setFont(QFont("Monaco", self.font_size))
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A2A;
                color: #CCCCCC;
                border: none;
                font-family: Monaco, Consolas, monospace;
                font-size: {self.font_size}pt;
                padding: 10px;
                selection-background-color: #4A4A4A;
            }}
        """)
        # Update label
        if hasattr(self, 'font_size_label'):
            self.font_size_label.setText(f"{self.font_size}pt")
        # Save font size preference
        settings = QSettings("NoodleStudio", "FloatingTextEditor")
        settings.setValue("font_size", self.font_size)

    def copy_to_clipboard(self):
        """Copy current text content to clipboard."""
        from PyQt6.QtWidgets import QApplication
        text = self.text_edit.toPlainText()
        QApplication.clipboard().setText(text)

    def toggle_maximize(self):
        """Toggle between maximized and normal window size (double-click header)."""
        if self.is_maximized:
            # Restore to normal size
            if self.normal_geometry:
                self.setGeometry(self.normal_geometry)
            self.showNormal()
            self.is_maximized = False
        else:
            # Save current geometry and maximize
            self.normal_geometry = self.geometry()
            self.showMaximized()
            self.is_maximized = True

    def has_unsaved_changes(self) -> bool:
        """Check if text has been modified."""
        return self.text_edit.toPlainText() != self.initial_value

    def apply_and_close(self):
        """Apply changes and close dialog."""
        if not self.read_only:
            new_value = self.text_edit.toPlainText()
            self.textApplied.emit(self.field_key, new_value)
        self.accept()

    def closeEvent(self, event):
        """Handle dialog close - check for unsaved changes."""
        if not self.read_only and self.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Apply them?",
                QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Apply:
                new_value = self.text_edit.toPlainText()
                self.textApplied.emit(self.field_key, new_value)
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def eventFilter(self, obj, event):
        """
        Event filter to catch Cmd+/- before QTextEdit processes them.

        This is necessary because QTextEdit has its own key handling that
        might consume these events before our keyPressEvent sees them.
        """
        from PyQt6.QtCore import QEvent, Qt

        if event.type() == QEvent.Type.KeyPress:
            # Check for Cmd/Ctrl modifier
            is_cmd_ctrl = event.modifiers() & Qt.KeyboardModifier.ControlModifier or \
                          event.modifiers() & Qt.KeyboardModifier.MetaModifier

            if is_cmd_ctrl and event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                # Cmd/Ctrl + Plus or Equal
                self.increase_font_size()
                return True  # Event handled, don't propagate
            elif is_cmd_ctrl and event.key() == Qt.Key.Key_Minus:
                # Cmd/Ctrl + Minus
                self.decrease_font_size()
                return True  # Event handled
            elif is_cmd_ctrl and event.key() == Qt.Key.Key_0:
                # Cmd/Ctrl + 0
                self.reset_font_size()
                return True  # Event handled

        # Let other events pass through
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle key events at dialog level."""
        from PyQt6.QtCore import Qt

        if event.key() == Qt.Key.Key_Escape:
            # ESC triggers closeEvent which checks for unsaved changes
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    """Test floating editor."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Test editable field
    editor = FloatingTextEditor(
        "Processing Prompt",
        "prompt",
        "You are a fire imp made of crimson flame.\n\nAnalyze the context...",
        read_only=False
    )

    def on_applied(key, value):
        print(f"Field '{key}' updated:")
        print(value)

    editor.textApplied.connect(on_applied)
    editor.show()

    sys.exit(app.exec())
