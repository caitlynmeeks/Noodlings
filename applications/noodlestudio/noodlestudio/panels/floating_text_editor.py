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
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QFont, QColor, QPalette
from typing import Optional, Callable


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
        self.font_size = 12  # Base font size

        self.init_ui()

    def init_ui(self):
        """Initialize user interface."""
        # Standard window with title bar
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle(f"Edit: {self.field_name}")

        # Set size
        self.setFixedSize(600, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar (custom, matches theme)
        header = QLabel(f"  ✎  {self.field_name}")
        header.setStyleSheet("""
            QLabel {
                color: #CCCCCC;
                font-size: 13px;
                font-weight: bold;
                background-color: #2D2D2D;
                padding: 10px;
                border-bottom: 1px solid #444;
            }
        """)
        layout.addWidget(header)

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
                font-size: 12px;
                padding: 10px;
                selection-background-color: #4A4A4A;
            }
        """)
        self.text_edit.setFont(QFont("Monaco", 12))
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

        # Apply button (always rightmost)
        apply_btn = QPushButton("Close" if self.read_only else "Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.apply_and_close)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5c8f;
                color: #FFFFFF;
                border: 1px solid #4a7cba;
                padding: 8px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d6c9f;
            }
        """)
        button_layout.addWidget(apply_btn)

        layout.addWidget(button_bar)

        # Set dialog background
        self.setStyleSheet("""
            QDialog {
                background-color: #2D2D2D;
            }
        """)

        # Setup keyboard shortcuts for font scaling
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for font scaling."""
        from PyQt6.QtGui import QShortcut, QKeySequence

        # Cmd/Ctrl + Plus - Increase font size
        zoom_in = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        zoom_in.activated.connect(self.increase_font_size)

        # Cmd/Ctrl + Minus - Decrease font size
        zoom_out = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        zoom_out.activated.connect(self.decrease_font_size)

        # Cmd/Ctrl + 0 - Reset to default
        reset_zoom = QShortcut(QKeySequence("Ctrl+0"), self)
        reset_zoom.activated.connect(self.reset_font_size)

    def increase_font_size(self):
        """Increase editor font size (Cmd/Ctrl +)."""
        self.font_size = min(self.font_size + 2, 32)  # Max 32pt
        self.update_font()

    def decrease_font_size(self):
        """Decrease editor font size (Cmd/Ctrl -)."""
        self.font_size = max(self.font_size - 2, 8)  # Min 8pt
        self.update_font()

    def reset_font_size(self):
        """Reset font to default size (Cmd/Ctrl 0)."""
        self.font_size = 12
        self.update_font()

    def update_font(self):
        """Update text editor font size."""
        font = QFont("Monaco", self.font_size)
        self.text_edit.setFont(font)

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

    def keyPressEvent(self, event):
        """Handle key events."""
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
