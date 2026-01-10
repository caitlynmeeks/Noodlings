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
#   Script Editor Dialog - Inline JavaScript editor for UI events
#
#   Modal dialog for editing QuickJS scripts that respond to ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.dialogs.script_editor_dialog
# PURPOSE:  Script Editor Dialog
# LAYER:    Studio / Dialogs
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   JavaScriptHighlighter, ScriptEditorDialog
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QSplitter, QWidget, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
import re


class JavaScriptHighlighter(QSyntaxHighlighter):
    """Syntax highlighting for JavaScript/QuickJS code."""

    def __init__(self, document):
        super().__init__(document)

        # Keyword format (blue)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#569CD6"))
        self.keyword_format.setFontWeight(QFont.Weight.Bold)

        # String format (orange)
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#CE9178"))

        # Comment format (green)
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#6A9955"))

        # Function format (yellow)
        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#DCDCAA"))

        # Number format (light green)
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#B5CEA8"))

        # API object format (cyan) - ui, event, console, app
        self.api_format = QTextCharFormat()
        self.api_format.setForeground(QColor("#4EC9B0"))

        # Keywords
        self.keywords = [
            'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while',
            'return', 'true', 'false', 'null', 'undefined', 'this',
            'try', 'catch', 'throw', 'new', 'typeof', 'instanceof'
        ]

        # API objects
        self.api_objects = ['ui', 'event', 'console', 'app', 'storage', 'audio']

    def highlightBlock(self, text):
        """Highlight a block of text."""
        # Keywords
        for word in self.keywords:
            pattern = f'\\b{word}\\b'
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.keyword_format)

        # API objects
        for obj in self.api_objects:
            pattern = f'\\b{obj}\\b'
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.api_format)

        # Double-quoted strings
        for match in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)

        # Single-quoted strings
        for match in re.finditer(r"'(?:[^'\\]|\\.)*'", text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)

        # Template strings
        for match in re.finditer(r'`(?:[^`\\]|\\.)*`', text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)

        # Numbers
        for match in re.finditer(r'\b\d+\.?\d*\b', text):
            self.setFormat(match.start(), match.end() - match.start(), self.number_format)

        # Single-line comments
        for match in re.finditer(r'//[^\n]*', text):
            self.setFormat(match.start(), match.end() - match.start(), self.comment_format)

        # Function calls
        for match in re.finditer(r'\b(\w+)\s*\(', text):
            name = match.group(1)
            if name not in self.keywords and name not in self.api_objects:
                self.setFormat(match.start(), len(name), self.function_format)


class ScriptEditorDialog(QDialog):
    """
    Dialog for editing inline JavaScript scripts for UI events.

    Features:
    - Syntax highlighting
    - API reference sidebar
    - Event context info
    - Basic validation
    """

    def __init__(
        self,
        script_content: str = "",
        event_name: str = "onClick",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.event_name = event_name
        self._result_script = script_content

        self.setWindowTitle(f"Edit Script - {event_name}")
        self.setModal(True)
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        self._init_style()
        self._init_ui()
        self._connect_signals()

        # Load initial content
        self.code_editor.setPlainText(script_content)

    def _init_style(self):
        """Set up dialog styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #cccccc;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                font-family: "SF Mono", "Menlo", "Monaco", "Courier New", monospace;
                font-size: 13px;
            }
            QTextEdit:focus {
                border-color: #76AF6A;
            }
            QTreeWidget {
                background-color: #252525;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #3d3d3d;
            }
            QSplitter::handle {
                background-color: #3d3d3d;
            }
        """)

    def _init_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel(f"Script for {self.event_name}")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #76AF6A;")
        layout.addWidget(header)

        # Info label
        info = QLabel("Scripts run in a QuickJS sandbox. Use ui.* to interact with components.")
        info.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(info)

        # Splitter: code editor | API reference
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Code editor
        editor_container = QWidget()
        editor_layout = QVBoxLayout(editor_container)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        self.code_editor = QTextEdit()
        self.code_editor.setPlaceholderText(self._get_placeholder())
        self.code_editor.setTabStopDistance(28)  # 4 spaces
        self.highlighter = JavaScriptHighlighter(self.code_editor.document())
        editor_layout.addWidget(self.code_editor)

        splitter.addWidget(editor_container)

        # API reference tree
        self.api_tree = self._build_api_tree()
        splitter.addWidget(self.api_tree)

        splitter.setSizes([550, 250])
        layout.addWidget(splitter, stretch=1)

        # Error display
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumWidth(80)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        button_layout.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("Save Script")
        self.save_btn.setMinimumWidth(100)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #76AF6A;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #86BF7A;
            }
        """)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def _get_placeholder(self) -> str:
        """Get placeholder text based on event type."""
        if self.event_name == "onClick":
            return """// Handle click event
// Access event data: event.x, event.y, event.button
// Access UI: ui.get('name'), ui.set('name', value)

console.log('Clicked at', event.x, event.y);
ui.set('output', 'Button clicked!');"""

        elif self.event_name in ("onChange", "onSubmit"):
            return """// Handle value change
// event.value = current value
// event.previousValue = previous value

console.log('Value changed to:', event.value);
ui.set('output', event.value);"""

        elif self.event_name.startswith("onKey"):
            return """// Handle keyboard event
// event.key = key name (e.g., "Enter", "a")
// event.modifiers.ctrl, .shift, .alt, .meta

if (event.key === 'Enter') {
    console.log('Enter pressed!');
}"""

        else:
            return """// Handle event
// event.type = event name
// event.source = component name

console.log(event.type, 'from', event.source);"""

    def _build_api_tree(self) -> QTreeWidget:
        """Build the API reference tree."""
        tree = QTreeWidget()
        tree.setHeaderLabel("API Reference")
        tree.setMaximumWidth(300)

        # ui object
        ui_item = QTreeWidgetItem(tree, ["ui"])
        ui_item.setForeground(0, QColor("#4EC9B0"))
        QTreeWidgetItem(ui_item, ["get(name) - Get component value"])
        QTreeWidgetItem(ui_item, ["set(name, value) - Set component value"])
        QTreeWidgetItem(ui_item, ["show(name) - Show component"])
        QTreeWidgetItem(ui_item, ["hide(name) - Hide component"])
        QTreeWidgetItem(ui_item, ["toggle(name) - Toggle visibility"])
        QTreeWidgetItem(ui_item, ["enable(name) - Enable component"])
        QTreeWidgetItem(ui_item, ["disable(name) - Disable component"])
        ui_item.setExpanded(True)

        # event object
        event_item = QTreeWidgetItem(tree, ["event"])
        event_item.setForeground(0, QColor("#4EC9B0"))
        QTreeWidgetItem(event_item, ["type - Event type name"])
        QTreeWidgetItem(event_item, ["source - Component name"])
        QTreeWidgetItem(event_item, ["value - Current value"])
        QTreeWidgetItem(event_item, ["previousValue - Previous value"])
        QTreeWidgetItem(event_item, ["x, y - Mouse position"])
        QTreeWidgetItem(event_item, ["button - Mouse button"])
        QTreeWidgetItem(event_item, ["key - Key name"])
        QTreeWidgetItem(event_item, ["modifiers - {ctrl, shift, alt, meta}"])
        event_item.setExpanded(True)

        # console object
        console_item = QTreeWidgetItem(tree, ["console"])
        console_item.setForeground(0, QColor("#4EC9B0"))
        QTreeWidgetItem(console_item, ["log(...) - Log message"])
        QTreeWidgetItem(console_item, ["warn(...) - Warning"])
        QTreeWidgetItem(console_item, ["error(...) - Error"])
        console_item.setExpanded(True)

        # app object (future)
        app_item = QTreeWidgetItem(tree, ["app (future)"])
        app_item.setForeground(0, QColor("#888888"))
        QTreeWidgetItem(app_item, ["sendToNoodling(name, msg)"])
        QTreeWidgetItem(app_item, ["getNoodling(name)"])

        return tree

    def _connect_signals(self):
        """Connect signals."""
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self._on_save)
        self.api_tree.itemDoubleClicked.connect(self._insert_api_snippet)

    def _on_save(self):
        """Validate and save the script."""
        script = self.code_editor.toPlainText()

        # Basic validation - check for obvious syntax errors
        error = self._validate_script(script)
        if error:
            self.error_label.setText(error)
            self.error_label.setVisible(True)
            return

        self._result_script = script
        self.accept()

    def _validate_script(self, script: str) -> Optional[str]:
        """
        Basic script validation.

        Returns error message or None if valid.
        """
        if not script.strip():
            return None  # Empty script is valid (will clear the binding)

        # Check for balanced braces
        brace_count = 0
        paren_count = 0
        bracket_count = 0

        in_string = False
        string_char = None

        for i, char in enumerate(script):
            # Track string state
            if char in ('"', "'", '`') and (i == 0 or script[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1

        if brace_count != 0:
            return f"Unbalanced braces: {'+' if brace_count > 0 else ''}{brace_count}"
        if paren_count != 0:
            return f"Unbalanced parentheses: {'+' if paren_count > 0 else ''}{paren_count}"
        if bracket_count != 0:
            return f"Unbalanced brackets: {'+' if bracket_count > 0 else ''}{bracket_count}"
        if in_string:
            return "Unclosed string literal"

        return None

    def _insert_api_snippet(self, item: QTreeWidgetItem, column: int):
        """Insert API snippet on double-click."""
        text = item.text(0)

        # Extract method name if it's a method
        if " - " in text:
            snippet = text.split(" - ")[0]

            # Build full call
            parent = item.parent()
            if parent:
                obj = parent.text(0).split(" ")[0]  # Remove "(future)" etc.
                if "(" in snippet:
                    snippet = f"{obj}.{snippet}"
                else:
                    snippet = f"{obj}.{snippet}"

            cursor = self.code_editor.textCursor()
            cursor.insertText(snippet)
            self.code_editor.setFocus()

    def get_script(self) -> str:
        """Get the edited script content."""
        return self._result_script

    @staticmethod
    def edit_script(
        script_content: str = "",
        event_name: str = "onClick",
        parent: Optional[QWidget] = None
    ) -> Optional[str]:
        """
        Static method to show dialog and return edited script.

        Returns:
            Edited script string or None if cancelled.
        """
        dialog = ScriptEditorDialog(script_content, event_name, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_script()
        return None

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
