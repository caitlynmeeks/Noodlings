"""
Script Editor Widget - Code editor for NoodleScripts

Like Unity's script editor but embedded in Inspector!

Features:
- Syntax highlighting (Python)
- Line numbers
- Auto-indent
- Compile button
- Error display

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QLabel, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
import re


class PythonHighlighter(QSyntaxHighlighter):
    """Basic Python syntax highlighting."""

    def __init__(self, document):
        super().__init__(document)

        # Define formats
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#569CD6"))  # Blue
        self.keyword_format.setFontWeight(QFont.Weight.Bold)

        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#CE9178"))  # Orange

        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#6A9955"))  # Green

        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#DCDCAA"))  # Yellow

        self.class_format = QTextCharFormat()
        self.class_format.setForeground(QColor("#4EC9B0"))  # Cyan

        # Keywords
        self.keywords = [
            'def', 'class', 'if', 'elif', 'else', 'for', 'while',
            'return', 'import', 'from', 'as', 'try', 'except',
            'True', 'False', 'None', 'self', 'pass', 'break', 'continue'
        ]

    def highlightBlock(self, text):
        """Highlight a block of text."""
        # Keywords
        for word in self.keywords:
            pattern = f'\\b{word}\\b'
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.keyword_format)

        # Strings
        for match in re.finditer(r'"[^"]*"', text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)
        for match in re.finditer(r"'[^']*'", text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)

        # Comments
        for match in re.finditer(r'#[^\n]*', text):
            self.setFormat(match.start(), match.end() - match.start(), self.comment_format)

        # Function definitions
        for match in re.finditer(r'\bdef\s+(\w+)', text):
            self.setFormat(match.start(1), match.end(1) - match.start(1), self.function_format)

        # Class definitions
        for match in re.finditer(r'\bclass\s+(\w+)', text):
            self.setFormat(match.start(1), match.end(1) - match.start(1), self.class_format)


class ScriptEditor(QWidget):
    """
    Script editor widget for NoodleScripts.

    Like Unity's Inspector script editor!
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()

        # Script dropdown
        self.script_selector = QComboBox()
        self.script_selector.addItems([
            "New Script...",
            "ClickableBox.py",
            "QuestGiver.py",
            "VendingMachine.py"
        ])
        self.script_selector.currentTextChanged.connect(self.load_script_template)
        toolbar.addWidget(QLabel("Script:"))
        toolbar.addWidget(self.script_selector, stretch=1)

        # Compile button
        self.compile_btn = QPushButton("▶ Compile & Attach")
        self.compile_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        self.compile_btn.clicked.connect(self.compile_script)
        toolbar.addWidget(self.compile_btn)

        layout.addLayout(toolbar)

        # Code editor
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Courier New", 11))
        self.code_editor.setStyleSheet("""
            QTextEdit {
                background: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        self.code_editor.setPlaceholderText("# Write your NoodleScript here...\n# Inherit from NoodleScript")

        # Syntax highlighting
        self.highlighter = PythonHighlighter(self.code_editor.document())

        layout.addWidget(self.code_editor)

        # Error display
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #EF5350; font-family: 'Courier New'; font-size: 10px;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def load_script_template(self, script_name: str):
        """Load script template."""
        if script_name == "New Script...":
            self.code_editor.setPlainText(self.get_empty_template())

        elif script_name == "ClickableBox.py":
            # Load example script
            try:
                from pathlib import Path
                script_path = Path(__file__).parent.parent.parent / "example_scripts" / "ClickableBox.py"
                if script_path.exists():
                    with open(script_path, 'r') as f:
                        self.code_editor.setPlainText(f.read())
            except:
                self.code_editor.setPlainText(self.get_clickable_box_template())

        # Similar for other templates...

    def get_empty_template(self) -> str:
        """Get empty script template."""
        return """from noodlestudio.scripting import NoodleScript, Noodlings, Debug


class MyScript(NoodleScript):
    \"\"\"
    Custom script for [describe what it does].
    \"\"\"

    def Start(self):
        \"\"\"Called when script first loads.\"\"\"
        Debug.Log("MyScript started!")

    def OnClick(self, clicker):
        \"\"\"Called when prim is clicked.\"\"\"
        Debug.Log(f"{clicker} clicked me!")
        # Your code here

    def OnUse(self, user):
        \"\"\"Called when prim is used.\"\"\"
        Debug.Log(f"{user} used me!")
        # Your code here
"""

    def get_clickable_box_template(self) -> str:
        """Get ClickableBox template."""
        return """from noodlestudio.scripting import NoodleScript, Noodlings, Debug


class ClickableBox(NoodleScript):
    \"\"\"Rezzes Anklebiters when clicked. DON'T CLICK!\"\"\"

    def Start(self):
        Debug.Log("Mysterious box initialized. (Don't click it!)")
        self.rez_count = 0
        self.max_rezzes = 10

    def OnClick(self, clicker):
        if self.rez_count >= self.max_rezzes:
            Debug.LogWarning("Box is exhausted!")
            return

        # REZ AN ANKLEBITER!
        anklebiter = Noodlings.Rez("anklebiter.noodling", room=self.prim.room)
        self.rez_count += 1

        Debug.Log(f"Anklebiter #{self.rez_count} rezzed!")

        if self.rez_count >= self.max_rezzes:
            self.Destroy(delay=2.0)  # Box crumbles to dust
"""

    def compile_script(self):
        """Compile and attach script to current prim."""
        script_code = self.code_editor.toPlainText()

        if not script_code.strip():
            self.show_error("Script is empty!")
            return

        try:
            from ..scripting.script_executor import SCRIPT_EXECUTOR

            # Extract class name
            class_name = self.extract_class_name(script_code)
            if not class_name:
                self.show_error("No NoodleScript class found in script!")
                return

            # Compile
            success = SCRIPT_EXECUTOR.compile_script(class_name, script_code)

            if success:
                self.error_label.setVisible(False)
                Debug.Log(f"Script compiled successfully: {class_name}")

                # TODO: Attach to selected prim
                # For now, just show success
                self.compile_btn.setText("✓ Compiled!")
                self.compile_btn.setStyleSheet("""
                    QPushButton {
                        background: #45a049;
                        color: white;
                        padding: 6px 12px;
                        border-radius: 3px;
                        font-weight: bold;
                    }
                """)

            else:
                self.show_error("Compilation failed! Check console for errors.")

        except Exception as e:
            self.show_error(f"Error: {e}")

    def extract_class_name(self, code: str) -> Optional[str]:
        """Extract class name from script code."""
        match = re.search(r'class\s+(\w+)\s*\(', code)
        return match.group(1) if match else None

    def show_error(self, error: str):
        """Show error message."""
        self.error_label.setText(f"⚠️ {error}")
        self.error_label.setVisible(True)

        self.compile_btn.setText("▶ Compile & Attach")
        self.compile_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
