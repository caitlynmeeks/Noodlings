"""
Event Binding Widget - Single row for event configuration

The Delphi Object Inspector's Events tab equivalent. Each row represents
one event binding (e.g., onClick -> send_to_noodling).

Author: Caitlyn + Claude
Date: January 2026
"""

from typing import Optional, Dict, Any, List, Callable

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal


# Available actions and their required parameters
ACTIONS = {
    "send_to_noodling": {
        "label": "Send to Noodling",
        "params": ["target", "message_source", "chat_history"],
    },
    "call_script": {
        "label": "Call Script",
        "params": ["script"],  # or script_file
    },
    "set_value": {
        "label": "Set Value",
        "params": ["target", "value"],
    },
    "show": {
        "label": "Show",
        "params": ["target"],
    },
    "hide": {
        "label": "Hide",
        "params": ["target"],
    },
    "toggle_visible": {
        "label": "Toggle Visible",
        "params": ["target"],
    },
}

# Common event types (most frequently used)
COMMON_EVENTS = [
    "onClick",
    "onDoubleClick",
    "onChange",
    "onSubmit",
    "onMouseEnter",
    "onMouseLeave",
    "onFocus",
    "onBlur",
    "onKeyDown",
    "onKeyUp",
]


class EventBindingWidget(QWidget):
    """
    Widget for editing a single event binding.

    Layout:
    +-------------------------------------------------------+
    | onClick        [send_to_noodling v]  [x]              |
    |                Target: [red        v]                 |
    |                Message: [input     v]                 |
    +-------------------------------------------------------+

    Signals:
        changed: Emitted when any value changes
        delete_requested: Emitted when delete button clicked
        edit_script_requested: Emitted when Edit Script clicked
    """

    changed = pyqtSignal()
    delete_requested = pyqtSignal()
    edit_script_requested = pyqtSignal(str)  # current script content

    def __init__(
        self,
        event_name: str,
        binding_data: Optional[Dict[str, Any]] = None,
        available_components: Optional[List[str]] = None,
        available_noodlings: Optional[List[str]] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize event binding widget.

        Args:
            event_name: Event type (onClick, onChange, etc.)
            binding_data: Existing binding data dict or None for new
            available_components: List of component names for dropdowns
            available_noodlings: List of noodling names for dropdowns
            parent: Parent widget
        """
        super().__init__(parent)

        self.event_name = event_name
        self.available_components = available_components or []
        self.available_noodlings = available_noodlings or []
        self._updating = False

        self._init_ui()
        self._connect_signals()

        if binding_data:
            self._load_binding(binding_data)

    def _init_ui(self):
        """Build the UI."""
        self.setStyleSheet("""
            EventBindingWidget {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header row: event name, action dropdown, delete button
        header = QHBoxLayout()
        header.setSpacing(8)

        # Event name label
        self.event_label = QLabel(self.event_name)
        self.event_label.setStyleSheet("""
            font-weight: bold;
            color: #76AF6A;
            font-family: monospace;
        """)
        self.event_label.setMinimumWidth(100)
        header.addWidget(self.event_label)

        # Action dropdown
        self.action_combo = QComboBox()
        self.action_combo.setMinimumWidth(140)
        for action_id, action_info in ACTIONS.items():
            self.action_combo.addItem(action_info["label"], action_id)
        self.action_combo.setStyleSheet(self._combo_style())
        header.addWidget(self.action_combo, stretch=1)

        # Delete button
        self.delete_btn = QPushButton("x")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888888;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff6b6b;
            }
        """)
        self.delete_btn.setToolTip("Remove this event binding")
        header.addWidget(self.delete_btn)

        layout.addLayout(header)

        # Parameters container (shown/hidden based on action)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(100, 0, 24, 0)  # Indent under action
        self.params_layout.setSpacing(4)
        layout.addWidget(self.params_container)

        # Build parameter fields
        self._build_param_fields()

    def _combo_style(self) -> str:
        """Standard combo box styling."""
        return """
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QComboBox:focus {
                border-color: #76AF6A;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #cccccc;
                selection-background-color: #3d3d3d;
            }
        """

    def _line_edit_style(self) -> str:
        """Standard line edit styling."""
        return """
            QLineEdit {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QLineEdit:focus {
                border-color: #76AF6A;
            }
        """

    def _build_param_fields(self):
        """Build parameter input fields."""
        # Clear existing
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Target field (for most actions)
        self.target_row = self._create_param_row("Target:")
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.setStyleSheet(self._combo_style())
        self.target_row.layout().addWidget(self.target_combo, stretch=1)
        self.params_layout.addWidget(self.target_row)

        # Message source (for send_to_noodling)
        self.message_row = self._create_param_row("Message:")
        self.message_combo = QComboBox()
        self.message_combo.setEditable(True)
        self.message_combo.addItem("self", "self")
        for comp in self.available_components:
            self.message_combo.addItem(comp, comp)
        self.message_combo.setStyleSheet(self._combo_style())
        self.message_row.layout().addWidget(self.message_combo, stretch=1)
        self.params_layout.addWidget(self.message_row)

        # Chat history (for send_to_noodling)
        self.chat_history_row = self._create_param_row("Chat History:")
        self.chat_history_combo = QComboBox()
        self.chat_history_combo.setEditable(True)
        self.chat_history_combo.addItem("chat_history", "chat_history")
        for comp in self.available_components:
            if "chat" in comp.lower() or "history" in comp.lower():
                self.chat_history_combo.addItem(comp, comp)
        self.chat_history_combo.setStyleSheet(self._combo_style())
        self.chat_history_row.layout().addWidget(self.chat_history_combo, stretch=1)
        self.params_layout.addWidget(self.chat_history_row)

        # Value field (for set_value)
        self.value_row = self._create_param_row("Value:")
        self.value_edit = QLineEdit()
        self.value_edit.setStyleSheet(self._line_edit_style())
        self.value_edit.setPlaceholderText("Value to set")
        self.value_row.layout().addWidget(self.value_edit, stretch=1)
        self.params_layout.addWidget(self.value_row)

        # Script button (for call_script)
        self.script_row = self._create_param_row("Script:")
        self.script_btn = QPushButton("Edit Script...")
        self.script_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: none;
                border-radius: 3px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        self.script_row.layout().addWidget(self.script_btn)
        self.script_row.layout().addStretch()
        self.params_layout.addWidget(self.script_row)

        # Store current script content
        self._script_content = ""

        # Update visibility
        self._update_param_visibility()

    def _create_param_row(self, label_text: str) -> QWidget:
        """Create a parameter row with label."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        label = QLabel(label_text)
        label.setStyleSheet("color: #888888; font-size: 11px;")
        label.setMinimumWidth(70)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(label)

        return row

    def _connect_signals(self):
        """Connect internal signals."""
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        self.delete_btn.clicked.connect(self.delete_requested.emit)
        self.script_btn.clicked.connect(self._on_edit_script)

        # Emit changed on any value change
        self.target_combo.currentTextChanged.connect(self._emit_changed)
        self.message_combo.currentTextChanged.connect(self._emit_changed)
        self.chat_history_combo.currentTextChanged.connect(self._emit_changed)
        self.value_edit.textChanged.connect(self._emit_changed)

    def _on_action_changed(self, index: int):
        """Handle action selection change."""
        self._update_param_visibility()
        self._update_target_options()
        if not self._updating:
            self.changed.emit()

    def _update_param_visibility(self):
        """Show/hide parameter fields based on selected action."""
        action = self.action_combo.currentData()

        # Hide all first
        self.target_row.setVisible(False)
        self.message_row.setVisible(False)
        self.chat_history_row.setVisible(False)
        self.value_row.setVisible(False)
        self.script_row.setVisible(False)

        if action == "send_to_noodling":
            self.target_row.setVisible(True)
            self.message_row.setVisible(True)
            self.chat_history_row.setVisible(True)
        elif action == "call_script":
            self.script_row.setVisible(True)
        elif action == "set_value":
            self.target_row.setVisible(True)
            self.value_row.setVisible(True)
        elif action in ("show", "hide", "toggle_visible"):
            self.target_row.setVisible(True)

    def _update_target_options(self):
        """Update target dropdown options based on action."""
        action = self.action_combo.currentData()
        current = self.target_combo.currentText()

        self.target_combo.clear()

        if action == "send_to_noodling":
            # Show noodlings
            for noodling in self.available_noodlings:
                self.target_combo.addItem(noodling, noodling)
        else:
            # Show components
            for comp in self.available_components:
                self.target_combo.addItem(comp, comp)

        # Restore previous selection if valid
        idx = self.target_combo.findText(current)
        if idx >= 0:
            self.target_combo.setCurrentIndex(idx)

    def _on_edit_script(self):
        """Request script editing."""
        self.edit_script_requested.emit(self._script_content)

    def _emit_changed(self):
        """Emit changed signal if not updating programmatically."""
        if not self._updating:
            self.changed.emit()

    def _load_binding(self, data: Dict[str, Any]):
        """Load binding data into the widget."""
        self._updating = True
        try:
            # Set action
            action = data.get("action", "send_to_noodling")
            for i in range(self.action_combo.count()):
                if self.action_combo.itemData(i) == action:
                    self.action_combo.setCurrentIndex(i)
                    break

            self._update_param_visibility()
            self._update_target_options()

            # Set target
            if data.get("target"):
                self.target_combo.setCurrentText(data["target"])

            # Set message source
            if data.get("message_source"):
                self.message_combo.setCurrentText(data["message_source"])

            # Set chat history
            if data.get("chat_history"):
                self.chat_history_combo.setCurrentText(data["chat_history"])

            # Set value
            if data.get("value"):
                self.value_edit.setText(str(data["value"]))

            # Set script
            if data.get("script"):
                self._script_content = data["script"]
                self.script_btn.setText("Edit Script...")
                self.script_btn.setToolTip(f"Script: {len(self._script_content)} chars")

        finally:
            self._updating = False

    def get_binding_data(self) -> Dict[str, Any]:
        """Get current binding data as dictionary."""
        action = self.action_combo.currentData()
        data = {"action": action}

        if action == "send_to_noodling":
            target = self.target_combo.currentText().strip()
            if target:
                data["target"] = target

            msg_src = self.message_combo.currentText().strip()
            if msg_src and msg_src != "self":
                data["message_source"] = msg_src

            chat_hist = self.chat_history_combo.currentText().strip()
            if chat_hist and chat_hist != "chat_history":
                data["chat_history"] = chat_hist

        elif action == "call_script":
            if self._script_content:
                data["script"] = self._script_content

        elif action == "set_value":
            target = self.target_combo.currentText().strip()
            if target:
                data["target"] = target
            value = self.value_edit.text()
            if value:
                data["value"] = value

        elif action in ("show", "hide", "toggle_visible"):
            target = self.target_combo.currentText().strip()
            if target:
                data["target"] = target

        return data

    def set_script_content(self, content: str):
        """Set script content (called after script editor dialog closes)."""
        self._script_content = content
        self.script_btn.setToolTip(f"Script: {len(content)} chars" if content else "No script")
        self._emit_changed()

    def set_available_components(self, components: List[str]):
        """Update available component names."""
        self.available_components = components
        self._update_target_options()

    def set_available_noodlings(self, noodlings: List[str]):
        """Update available noodling names."""
        self.available_noodlings = noodlings
        self._update_target_options()
