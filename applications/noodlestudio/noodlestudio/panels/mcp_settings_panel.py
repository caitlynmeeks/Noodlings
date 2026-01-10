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
#   MCP Settings Panel - Configuration UI for Model Context Protocol servers
#
#   Provides interface to: - View configured MCP servers - Ad...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.mcp_settings_panel
# PURPOSE:  mcp settings panel panel UI
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MCPSettingsPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QGroupBox, QFormLayout,
    QLineEdit, QTextEdit, QComboBox, QSplitter, QScrollArea,
    QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

from ..core.mcp_manager import (
    MCPManager, MCPServerConfig, MCPServerType, MCPConnectionStatus
)


class MCPSettingsPanel(QWidget):
    """
    Panel for configuring MCP servers.

    Layout:
    - Left: Server list with status indicators
    - Right: Server details and tools
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mcp_manager = MCPManager.instance()

        self._selected_server: str = ""
        self._setup_ui()
        self._connect_signals()
        self._refresh_server_list()

    def _setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("MCP Servers")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Configure Model Context Protocol servers to give agents access to tools.\n"
            "MCP servers provide file access, web search, databases, and more."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(desc)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Server list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        # Server list
        self.server_list = QListWidget()
        self.server_list.setMinimumWidth(200)
        self.server_list.currentItemChanged.connect(self._on_server_selected)
        left_layout.addWidget(self.server_list)

        # Add/Remove buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._on_add_server)
        self.remove_btn = QPushButton("- Remove")
        self.remove_btn.clicked.connect(self._on_remove_server)
        self.remove_btn.setEnabled(False)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_panel)

        # Right panel: Server details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        # Scroll area for details
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        self.details_layout = QVBoxLayout(scroll_content)
        self.details_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Server configuration group
        config_group = QGroupBox("Configuration")
        config_form = QFormLayout(config_group)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Server name")
        config_form.addRow("Name:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItem("Local Command", MCPServerType.LOCAL)
        self.type_combo.addItem("Docker Container", MCPServerType.DOCKER)
        self.type_combo.addItem("Remote URL", MCPServerType.REMOTE)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        config_form.addRow("Type:", self.type_combo)

        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("npx, python, node, etc.")
        config_form.addRow("Command:", self.command_edit)

        self.args_edit = QLineEdit()
        self.args_edit.setPlaceholderText("-y @modelcontextprotocol/server-filesystem /path")
        config_form.addRow("Arguments:", self.args_edit)

        self.env_edit = QTextEdit()
        self.env_edit.setPlaceholderText("KEY=value (one per line)")
        self.env_edit.setMaximumHeight(80)
        config_form.addRow("Env Vars:", self.env_edit)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://example.com/mcp")
        self.url_edit.setVisible(False)
        self.url_label = QLabel("URL:")
        self.url_label.setVisible(False)
        config_form.addRow(self.url_label, self.url_edit)

        self.details_layout.addWidget(config_group)

        # Connection group
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Status: Disconnected")
        status_row.addWidget(self.status_label)
        status_row.addStretch()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        status_row.addWidget(self.connect_btn)

        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self._on_test)
        status_row.addWidget(self.test_btn)

        conn_layout.addLayout(status_row)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #FF6B6B;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        conn_layout.addWidget(self.error_label)

        self.details_layout.addWidget(conn_group)

        # Tools group
        tools_group = QGroupBox("Available Tools")
        tools_layout = QVBoxLayout(tools_group)

        self.tools_list = QListWidget()
        self.tools_list.setMinimumHeight(150)
        tools_layout.addWidget(self.tools_list)

        self.tool_desc_label = QLabel("Select a tool to see its description")
        self.tool_desc_label.setWordWrap(True)
        self.tool_desc_label.setStyleSheet("color: #888; font-style: italic;")
        tools_layout.addWidget(self.tool_desc_label)

        self.details_layout.addWidget(tools_group)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.clicked.connect(self._on_save)
        save_layout.addWidget(self.save_btn)
        self.details_layout.addLayout(save_layout)

        self.details_layout.addStretch()

        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)

        splitter.addWidget(right_panel)
        splitter.setSizes([250, 450])

        layout.addWidget(splitter)

        # Config file path
        config_path = QLabel(f"Config: {self.mcp_manager.get_config_path()}")
        config_path.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(config_path)

        # Initially disable details until a server is selected
        self._set_details_enabled(False)

    def _connect_signals(self):
        """Connect MCP manager signals."""
        self.mcp_manager.server_connected.connect(self._on_server_status_changed)
        self.mcp_manager.server_disconnected.connect(self._on_server_status_changed)
        self.mcp_manager.server_error.connect(self._on_server_error)
        self.mcp_manager.tools_updated.connect(self._on_tools_updated)

        self.tools_list.currentItemChanged.connect(self._on_tool_selected)

    def _refresh_server_list(self):
        """Refresh the server list."""
        self.server_list.clear()

        for config in self.mcp_manager.get_servers():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, config.name)

            # Status indicator
            if config.status == MCPConnectionStatus.CONNECTED:
                indicator = "\u2713"  # Check mark
                color = "#4CAF50"
            elif config.status == MCPConnectionStatus.ERROR:
                indicator = "\u2717"  # X mark
                color = "#FF6B6B"
            elif config.status == MCPConnectionStatus.CONNECTING:
                indicator = "\u2022"  # Bullet
                color = "#FFA726"
            else:
                indicator = "\u25CB"  # Empty circle
                color = "#888"

            item.setText(f"{indicator} {config.name}")
            item.setForeground(QColor(color))

            self.server_list.addItem(item)

    def _set_details_enabled(self, enabled: bool):
        """Enable/disable the details panel."""
        self.name_edit.setEnabled(enabled)
        self.type_combo.setEnabled(enabled)
        self.command_edit.setEnabled(enabled)
        self.args_edit.setEnabled(enabled)
        self.env_edit.setEnabled(enabled)
        self.url_edit.setEnabled(enabled)
        self.connect_btn.setEnabled(enabled)
        self.test_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.remove_btn.setEnabled(enabled)

    def _load_server_details(self, name: str):
        """Load server details into the form."""
        config = self.mcp_manager.get_server(name)
        if not config:
            return

        self._selected_server = name

        self.name_edit.setText(config.name)

        # Set type combo
        for i in range(self.type_combo.count()):
            if self.type_combo.itemData(i) == config.type:
                self.type_combo.setCurrentIndex(i)
                break

        self.command_edit.setText(config.command)
        self.args_edit.setText(" ".join(config.args))

        # Format env vars
        env_lines = [f"{k}={v}" for k, v in config.env.items()]
        self.env_edit.setPlainText("\n".join(env_lines))

        self.url_edit.setText(config.url)

        # Update status
        self._update_status_display(config)

        # Update tools
        self._update_tools_display(config)

        self._set_details_enabled(True)

    def _update_status_display(self, config: MCPServerConfig):
        """Update the connection status display."""
        status_text = {
            MCPConnectionStatus.CONNECTED: "Connected",
            MCPConnectionStatus.CONNECTING: "Connecting...",
            MCPConnectionStatus.DISCONNECTED: "Disconnected",
            MCPConnectionStatus.ERROR: "Error"
        }

        status_color = {
            MCPConnectionStatus.CONNECTED: "#4CAF50",
            MCPConnectionStatus.CONNECTING: "#FFA726",
            MCPConnectionStatus.DISCONNECTED: "#888",
            MCPConnectionStatus.ERROR: "#FF6B6B"
        }

        self.status_label.setText(f"Status: {status_text[config.status]}")
        self.status_label.setStyleSheet(f"color: {status_color[config.status]};")

        if config.error_message:
            self.error_label.setText(config.error_message)
            self.error_label.setVisible(True)
        else:
            self.error_label.setVisible(False)

        # Update button text
        if config.status == MCPConnectionStatus.CONNECTED:
            self.connect_btn.setText("Disconnect")
        else:
            self.connect_btn.setText("Connect")

    def _update_tools_display(self, config: MCPServerConfig):
        """Update the tools list display."""
        self.tools_list.clear()

        for tool in config.tools:
            item = QListWidgetItem(tool.name)
            item.setData(Qt.ItemDataRole.UserRole, tool)
            self.tools_list.addItem(item)

        if not config.tools:
            if config.status == MCPConnectionStatus.CONNECTED:
                self.tool_desc_label.setText("No tools available")
            else:
                self.tool_desc_label.setText("Connect to discover tools")
        else:
            self.tool_desc_label.setText(f"{len(config.tools)} tools available")

    # === Slots ===

    def _on_server_selected(self, current, previous):
        """Handle server selection."""
        if not current:
            self._set_details_enabled(False)
            return

        name = current.data(Qt.ItemDataRole.UserRole)
        self._load_server_details(name)

    def _on_type_changed(self, index):
        """Handle server type change."""
        server_type = self.type_combo.itemData(index)

        # Show/hide URL field
        is_remote = server_type == MCPServerType.REMOTE
        self.url_edit.setVisible(is_remote)
        self.url_label.setVisible(is_remote)

        # Show/hide command fields
        is_local = server_type in (MCPServerType.LOCAL, MCPServerType.DOCKER)
        self.command_edit.setVisible(is_local)
        self.args_edit.setVisible(is_local)

    def _on_add_server(self):
        """Add a new server."""
        # Generate unique name
        base_name = "new-server"
        name = base_name
        counter = 1
        while self.mcp_manager.get_server(name):
            name = f"{base_name}-{counter}"
            counter += 1

        # Create default config
        config = MCPServerConfig(
            name=name,
            type=MCPServerType.LOCAL,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."]
        )

        self.mcp_manager.add_server(config)
        self._refresh_server_list()

        # Select the new server
        for i in range(self.server_list.count()):
            item = self.server_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == name:
                self.server_list.setCurrentItem(item)
                break

    def _on_remove_server(self):
        """Remove the selected server."""
        if not self._selected_server:
            return

        reply = QMessageBox.question(
            self,
            "Remove Server",
            f"Remove server '{self._selected_server}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.mcp_manager.remove_server(self._selected_server)
            self._selected_server = ""
            self._refresh_server_list()
            self._set_details_enabled(False)

    def _on_save(self):
        """Save changes to the selected server."""
        if not self._selected_server:
            return

        # Parse env vars
        env = {}
        for line in self.env_edit.toPlainText().strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()

        # Parse args
        args = self.args_edit.text().split()

        config = MCPServerConfig(
            name=self.name_edit.text(),
            type=self.type_combo.currentData(),
            command=self.command_edit.text(),
            args=args,
            env=env,
            url=self.url_edit.text()
        )

        self.mcp_manager.update_server(self._selected_server, config)
        self._selected_server = config.name
        self._refresh_server_list()

        print(f"[MCP Settings] Saved server: {config.name}")

    def _on_connect(self):
        """Connect/disconnect from the selected server."""
        if not self._selected_server:
            return

        config = self.mcp_manager.get_server(self._selected_server)
        if not config:
            return

        import asyncio

        if config.status == MCPConnectionStatus.CONNECTED:
            # Disconnect
            asyncio.create_task(
                self.mcp_manager._disconnect_server(self._selected_server)
            )
        else:
            # Connect
            asyncio.create_task(
                self.mcp_manager.connect_server(self._selected_server)
            )

    def _on_test(self):
        """Test connection to the selected server."""
        self._on_connect()

    def _on_server_status_changed(self, server_name: str):
        """Handle server status change."""
        self._refresh_server_list()

        if server_name == self._selected_server:
            config = self.mcp_manager.get_server(server_name)
            if config:
                self._update_status_display(config)
                self._update_tools_display(config)

    def _on_server_error(self, server_name: str, error: str):
        """Handle server error."""
        self._refresh_server_list()

        if server_name == self._selected_server:
            config = self.mcp_manager.get_server(server_name)
            if config:
                self._update_status_display(config)

    def _on_tools_updated(self, server_name: str):
        """Handle tools discovery."""
        if server_name == self._selected_server:
            config = self.mcp_manager.get_server(server_name)
            if config:
                self._update_tools_display(config)

    def _on_tool_selected(self, current, previous):
        """Handle tool selection."""
        if not current:
            self.tool_desc_label.setText("Select a tool to see its description")
            return

        tool = current.data(Qt.ItemDataRole.UserRole)
        if tool:
            desc = tool.description or "No description available"
            schema = tool.input_schema

            # Format schema
            if schema and 'properties' in schema:
                props = schema['properties']
                params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
                self.tool_desc_label.setText(f"{tool.name}({params})\n\n{desc}")
            else:
                self.tool_desc_label.setText(f"{tool.name}()\n\n{desc}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
