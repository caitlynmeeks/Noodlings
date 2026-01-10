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
#   MCP Manager - Connection handling for Model Context Protocol servers
#
#   Manages MCP server connections, tool discovery, and tool ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.mcp_manager
# PURPOSE:  Mcp Manager
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MCPServerType, MCPConnectionStatus, MCPToolDef, MCPServerConfig, MCPManager
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import AsyncExitStack

from PyQt6.QtCore import QObject, pyqtSignal


class MCPServerType(Enum):
    """Type of MCP server transport."""
    LOCAL = "local"      # Local command (npx, python, etc.)
    DOCKER = "docker"    # Docker container
    REMOTE = "remote"    # Remote URL (SSE/HTTP)


class MCPConnectionStatus(Enum):
    """Connection status for an MCP server."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPToolDef:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    type: MCPServerType = MCPServerType.LOCAL
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: str = ""  # For remote servers

    # Runtime state (not persisted)
    status: MCPConnectionStatus = MCPConnectionStatus.DISCONNECTED
    error_message: str = ""
    tools: List[MCPToolDef] = field(default_factory=list)


class MCPManager(QObject):
    """
    Manages MCP server connections and tool invocations.

    Signals:
        server_connected(str): Emitted when a server connects (server_name)
        server_disconnected(str): Emitted when a server disconnects (server_name)
        server_error(str, str): Emitted on error (server_name, error_message)
        tools_updated(str): Emitted when tools are discovered (server_name)
        config_changed(): Emitted when configuration changes
    """

    server_connected = pyqtSignal(str)
    server_disconnected = pyqtSignal(str)
    server_error = pyqtSignal(str, str)
    tools_updated = pyqtSignal(str)
    config_changed = pyqtSignal()

    _instance = None

    @classmethod
    def instance(cls) -> 'MCPManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)

        # Server configurations
        self._servers: Dict[str, MCPServerConfig] = {}

        # Active sessions (server_name -> ClientSession)
        self._sessions: Dict[str, Any] = {}

        # Exit stack for managing async contexts
        self._exit_stack: Optional[AsyncExitStack] = None

        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Config file path
        self._config_path = self._get_config_path()

        # Load configuration
        self._load_config()

    def _get_config_path(self) -> Path:
        """Get path to MCP servers configuration file."""
        # Check for project-specific config first
        project_config = Path.cwd() / ".noodlestudio" / "mcp_servers.yaml"
        if project_config.exists():
            return project_config

        # Fall back to user config
        user_config = Path.home() / ".noodlestudio" / "mcp_servers.yaml"
        user_config.parent.mkdir(parents=True, exist_ok=True)
        return user_config

    def _load_config(self):
        """Load MCP server configuration from YAML file."""
        if not self._config_path.exists():
            # Create default config with common servers
            self._create_default_config()
            return

        try:
            with open(self._config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            servers = data.get('servers', {})
            for name, config in servers.items():
                server_type = MCPServerType(config.get('type', 'local'))

                # Expand environment variables in command and args
                command = self._expand_env(config.get('command', ''))
                args = [self._expand_env(arg) for arg in config.get('args', [])]
                env = {k: self._expand_env(v) for k, v in config.get('env', {}).items()}

                self._servers[name] = MCPServerConfig(
                    name=name,
                    type=server_type,
                    command=command,
                    args=args,
                    env=env,
                    url=config.get('url', '')
                )

            print(f"[MCP] Loaded {len(self._servers)} server configurations")

        except Exception as e:
            print(f"[MCP] Error loading config: {e}")

    def _expand_env(self, value: str) -> str:
        """Expand environment variables in a string."""
        if not isinstance(value, str):
            return value

        # Handle ${VAR} syntax
        import re
        pattern = r'\$\{([^}]+)\}'

        def replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace, value)

    def _create_default_config(self):
        """Create default MCP configuration with common servers."""
        default_config = {
            'servers': {
                'filesystem': {
                    'type': 'local',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-filesystem', '${PROJECT_PATH:-.}'],
                    'env': {}
                },
                'fetch': {
                    'type': 'local',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-fetch'],
                    'env': {}
                }
            }
        }

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print(f"[MCP] Created default config at {self._config_path}")
        self._load_config()

    def save_config(self):
        """Save current configuration to YAML file."""
        servers = {}
        for name, config in self._servers.items():
            servers[name] = {
                'type': config.type.value,
                'command': config.command,
                'args': config.args,
                'env': config.env
            }
            if config.url:
                servers[name]['url'] = config.url

        data = {'servers': servers}

        with open(self._config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        self.config_changed.emit()
        print(f"[MCP] Saved config to {self._config_path}")

    def get_servers(self) -> List[MCPServerConfig]:
        """Get list of configured servers."""
        return list(self._servers.values())

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a specific server configuration."""
        return self._servers.get(name)

    def add_server(self, config: MCPServerConfig):
        """Add a new server configuration."""
        self._servers[config.name] = config
        self.save_config()

    def remove_server(self, name: str):
        """Remove a server configuration."""
        if name in self._servers:
            # Disconnect if connected
            if name in self._sessions:
                asyncio.create_task(self._disconnect_server(name))
            del self._servers[name]
            self.save_config()

    def update_server(self, name: str, config: MCPServerConfig):
        """Update a server configuration."""
        # Disconnect if connected
        if name in self._sessions:
            asyncio.create_task(self._disconnect_server(name))

        # Update config
        if name != config.name:
            # Name changed - remove old, add new
            if name in self._servers:
                del self._servers[name]

        self._servers[config.name] = config
        self.save_config()

    async def connect_server(self, name: str) -> bool:
        """
        Connect to an MCP server.

        Args:
            name: Server name from configuration

        Returns:
            True if connected successfully
        """
        config = self._servers.get(name)
        if not config:
            print(f"[MCP] Unknown server: {name}")
            return False

        if name in self._sessions:
            print(f"[MCP] Already connected to {name}")
            return True

        config.status = MCPConnectionStatus.CONNECTING
        config.error_message = ""

        try:
            # Import MCP SDK
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ImportError:
                config.status = MCPConnectionStatus.ERROR
                config.error_message = "MCP SDK not installed. Run: pip install mcp"
                self.server_error.emit(name, config.error_message)
                return False

            # Set up exit stack if needed
            if self._exit_stack is None:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.__aenter__()

            # Create server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env={**os.environ, **config.env} if config.env else None
            )

            # Connect via stdio
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            # Create session
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize
            await session.initialize()

            # Store session
            self._sessions[name] = session

            # Discover tools
            await self._discover_tools(name, session)

            config.status = MCPConnectionStatus.CONNECTED
            self.server_connected.emit(name)
            print(f"[MCP] Connected to {name}")
            return True

        except Exception as e:
            config.status = MCPConnectionStatus.ERROR
            config.error_message = str(e)
            self.server_error.emit(name, str(e))
            print(f"[MCP] Error connecting to {name}: {e}")
            return False

    async def _disconnect_server(self, name: str):
        """Disconnect from an MCP server."""
        if name in self._sessions:
            del self._sessions[name]

        config = self._servers.get(name)
        if config:
            config.status = MCPConnectionStatus.DISCONNECTED
            config.tools = []

        self.server_disconnected.emit(name)
        print(f"[MCP] Disconnected from {name}")

    async def _discover_tools(self, name: str, session):
        """Discover available tools from a connected server."""
        config = self._servers.get(name)
        if not config:
            return

        try:
            response = await session.list_tools()
            config.tools = []

            for tool in response.tools:
                tool_def = MCPToolDef(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    server_name=name
                )
                config.tools.append(tool_def)

            self.tools_updated.emit(name)
            print(f"[MCP] Discovered {len(config.tools)} tools from {name}")

        except Exception as e:
            print(f"[MCP] Error discovering tools from {name}: {e}")

    async def call_tool(self, server_name: str, tool_name: str,
                        arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Dict with 'success', 'result' or 'error' keys
        """
        session = self._sessions.get(server_name)
        if not session:
            return {
                'success': False,
                'error': f"Not connected to server: {server_name}"
            }

        try:
            from mcp import types

            result = await session.call_tool(tool_name, arguments=arguments)

            # Parse result content
            output = []
            for content in result.content:
                if hasattr(content, 'text'):
                    output.append(content.text)
                elif hasattr(content, 'data'):
                    output.append(content.data)

            return {
                'success': True,
                'result': output[0] if len(output) == 1 else output,
                'is_error': getattr(result, 'isError', False)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def call_tool_sync(self, server_name: str, tool_name: str,
                       arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for call_tool.

        Runs the async call in the event loop.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.call_tool(server_name, tool_name, arguments)
        )

    def get_all_tools(self) -> List[MCPToolDef]:
        """Get all tools from all connected servers."""
        tools = []
        for config in self._servers.values():
            if config.status == MCPConnectionStatus.CONNECTED:
                tools.extend(config.tools)
        return tools

    def get_tools_for_server(self, server_name: str) -> List[MCPToolDef]:
        """Get tools for a specific server."""
        config = self._servers.get(server_name)
        if config:
            return config.tools
        return []

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected."""
        return server_name in self._sessions

    async def connect_all(self):
        """Connect to all configured servers."""
        for name in self._servers:
            await self.connect_server(name)

    async def disconnect_all(self):
        """Disconnect from all servers."""
        for name in list(self._sessions.keys()):
            await self._disconnect_server(name)

        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None

    def get_config_path(self) -> Path:
        """Get the path to the configuration file."""
        return self._config_path

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
