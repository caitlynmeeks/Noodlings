"""
MCP Facet - Tool invocation via Model Context Protocol

Allows facet assemblies to call MCP tools for:
- File system operations
- Web search
- Database queries
- API integrations
- Custom tools

The facet dynamically generates input pads from the tool's schema.

Author: Caitlyn + Claude
Date: December 20, 2025
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .mcp_manager import MCPManager, MCPToolDef, MCPConnectionStatus


@dataclass
class MCPFacetConfig:
    """Configuration for an MCP facet instance."""
    server: str = ""          # MCP server name
    tool: str = ""            # Tool name on that server
    auto_connect: bool = True # Connect on first use if not connected


class MCPFacet:
    """
    Facet that invokes MCP tools.

    Usage in assembly YAML:
        - id: "read_file"
          type: "MCPFacet"
          name: "Read File"
          config:
            server: "filesystem"
            tool: "read_file"
          inputs:
            - name: path
              required: true
          outputs:
            - name: content
            - name: error

    The inputs are derived from the tool's input_schema.
    Outputs are always 'result' and 'error'.
    """

    def __init__(self, facet_id: str, config: Dict[str, Any]):
        """
        Initialize MCP facet.

        Args:
            facet_id: Unique facet identifier
            config: Configuration dict with 'server', 'tool', etc.
        """
        self.facet_id = facet_id
        self.server_name = config.get('server', '')
        self.tool_name = config.get('tool', '')
        self.auto_connect = config.get('auto_connect', True)

        self.mcp_manager = MCPManager.instance()

        # Cached tool definition
        self._tool_def: Optional[MCPToolDef] = None

    def get_tool_def(self) -> Optional[MCPToolDef]:
        """Get the tool definition from MCP manager."""
        if self._tool_def:
            return self._tool_def

        tools = self.mcp_manager.get_tools_for_server(self.server_name)
        for tool in tools:
            if tool.name == self.tool_name:
                self._tool_def = tool
                return tool
        return None

    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get input schema for this tool.

        Returns the tool's input_schema if connected, or empty schema.
        """
        tool = self.get_tool_def()
        if tool and tool.input_schema:
            return tool.input_schema
        return {'type': 'object', 'properties': {}}

    def get_required_inputs(self) -> List[str]:
        """Get list of required input parameter names."""
        schema = self.get_input_schema()
        return schema.get('required', [])

    async def ensure_connected(self) -> bool:
        """Ensure the MCP server is connected."""
        if self.mcp_manager.is_connected(self.server_name):
            return True

        if self.auto_connect:
            return await self.mcp_manager.connect_server(self.server_name)

        return False

    async def process_async(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the MCP tool with given inputs.

        Args:
            inputs: Input values mapped to tool parameters
            context: Execution context (agent info, cycle, etc.)

        Returns:
            Dict with 'result' and 'error' keys
        """
        # Ensure connected
        if not await self.ensure_connected():
            return {
                'result': None,
                'error': f"Not connected to MCP server: {self.server_name}",
                'success': False
            }

        # Build arguments from inputs
        # Filter to only include keys that are in the tool's schema
        schema = self.get_input_schema()
        properties = schema.get('properties', {})

        arguments = {}
        for key, value in inputs.items():
            if key in properties or not properties:
                # Include if it's a known property or if we don't have schema
                arguments[key] = value

        # Call the tool
        result = await self.mcp_manager.call_tool(
            self.server_name,
            self.tool_name,
            arguments
        )

        # Format output
        if result.get('success'):
            return {
                'result': result.get('result'),
                'error': None,
                'success': True,
                'out': result.get('result')  # Default output pad
            }
        else:
            return {
                'result': None,
                'error': result.get('error'),
                'success': False,
                'out': None
            }

    def process(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_async.

        Runs the async process in the event loop.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # We're already in an async context - create a task
            # This happens when called from facet executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.process_async(inputs, context)
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                self.process_async(inputs, context)
            )


def get_available_mcp_tools() -> List[Dict[str, Any]]:
    """
    Get list of available MCP tools for facet creation UI.

    Returns list of dicts with:
        - server: Server name
        - tool: Tool name
        - description: Tool description
        - input_schema: Tool input schema
    """
    manager = MCPManager.instance()
    tools = []

    for server_config in manager.get_servers():
        for tool in server_config.tools:
            tools.append({
                'server': server_config.name,
                'tool': tool.name,
                'description': tool.description,
                'input_schema': tool.input_schema,
                'display_name': f"{server_config.name}/{tool.name}"
            })

    return tools


def create_mcp_facet_from_tool(
    tool_info: Dict[str, Any],
    facet_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a facet definition dict from an MCP tool.

    This can be used to add MCP facets to assemblies.

    Args:
        tool_info: Dict from get_available_mcp_tools()
        facet_id: Optional facet ID (generated if not provided)

    Returns:
        Facet definition dict for assembly
    """
    import uuid

    if not facet_id:
        facet_id = str(uuid.uuid4())

    # Build input pads from schema
    input_pads = []
    schema = tool_info.get('input_schema', {})
    properties = schema.get('properties', {})
    required = schema.get('required', [])

    for prop_name, prop_def in properties.items():
        input_pads.append({
            'name': prop_name,
            'type': 'input',
            'description': prop_def.get('description', ''),
            'required': prop_name in required
        })

    return {
        'id': facet_id,
        'name': f"{tool_info['tool']}",
        'type': 'MCPFacet',
        'config': {
            'server': tool_info['server'],
            'tool': tool_info['tool']
        },
        'inputs': input_pads,
        'outputs': [
            {'name': 'result', 'type': 'output', 'description': 'Tool result'},
            {'name': 'error', 'type': 'output', 'description': 'Error message if failed'},
            {'name': 'out', 'type': 'output', 'description': 'Default output (same as result)'}
        ],
        'position': {'x': 0, 'y': 0},
        'color': '#9C27B0',  # Purple for MCP facets
        'enabled': True
    }
