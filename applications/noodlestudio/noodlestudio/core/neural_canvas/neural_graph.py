"""
Neural Graph - Complete neural network topology representation.

Manages nodes, connections, validation, and serialization.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Set, Tuple, Optional
import json
from datetime import datetime
from .neural_node import NeuralNode, NodeType, Connection


@dataclass
class ValidationResult:
    """Result of graph validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self):
        if self.valid:
            return "✅ Valid"
        else:
            msg = "❌ Invalid:\n"
            for error in self.errors:
                msg += f"  - {error}\n"
            if self.warnings:
                msg += "⚠️ Warnings:\n"
                for warning in self.warnings:
                    msg += f"  - {warning}\n"
            return msg


class NeuralGraph:
    """
    Complete neural network topology.

    Manages nodes, connections, hidden states, and provides validation,
    serialization, and code generation capabilities.
    """

    def __init__(self):
        self.nodes: Dict[str, NeuralNode] = {}  # node_id -> NeuralNode
        self.connections: List[Connection] = []
        self.hidden_states: Dict[str, Dict[str, Any]] = {}  # state_id -> {shape, initial_value}

        # Metadata
        self.name: str = "Untitled Network"
        self.description: str = ""
        self.version: str = "1.0"
        self.created: Optional[datetime] = None
        self.modified: Optional[datetime] = None
        self.author: str = ""

        # Export settings
        self.export_targets: Dict[str, bool] = {
            'mlx': True,
            'pytorch': False,
            'onnx': False
        }

    def add_node(self, node: NeuralNode) -> str:
        """
        Add a node to the graph.

        Args:
            node: NeuralNode to add

        Returns:
            Node ID
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists")

        self.nodes[node.id] = node
        self.modified = datetime.now()
        return node.id

    def remove_node(self, node_id: str):
        """Remove a node and all its connections."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        # Remove all connections involving this node
        self.connections = [
            conn for conn in self.connections
            if conn.from_node != node_id and conn.to_node != node_id
        ]

        del self.nodes[node_id]
        self.modified = datetime.now()

    def add_connection(self, connection: Connection):
        """
        Add a connection between nodes.

        Args:
            connection: Connection to add

        Raises:
            ValueError: If nodes don't exist or ports invalid
        """
        # Validate nodes exist
        if connection.from_node not in self.nodes:
            raise ValueError(f"Source node {connection.from_node} not found")
        if connection.to_node not in self.nodes:
            raise ValueError(f"Target node {connection.to_node} not found")

        from_node = self.nodes[connection.from_node]
        to_node = self.nodes[connection.to_node]

        # Validate ports exist
        if connection.from_port not in from_node.outputs:
            raise ValueError(f"Output port {connection.from_port} not found on {from_node.name}")
        if connection.to_port not in to_node.inputs:
            raise ValueError(f"Input port {connection.to_port} not found on {to_node.name}")

        # Check for duplicate connections
        for existing in self.connections:
            if (existing.to_node == connection.to_node and
                existing.to_port == connection.to_port):
                raise ValueError(f"Port {connection.to_port} on {to_node.name} already connected")

        self.connections.append(connection)
        self.modified = datetime.now()

    def remove_connection(self, from_node: str, from_port: str, to_node: str, to_port: str):
        """Remove a specific connection."""
        self.connections = [
            conn for conn in self.connections
            if not (conn.from_node == from_node and
                   conn.from_port == from_port and
                   conn.to_node == to_node and
                   conn.to_port == to_port)
        ]
        self.modified = datetime.now()

    def get_node_by_id(self, node_id: str) -> Optional[NeuralNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[NeuralNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.type == node_type]

    def get_input_nodes(self) -> List[NeuralNode]:
        """Get all INPUT nodes."""
        return self.get_nodes_by_type(NodeType.INPUT)

    def get_output_nodes(self) -> List[NeuralNode]:
        """Get all OUTPUT nodes."""
        return self.get_nodes_by_type(NodeType.OUTPUT)

    def get_connections_from_node(self, node_id: str) -> List[Connection]:
        """Get all connections originating from a node."""
        return [conn for conn in self.connections if conn.from_node == node_id]

    def get_connections_to_node(self, node_id: str) -> List[Connection]:
        """Get all connections targeting a node."""
        return [conn for conn in self.connections if conn.to_node == node_id]

    def validate(self) -> ValidationResult:
        """
        Validate the entire graph.

        Checks:
        - At least one entry point (INPUT or NUMBER_INPUT)
        - At least one exit point (OUTPUT or THRESHOLD_OUTPUT)
        - No cycles (DAG requirement)
        - All input ports connected
        - Type compatibility
        - Dimension matching
        - Parameter validation
        """
        result = ValidationResult(valid=True)

        # Check for entry points (INPUT or NUMBER_INPUT nodes)
        input_nodes = self.get_input_nodes()
        number_input_nodes = self.get_nodes_by_type(NodeType.NUMBER_INPUT)
        entry_points = input_nodes + number_input_nodes

        if len(entry_points) == 0:
            result.valid = False
            result.errors.append("No entry point found (need INPUT or NUMBER_INPUT)")

        # Check for exit points (OUTPUT or THRESHOLD_OUTPUT nodes)
        output_nodes = self.get_output_nodes()
        threshold_output_nodes = self.get_nodes_by_type(NodeType.THRESHOLD_OUTPUT)
        exit_points = output_nodes + threshold_output_nodes

        if len(exit_points) == 0:
            result.valid = False
            result.errors.append("No exit point found (need OUTPUT or THRESHOLD_OUTPUT)")

        # Check for cycles
        if self._has_cycle():
            result.valid = False
            result.errors.append("Graph contains cycles (must be DAG)")

        # Validate each node's parameters
        for node in self.nodes.values():
            param_errors = node.validate_params()
            if param_errors:
                result.valid = False
                for error in param_errors:
                    result.errors.append(f"Node '{node.name}': {error}")

        # Check all required input ports are connected
        for node in self.nodes.values():
            for port_name, port in node.inputs.items():
                if port.required:
                    connected = any(
                        conn.to_node == node.id and conn.to_port == port_name
                        for conn in self.connections
                    )
                    if not connected:
                        result.valid = False
                        result.errors.append(
                            f"Node '{node.name}': required port '{port_name}' not connected"
                        )

        # Check for unused nodes (except entry/exit points)
        entry_exit_types = (NodeType.INPUT, NodeType.OUTPUT,
                           NodeType.NUMBER_INPUT, NodeType.THRESHOLD_OUTPUT)
        for node in self.nodes.values():
            if node.type in entry_exit_types:
                continue

            has_incoming = any(conn.to_node == node.id for conn in self.connections)
            has_outgoing = any(conn.from_node == node.id for conn in self.connections)

            if not has_incoming and not has_outgoing:
                result.warnings.append(f"Node '{node.name}' is not connected")

        # Warn about untrained weights
        for node in self.nodes.values():
            for weight_info in node.weights.values():
                if weight_info.trainable and not weight_info.path:
                    result.warnings.append(
                        f"Node '{node.name}': weight '{weight_info.name}' has no checkpoint path (untrained)"
                    )

        return result

    def _has_cycle(self) -> bool:
        """
        Check if graph contains cycles using DFS.

        Returns:
            True if cycle detected
        """
        # Build adjacency list
        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for conn in self.connections:
            adjacency[conn.from_node].append(conn.to_node)

        # DFS with recursion stack
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in adjacency[node_id]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True  # Cycle detected

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def topological_sort(self) -> List[str]:
        """
        Get topological ordering of nodes (for execution order).

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        if self._has_cycle():
            raise ValueError("Cannot topologically sort graph with cycles")

        # Build adjacency list and in-degree map
        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections:
            adjacency[conn.from_node].append(conn.to_node)
            in_degree[conn.to_node] += 1

        # Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")

        return result

    def compute_total_parameters(self) -> int:
        """Calculate total trainable parameters in the network."""
        total = 0
        for node in self.nodes.values():
            total += node.compute_num_parameters()
        return total

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to dictionary (for .nncanvas JSON).

        Returns:
            Dictionary representation
        """
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'metadata': {
                'created': self.created.isoformat() if self.created else None,
                'modified': self.modified.isoformat() if self.modified else datetime.now().isoformat(),
                'author': self.author,
                'total_parameters': self.compute_total_parameters()
            },
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'connections': [
                {
                    'from_node': conn.from_node,
                    'from_port': conn.from_port,
                    'to_node': conn.to_node,
                    'to_port': conn.to_port
                }
                for conn in self.connections
            ],
            'hidden_states': self.hidden_states,
            'export_targets': self.export_targets
        }

    def to_json(self, filepath: str):
        """
        Save graph to .nncanvas JSON file.

        Args:
            filepath: Path to save to (should end with .nncanvas)
        """
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'NeuralGraph':
        """
        Deserialize graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            NeuralGraph instance
        """
        graph = NeuralGraph()

        graph.version = data.get('version', '1.0')
        graph.name = data.get('name', 'Untitled Network')
        graph.description = data.get('description', '')

        metadata = data.get('metadata', {})
        graph.author = metadata.get('author', '')
        if metadata.get('created'):
            graph.created = datetime.fromisoformat(metadata['created'])
        if metadata.get('modified'):
            graph.modified = datetime.fromisoformat(metadata['modified'])

        # Deserialize nodes
        for node_data in data.get('nodes', []):
            node = NeuralNode.from_dict(node_data)
            graph.nodes[node.id] = node

        # Deserialize connections
        for conn_data in data.get('connections', []):
            connection = Connection(
                from_node=conn_data['from_node'],
                from_port=conn_data['from_port'],
                to_node=conn_data['to_node'],
                to_port=conn_data['to_port']
            )
            graph.connections.append(connection)

        # Deserialize hidden states
        graph.hidden_states = data.get('hidden_states', {})

        # Deserialize export targets
        graph.export_targets = data.get('export_targets', {
            'mlx': True,
            'pytorch': False,
            'onnx': False
        })

        return graph

    @staticmethod
    def from_json(filepath: str) -> 'NeuralGraph':
        """
        Load graph from .nncanvas JSON file.

        Args:
            filepath: Path to .nncanvas file

        Returns:
            NeuralGraph instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return NeuralGraph.from_dict(data)

    def __str__(self):
        return (f"NeuralGraph('{self.name}', "
                f"{len(self.nodes)} nodes, "
                f"{len(self.connections)} connections, "
                f"{self.compute_total_parameters()} params)")
