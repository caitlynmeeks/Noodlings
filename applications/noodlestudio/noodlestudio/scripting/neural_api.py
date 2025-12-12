"""
Neural API - Scriptable interface to Neural Canvas.

Provides JavaScript-accessible methods for:
- Creating/removing nodes
- Connecting ports
- Setting node properties
- Generating MLX code
- Loading/saving .nncanvas files

Part of the unified Noodlings scripting API (context.noodle.neural).

Author: Commander Spock + Cadet Caity
Date: December 10, 2025
"""

from typing import Dict, List, Optional, Any, Tuple


class NeuralNetworkProxy:
    """
    Proxy object for a single neural network graph.

    Provides JavaScript-friendly interface to NeuralGraph.
    """

    def __init__(self, graph):
        """
        Initialize proxy.

        Args:
            graph: NeuralGraph instance
        """
        self._graph = graph

    def create_node(self, node_type: str, **properties) -> Optional[str]:
        """
        Create a new node in the network.

        Args:
            node_type: Node type (e.g., "LSTM", "GRU", "Linear")
            **properties: Node properties (hidden_dim, position, etc.)

        Returns:
            Node ID if created, None on failure

        Example (JavaScript):
            var node_id = network.create_node("LSTM", {
                hidden_dim: 32,
                position: [100, 200]
            });
        """
        try:
            from noodlestudio.core.neural_canvas.neural_node import NeuralNode, NodeType
            from noodlestudio.core.neural_canvas.node_definitions import NODE_DEFINITIONS
            import uuid

            # Find node definition
            node_def = None
            for definition in NODE_DEFINITIONS:
                if definition['id'].upper() == node_type.upper():
                    node_def = definition
                    break

            if not node_def:
                return None

            # Extract position
            position = properties.pop('position', [0, 0])
            pos_x, pos_y = position[0], position[1]

            # Create node
            node_id = str(uuid.uuid4())
            node = NeuralNode(
                id=node_id,
                type=NodeType[node_def['id']],
                name=node_def['default_name'],
                position=(pos_x, pos_y)
            )

            # Set properties
            for key, value in properties.items():
                if key in node.properties:
                    node.properties[key] = value

            # Add to graph
            self._graph.add_node(node)
            return node_id

        except Exception as e:
            return None

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the network.

        Args:
            node_id: Node UUID

        Returns:
            True if removed successfully
        """
        try:
            self._graph.remove_node(node_id)
            return True
        except:
            return False

    def connect(self, from_node: str, from_port: str, to_node: str, to_port: str) -> bool:
        """
        Connect two nodes.

        Args:
            from_node: Source node ID
            from_port: Source port name
            to_node: Target node ID
            to_port: Target port name

        Returns:
            True if connected successfully

        Example (JavaScript):
            network.connect(lstm_id, "out", gru_id, "input");
        """
        try:
            from noodlestudio.core.neural_canvas.neural_node import Connection

            conn = Connection(
                from_node=from_node,
                from_port=from_port,
                to_node=to_node,
                to_port=to_port
            )
            self._graph.add_connection(conn)
            return True
        except:
            return False

    def disconnect(self, from_node: str, from_port: str, to_node: str, to_port: str) -> bool:
        """
        Disconnect two nodes.

        Args:
            from_node: Source node ID
            from_port: Source port name
            to_node: Target node ID
            to_port: Target port name

        Returns:
            True if disconnected successfully
        """
        try:
            self._graph.remove_connection(from_node, from_port, to_node, to_port)
            return True
        except:
            return False

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node information.

        Args:
            node_id: Node UUID

        Returns:
            Dict with node info {id, type, name, properties, position} or None

        Example (JavaScript):
            var node = network.get_node(lstm_id);
            console.log(node.properties.hidden_dim);  // 32
        """
        node = self._graph.get_node_by_id(node_id)
        if not node:
            return None

        return {
            'id': node.id,
            'type': node.type.name,
            'name': node.name,
            'properties': dict(node.properties),
            'position': list(node.position)
        }

    def get_node_by_name(self, name: str) -> Optional[str]:
        """
        Find node ID by name.

        Args:
            name: Node name (e.g., "Fast_LSTM")

        Returns:
            Node ID or None if not found
        """
        for node in self._graph.nodes.values():
            if node.name == name:
                return node.id
        return None

    def set_node_property(self, node_id: str, property_name: str, value: Any) -> bool:
        """
        Set a node property.

        Args:
            node_id: Node UUID
            property_name: Property name (e.g., "hidden_dim")
            value: New value

        Returns:
            True if set successfully

        Example (JavaScript):
            network.set_node_property(lstm_id, "hidden_dim", 64);
        """
        try:
            node = self._graph.get_node_by_id(node_id)
            if node and property_name in node.properties:
                node.properties[property_name] = value
                return True
            return False
        except:
            return False

    def set_node_position(self, node_id: str, x: float, y: float) -> bool:
        """
        Set node position in canvas.

        Args:
            node_id: Node UUID
            x: X coordinate
            y: Y coordinate

        Returns:
            True if set successfully
        """
        try:
            node = self._graph.get_node_by_id(node_id)
            if node:
                node.position = (x, y)
                return True
            return False
        except:
            return False

    def generate_mlx_code(self) -> Optional[str]:
        """
        Generate MLX Python code from topology.

        Returns:
            Python source code string or None on failure

        Example (JavaScript):
            var code = network.generate_mlx_code();
            context.log("Generated " + code.length + " characters of code");
        """
        try:
            from noodlestudio.core.neural_canvas.mlx_codegen import MLXCodeGenerator
            generator = MLXCodeGenerator(self._graph)
            return generator.generate()
        except:
            return None

    def save(self, filepath: str) -> bool:
        """
        Save network to .nncanvas file.

        Args:
            filepath: Path to save file

        Returns:
            True if saved successfully

        Example (JavaScript):
            network.save("custom_topology.nncanvas");
        """
        try:
            self._graph.save(filepath)
            return True
        except:
            return False

    def get_parameter_count(self) -> int:
        """
        Calculate total trainable parameters.

        Returns:
            Parameter count
        """
        try:
            return self._graph.calculate_total_parameters()
        except:
            return 0


class NeuralAPI:
    """
    Scriptable interface to Neural Canvas system.

    Available to JavaScript via context.noodle.neural
    """

    def __init__(self):
        """Initialize Neural API."""
        self._graphs: Dict[str, Any] = {}  # graph_id -> NeuralGraph

    def get_network(self, graph_id: str) -> Optional[NeuralNetworkProxy]:
        """
        Get network by ID.

        Args:
            graph_id: Graph UUID

        Returns:
            NeuralNetworkProxy or None

        Example (JavaScript):
            var network = context.noodle.neural.get_network(graph_id);
            network.create_node("LSTM", {hidden_dim: 32});
        """
        if graph_id in self._graphs:
            return NeuralNetworkProxy(self._graphs[graph_id])
        return None

    def load(self, filepath: str) -> Optional[NeuralNetworkProxy]:
        """
        Load network from .nncanvas file.

        Args:
            filepath: Path to .nncanvas file

        Returns:
            NeuralNetworkProxy or None

        Example (JavaScript):
            var network = context.noodle.neural.load("custom.nncanvas");
        """
        try:
            from noodlestudio.core.neural_canvas.neural_graph import NeuralGraph
            graph = NeuralGraph.load(filepath)
            # Store with filename as ID
            import os
            graph_id = os.path.basename(filepath)
            self._graphs[graph_id] = graph
            return NeuralNetworkProxy(graph)
        except:
            return None

    def create_network(self, name: str = "Untitled") -> NeuralNetworkProxy:
        """
        Create a new empty network.

        Args:
            name: Network name

        Returns:
            NeuralNetworkProxy

        Example (JavaScript):
            var network = context.noodle.neural.create_network("MyNetwork");
        """
        from noodlestudio.core.neural_canvas.neural_graph import NeuralGraph
        graph = NeuralGraph()
        graph.name = name
        self._graphs[name] = graph
        return NeuralNetworkProxy(graph)

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with method names as keys
        """
        return {
            'get_network': '__neural_get_network__',
            'load': '__neural_load__',
            'create_network': '__neural_create_network__'
        }
