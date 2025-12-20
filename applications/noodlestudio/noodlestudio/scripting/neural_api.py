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

            # Convert string to NodeType enum
            try:
                node_type_enum = NodeType[node_type.upper()]
            except KeyError:
                return None

            # Get node definition from dict
            node_def = NODE_DEFINITIONS.get(node_type_enum)
            if not node_def:
                return None

            # Extract position
            position = properties.pop('position', [0, 0])
            pos_x, pos_y = position[0], position[1]

            # Create node
            node_id = str(uuid.uuid4())
            node = NeuralNode(
                id=node_id,
                type=node_type_enum,
                name=node_def.get('default_name', node_type),
                position=(pos_x, pos_y)
            )

            # Set params from definition defaults
            if 'params' in node_def:
                for key, value in node_def['params'].items():
                    node.params[key] = value

            # Override with user-provided params
            for key, value in properties.items():
                node.params[key] = value

            # Add to graph
            self._graph.add_node(node)
            return node_id

        except Exception as e:
            print(f"[NeuralAPI] create_node error: {e}")
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
            console.log(node.params.hidden_dim);  // 32
        """
        node = self._graph.get_node_by_id(node_id)
        if not node:
            return None

        return {
            'id': node.id,
            'type': node.type.name,
            'name': node.name,
            'params': dict(node.params),
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
            if node and property_name in node.params:
                node.params[property_name] = value
                return True
            return False
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
            return False

    def get_parameter_count(self) -> int:
        """
        Calculate total trainable parameters.

        Returns:
            Parameter count
        """
        try:
            return self._graph.calculate_total_parameters()
        except Exception:
            return 0

    def list_nodes(self) -> List[Dict[str, Any]]:
        """
        List all nodes in network.

        Returns:
            List of {id, type, name, position}

        Example (JavaScript):
            var nodes = network.list_nodes();
            nodes.forEach(function(n) {
                console.log(n.name + " (" + n.type + ")");
            });
        """
        result = []
        for node in self._graph.nodes.values():
            result.append({
                'id': node.id,
                'type': node.type.name,
                'name': node.name,
                'position': list(node.position)
            })
        return result

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type (like Unity's GetComponents<T>).

        Args:
            node_type: Type name (e.g., "LSTM", "GRU", "LINEAR")

        Returns:
            List of node info dicts

        Example (JavaScript):
            var lstms = network.get_nodes_by_type("LSTM");
            lstms.forEach(function(n) {
                network.set_node_property(n.id, "hidden_dim", 64);
            });
        """
        result = []
        for node in self._graph.nodes.values():
            if node.type.name == node_type.upper():
                result.append({
                    'id': node.id,
                    'type': node.type.name,
                    'name': node.name,
                    'params': dict(node.params),
                    'position': list(node.position)
                })
        return result

    def find_nodes(self, predicate: dict) -> List[Dict[str, Any]]:
        """
        Find nodes matching criteria.

        Args:
            predicate: Dict of field->value to match
                      Supports: type, name

        Returns:
            List of matching node dicts

        Example (JavaScript):
            var fast_lstms = network.find_nodes({type: "LSTM", name: "Fast_LSTM"});
        """
        result = []
        for node in self._graph.nodes.values():
            match = True
            if 'type' in predicate and node.type.name != predicate['type'].upper():
                match = False
            if 'name' in predicate and node.name != predicate['name']:
                match = False
            if match:
                result.append({
                    'id': node.id,
                    'type': node.type.name,
                    'name': node.name,
                    'params': dict(node.params),
                    'position': list(node.position)
                })
        return result

    def get_connections(self) -> List[Dict[str, str]]:
        """
        Get all connections in the network.

        Returns:
            List of {from_node, from_port, to_node, to_port}
        """
        result = []
        for conn in self._graph.connections:
            result.append({
                'from_node': conn.from_node,
                'from_port': conn.from_port,
                'to_node': conn.to_node,
                'to_port': conn.to_port
            })
        return result

    def get_connections_from(self, node_id: str) -> List[Dict[str, str]]:
        """
        Get all connections originating from a node.

        Args:
            node_id: Source node ID

        Returns:
            List of connection dicts
        """
        result = []
        for conn in self._graph.connections:
            if conn.from_node == node_id:
                result.append({
                    'from_node': conn.from_node,
                    'from_port': conn.from_port,
                    'to_node': conn.to_node,
                    'to_port': conn.to_port
                })
        return result

    def get_connections_to(self, node_id: str) -> List[Dict[str, str]]:
        """
        Get all connections going to a node.

        Args:
            node_id: Target node ID

        Returns:
            List of connection dicts
        """
        result = []
        for conn in self._graph.connections:
            if conn.to_node == node_id:
                result.append({
                    'from_node': conn.from_node,
                    'from_port': conn.from_port,
                    'to_node': conn.to_node,
                    'to_port': conn.to_port
                })
        return result

    def duplicate_node(self, node_id: str, new_name: Optional[str] = None) -> Optional[str]:
        """
        Duplicate a node (like Unity's Instantiate).

        Args:
            node_id: ID of node to clone
            new_name: Optional name for the clone

        Returns:
            New node ID or None

        Example (JavaScript):
            var clone_id = network.duplicate_node(lstm_id, "Fast_LSTM_Clone");
        """
        try:
            import uuid

            original = self._graph.get_node_by_id(node_id)
            if not original:
                return None

            # Create clone with new position
            clone_id = self.create_node(
                original.type.name,
                position=[original.position[0] + 50, original.position[1] + 50],
                **{k: v for k, v in original.params.items()}
            )

            if clone_id and new_name:
                node = self._graph.get_node_by_id(clone_id)
                if node:
                    node.name = new_name

            return clone_id
        except Exception:
            return None

    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._graph.nodes)

    def get_connection_count(self) -> int:
        """Get total number of connections."""
        return len(self._graph.connections)

    def get_inputs(self, node_id: str) -> List[Dict[str, str]]:
        """
        Get input ports for a node.

        Args:
            node_id: Node UUID

        Returns:
            List of {name, data_type, label}
        """
        try:
            from noodlestudio.core.neural_canvas.node_definitions import NODE_DEFINITIONS

            node = self._graph.get_node_by_id(node_id)
            if not node:
                return []

            # Get node definition from dict using type enum
            node_def = NODE_DEFINITIONS.get(node.type)
            if not node_def:
                return []

            inputs = node_def.get('inputs', {})
            result = []
            for name, port in inputs.items():
                result.append({
                    'name': name,
                    'data_type': port.data_type.name if hasattr(port, 'data_type') else 'TENSOR',
                    'label': port.label if hasattr(port, 'label') else name
                })
            return result
        except Exception:
            return []

    def get_outputs(self, node_id: str) -> List[Dict[str, str]]:
        """
        Get output ports for a node.

        Args:
            node_id: Node UUID

        Returns:
            List of {name, data_type, label}
        """
        try:
            from noodlestudio.core.neural_canvas.node_definitions import NODE_DEFINITIONS

            node = self._graph.get_node_by_id(node_id)
            if not node:
                return []

            # Get node definition from dict using type enum
            node_def = NODE_DEFINITIONS.get(node.type)
            if not node_def:
                return []

            outputs = node_def.get('outputs', {})
            result = []
            for name, port in outputs.items():
                result.append({
                    'name': name,
                    'data_type': port.data_type.name if hasattr(port, 'data_type') else 'TENSOR',
                    'label': port.label if hasattr(port, 'label') else name
                })
            return result
        except Exception:
            return []

    def set_node_name(self, node_id: str, name: str) -> bool:
        """
        Set node display name.

        Args:
            node_id: Node UUID
            name: New name

        Returns:
            True if set successfully
        """
        try:
            node = self._graph.get_node_by_id(node_id)
            if node:
                node.name = name
                return True
            return False
        except Exception:
            return False


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
        except Exception:
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

    def get_node_types(self) -> List[str]:
        """
        Get list of available node types.

        Returns:
            List of node type names

        Example (JavaScript):
            var types = context.noodle.neural.get_node_types();
            // ["LSTM", "GRU", "LINEAR", "SCRIPTED_NODE", ...]
        """
        from noodlestudio.core.neural_canvas.neural_node import NodeType
        return [nt.name for nt in NodeType]

    def register_custom_node(
        self,
        name: str,
        description: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        params: Dict[str, Any],
        script: str,
        color: str = '#607D8B'
    ) -> bool:
        """
        Register a new custom node type.

        Creates a SCRIPTED_NODE template that can be added via context menu.

        Args:
            name: Display name for the node (e.g., "My Custom Node")
            description: Brief description shown in help
            inputs: Dict of input port names to data types
                    {"a": "TENSOR", "b": "SCALAR"}
            outputs: Dict of output port names to data types
                     {"out": "TENSOR", "flag": "SCALAR"}
            params: Dict of default parameter values
                    {"multiplier": 1.0, "threshold": 0.5}
            script: JavaScript code to execute (receives inputs, params)
            color: Hex color for node header (default: blue-gray)

        Returns:
            True if registered successfully

        Example (JavaScript):
            context.noodle.neural.register_custom_node(
                "Clamp Node",
                "Clamps input between min and max",
                {"x": "TENSOR"},
                {"out": "TENSOR"},
                {"min_val": 0.0, "max_val": 1.0},
                `
                var val = inputs.x;
                var clamped = Math.max(params.min_val, Math.min(params.max_val, val));
                return { out: clamped };
                `,
                "#4CAF50"
            );
        """
        try:
            from noodlestudio.core.neural_canvas.node_definitions import (
                NODE_DEFINITIONS, _custom_node_registry
            )
            from noodlestudio.core.neural_canvas.neural_node import NodeType, DataType, Port

            # Create the custom node definition
            node_def = {
                'type': NodeType.SCRIPTED_NODE,
                'name': name,
                'description': description,
                'how_it_works': f'''CUSTOM NODE: {name}

{description}

INPUTS:
{chr(10).join(f"- {k}: {v}" for k, v in inputs.items()) if inputs else "None"}

OUTPUTS:
{chr(10).join(f"- {k}: {v}" for k, v in outputs.items()) if outputs else "None"}

PARAMS:
{chr(10).join(f"- {k}: {v}" for k, v in params.items()) if params else "None"}

SCRIPT:
{script[:200]}{"..." if len(script) > 200 else ""}''',
                'params': {
                    'script': script,
                    'num_inputs': len(inputs),
                    'num_outputs': len(outputs),
                    **params
                },
                'inputs': {
                    port_name: Port(
                        port_name,
                        DataType[dtype] if dtype in DataType.__members__ else DataType.TENSOR,
                        label=port_name.title()
                    )
                    for port_name, dtype in inputs.items()
                },
                'outputs': {
                    port_name: Port(
                        port_name,
                        DataType[dtype] if dtype in DataType.__members__ else DataType.TENSOR,
                        label=port_name.title()
                    )
                    for port_name, dtype in outputs.items()
                },
                'weights': {},
                'color': color,
                'icon': 'script',
                'is_custom': True,  # Mark as user-registered
                'custom_name': name
            }

            # Register in the custom node registry
            _custom_node_registry[name] = node_def
            print(f"[FACET] Registered custom node: {name}")
            return True

        except Exception as e:
            print(f"[FACET] Failed to register custom node: {e}")
            return False

    def get_custom_nodes(self) -> List[str]:
        """
        Get list of registered custom node names.

        Returns:
            List of custom node names

        Example (JavaScript):
            var customs = context.noodle.neural.get_custom_nodes();
            // ["Clamp Node", "My Gate", ...]
        """
        try:
            from noodlestudio.core.neural_canvas.node_definitions import _custom_node_registry
            return list(_custom_node_registry.keys())
        except Exception:
            return []

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with method names as keys
        """
        return {
            'get_network': '__neural_get_network__',
            'load': '__neural_load__',
            'create_network': '__neural_create_network__',
            'get_node_types': '__neural_get_node_types__',
            'register_custom_node': '__neural_register_custom_node__',
            'get_custom_nodes': '__neural_get_custom_nodes__'
        }
