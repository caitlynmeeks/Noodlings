# NeuralAPI

Scriptable interface to Neural Canvas system.

**Location**: `noodlestudio/scripting/neural_api.py`

**Access**: `context.noodle.neural`

## Overview

The NeuralAPI allows scripts to:

- Create, load, and save neural network topologies
- Add/remove nodes (LSTM, GRU, Linear, etc.)
- Connect ports between nodes
- Modify node properties (hidden dimensions, etc.)
- Generate MLX Python code from visual topology
- Calculate parameter counts

All modifications happen through **NeuralNetworkProxy** objects.

## NeuralAPI Methods

### `create_network(name)`

Create a new empty neural network.

**Parameters**:

- `name` (string, optional) - Network name (default: `"Untitled"`)

**Returns**: `NeuralNetworkProxy` instance

**Example**:
```javascript
var network = context.noodle.neural.create_network("MyNetwork");
context.log("Created network: " + network);
```

---

### `load(filepath)`

Load a network from .nncanvas file.

**Parameters**:

- `filepath` (string) - Path to `.nncanvas` file

**Returns**: `NeuralNetworkProxy` instance or `null` on failure

**Example**:
```javascript
var network = context.noodle.neural.load("facet_assemblies/charm_networks/default.nncanvas");
if (network) {
    var params = network.get_parameter_count();
    context.log("Loaded network with " + params + " parameters");
}
```

---

### `get_network(graph_id)`

Get network by ID (for networks created in current session).

**Parameters**:

- `graph_id` (string) - Graph UUID or name

**Returns**: `NeuralNetworkProxy` instance or `null`

**Example**:
```javascript
var network = context.noodle.neural.get_network("default");
if (network) {
    context.log("Found network");
}
```

---

## NeuralNetworkProxy Methods

Proxy object for a single neural network graph.

### `create_node(node_type, properties)`

Create a new node in the network.

**Parameters**:

- `node_type` (string) - Node type (e.g., `"LSTM"`, `"GRU"`, `"Linear"`)
- `properties` (object) - Node properties:
    - `hidden_dim` (number) - Hidden dimension size
    - `position` (array) - Canvas position `[x, y]`
    - Additional properties depending on node type

**Returns**: Node ID (UUID string) or `null` on failure

**Example**:
```javascript
var lstm_id = network.create_node("LSTM", {
    hidden_dim: 32,
    position: [100, 200]
});

var gru_id = network.create_node("GRU", {
    hidden_dim: 16,
    position: [300, 200]
});

context.log("Created LSTM: " + lstm_id);
```

**Supported Node Types**: See [Node Types Reference](../node-types.md) for complete list (26 types)

---

### `remove_node(node_id)`

Remove a node from the network.

**Parameters**:

- `node_id` (string) - Node UUID

**Returns**: `true` if removed successfully, `false` on error

**Example**:
```javascript
var removed = network.remove_node(lstm_id);
if (removed) {
    context.log("Node removed");
}
```

---

### `connect(from_node, from_port, to_node, to_port)`

Connect two nodes via their ports.

**Parameters**:

- `from_node` (string) - Source node ID
- `from_port` (string) - Source port name (e.g., `"out"`)
- `to_node` (string) - Target node ID
- `to_port` (string) - Target port name (e.g., `"input"`)

**Returns**: `true` if connected successfully, `false` on error

**Example**:
```javascript
// Connect LSTM output to GRU input
var success = network.connect(lstm_id, "out", gru_id, "input");
if (success) {
    context.log("Connected LSTM → GRU");
}
```

**Common Port Names**:

- Recurrent nodes (LSTM/GRU): `input`, `out`
- Linear nodes: `input`, `out`
- Affect Head: `hidden_state`, `affect_output`

---

### `disconnect(from_node, from_port, to_node, to_port)`

Disconnect two nodes.

**Parameters**: Same as `connect()`

**Returns**: `true` if disconnected successfully, `false` on error

**Example**:
```javascript
network.disconnect(lstm_id, "out", gru_id, "input");
```

---

### `get_node(node_id)`

Get node information.

**Parameters**:

- `node_id` (string) - Node UUID

**Returns**: Object with `{id, type, name, properties, position}` or `null`

**Example**:
```javascript
var node = network.get_node(lstm_id);
if (node) {
    context.log("Type: " + node.type);
    context.log("Name: " + node.name);
    context.log("Hidden dim: " + node.properties.hidden_dim);
    context.log("Position: [" + node.position[0] + ", " + node.position[1] + "]");
}
```

---

### `get_node_by_name(name)`

Find node ID by name.

**Parameters**:

- `name` (string) - Node name (e.g., `"Fast_LSTM"`, `"Slow_GRU"`)

**Returns**: Node ID (UUID) or `null` if not found

**Example**:
```javascript
var fast_lstm_id = network.get_node_by_name("Fast_LSTM");
if (fast_lstm_id) {
    var node = network.get_node(fast_lstm_id);
    context.log("Found Fast LSTM with hidden_dim: " + node.properties.hidden_dim);
}
```

**Note**: Node names can be changed in Neural Canvas by double-clicking

---

### `set_node_property(node_id, property_name, value)`

Set a node property.

**Parameters**:

- `node_id` (string) - Node UUID
- `property_name` (string) - Property name (e.g., `"hidden_dim"`)
- `value` (any) - New value

**Returns**: `true` if set successfully, `false` on error

**Example**:
```javascript
// Double the hidden dimension of an LSTM
var node = network.get_node(lstm_id);
var new_dim = node.properties.hidden_dim * 2;

var success = network.set_node_property(lstm_id, "hidden_dim", new_dim);
if (success) {
    context.log("Increased hidden_dim to " + new_dim);
}
```

**Common Properties**:

- `hidden_dim` - Hidden layer size (LSTM, GRU)
- `num_heads` - Attention heads (MultiHeadAttention)
- `dropout` - Dropout rate
- `activation` - Activation function

---

### `set_node_position(node_id, x, y)`

Set node position in canvas.

**Parameters**:

- `node_id` (string) - Node UUID
- `x` (number) - X coordinate
- `y` (number) - Y coordinate

**Returns**: `true` if set successfully, `false` on error

**Example**:
```javascript
network.set_node_position(lstm_id, 150, 250);
```

---

### `generate_mlx_code()`

Generate MLX Python code from the visual topology.

**Parameters**: None

**Returns**: Python source code string or `null` on failure

**Example**:
```javascript
var code = network.generate_mlx_code();
if (code) {
    context.log("Generated " + code.length + " characters of MLX code");
    // Code is a complete Python class with forward() method
}
```

**Generated Code Structure**:
```python
import mlx.core as mx
import mlx.nn as nn

class GeneratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_lstm = nn.LSTM(input_dim=..., hidden_dim=16)
        self.medium_lstm = nn.LSTM(input_dim=..., hidden_dim=16)
        # ... etc

    def forward(self, x):
        # Generated forward pass
        return output
```

---

### `save(filepath)`

Save network to .nncanvas file.

**Parameters**:

- `filepath` (string) - Path to save file (e.g., `"custom_topology.nncanvas"`)

**Returns**: `true` if saved successfully, `false` on error

**Example**:
```javascript
var saved = network.save("modified_network.nncanvas");
if (saved) {
    context.log("Network saved successfully");
}
```

**Format**: JSON file with nodes, connections, and metadata

---

### `get_parameter_count()`

Calculate total trainable parameters in the network.

**Parameters**: None

**Returns**: Integer parameter count

**Example**:
```javascript
var params = network.get_parameter_count();
context.log("Network has " + params + " trainable parameters");
```

**Calculation**: Sums all weight matrices and bias vectors across all nodes

---

## Complete Example: Procedural Network Generation

```javascript
function process(inputs, context) {
    // Create network
    var network = context.noodle.neural.create_network("AdaptiveNetwork");

    // Create input node
    var input_id = network.create_node("Input", {
        output_dim: 64,
        position: [50, 200]
    });

    // Create LSTM layers based on complexity requirement
    var depth = inputs.required_depth || 2;
    var prev_id = input_id;

    for (var i = 0; i < depth; i++) {
        var lstm_id = network.create_node("LSTM", {
            hidden_dim: 32,
            position: [200 + (i * 150), 200]
        });

        // Connect previous node to this LSTM
        network.connect(prev_id, "out", lstm_id, "input");
        prev_id = lstm_id;
    }

    // Create output node
    var output_id = network.create_node("Linear", {
        output_dim: 5,  // 5 affect dimensions
        position: [200 + (depth * 150), 200]
    });

    network.connect(prev_id, "out", output_id, "input");

    // Generate code
    var code = network.generate_mlx_code();
    var params = network.get_parameter_count();

    context.log("Generated " + depth + "-layer network with " + params + " parameters");

    // Save
    network.save("adaptive_network_depth" + depth + ".nncanvas");

    return {
        depth: depth,
        parameters: params,
        code_length: code ? code.length : 0
    };
}
```

## Complete Example: Topology Inspector

```javascript
function process(inputs, context) {
    // Load default CharmNetwork topology
    var network = context.noodle.neural.load("facet_assemblies/charm_networks/default.nncanvas");

    if (!network) {
        context.log("ERROR: Could not load network");
        return {error: true};
    }

    // Analyze structure
    var fast_lstm = network.get_node_by_name("Fast_LSTM");
    var medium_lstm = network.get_node_by_name("Medium_LSTM");
    var slow_gru = network.get_node_by_name("Slow_GRU");

    context.log("=== CharmNetwork Analysis ===");

    if (fast_lstm) {
        var node = network.get_node(fast_lstm);
        context.log("Fast LSTM: " + node.properties.hidden_dim + "-D");
    }

    if (medium_lstm) {
        var node = network.get_node(medium_lstm);
        context.log("Medium LSTM: " + node.properties.hidden_dim + "-D");
    }

    if (slow_gru) {
        var node = network.get_node(slow_gru);
        context.log("Slow GRU: " + node.properties.hidden_dim + "-D");
    }

    var total_params = network.get_parameter_count();
    context.log("Total parameters: " + total_params);

    return {
        total_parameters: total_params,
        analyzed: true
    };
}
```

## See Also

- [Node Types Reference](../node-types.md) - Complete list of 26 node types
- [ModelsAPI Reference](models-api.md)
- [AgentsAPI Reference](agents-api.md)
- [Complete Examples](../examples.md)
