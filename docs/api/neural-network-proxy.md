# NeuralNetworkProxy

class in Noodlings.Scripting

## Description

Proxy object for a neural network graph. Provides methods to create nodes, connect them, modify properties, and generate MLX code.

## Methods

| Method | Description |
|--------|-------------|
| [create_node()](#create_node) | Create a new node in the network |
| [remove_node()](#remove_node) | Remove a node from the network |
| [connect()](#connect) | Connect two nodes via ports |
| [disconnect()](#disconnect) | Disconnect two nodes |
| [get_node()](#get_node) | Get node information |
| [get_node_by_name()](#get_node_by_name) | Find node ID by name |
| [set_node_property()](#set_node_property) | Set a node property |
| [set_node_position()](#set_node_position) | Set node position in canvas |
| [generate_mlx_code()](#generate_mlx_code) | Generate MLX Python code from topology |
| [save()](#save) | Save network to .nncanvas file |
| [get_parameter_count()](#get_parameter_count) | Calculate total trainable parameters |

---

## create_node()

Create a new node in the network.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| node_type | string | Node type (e.g., "LSTM", "GRU", "Linear") |
| properties | object | Node properties (hidden_dim, position, etc.) |

**Returns:** Node ID (UUID string) or null on failure

**Example:**
```javascript
var lstm_id = network.create_node("LSTM", {
    hidden_dim: 32,
    position: [100, 200]
});
```

---

## remove_node()

Remove a node from the network.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| node_id | string | Node UUID |

**Returns:** true if removed successfully, false on error

**Example:**
```javascript
network.remove_node(lstm_id);
```

---

## connect()

Connect two nodes via their ports.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from_node | string | Source node ID |
| from_port | string | Source port name (e.g., "out") |
| to_node | string | Target node ID |
| to_port | string | Target port name (e.g., "input") |

**Returns:** true if connected successfully, false on error

**Example:**
```javascript
network.connect(lstm_id, "out", gru_id, "input");
```

---

## disconnect()

Disconnect two nodes.

**Parameters:** Same as connect()

**Returns:** true if disconnected successfully, false on error

**Example:**
```javascript
network.disconnect(lstm_id, "out", gru_id, "input");
```

---

## get_node()

Get node information.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| node_id | string | Node UUID |

**Returns:** Object with {id, type, name, properties, position} or null

**Example:**
```javascript
var node = network.get_node(lstm_id);
context.log("Hidden dim: " + node.properties.hidden_dim);
```

---

## get_node_by_name()

Find node ID by name.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| name | string | Node name (e.g., "Fast_LSTM") |

**Returns:** Node ID (UUID) or null if not found

**Example:**
```javascript
var fast_lstm_id = network.get_node_by_name("Fast_LSTM");
```

---

## set_node_property()

Set a node property.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| node_id | string | Node UUID |
| property_name | string | Property name (e.g., "hidden_dim") |
| value | any | New value |

**Returns:** true if set successfully, false on error

**Example:**
```javascript
network.set_node_property(lstm_id, "hidden_dim", 64);
```

---

## set_node_position()

Set node position in canvas.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| node_id | string | Node UUID |
| x | number | X coordinate |
| y | number | Y coordinate |

**Returns:** true if set successfully, false on error

**Example:**
```javascript
network.set_node_position(lstm_id, 150, 250);
```

---

## generate_mlx_code()

Generate MLX Python code from the visual topology.

**Parameters:** None

**Returns:** Python source code string or null on failure

**Example:**
```javascript
var code = network.generate_mlx_code();
context.log("Generated " + code.length + " characters");
```

---

## save()

Save network to .nncanvas file.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| filepath | string | Path to save file |

**Returns:** true if saved successfully, false on error

**Example:**
```javascript
network.save("custom_topology.nncanvas");
```

---

## get_parameter_count()

Calculate total trainable parameters in the network.

**Parameters:** None

**Returns:** Integer parameter count

**Example:**
```javascript
var params = network.get_parameter_count();
```
