# get_node()

Get node information.

**Class**: NeuralNetworkProxy

**Access**: `network.get_node(node_id)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `node_id` | string | Node UUID |

## Returns

Object with `{id, type, name, properties, position}` or `null`

## Example

```javascript
var node = network.get_node(lstm_id);

if (node) {
    context.log("Type: " + node.type);
    context.log("Name: " + node.name);
    context.log("Hidden dim: " + node.properties.hidden_dim);
    context.log("Position: [" + node.position[0] + ", " + node.position[1] + "]");
}
```

## See Also

- [get_node_by_name()](get_node_by_name.md) - Find by name
- [set_node_property()](set_node_property.md) - Modify properties
- [create_node()](create_node.md) - Create nodes
