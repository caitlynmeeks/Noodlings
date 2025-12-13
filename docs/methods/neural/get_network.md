# get_network()

Get network by ID (for networks created in current session).

**Class**: NeuralAPI

**Access**: `context.noodle.neural.get_network(graph_id)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `graph_id` | string | Graph UUID or name |

## Returns

NeuralNetworkProxy instance or `null`

## Example

```javascript
var network = context.noodle.neural.get_network("default");

if (network) {
    context.log("Found network");
    var params = network.get_parameter_count();
    context.log("Parameters: " + params);
}
```

## See Also

- [create_network()](create_network.md) - Create new network
- [load()](load.md) - Load from file
- [NeuralAPI](../../api/neural-api.md) - Complete class reference
