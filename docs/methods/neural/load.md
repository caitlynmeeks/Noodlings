# load()

Load a network from .nncanvas file.

**Class**: NeuralAPI

**Access**: `context.noodle.neural.load(filepath)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `filepath` | string | Path to .nncanvas file |

## Returns

NeuralNetworkProxy instance or `null` on failure

## Example

```javascript
var network = context.noodle.neural.load("facet_assemblies/charm_networks/default.nncanvas");

if (network) {
    var params = network.get_parameter_count();
    context.log("Loaded network with " + params + " parameters");
}
```

## See Also

- [create_network()](create_network.md) - Create new network
- [get_network()](get_network.md) - Get by ID
- [NeuralAPI](../../api/neural-api.md) - Complete class reference
