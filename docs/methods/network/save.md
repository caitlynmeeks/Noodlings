# save()

Save network to .nncanvas file.

**Class**: NeuralNetworkProxy

**Access**: `network.save(filepath)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `filepath` | string | Path to save file (e.g., "custom.nncanvas") |

## Returns

`true` if saved successfully, `false` on error

## Format

JSON file with nodes, connections, and metadata

## Example

```javascript
var saved = network.save("modified_network.nncanvas");

if (saved) {
    context.log("Network saved successfully");
}
```

## Warning

Don't overwrite default topologies. Use unique filenames:

```javascript
// Good - create new file
network.save("custom_topologies/my_network.nncanvas");

// Bad - overwrites default
network.save("facet_assemblies/charm_networks/default.nncanvas");
```

## See Also

- [load()](../neural/load.md) - Load saved network
- [generate_mlx_code()](generate_mlx_code.md) - Export as code
