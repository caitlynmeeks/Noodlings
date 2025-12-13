# disconnect()

Remove connection between facets.

**Class**: FacetAssemblyProxy

**Access**: `assembly.disconnect(from_facet, from_pad, to_facet, to_pad)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `from_facet` | string | Source facet UUID or name |
| `from_pad` | string | Output pad name |
| `to_facet` | string | Target facet UUID or name |
| `to_pad` | string | Input pad name |

## Returns

`true` if disconnected, `false` if connection not found.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

// Disconnect specific connection
var disconnected = assembly.disconnect(
    "CHARM_NET",
    "affect_valence",
    "Red's Mind",
    "affect_input"
);

if (disconnected) {
    context.log("Connection removed");
    assembly.save("modified_topology.yaml");
}
```

## See Also

- [connect()](connect.md) - Create connection
- [remove_facet()](remove_facet.md) - Remove entire facet
- [save()](save.md) - Save changes
