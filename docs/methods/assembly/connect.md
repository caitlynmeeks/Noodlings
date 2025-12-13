# connect()

Connect output pad of one facet to input pad of another.

**Class**: FacetAssemblyProxy

**Access**: `assembly.connect(from_facet, from_pad, to_facet, to_pad)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `from_facet` | string | Source facet UUID or name |
| `from_pad` | string | Output pad name |
| `to_facet` | string | Target facet UUID or name |
| `to_pad` | string | Input pad name |

## Returns

`true` if connected, `false` on failure.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

// Connect CHARM_NET affect output to reasoner input
var connected = assembly.connect(
    "CHARM_NET",
    "affect_valence",
    "Red's Mind",
    "affect_input"
);

if (connected) {
    context.log("Facets connected successfully");
    assembly.save("modified_topology.yaml");
}

// Chain multiple facets
assembly.connect("INCOMING", "data", "Filter", "input");
assembly.connect("Filter", "output", "Reasoner", "data");
assembly.connect("Reasoner", "result", "OUTGOING", "text");
```

## See Also

- [disconnect()](disconnect.md) - Remove connection
- [add_facet()](add_facet.md) - Add facets to connect
- [save()](save.md) - Save topology
