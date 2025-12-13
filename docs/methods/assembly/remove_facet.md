# remove_facet()

Remove facet from the assembly.

**Class**: FacetAssemblyProxy

**Access**: `assembly.remove_facet(facet_id)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `facet_id` | string | Facet UUID or name |

## Returns

`true` if removed, `false` if not found.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

// Remove by UUID
var removed = assembly.remove_facet("550e8400-e29b-41d4-a716-446655440000");

// Remove by name
var removed = assembly.remove_facet("Old Reasoner");

if (removed) {
    context.log("Facet removed successfully");
    assembly.save("modified_agent.yaml");
}
```

## Warning

Removing a facet will disconnect all its connections. Make sure to rewire the topology if needed.

## See Also

- [add_facet()](add_facet.md) - Add new facet
- [disconnect()](disconnect.md) - Disconnect specific connections
- [save()](save.md) - Save changes
