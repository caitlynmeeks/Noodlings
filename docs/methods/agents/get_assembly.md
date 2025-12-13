# get_assembly()

Get facet assembly for an agent by UUID or name.

**Class**: AgentsAPI

**Access**: `context.noodle.agents.get_assembly(identifier)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `identifier` | string | Agent UUID or name (e.g., "red-fire-anklebiter") |

## Returns

[FacetAssemblyProxy](../../api/facet-assembly-proxy.md) object or `null` if not found.

## Example

```javascript
// By name
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

// By UUID
var assembly = context.noodle.agents.get_assembly("550e8400-e29b-41d4-a716-446655440000");

// Access facets
var facet = assembly.get_facet_by_name("CHARM_NET");
context.log("Found facet: " + facet.get_name());
```

## See Also

- [load_assembly()](load_assembly.md) - Load assembly from YAML
- [FacetAssemblyProxy](../../api/facet-assembly-proxy.md) - Assembly manipulation
- [AgentsAPI](../../api/agents-api.md) - Complete class reference
