# get_facet()

Get facet by UUID.

**Class**: FacetAssemblyProxy

**Access**: `assembly.get_facet(facet_uuid)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `facet_uuid` | string | Facet UUID |

## Returns

[FacetProxy](../../api/facet-proxy.md) object or `null` if not found.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

var facet = assembly.get_facet("550e8400-e29b-41d4-a716-446655440000");

if (facet) {
    context.log("Facet name: " + facet.get_name());
    context.log("Facet type: " + facet.get_type());
}
```

## See Also

- [get_facet_by_name()](get_facet_by_name.md) - Get facet by name
- [list_facets()](list_facets.md) - List all facets
- [FacetProxy](../../api/facet-proxy.md) - Facet manipulation
