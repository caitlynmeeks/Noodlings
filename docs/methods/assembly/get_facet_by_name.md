# get_facet_by_name()

Get facet by name.

**Class**: FacetAssemblyProxy

**Access**: `assembly.get_facet_by_name(name)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `name` | string | Facet name (e.g., "CHARM_NET", "Red's Mind") |

## Returns

[FacetProxy](../../api/facet-proxy.md) object or `null` if not found.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

var charm_net = assembly.get_facet_by_name("CHARM_NET");
var reasoner = assembly.get_facet_by_name("Red's Mind");

if (charm_net) {
    context.log("Found CHARM_NET facet");
    var model = charm_net.get_property("model");
    context.log("Using model: " + model);
}
```

## See Also

- [get_facet()](get_facet.md) - Get facet by UUID
- [list_facets()](list_facets.md) - List all facets
- [FacetProxy](../../api/facet-proxy.md) - Facet manipulation
