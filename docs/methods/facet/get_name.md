# get_name()

Get the facet name.

**Class**: FacetProxy

**Access**: `facet.get_name()`

## Parameters

None

## Returns

String representing facet name.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

var facets = assembly.list_facets();

context.log("Facets in assembly:");

for (var i = 0; i < facets.length; i++) {
    var facet = assembly.get_facet(facets[i].uuid);
    context.log("  " + facet.get_name());
}
```

## Output Example

```
Facets in assembly:
  INCOMING
  CHARM_NET
  Red's Mind
  Fire Body
  OUTGOING
```

## See Also

- [get_id()](get_id.md) - Get facet UUID
- [get_type()](get_type.md) - Get facet type
- [get_facet_by_name()](../assembly/get_facet_by_name.md) - Get facet by name
