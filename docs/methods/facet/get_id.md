# get_id()

Get the facet UUID.

**Class**: FacetProxy

**Access**: `facet.get_id()`

## Parameters

None

## Returns

UUID string (e.g., "550e8400-e29b-41d4-a716-446655440000")

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
var reasoner = assembly.get_facet_by_name("Red's Mind");

var id = reasoner.get_id();
context.log("Facet UUID: " + id);

// Use UUID to get facet directly
var same_facet = assembly.get_facet(id);
context.log("Same facet: " + (same_facet.get_name() === "Red's Mind"));
```

## See Also

- [get_name()](get_name.md) - Get facet name
- [get_type()](get_type.md) - Get facet type
- [get_facet()](../assembly/get_facet.md) - Get facet by UUID
