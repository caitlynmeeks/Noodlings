# get_all_properties()

Get all properties of a facet.

**Class**: FacetProxy

**Access**: `facet.get_all_properties()`

## Parameters

None

## Returns

Object containing all property key-value pairs.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
var reasoner = assembly.get_facet_by_name("Red's Mind");

var props = reasoner.get_all_properties();

context.log("Facet properties:");
for (var key in props) {
    context.log("  " + key + ": " + props[key]);
}
```

## Output Example

```
Facet properties:
  model: LARGE
  temperature: 0.9
  max_tokens: 1000
  prompt: You are Red Fire Anklebiter...
  salience_script: function process...
```

## See Also

- [get_property()](get_property.md) - Get single property
- [set_property()](set_property.md) - Set property value
- [FacetProxy](../../api/facet-proxy.md) - Complete class reference
