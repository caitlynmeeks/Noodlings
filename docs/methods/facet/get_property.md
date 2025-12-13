# get_property()

Get value of a facet property.

**Class**: FacetProxy

**Access**: `facet.get_property(property_name)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `property_name` | string | Property name (e.g., "model", "temperature", "prompt") |

## Returns

Property value (type varies) or `null` if not found.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
var reasoner = assembly.get_facet_by_name("Red's Mind");

var model = reasoner.get_property("model");
var temp = reasoner.get_property("temperature");
var prompt = reasoner.get_property("prompt");

context.log("Model: " + model);           // "LARGE"
context.log("Temperature: " + temp);      // 0.9
context.log("Prompt length: " + prompt.length);
```

## See Also

- [set_property()](set_property.md) - Set property value
- [get_all_properties()](get_all_properties.md) - Get all properties
- [FacetProxy](../../api/facet-proxy.md) - Complete class reference
