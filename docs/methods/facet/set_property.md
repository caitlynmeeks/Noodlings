# set_property()

Set value of a facet property.

**Class**: FacetProxy

**Access**: `facet.set_property(property_name, value)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `property_name` | string | Property name |
| `value` | any | New value |

## Returns

`true` if set successfully, `false` on failure.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
var reasoner = assembly.get_facet_by_name("Red's Mind");

// Change model label
reasoner.set_property("model", "LARGE");

// Adjust temperature
reasoner.set_property("temperature", 0.9);

// Update prompt
reasoner.set_property("prompt", "You are a creative thinker...");

// Save changes
assembly.save("modified_agent.yaml");

context.log("Properties updated");
```

## Common Properties by Facet Type

### LLMFacet
- `model` (string) - Model label (SMALL/MEDIUM/LARGE)
- `temperature` (number) - 0.0 to 1.0
- `max_tokens` (number) - Max response length
- `prompt` (string) - System prompt

### ScriptedFacet
- `language` (string) - "javascript" or "python"
- `code` (string) - Source code

### CharmNetworkFacet
- `topology` (string) - Path to .nncanvas file

## See Also

- [get_property()](get_property.md) - Get property value
- [get_all_properties()](get_all_properties.md) - Get all properties
- [save()](../assembly/save.md) - Save changes
