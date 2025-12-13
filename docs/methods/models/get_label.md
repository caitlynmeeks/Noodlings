# get_label()

Get the (provider, model) assigned to a label.

**Class**: ModelsAPI

**Access**: `context.noodle.models.get_label(label)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `label` | string | Label name (e.g., "SMALL", "MEDIUM", "LARGE") |

## Returns

Object with `{provider, model}` keys. Returns `null` if unassigned.

## Example

```javascript
var assignment = context.noodle.models.get_label("SMALL");

context.log("Provider: " + assignment.provider);  // "ollama"
context.log("Model: " + assignment.model);        // "deepseek-r1:7b"
```

## See Also

- [set_label()](set_label.md) - Assign model to label
- [get_all_labels()](get_all_labels.md) - Get all assignments
- [ModelsAPI](../../api/models-api.md) - Complete class reference
