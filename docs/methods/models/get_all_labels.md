# get_all_labels()

Get all label assignments.

**Class**: ModelsAPI

**Access**: `context.noodle.models.get_all_labels()`

## Parameters

None

## Returns

Object mapping labels to `{provider, model}`. Excludes unassigned labels.

## Example

```javascript
var labels = context.noodle.models.get_all_labels();

for (var label in labels) {
    var assignment = labels[label];
    context.log(label + " → " + assignment.provider + "/" + assignment.model);
}

// Output:
// SMALL → ollama/deepseek-r1:7b
// MEDIUM → ollama/deepseek-r1:14b
// LARGE → anthropic/claude-opus-4.5
```

## See Also

- [get_label()](get_label.md) - Get single label
- [set_label()](set_label.md) - Set label assignment
- [ModelsAPI](../../api/models-api.md) - Complete class reference
