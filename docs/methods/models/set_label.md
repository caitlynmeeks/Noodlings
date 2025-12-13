# set_label()

Assign a (provider, model) to a label.

**Class**: ModelsAPI

**Access**: `context.noodle.models.set_label(label, provider, model)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `label` | string | Label name (e.g., "MEDIUM", "LARGE") |
| `provider` | string | Provider ID (e.g., "anthropic", "ollama", "openrouter") |
| `model` | string | Model name (e.g., "claude-opus-4.5", "deepseek-r1:70b") |

## Returns

`true` if set successfully, `false` on error

## Persistence

Changes are immediately saved to `model_labels.json`

## Example

```javascript
var success = context.noodle.models.set_label(
    "LARGE",
    "anthropic",
    "claude-opus-4.5"
);

if (success) {
    context.log("Successfully switched to Claude Opus");
} else {
    context.log("Failed to change label");
}
```

## See Also

- [get_label()](get_label.md) - Get current assignment
- [list_available()](list_available.md) - See available models
- [ModelsAPI](../../api/models-api.md) - Complete class reference
