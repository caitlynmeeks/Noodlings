# list_available()

List all models available from a provider.

**Class**: ModelsAPI

**Access**: `context.noodle.models.list_available(provider)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `provider` | string | Provider ID (e.g., "openrouter", "anthropic", "ollama") |

## Returns

Array of model name strings

## Example

```javascript
var anthropic_models = context.noodle.models.list_available("anthropic");

context.log("Anthropic offers " + anthropic_models.length + " models");
// ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.0"]

// List all models from OpenRouter
var openrouter = context.noodle.models.list_available("openrouter");
context.log("OpenRouter has " + openrouter.length + " models");
// 200+ models
```

## Note

For Ollama, this returns **downloaded models only**, not all available from the Ollama library.

## See Also

- [list_providers()](list_providers.md) - See all providers
- [set_label()](set_label.md) - Assign model to label
- [ModelsAPI](../../api/models-api.md) - Complete class reference
