# list_providers()

List all configured providers.

**Class**: ModelsAPI

**Access**: `context.noodle.models.list_providers()`

## Parameters

None

## Returns

Array of `{id, name, type}` objects

## Example

```javascript
var providers = context.noodle.models.list_providers();

providers.forEach(function(p) {
    context.log("Provider: " + p.name + " (type: " + p.type + ")");
});

// Output:
// Provider: Internal (Ollama) (type: ollama)
// Provider: Anthropic (type: anthropic)
// Provider: OpenRouter (type: openrouter)
// Provider: LM Studio (type: lmstudio)
// Provider: Groq (type: groq)
// Provider: Together AI (type: together)
// Provider: Mistral AI (type: mistral)
```

## Supported Providers

| Provider ID | Name | Type |
|------------|------|------|
| `ollama` | Internal (Ollama) | Local |
| `anthropic` | Anthropic | Cloud |
| `openai` | OpenAI | Cloud |
| `openrouter` | OpenRouter | Cloud |
| `lmstudio` | LM Studio | Local |
| `groq` | Groq | Cloud |
| `together` | Together AI | Cloud |
| `mistral` | Mistral AI | Cloud |

## See Also

- [list_available()](list_available.md) - List models from a provider
- [configure_provider()](configure_provider.md) - Configure provider settings
- [ModelsAPI](../../api/models-api.md) - Complete class reference
