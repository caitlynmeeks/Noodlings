# configure_provider()

Configure provider settings (API keys, endpoints, ports).

**Class**: ModelsAPI

**Access**: `context.noodle.models.configure_provider(provider, options)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `provider` | string | Provider ID (e.g., "anthropic") |
| `options` | object | Configuration options (see below) |

### Options Object

| Property | Type | Description |
|----------|------|-------------|
| `api_key` | string | API key for the provider |
| `base_url` | string | Custom base URL (for compatible APIs) |
| `port` | number | Custom port (for local servers) |

## Returns

`true` if configured successfully, `false` on error

## Examples

### Configure Anthropic API Key

```javascript
var success = context.noodle.models.configure_provider("anthropic", {
    api_key: "sk-ant-api03-..."
});

if (success) {
    context.log("Anthropic configured");
}
```

### Configure Local LM Studio Server

```javascript
context.noodle.models.configure_provider("lmstudio", {
    base_url: "http://localhost:1234",
    port: 1234
});
```

## Security Note

API keys are stored in provider configuration files. Ensure these files are gitignored.

## See Also

- [list_providers()](list_providers.md) - See all providers
- [list_available()](list_available.md) - List provider's models
- [ModelsAPI](../../api/models-api.md) - Complete class reference
