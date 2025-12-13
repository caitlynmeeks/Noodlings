# ModelsAPI

Scriptable interface to model and provider configuration.

**Location**: `noodlestudio/scripting/models_api.py`

**Access**: `context.noodle.models`

## Overview

The ModelsAPI allows scripts to:

- Query which models are assigned to labels (SMALL/MEDIUM/LARGE)
- Dynamically reassign labels to different providers/models
- List available models from each provider
- Configure provider settings (API keys, endpoints)

## Methods

### `get_label(label)`

Get the (provider, model) assigned to a label.

**Parameters**:

- `label` (string) - Label name (e.g., `"SMALL"`, `"MEDIUM"`, `"LARGE"`)

**Returns**: Object with `{provider, model}` keys (null if unassigned)

**Example**:
```javascript
var assignment = context.noodle.models.get_label("SMALL");
context.log("Provider: " + assignment.provider);  // "ollama"
context.log("Model: " + assignment.model);        // "deepseek-r1:7b"
```

---

### `set_label(label, provider, model)`

Assign a (provider, model) to a label.

**Parameters**:

- `label` (string) - Label name (e.g., `"LARGE"`)
- `provider` (string) - Provider ID (e.g., `"anthropic"`, `"ollama"`, `"openrouter"`)
- `model` (string) - Model name (e.g., `"claude-opus-4.5"`, `"deepseek-r1:70b"`)

**Returns**: `true` if set successfully, `false` on error

**Example**:
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

**Persistence**: Changes are immediately saved to `model_labels.json`

---

### `get_all_labels()`

Get all label assignments.

**Parameters**: None

**Returns**: Object mapping labels to `{provider, model}` (excludes unassigned labels)

**Example**:
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

---

### `list_available(provider)`

List all models available from a provider.

**Parameters**:

- `provider` (string) - Provider ID (e.g., `"openrouter"`, `"anthropic"`, `"ollama"`)

**Returns**: Array of model name strings

**Example**:
```javascript
var anthropic_models = context.noodle.models.list_available("anthropic");
context.log("Anthropic offers " + anthropic_models.length + " models");

// ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.0"]
```

**Note**: For Ollama, this returns downloaded models only (not all available from library)

---

### `list_providers()`

List all configured providers.

**Parameters**: None

**Returns**: Array of `{id, name, type}` objects

**Example**:
```javascript
var providers = context.noodle.models.list_providers();

providers.forEach(function(p) {
    context.log("Provider: " + p.name + " (type: " + p.type + ")");
});

// Output:
// Provider: Internal (Ollama) (type: ollama)
// Provider: Anthropic (type: anthropic)
// Provider: OpenRouter (type: openrouter)
```

---

### `configure_provider(provider, options)`

Configure provider settings.

**Parameters**:

- `provider` (string) - Provider ID (e.g., `"anthropic"`)
- `options` (object) - Configuration options:
    - `api_key` (string) - API key for the provider
    - `base_url` (string) - Custom base URL (for compatible APIs)
    - `port` (number) - Custom port (for local servers)

**Returns**: `true` if configured successfully, `false` on error

**Example**:
```javascript
// Configure Anthropic API key
var success = context.noodle.models.configure_provider("anthropic", {
    api_key: "sk-ant-api03-..."
});

// Configure local LM Studio server
context.noodle.models.configure_provider("lmstudio", {
    base_url: "http://localhost:1234",
    port: 1234
});
```

**Security Note**: API keys are stored in provider configuration files. Ensure these files are gitignored!

## Supported Providers

| Provider ID | Name | Type | Notes |
|------------|------|------|-------|
| `ollama` | Internal (Ollama) | Local | Downloaded models only |
| `anthropic` | Anthropic | Cloud | Requires API key |
| `openai` | OpenAI | Cloud | Requires API key |
| `openrouter` | OpenRouter | Cloud | 200+ models, requires API key |
| `lmstudio` | LM Studio | Local | OpenAI-compatible |
| `groq` | Groq | Cloud | Super fast LPU inference |
| `together` | Together AI | Cloud | Open source models |
| `mistral` | Mistral AI | Cloud | Direct Mistral access |

## Complete Example: Dynamic Model Selection

```javascript
function process(inputs, context) {
    // Analyze task complexity
    var wordCount = inputs.text ? inputs.text.split(' ').length : 0;
    var isComplex = wordCount > 500;

    var models = context.noodle.models;

    if (isComplex) {
        // Use powerful cloud model for complex tasks
        models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        context.log("Complex task detected - using Claude Opus");
    } else {
        // Use local model for simple tasks (save API costs)
        models.set_label("LARGE", "ollama", "deepseek-r1:70b");
        context.log("Simple task - using local DeepSeek");
    }

    // Verify the change
    var current = models.get_label("LARGE");
    context.log("Now using: " + current.provider + "/" + current.model);

    return {
        complexity: isComplex ? "high" : "low",
        provider: current.provider,
        model: current.model
    };
}
```

## Complete Example: Night Mode

```javascript
function process(inputs, context) {
    var models = context.noodle.models;
    var hour = new Date().getHours();
    var isNightTime = (hour >= 22 || hour < 6);

    if (isNightTime) {
        // Switch to local models at night (save API costs while you sleep)
        models.set_label("SMALL", "ollama", "deepseek-r1:7b");
        models.set_label("MEDIUM", "ollama", "deepseek-r1:14b");
        models.set_label("LARGE", "ollama", "deepseek-r1:70b");
        context.log("Night mode activated - using local models");
    } else {
        // Day mode: Use cloud models for best quality
        models.set_label("SMALL", "anthropic", "claude-haiku-4.0");
        models.set_label("MEDIUM", "anthropic", "claude-sonnet-4.5");
        models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        context.log("Day mode activated - using Claude models");
    }

    return {night_mode: isNightTime};
}
```

## See Also

- [Quick Start Guide](../quick-start.md)
- [NeuralAPI Reference](neural-api.md)
- [AgentsAPI Reference](agents-api.md)
- [Complete Examples](../examples.md)
