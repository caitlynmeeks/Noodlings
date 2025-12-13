# Complete API Reference

Comprehensive single-page reference for the Noodlings Scripting API.

**Access**: Available in all ScriptedFacets via `context.noodle`

---

## NoodleAPI

Main entry point for all scripting capabilities.

### Properties

#### `.models` → ModelsAPI
Access model and provider configuration methods.

#### `.neural` → NeuralAPI
Access neural topology manipulation methods.

#### `.agents` → AgentsAPI
Access facet assembly modification methods.

### Methods

#### `get_by_uuid(uuid)` → object | null
Get any entity by UUID (future enhancement).

---

## ModelsAPI

**Access**: `context.noodle.models`

Configure LLM providers and model label assignments.

### Methods

#### `get_label(label)` → {provider, model}
Get the (provider, model) assigned to a label.

```javascript
var assignment = context.noodle.models.get_label("SMALL");
// → {provider: "ollama", model: "deepseek-r1:7b"}
```

---

#### `set_label(label, provider, model)` → boolean
Assign a (provider, model) to a label.

```javascript
context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
// Returns: true
```

**Persistence**: Changes saved immediately to `model_labels.json`

---

#### `get_all_labels()` → object
Get all label assignments.

```javascript
var labels = context.noodle.models.get_all_labels();
// → {
//   "SMALL": {provider: "ollama", model: "deepseek-r1:7b"},
//   "LARGE": {provider: "anthropic", model: "claude-opus-4.5"}
// }
```

---

#### `list_available(provider)` → string[]
List models available from a provider.

```javascript
var models = context.noodle.models.list_available("anthropic");
// → ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.0"]
```

---

#### `list_providers()` → object[]
List all configured providers.

```javascript
var providers = context.noodle.models.list_providers();
// → [{id: "ollama", name: "Internal (Ollama)", type: "ollama"}, ...]
```

---

#### `configure_provider(provider, options)` → boolean
Configure provider settings (API keys, endpoints).

```javascript
context.noodle.models.configure_provider("anthropic", {
    api_key: "sk-ant-api03-..."
});
```

---

## NeuralAPI

**Access**: `context.noodle.neural`

Create and manipulate neural network topologies.

### Methods

#### `create_network(name)` → NeuralNetworkProxy
Create a new empty network.

```javascript
var network = context.noodle.neural.create_network("MyNetwork");
```

---

#### `load(filepath)` → NeuralNetworkProxy | null
Load network from .nncanvas file.

```javascript
var network = context.noodle.neural.load("facet_assemblies/charm_networks/default.nncanvas");
```

---

#### `get_network(graph_id)` → NeuralNetworkProxy | null
Get network by ID.

```javascript
var network = context.noodle.neural.get_network("default");
```

---

## NeuralNetworkProxy

**Returned by**: `NeuralAPI` methods

Proxy object for manipulating a neural network.

### Methods

#### `create_node(node_type, properties)` → string | null
Create a new node. Returns node ID.

```javascript
var lstm_id = network.create_node("LSTM", {
    hidden_dim: 32,
    position: [100, 200]
});
```

**Node Types**: LSTM, GRU, RNN, Linear, Attention, AffectHead, etc. (26 total - see [Node Types](node-types.md))

---

#### `remove_node(node_id)` → boolean
Remove a node from the network.

```javascript
network.remove_node(lstm_id);
```

---

#### `connect(from_node, from_port, to_node, to_port)` → boolean
Connect two nodes via ports.

```javascript
network.connect(lstm_id, "out", gru_id, "input");
```

---

#### `disconnect(from_node, from_port, to_node, to_port)` → boolean
Disconnect two nodes.

```javascript
network.disconnect(lstm_id, "out", gru_id, "input");
```

---

#### `get_node(node_id)` → object | null
Get node information.

```javascript
var node = network.get_node(lstm_id);
// → {id, type, name, properties, position}
context.log("Hidden dim: " + node.properties.hidden_dim);
```

---

#### `get_node_by_name(name)` → string | null
Find node ID by name.

```javascript
var fast_lstm_id = network.get_node_by_name("Fast_LSTM");
```

---

#### `set_node_property(node_id, property_name, value)` → boolean
Set a node property.

```javascript
network.set_node_property(lstm_id, "hidden_dim", 64);
```

---

#### `set_node_position(node_id, x, y)` → boolean
Set node position in canvas.

```javascript
network.set_node_position(lstm_id, 150, 250);
```

---

#### `generate_mlx_code()` → string | null
Generate MLX Python code from topology.

```javascript
var code = network.generate_mlx_code();
context.log("Generated " + code.length + " characters");
```

---

#### `save(filepath)` → boolean
Save network to .nncanvas file.

```javascript
network.save("custom_topology.nncanvas");
```

---

#### `get_parameter_count()` → number
Calculate total trainable parameters.

```javascript
var params = network.get_parameter_count();
// → 54280
```

---

## AgentsAPI

**Access**: `context.noodle.agents`

Access and modify agent facet assemblies.

### Methods

#### `get(agent_id)` → object | null
Get agent information.

```javascript
var agent = context.noodle.agents.get("red-fire-anklebiter");
// → {id, name, species, assembly}
```

---

#### `get_assembly(agent_id)` → FacetAssemblyProxy | null
Get facet assembly for an agent.

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
```

---

#### `load_assembly(agent_id, filepath)` → FacetAssemblyProxy | null
Load assembly from YAML file.

```javascript
var assembly = context.noodle.agents.load_assembly("custom-agent", "custom.yaml");
```

---

#### `list_all()` → string[]
List all registered agent IDs.

```javascript
var agents = context.noodle.agents.list_all();
// → ["red-fire-anklebiter", "callie-wisdom-keeper", ...]
```

---

## FacetAssemblyProxy

**Returned by**: `AgentsAPI.get_assembly()`

Proxy object for a facet assembly (cognitive topology).

### Methods

#### `get_facet(facet_id)` → FacetProxy | null
Get facet by ID.

```javascript
var charm = assembly.get_facet("CHARM_NET");
```

---

#### `get_facet_by_name(name)` → FacetProxy | null
Get facet by display name.

```javascript
var mind = assembly.get_facet_by_name("Red's Mind");
```

---

#### `list_facets()` → object[]
List all facets in assembly.

```javascript
var facets = assembly.list_facets();
// → [{id: "CHARM_NET", name: "CharmNetwork", type: "CharmNetworkFacet"}, ...]
```

---

#### `add_facet(facet_type, name, properties)` → string | null
Add new facet to assembly. Returns facet ID.

```javascript
var facet_id = assembly.add_facet("LLMFacet", "Custom Reasoner", {
    model: "LARGE",
    temperature: 0.8
});
```

**Facet Types**: LLMFacet, ScriptedFacet, CharmNetworkFacet, ContextIntelligenceFacet, ConvergenceFacet

---

#### `remove_facet(facet_id)` → boolean
Remove facet from assembly.

```javascript
assembly.remove_facet("OLD_FACET");
```

---

#### `connect(from_facet, from_pad, to_facet, to_pad)` → boolean
Connect two facets via data pads.

```javascript
assembly.connect("CHARM_NET", "affect_valence", "RED_MIND", "affect");
```

---

#### `disconnect(from_facet, from_pad, to_facet, to_pad)` → boolean
Disconnect two facets.

```javascript
assembly.disconnect("CHARM_NET", "affect_valence", "RED_MIND", "affect");
```

---

#### `save(filepath)` → boolean
Save modified assembly to YAML file.

```javascript
assembly.save("facet_assemblies/red_modified.yaml");
```

---

## FacetProxy

**Returned by**: `FacetAssemblyProxy.get_facet()`

Proxy object for a single facet.

### Methods

#### `get_property(name)` → any
Get facet property value.

```javascript
var facet = assembly.get_facet("RED_MIND");
var model = facet.get_property("model");      // "LARGE"
var temp = facet.get_property("temperature"); // 0.9
```

---

#### `set_property(name, value)` → boolean
Set facet property value.

```javascript
facet.set_property("temperature", 0.95);
facet.set_property("model", "LARGE");
```

**Common Properties**:
- `model` - Model label (SMALL/MEDIUM/LARGE)
- `temperature` - Sampling temperature (0.0-2.0)
- `max_tokens` - Max response length
- `prompt` - System prompt text
- `script` - JavaScript code (ScriptedFacet only)

---

#### `get_all_properties()` → object
Get all facet properties.

```javascript
var props = facet.get_all_properties();
for (var key in props) {
    context.log(key + ": " + props[key]);
}
```

---

#### `get_type()` → string
Get facet type.

```javascript
var type = facet.get_type();
// → "LLMFacet", "ScriptedFacet", etc.
```

---

#### `get_id()` → string
Get facet ID.

```javascript
var id = facet.get_id();
```

---

#### `get_name()` → string
Get facet display name.

```javascript
var name = facet.get_name();
// → "Red's Mind"
```

---

## Quick Reference Table

### ModelsAPI Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `get_label` | label | {provider, model} | Get label assignment |
| `set_label` | label, provider, model | boolean | Set label assignment |
| `get_all_labels` | - | object | Get all assignments |
| `list_available` | provider | string[] | List provider's models |
| `list_providers` | - | object[] | List all providers |
| `configure_provider` | provider, options | boolean | Set API keys/endpoints |

### NeuralAPI Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `create_network` | name | NeuralNetworkProxy | Create empty network |
| `load` | filepath | NeuralNetworkProxy | Load from .nncanvas |
| `get_network` | graph_id | NeuralNetworkProxy | Get by ID |

### NeuralNetworkProxy Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `create_node` | type, properties | string | Create node, return ID |
| `remove_node` | node_id | boolean | Remove node |
| `connect` | from, from_port, to, to_port | boolean | Connect nodes |
| `disconnect` | from, from_port, to, to_port | boolean | Disconnect nodes |
| `get_node` | node_id | object | Get node info |
| `get_node_by_name` | name | string | Find node by name |
| `set_node_property` | node_id, name, value | boolean | Set property |
| `set_node_position` | node_id, x, y | boolean | Set position |
| `generate_mlx_code` | - | string | Generate Python code |
| `save` | filepath | boolean | Save to .nncanvas |
| `get_parameter_count` | - | number | Count parameters |

### AgentsAPI Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `get` | agent_id | object | Get agent info |
| `get_assembly` | agent_id | FacetAssemblyProxy | Get assembly |
| `load_assembly` | agent_id, filepath | FacetAssemblyProxy | Load from YAML |
| `list_all` | - | string[] | List all agents |

### FacetAssemblyProxy Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `get_facet` | facet_id | FacetProxy | Get by ID |
| `get_facet_by_name` | name | FacetProxy | Get by name |
| `list_facets` | - | object[] | List all facets |
| `add_facet` | type, name, properties | string | Add facet, return ID |
| `remove_facet` | facet_id | boolean | Remove facet |
| `connect` | from, from_pad, to, to_pad | boolean | Connect facets |
| `disconnect` | from, from_pad, to, to_pad | boolean | Disconnect facets |
| `save` | filepath | boolean | Save to YAML |

### FacetProxy Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `get_property` | name | any | Get property value |
| `set_property` | name, value | boolean | Set property value |
| `get_all_properties` | - | object | Get all properties |
| `get_type` | - | string | Get facet type |
| `get_id` | - | string | Get facet ID |
| `get_name` | - | string | Get facet name |

---

## Complete Example: All APIs Together

```javascript
function process(inputs, context) {
    // ========== MODELS API ==========
    var models = context.noodle.models;

    // Check current model
    var large = models.get_label("LARGE");
    context.log("Current LARGE: " + large.provider + "/" + large.model);

    // Switch to Opus for complex tasks
    if (inputs.complexity > 0.8) {
        models.set_label("LARGE", "anthropic", "claude-opus-4.5");
    }

    // ========== NEURAL API ==========
    var neural = context.noodle.neural;

    // Load CharmNetwork
    var network = neural.load("facet_assemblies/charm_networks/default.nncanvas");

    if (network) {
        // Get node info
        var fast_lstm_id = network.get_node_by_name("Fast_LSTM");
        var node = network.get_node(fast_lstm_id);
        context.log("Fast LSTM hidden_dim: " + node.properties.hidden_dim);

        // Modify topology
        var new_lstm = network.create_node("LSTM", {
            hidden_dim: 64,
            position: [400, 200]
        });

        // Connect to existing network
        network.connect(fast_lstm_id, "out", new_lstm, "input");

        // Generate code
        var code = network.generate_mlx_code();
        var params = network.get_parameter_count();
        context.log("Network: " + params + " parameters");
    }

    // ========== AGENTS API ==========
    var agents = context.noodle.agents;

    // Get this agent's assembly
    var assembly = agents.get_assembly(context.agent.id);

    if (assembly) {
        // List all facets
        var facets = assembly.list_facets();
        context.log("Assembly has " + facets.length + " facets");

        // Modify a facet
        var mind = assembly.get_facet_by_name("Red's Mind");
        if (mind) {
            // Get current settings
            var temp = mind.get_property("temperature");
            context.log("Current temperature: " + temp);

            // Increase creativity
            mind.set_property("temperature", 0.95);

            // Verify change
            var new_temp = mind.get_property("temperature");
            context.log("New temperature: " + new_temp);
        }

        // Add a new facet
        var memory_id = assembly.add_facet("ScriptedFacet", "Memory Bank", {
            script: "function process(inputs, context) { return {stored: true}; }"
        });

        if (memory_id) {
            // Connect to data flow
            assembly.connect("CONTEXT_INTELLIGENCE", "result", memory_id, "data");
            context.log("Added memory facet");
        }

        // Save modified assembly
        assembly.save("facet_assemblies/" + context.agent.id + "_modified.yaml");
    }

    return {
        apis_tested: 3,
        all_working: true
    };
}
```

---

## Supported Providers

| Provider | ID | Type | Notes |
|----------|-----|------|-------|
| Internal (Ollama) | `ollama` | Local | Downloaded models |
| Anthropic | `anthropic` | Cloud | Requires API key |
| OpenAI | `openai` | Cloud | Requires API key |
| OpenRouter | `openrouter` | Cloud | 200+ models |
| LM Studio | `lmstudio` | Local | OpenAI-compatible |
| Groq | `groq` | Cloud | Fast LPU inference |
| Together AI | `together` | Cloud | Open source models |
| Mistral AI | `mistral` | Cloud | Direct Mistral access |

---

## Neural Canvas Node Types (26 Total)

### Recurrent
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit
- **RNN** - Simple RNN

### Feedforward
- **Linear** - Fully connected layer
- **Conv1D** - 1D convolution

### Attention
- **Attention** - Scaled dot-product
- **Multi-Head Attention** - Transformer attention

### Activation
- **Tanh**, **ReLU**, **GELU**, **Sigmoid**, **Softmax**

### Normalization
- **Layer Norm**, **Batch Norm**

### Regularization
- **Dropout**

### CharmNetwork Special
- **State Concat** - Combine hidden states
- **State Split** - Split phenomenal state
- **Affect Head** - Generate PAD affect values

### Quantum
- **Quantum Microtubule** - Penrose-Hameroff simulation
- **IBM Quantum** - Real quantum hardware
- **Entropy Injection** - TrueRNG randomness

### I/O
- **Input**, **Output**

### Assets
- **Checkpoint** - Load trained weights

See [Node Types](node-types.md) for detailed specifications.

---

## Common Patterns

### Pattern: Time-Based Model Switching
```javascript
var hour = new Date().getHours();
var isNight = (hour >= 22 || hour < 6);

if (isNight) {
    context.noodle.models.set_label("LARGE", "ollama", "deepseek-r1:70b");
} else {
    context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
}
```

### Pattern: Affect-Driven Temperature
```javascript
var valence = inputs.affect_valence || 0.0;
var arousal = inputs.affect_arousal || 0.5;

var temp = 0.7 + (arousal * 0.3) + (Math.max(0, valence) * 0.2);
temp = Math.min(temp, 1.2);

var mind = assembly.get_facet_by_name("Red's Mind");
mind.set_property("temperature", temp);
```

### Pattern: Procedural Topology
```javascript
var network = context.noodle.neural.create_network("Adaptive");
var prev_id = network.create_node("Input", {output_dim: 64});

for (var i = 0; i < 3; i++) {
    var lstm = network.create_node("LSTM", {hidden_dim: 32});
    network.connect(prev_id, "out", lstm, "input");
    prev_id = lstm;
}

var code = network.generate_mlx_code();
```

---

## See Also

- [Quick Start Guide](quick-start.md) - Get started in 5 minutes
- [Complete Examples](examples.md) - Real-world usage patterns
- [Node Types](node-types.md) - Detailed node specifications
- [Troubleshooting](troubleshooting.md) - Common issues
