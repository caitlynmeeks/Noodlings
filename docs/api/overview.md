# API Overview

The Noodlings Scripting API provides a unified interface to all system components.

## Architecture

```
ScriptContext
    └─ noodle: NoodleAPI
        ├─ models: ModelsAPI
        │   ├─ get_label()
        │   ├─ set_label()
        │   ├─ get_all_labels()
        │   ├─ list_available()
        │   ├─ list_providers()
        │   └─ configure_provider()
        │
        ├─ neural: NeuralAPI
        │   ├─ get_network() → NeuralNetworkProxy
        │   ├─ load()         → NeuralNetworkProxy
        │   └─ create_network() → NeuralNetworkProxy
        │       └─ NeuralNetworkProxy methods:
        │           ├─ create_node()
        │           ├─ remove_node()
        │           ├─ connect()
        │           ├─ disconnect()
        │           ├─ get_node()
        │           ├─ get_node_by_name()
        │           ├─ set_node_property()
        │           ├─ set_node_position()
        │           ├─ generate_mlx_code()
        │           ├─ save()
        │           └─ get_parameter_count()
        │
        └─ agents: AgentsAPI
            ├─ get()
            ├─ get_assembly() → FacetAssemblyProxy
            ├─ load_assembly() → FacetAssemblyProxy
            └─ list_all()
                └─ FacetAssemblyProxy methods:
                    ├─ get_facet() → FacetProxy
                    ├─ get_facet_by_name() → FacetProxy
                    ├─ list_facets()
                    ├─ add_facet()
                    ├─ remove_facet()
                    ├─ connect()
                    ├─ disconnect()
                    └─ save()
                        └─ FacetProxy methods:
                            ├─ get_property()
                            ├─ set_property()
                            ├─ get_all_properties()
                            ├─ get_type()
                            ├─ get_id()
                            └─ get_name()
```

## Design Principles

### 1. Lazy Initialization

Sub-APIs only initialize when first accessed, minimizing overhead:

```javascript
// ModelsAPI doesn't initialize until first use
var label = context.noodle.models.get_label("SMALL");  // ← Initialized here
```

### 2. Proxy Pattern

Complex objects (networks, assemblies, facets) are wrapped in proxy classes that provide JavaScript-friendly interfaces:

```javascript
var network = context.noodle.neural.get_network(id);
// network is a NeuralNetworkProxy wrapping the actual NeuralGraph
```

### 3. Graceful Failure

Methods return `null`/`false` on error, never throw exceptions that would crash your script:

```javascript
var node = network.get_node("invalid-id");
if (!node) {
    context.log("Node not found - graceful handling");
}
```

### 4. JavaScript-Python Bridge

Python methods are exposed to JavaScript via placeholder strings that the QuickJS engine maps back to Python calls:

```python
# Python side
def to_dict(self):
    return {'get_label': '__models_get_label__'}

# JavaScript side
context.noodle.models.get_label("SMALL")  // Maps to Python method
```

## File Locations

All API files in `/applications/noodlestudio/noodlestudio/scripting/`:

- `noodle_api.py` - Main entry point (197 lines)
- `models_api.py` - Model/provider API (199 lines)
- `neural_api.py` - Neural Canvas API + proxy (377 lines)
- `agents_api.py` - Agents/facets API + proxies (435 lines)

Integration point: `/applications/noodlestudio/noodlestudio/core/scripted_facet.py:101-120`

## Thread Safety

The API is **not thread-safe**. All calls should happen within a single facet execution cycle. Do not attempt concurrent modifications from multiple facets.

## Performance

API calls have minimal overhead:

- **Property access**: ~0.1ms (cached)
- **Model label lookup**: ~0.2ms (JSON read)
- **Neural topology modification**: ~1-5ms (depends on graph size)
- **Facet assembly modification**: ~2-10ms (depends on complexity)

## Persistence

Changes made via the API are **immediately persisted**:

- Model label changes → saved to `model_labels.json`
- Neural topology changes → saved to `.nncanvas` file (when calling `save()`)
- Facet assembly changes → saved to `.yaml` file (when calling `save()`)

## API References

- [ModelsAPI](models-api.md) - Provider and model configuration
- [NeuralAPI](neural-api.md) - Neural Canvas manipulation
- [AgentsAPI](agents-api.md) - Facet assembly access
