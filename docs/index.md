# Noodlings Scripting API

Unity-like programmatic access to all Noodlings systems.

## What is `context.noodle`?

The Noodlings Scripting API provides JavaScript-accessible methods for configuring the entire Noodlings system from within ScriptedFacets. Every entity is addressable, all properties are gettable/settable.

```javascript
function process(inputs, context) {
    // Change which model a label uses
    context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");

    // Modify neural topology
    var network = context.noodle.neural.get_network(graph_id);
    var lstm = network.create_node("LSTM", {hidden_dim: 64});

    // Reconfigure facet assemblies
    var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
    var facet = assembly.get_facet("CHARM_NET");
    facet.set_property("model", "LARGE");

    return {modified: true};
}
```

## API Structure

```
context.noodle
  ├─ .models      // Model/provider configuration
  ├─ .neural      // Neural Canvas manipulation
  ├─ .agents      // Facet assembly access
  └─ .get_by_uuid // Universal entity lookup (future)
```

## Where is it available?

The API is injected into **ScriptedFacet** contexts automatically:

- **Python ScriptedFacets**: `context._noodle_api` (Python object)
- **JavaScript ScriptedFacets**: `context.noodle` (JavaScript bridge)

## Key Capabilities

### Dynamic Model Configuration
Switch between LLM providers and models at runtime based on task complexity, time of day, or any other criteria.

### Procedural Neural Topologies
Generate LSTM/GRU networks programmatically, modify existing topologies, and export MLX code.

### Self-Modifying Agents
Facets can reconfigure their own assemblies, creating truly adaptive cognitive architectures.

## Implementation Stats

- **Total Lines**: 1,789 across 8 files
- **Main APIs**: NoodleAPI, ModelsAPI, NeuralAPI, AgentsAPI
- **Proxy Classes**: NeuralNetworkProxy, FacetAssemblyProxy, FacetProxy
- **Methods**: 40+ across all APIs
- **Test Coverage**: 3 comprehensive test suites

## Philosophy

This API follows Noodlings' Christopher Alexander-inspired design philosophy:

- **Emergent complexity**: Simple primitives combine into sophisticated behaviors
- **Self-modification**: Systems can reconfigure themselves based on experience
- **No artificial limits**: If it exists in Noodlings, scripts can access it
- **Graceful failure**: Methods return `None`/`false` on error, never crash

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started in 5 minutes
- [API Reference](api/overview.md) - Complete method documentation
- [Complete Examples](examples.md) - Real-world usage patterns
