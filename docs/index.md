# Noodlings Scripting API

Programmatic access to all Noodlings systems from ScriptedFacets.

## Quick Access

- **[Quick Start](quick-start.md)** - Get started in 5 minutes
- **[Complete API Reference](api-reference.md)** - All methods on one page
- **Browse the sidebar** → Click "MUSH API" or "Studio API" to see all methods

## What is `context.noodle`?

The Noodlings Scripting API provides JavaScript-accessible methods for configuring the Noodlings system from within ScriptedFacets.

```javascript
function process(inputs, context) {
    // Change which model a label uses
    context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");

    // Modify neural topology
    var network = context.noodle.neural.get_network(graph_id);
    var lstm = network.create_node("LSTM", {hidden_dim: 64});

    // Reconfigure facet assemblies
    var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
    facet.set_property("model", "LARGE");

    return {modified: true};
}
```

## API Structure

```
context.noodle
  ├─ .models      // Model/provider configuration
  ├─ .neural      // Neural Canvas manipulation
  └─ .agents      // Facet assembly access
```

## Browse Methods

**Look at the LEFT SIDEBAR** - expand "MUSH API" or "Studio API" to see all available methods!
