# Quick Start

Get up and running with the Noodlings Scripting API in 5 minutes.

## Installation

The API is automatically available in all ScriptedFacets. No installation needed!

## Your First Script

Create a ScriptedFacet in your facet assembly YAML:

```yaml
facets:
  - id: MY_SCRIPT
    type: ScriptedFacet
    name: My First Script
    script: |
      function process(inputs, context) {
          // Check which model is assigned to LARGE
          var large = context.noodle.models.get_label("LARGE");
          context.log("LARGE uses: " + large.model);

          return {checked: true};
      }
```

## Test It

1. Load your facet assembly in NoodleStudio
2. Check the DEBUG console (Console panel → DEBUG button)
3. You should see: `[My First Script] LARGE uses: claude-opus-4.5`

## Common Patterns

### Pattern 1: Dynamic Model Switching

```javascript
function process(inputs, context) {
    var hour = new Date().getHours();

    if (hour >= 22 || hour < 6) {
        // Night mode: Use smaller model
        context.noodle.models.set_label("LARGE", "ollama", "deepseek-r1:7b");
        context.log("Night mode: Switched to local model");
    } else {
        // Day mode: Use powerful model
        context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        context.log("Day mode: Using Opus");
    }

    return {night_mode: hour >= 22 || hour < 6};
}
```

### Pattern 2: Inspect Neural Topology

```javascript
function process(inputs, context) {
    // Load default CharmNetwork topology
    var network = context.noodle.neural.load("facet_assemblies/charm_networks/default.nncanvas");

    if (network) {
        var params = network.get_parameter_count();
        context.log("CharmNetwork has " + params + " parameters");

        // Get specific node
        var fast_lstm_id = network.get_node_by_name("Fast_LSTM");
        if (fast_lstm_id) {
            var node = network.get_node(fast_lstm_id);
            context.log("Fast LSTM hidden_dim: " + node.properties.hidden_dim);
        }
    }

    return {inspected: true};
}
```

### Pattern 3: Modify Facet Properties

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

    if (assembly) {
        // Get Red's reasoning facet
        var mind = assembly.get_facet_by_name("Red's Mind");

        if (mind) {
            // Increase creativity
            mind.set_property("temperature", 0.95);
            context.log("Increased Red's creativity to 0.95");
        }
    }

    return {modified: true};
}
```

## Debugging Tips

### Enable Debug Console

1. Open Console panel in NoodleStudio
2. Click **DEBUG** button (alongside MUSH/STUDIO/FACETS)
3. Your `context.log()` calls appear there

### Check Return Values

All API methods return values indicating success/failure:

```javascript
var success = context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
if (!success) {
    context.log("ERROR: Failed to set label");
}
```

### Test in Isolation

Create a dedicated test ScriptedFacet to experiment with API calls without affecting your main agent logic.

## Next Steps

- [API Overview](api/overview.md) - Architecture and design principles
- [ModelsAPI Reference](api/models-api.md) - Provider/model configuration
- [NeuralAPI Reference](api/neural-api.md) - Neural topology manipulation
- [AgentsAPI Reference](api/agents-api.md) - Facet assembly modification
- [Complete Examples](examples.md) - Real-world patterns
