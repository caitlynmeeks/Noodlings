# Troubleshooting

Common issues and solutions when using the Noodlings Scripting API.

## API Not Available

### Problem
`context.noodle` is `undefined` or `null` in JavaScript.

### Symptoms
```javascript
// ERROR: context.noodle is undefined
var models = context.noodle.models;
```

### Solutions

1. **Check facet type**: Only ScriptedFacets have access to the Noodle API. LLMFacets and other facet types do not.

2. **Check Python version**: In Python ScriptedFacets, access via `context._noodle_api`:
   ```python
   noodle = context._noodle_api
   if noodle:
       models = noodle.models
   ```

3. **Check NoodleStudio version**: The API was added in December 2025. Ensure you're running the latest version.

4. **Initialize manually** (if needed):
   ```python
   from noodlestudio.scripting.noodle_api import get_noodle_api
   noodle = get_noodle_api()
   ```

---

## Method Returns Null/False

### Problem
API methods return `null` (JavaScript) or `None` (Python) unexpectedly.

### Symptoms
```javascript
var network = context.noodle.neural.load("topology.nncanvas");
// network is null
```

### Solutions

1. **Check file paths**: Paths must be absolute or relative to NoodleStudio working directory:
   ```javascript
   // ✅ Good
   var network = context.noodle.neural.load("facet_assemblies/charm_networks/default.nncanvas");

   // ❌ Bad
   var network = context.noodle.neural.load("default.nncanvas");
   ```

2. **Check return values**: API methods return `false`/`null` on error for graceful failure:
   ```javascript
   var success = context.noodle.models.set_label("LARGE", "invalid_provider", "model");
   if (!success) {
       context.log("ERROR: Provider not found");
   }
   ```

3. **Verify entity existence**:
   ```javascript
   var assembly = context.noodle.agents.get_assembly("nonexistent-agent");
   if (!assembly) {
       context.log("ERROR: Agent not found");
       return {error: true};
   }
   ```

---

## Model Label Changes Don't Persist

### Problem
Model label changes via `set_label()` don't persist across sessions.

### Symptoms
```javascript
context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
// After restart, LARGE is back to original value
```

### Solutions

1. **Check write permissions**: Ensure `model_labels.json` is writable:
   ```bash
   ls -l ~/.noodlings/model_labels.json
   # Should NOT be read-only
   ```

2. **Check file locks**: Another NoodleStudio instance may have the file locked. Close all instances and retry.

3. **Verify changes in Model Manager**: Open Settings → Models tab in NoodleStudio to verify the change took effect.

---

## Neural Topology Changes Lost

### Problem
Created nodes or connections disappear after script finishes.

### Symptoms
```javascript
var network = context.noodle.neural.load("default.nncanvas");
network.create_node("LSTM", {hidden_dim: 64});
// Node disappears after script exits
```

### Solutions

1. **Save explicitly**: Changes to neural networks are not automatically saved:
   ```javascript
   var network = context.noodle.neural.load("default.nncanvas");
   network.create_node("LSTM", {hidden_dim: 64});
   network.save("modified_network.nncanvas");  // ✅ REQUIRED
   ```

2. **Use unique filenames**: Don't overwrite default topologies:
   ```javascript
   // ✅ Good - create new file
   network.save("custom_topologies/my_network.nncanvas");

   // ❌ Bad - overwrites default
   network.save("facet_assemblies/charm_networks/default.nncanvas");
   ```

---

## Facet Property Changes Don't Apply

### Problem
Changed facet properties (temperature, model, etc.) don't affect behavior.

### Symptoms
```javascript
facet.set_property("temperature", 0.9);
// LLM still uses old temperature
```

### Solutions

1. **Save assembly**: Facet changes must be saved to YAML:
   ```javascript
   var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
   var facet = assembly.get_facet("RED_MIND");
   facet.set_property("temperature", 0.9);
   assembly.save("facet_assemblies/red_fire_anklebiter.yaml");  // ✅ REQUIRED
   ```

2. **Reload assembly**: In NoodleStudio, reload the agent or restart the server to pick up changes.

3. **Check property names**: Property names are case-sensitive:
   ```javascript
   // ✅ Correct
   facet.set_property("temperature", 0.9);

   // ❌ Wrong
   facet.set_property("Temperature", 0.9);
   ```

---

## JavaScript Bridge Errors

### Problem
Placeholder strings appear instead of method results.

### Symptoms
```javascript
var label = context.noodle.models.get_label("SMALL");
// label is "__models_get_label__" (string)
```

### Solutions

1. **Check QuickJS version**: The JavaScript bridge requires PyMiniRacer or QuickJS. Ensure it's installed:
   ```bash
   pip install py-mini-racer
   ```

2. **Use Python directly**: If JavaScript bridge fails, write Python scripts instead:
   ```python
   def execute(context):
       noodle = context._noodle_api
       label = noodle.models.get_label("SMALL")
       return {"provider": label["provider"]}
   ```

---

## DEBUG Console Not Showing Logs

### Problem
`context.log()` calls don't appear in DEBUG console.

### Symptoms
```javascript
context.log("This should appear");
// Nothing in DEBUG console
```

### Solutions

1. **Click DEBUG button**: Open Console panel and click DEBUG (alongside MUSH/STUDIO/FACETS).

2. **Check facet execution**: Logs only appear after facet executes. Verify your facet is running:
   ```javascript
   function process(inputs, context) {
       context.log("Facet executed at cycle " + context.cycle);
       return {executed: true};
   }
   ```

3. **Check noodleScope server**: DEBUG console requires noodleScope API running on port 8081. Check logs:
   ```bash
   tail -f applications/cmush/logs/server_*.log
   ```

---

## Node Creation Fails

### Problem
`create_node()` returns `null`, node not added to network.

### Symptoms
```javascript
var lstm_id = network.create_node("LSTM", {hidden_dim: 64});
// lstm_id is null
```

### Solutions

1. **Check node type spelling**: Node types are case-sensitive:
   ```javascript
   // ✅ Correct
   network.create_node("LSTM", {...});

   // ❌ Wrong
   network.create_node("lstm", {...});
   network.create_node("Lstm", {...});
   ```

2. **Check required properties**: Some nodes require specific properties. See [Node Types](node-types.md) reference.

3. **Check position format**: Position must be `[x, y]` array:
   ```javascript
   // ✅ Correct
   network.create_node("LSTM", {position: [100, 200]});

   // ❌ Wrong
   network.create_node("LSTM", {position: {x: 100, y: 200}});
   ```

---

## MLX Code Generation Fails

### Problem
`generate_mlx_code()` returns `null` or invalid Python.

### Symptoms
```javascript
var code = network.generate_mlx_code();
// code is null or has syntax errors
```

### Solutions

1. **Check topology validity**: Network must have valid data flow (no disconnected nodes, no cycles in feedforward sections):
   ```javascript
   // Ensure all nodes are connected
   var nodes = network.list_nodes();  // Check node list
   ```

2. **Check node types**: Some experimental node types may not support code generation yet.

3. **Test generated code**: Save code to file and test:
   ```python
   # Save in JavaScript
   // var code = network.generate_mlx_code();
   // save_to_file("test_network.py", code);

   # Test in Python
   python test_network.py
   ```

---

## Performance Issues

### Problem
API calls are slow or cause UI freezing.

### Symptoms
- Script execution takes seconds
- NoodleStudio UI freezes
- High CPU usage

### Solutions

1. **Limit neural operations**: Creating many nodes or generating code is expensive:
   ```javascript
   // ❌ Bad - creates 100 nodes every cycle
   for (var i = 0; i < 100; i++) {
       network.create_node("LSTM", {...});
   }

   // ✅ Good - check if already created
   if (!context.storage.nodes_created) {
       for (var i = 0; i < 100; i++) {
           network.create_node("LSTM", {...});
       }
       context.storage.nodes_created = true;
   }
   ```

2. **Cache lookups**: Store frequently accessed objects in `context.storage`:
   ```javascript
   // ✅ Cache assembly lookup
   var assembly = context.storage.assembly;
   if (!assembly) {
       assembly = context.noodle.agents.get_assembly(context.agent.id);
       context.storage.assembly = assembly;
   }
   ```

3. **Use conditional execution**: Only run expensive operations when needed:
   ```javascript
   // Only regenerate code if topology changed
   if (inputs.topology_changed) {
       var code = network.generate_mlx_code();
       context.storage.last_code = code;
   }
   ```

---

## Common Error Messages

### "Provider not found"
Provider ID is invalid or provider not configured. Check [ModelsAPI](api/models-api.md) for valid provider IDs.

### "Model not assigned"
Label has no model assigned. Use Model Manager UI or `set_label()` to assign one.

### "Facet not found"
Facet ID or name doesn't exist in assembly. Use `list_facets()` to see available facets.

### "Invalid node type"
Node type name is misspelled or doesn't exist. See [Node Types](node-types.md) for valid types.

### "Cannot connect ports"
Port names don't exist on specified nodes, or data types are incompatible. Check node definitions.

---

## Getting Help

If your issue isn't covered here:

1. **Check logs**: Look for errors in facet execution logs
2. **Test in isolation**: Create a minimal test ScriptedFacet to reproduce the issue
3. **Verify API version**: Ensure you're using the latest Noodlings version
4. **Check CLAUDE.md**: See if there are known issues or workarounds

## See Also

- [Quick Start Guide](quick-start.md)
- [Complete Examples](examples.md)
- [API Reference](api/overview.md)
