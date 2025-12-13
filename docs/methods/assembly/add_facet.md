# add_facet()

Add new facet to the assembly.

**Class**: FacetAssemblyProxy

**Access**: `assembly.add_facet(facet_type, name, properties)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `facet_type` | string | Facet type (e.g., "LLMFacet", "ScriptedFacet") |
| `name` | string | Facet name |
| `properties` | object | Facet properties (optional) |

## Returns

[FacetProxy](../../api/facet-proxy.md) object or `null` on failure.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

// Add LLM facet
var reasoner = assembly.add_facet("LLMFacet", "Custom Reasoner", {
    model: "LARGE",
    temperature: 0.9,
    prompt: "You are a creative thinker..."
});

if (reasoner) {
    context.log("Created facet: " + reasoner.get_id());
}

// Add scripted facet
var filter = assembly.add_facet("ScriptedFacet", "Data Filter", {
    language: "javascript",
    code: "function process(inputs, context) { return inputs; }"
});
```

## Supported Facet Types

- `LLMFacet` - Language model reasoner
- `ScriptedFacet` - JavaScript/Python code
- `ConvergenceFacet` - Multi-input synthesis
- `ContextIntelligenceFacet` - Social context parser

## See Also

- [remove_facet()](remove_facet.md) - Remove facet
- [connect()](connect.md) - Connect facets
- [save()](save.md) - Save assembly
