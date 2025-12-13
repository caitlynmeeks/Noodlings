# save()

Save facet assembly to YAML file.

**Class**: FacetAssemblyProxy

**Access**: `assembly.save(file_path)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `file_path` | string | Path to save YAML file (relative or absolute) |

## Returns

`true` if saved successfully, `false` on failure.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

// Modify assembly
var reasoner = assembly.get_facet_by_name("Red's Mind");
reasoner.set_property("model", "LARGE");
reasoner.set_property("temperature", 0.9);

// Add new facet
assembly.add_facet("LLMFacet", "Roast Engine", {
    model: "MEDIUM",
    prompt: "Generate creative roasts..."
});

// Save modified assembly
var saved = assembly.save("modified_red.yaml");

if (saved) {
    context.log("Assembly saved successfully");
} else {
    context.log("Failed to save assembly");
}
```

## See Also

- [load_assembly()](../agents/load_assembly.md) - Load assembly from file
- [add_facet()](add_facet.md) - Add facets
- [connect()](connect.md) - Connect facets
