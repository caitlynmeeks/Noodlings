# load_assembly()

Load facet assembly from YAML file.

**Class**: AgentsAPI

**Access**: `context.noodle.agents.load_assembly(file_path)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `file_path` | string | Path to YAML file (relative or absolute) |

## Returns

[FacetAssemblyProxy](../../api/facet-assembly-proxy.md) object or `null` on failure.

## Example

```javascript
var assembly = context.noodle.agents.load_assembly("custom_agent.yaml");

if (assembly) {
    var facets = assembly.list_facets();
    context.log("Loaded " + facets.length + " facets");
} else {
    context.log("Failed to load assembly");
}
```

## See Also

- [get_assembly()](get_assembly.md) - Get existing assembly
- [FacetAssemblyProxy.save()](../assembly/save.md) - Save assembly to YAML
- [AgentsAPI](../../api/agents-api.md) - Complete class reference
