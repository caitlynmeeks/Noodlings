# list_all()

List all agents in the system.

**Class**: AgentsAPI

**Access**: `context.noodle.agents.list_all()`

## Parameters

None

## Returns

Array of agent objects with `{uuid, name}` properties.

## Example

```javascript
var agents = context.noodle.agents.list_all();

context.log("Found " + agents.length + " agents:");

for (var i = 0; i < agents.length; i++) {
    context.log("- " + agents[i].name + " (" + agents[i].uuid + ")");
}
```

## See Also

- [get()](get.md) - Get specific agent
- [get_assembly()](get_assembly.md) - Get agent's facet assembly
- [AgentsAPI](../../api/agents-api.md) - Complete class reference
