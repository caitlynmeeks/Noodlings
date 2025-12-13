# get()

Get an agent by UUID.

**Class**: AgentsAPI

**Access**: `context.noodle.agents.get(agent_uuid)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `agent_uuid` | string | Agent UUID |

## Returns

Agent object or `null` if not found.

## Example

```javascript
var agent = context.noodle.agents.get("550e8400-e29b-41d4-a716-446655440000");

if (agent) {
    context.log("Found agent: " + agent.name);
} else {
    context.log("Agent not found");
}
```

## See Also

- [get_assembly()](get_assembly.md) - Get facet assembly for agent
- [list_all()](list_all.md) - List all agents
- [AgentsAPI](../../api/agents-api.md) - Complete class reference
