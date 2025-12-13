# get_type()

Get the facet type.

**Class**: FacetProxy

**Access**: `facet.get_type()`

## Parameters

None

## Returns

String representing facet type.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

var facets = assembly.list_facets();

for (var i = 0; i < facets.length; i++) {
    var facet = assembly.get_facet(facets[i].uuid);
    var type = facet.get_type();
    var name = facet.get_name();

    context.log(name + " is a " + type);
}
```

## Output Example

```
INCOMING is a IncomingNode
CHARM_NET is a CharmNetworkFacet
Red's Mind is a LLMFacet
Fire Body is a LLMFacet
OUTGOING is a OutgoingNode
```

## Facet Types

- `LLMFacet` - Language model reasoner
- `ScriptedFacet` - JavaScript/Python code
- `CharmNetworkFacet` - Neural network
- `ContextIntelligenceFacet` - Social context parser
- `ConvergenceFacet` - Multi-input synthesis
- `IncomingNode` - Entry point (system)
- `OutgoingNode` - Exit point (system)

## See Also

- [get_name()](get_name.md) - Get facet name
- [get_id()](get_id.md) - Get facet UUID
- [FacetProxy](../../api/facet-proxy.md) - Complete class reference
