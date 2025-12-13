# list_facets()

List all facets in the assembly.

**Class**: FacetAssemblyProxy

**Access**: `assembly.list_facets()`

## Parameters

None

## Returns

Array of objects with `{uuid, name, type}` properties.

## Example

```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

var facets = assembly.list_facets();

context.log("Assembly contains " + facets.length + " facets:");

for (var i = 0; i < facets.length; i++) {
    var f = facets[i];
    context.log("- " + f.name + " (" + f.type + ")");
}
```

## Output Example

```
Assembly contains 5 facets:
- INCOMING (IncomingNode)
- CHARM_NET (CharmNetworkFacet)
- Red's Mind (LLMFacet)
- Fire Body (LLMFacet)
- OUTGOING (OutgoingNode)
```

## See Also

- [get_facet()](get_facet.md) - Get specific facet
- [add_facet()](add_facet.md) - Add new facet
- [remove_facet()](remove_facet.md) - Remove facet
