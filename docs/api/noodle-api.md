# NoodleAPI

Main entry point for the Noodlings Scripting API.

**Location**: `noodlestudio/scripting/noodle_api.py`

## Overview

The `NoodleAPI` class provides access to all sub-APIs:

```javascript
context.noodle
  ├─ .models      // ModelsAPI
  ├─ .neural      // NeuralAPI
  ├─ .agents      // AgentsAPI
  └─ .get_by_uuid // Universal lookup (future)
```

## Properties

### `models`

Access the ModelsAPI for provider/model configuration.

**Type**: `ModelsAPI`

**Example**:
```javascript
var assignment = context.noodle.models.get_label("SMALL");
```

---

### `neural`

Access the NeuralAPI for neural topology manipulation.

**Type**: `NeuralAPI`

**Example**:
```javascript
var network = context.noodle.neural.create_network("MyNetwork");
```

---

### `agents`

Access the AgentsAPI for facet assembly modification.

**Type**: `AgentsAPI`

**Example**:
```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
```

## Methods

### `get_by_uuid(uuid)`

Get any entity by UUID (future enhancement).

**Parameters**:

- `uuid` (string) - Universal unique identifier

**Returns**: Entity dict with `{type, properties, methods}` or `null`

**Example**:
```javascript
var entity = context.noodle.get_by_uuid("550e8400-e29b-41d4-a716-446655440000");
if (entity) {
    context.log("Found " + entity.type);
}
```

**Status**: Not yet implemented (returns `null`)

## Implementation Details

### Singleton Pattern

The API uses a global singleton accessible via `get_noodle_api()`:

```python
# Python
from noodlestudio.scripting.noodle_api import get_noodle_api

api = get_noodle_api()
```

### Lazy Initialization

Sub-APIs initialize only when first accessed, minimizing startup overhead.

### JavaScript Bridge

The `to_dict()` method converts the API to JavaScript-compatible placeholders:

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        'models': self.models.to_dict(),
        'neural': self.neural.to_dict(),
        'agents': self.agents.to_dict(),
        'get_by_uuid': '__noodle_get_by_uuid__'
    }
```

## Usage Example

```javascript
function process(inputs, context) {
    // Access all sub-APIs through context.noodle
    var models = context.noodle.models;
    var neural = context.noodle.neural;
    var agents = context.noodle.agents;

    // Check what model is assigned to LARGE
    var large = models.get_label("LARGE");
    context.log("Using: " + large.provider + "/" + large.model);

    // Load a neural topology
    var network = neural.load("default.nncanvas");
    if (network) {
        context.log("Loaded network with " + network.get_parameter_count() + " params");
    }

    // Get agent assembly
    var assembly = agents.get_assembly("red-fire-anklebiter");
    var facets = assembly.list_facets();
    context.log("Assembly has " + facets.length + " facets");

    return {explored: true};
}
```

## See Also

- [ModelsAPI Reference](models-api.md)
- [NeuralAPI Reference](neural-api.md)
- [AgentsAPI Reference](agents-api.md)
