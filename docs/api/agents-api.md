# AgentsAPI

Scriptable interface to agent facet assemblies.

**Location**: `noodlestudio/scripting/agents_api.py`

**Access**: `context.noodle.agents`

## Overview

The AgentsAPI allows scripts to:

- Access agent facet assemblies
- Query facet properties (model, temperature, prompts)
- Modify facet properties dynamically
- Add/remove facets from assemblies
- Connect/disconnect facet data flows
- Save modified assemblies to YAML

All modifications happen through **FacetAssemblyProxy** and **FacetProxy** objects.

## AgentsAPI Methods

### `get(agent_id)`

Get agent information by ID.

**Parameters**:

- `agent_id` (string) - Agent identifier (e.g., `"red-fire-anklebiter"`)

**Returns**: Object with `{id, name, species, assembly}` or `null`

**Example**:
```javascript
var agent = context.noodle.agents.get("red-fire-anklebiter");
if (agent) {
    context.log("Agent: " + agent.name);
    context.log("Species: " + agent.species);
}
```

---

### `get_assembly(agent_id)`

Get facet assembly for an agent.

**Parameters**:

- `agent_id` (string) - Agent identifier

**Returns**: `FacetAssemblyProxy` instance or `null`

**Example**:
```javascript
var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
if (assembly) {
    var facets = assembly.list_facets();
    context.log("Assembly has " + facets.length + " facets");
}
```

---

### `load_assembly(agent_id, filepath)`

Load assembly from YAML file.

**Parameters**:

- `agent_id` (string) - Agent identifier (for registration)
- `filepath` (string) - Path to `.yaml` facet assembly file

**Returns**: `FacetAssemblyProxy` instance or `null`

**Example**:
```javascript
var assembly = context.noodle.agents.load_assembly(
    "custom-agent",
    "facet_assemblies/custom_agent.yaml"
);
```

---

### `list_all()`

List all registered agent IDs.

**Parameters**: None

**Returns**: Array of agent ID strings

**Example**:
```javascript
var agents = context.noodle.agents.list_all();
agents.forEach(function(agent_id) {
    context.log("Agent: " + agent_id);
});
```

---

## FacetAssemblyProxy Methods

Proxy object for a facet assembly (cognitive topology).

### `get_facet(facet_id)`

Get facet by ID.

**Parameters**:

- `facet_id` (string) - Facet UUID or ID from YAML

**Returns**: `FacetProxy` instance or `null`

**Example**:
```javascript
var charm_facet = assembly.get_facet("CHARM_NET");
if (charm_facet) {
    context.log("Facet type: " + charm_facet.get_type());
}
```

---

### `get_facet_by_name(name)`

Get facet by display name.

**Parameters**:

- `name` (string) - Facet name (e.g., `"Red's Mind"`, `"Context Intelligence"`)

**Returns**: `FacetProxy` instance or `null`

**Example**:
```javascript
var mind = assembly.get_facet_by_name("Red's Mind");
if (mind) {
    var model = mind.get_property("model");
    context.log("Red's Mind uses: " + model);
}
```

---

### `list_facets()`

List all facets in the assembly.

**Parameters**: None

**Returns**: Array of `{id, name, type}` objects

**Example**:
```javascript
var facets = assembly.list_facets();
context.log("=== Facets in Assembly ===");
facets.forEach(function(f) {
    context.log(f.id + ": " + f.name + " (" + f.type + ")");
});

// Output:
// CHARM_NET: CharmNetwork (CharmNetworkFacet)
// CONTEXT_INTELLIGENCE: Context Intelligence (ContextIntelligenceFacet)
// RED_MIND: Red's Mind (LLMFacet)
```

---

### `add_facet(facet_type, name, properties)`

Add a new facet to the assembly.

**Parameters**:

- `facet_type` (string) - Facet type (e.g., `"LLMFacet"`, `"ScriptedFacet"`)
- `name` (string) - Display name for the facet
- `properties` (object, optional) - Initial properties:
    - `model` (string) - Model label (for LLMFacet)
    - `temperature` (number) - Temperature (for LLMFacet)
    - `prompt` (string) - System prompt (for LLMFacet)
    - `script` (string) - JavaScript code (for ScriptedFacet)

**Returns**: Facet ID (string) or `null` on failure

**Example**:
```javascript
var facet_id = assembly.add_facet("LLMFacet", "Custom Reasoner", {
    model: "LARGE",
    temperature: 0.8,
    prompt: "You are a creative problem solver."
});

if (facet_id) {
    context.log("Created facet: " + facet_id);
}
```

**Supported Facet Types**:

- `LLMFacet` - Language model reasoning
- `ScriptedFacet` - JavaScript/Python sandbox
- `CharmNetworkFacet` - Neural affect processing
- `ContextIntelligenceFacet` - Social context parsing
- `ConvergenceFacet` - Multi-input synthesis

---

### `remove_facet(facet_id)`

Remove a facet from the assembly.

**Parameters**:

- `facet_id` (string) - Facet ID

**Returns**: `true` if removed successfully, `false` on error

**Example**:
```javascript
var removed = assembly.remove_facet("OLD_FACET");
if (removed) {
    context.log("Facet removed");
}
```

**Warning**: Removing facets can break data flow if connections exist!

---

### `connect(from_facet, from_pad, to_facet, to_pad)`

Connect two facets via their data pads.

**Parameters**:

- `from_facet` (string) - Source facet ID
- `from_pad` (string) - Source pad name (e.g., `"affect_valence"`, `"result"`)
- `to_facet` (string) - Target facet ID
- `to_pad` (string) - Target pad name (e.g., `"affect"`, `"data"`)

**Returns**: `true` if connected successfully, `false` on error

**Example**:
```javascript
// Connect CharmNetwork affect to reasoning facet
var success = assembly.connect(
    "CHARM_NET",
    "affect_valence",
    "RED_MIND",
    "affect"
);

if (success) {
    context.log("Connected affect flow");
}
```

**Common Pad Names**:

- CharmNetworkFacet: `affect_valence`, `affect_arousal`, `affect_dominance`, `affect_boredom`, `affect_sorrow`
- LLMFacet: `incoming_data`, `observations`, `affect` (inputs); `result` (output)
- ScriptedFacet: Dynamic (defined in script)

---

### `disconnect(from_facet, from_pad, to_facet, to_pad)`

Disconnect two facets.

**Parameters**: Same as `connect()`

**Returns**: `true` if disconnected successfully, `false` on error

**Example**:
```javascript
assembly.disconnect("CHARM_NET", "affect_valence", "RED_MIND", "affect");
```

---

### `save(filepath)`

Save modified assembly to YAML file.

**Parameters**:

- `filepath` (string) - Path to save file (e.g., `"modified_red.yaml"`)

**Returns**: `true` if saved successfully, `false` on error

**Example**:
```javascript
var saved = assembly.save("facet_assemblies/red_modified.yaml");
if (saved) {
    context.log("Assembly saved");
}
```

**Format**: YAML file compatible with NoodleStudio facet system

---

## FacetProxy Methods

Proxy object for a single facet.

### `get_property(name)`

Get a facet property value.

**Parameters**:

- `name` (string) - Property name (e.g., `"model"`, `"temperature"`, `"prompt"`)

**Returns**: Property value (any type) or `null`

**Example**:
```javascript
var facet = assembly.get_facet("RED_MIND");

var model = facet.get_property("model");
var temp = facet.get_property("temperature");
var prompt = facet.get_property("prompt");

context.log("Model: " + model);
context.log("Temperature: " + temp);
context.log("Prompt length: " + prompt.length);
```

---

### `set_property(name, value)`

Set a facet property value.

**Parameters**:

- `name` (string) - Property name
- `value` (any) - New value

**Returns**: `true` if set successfully, `false` on error

**Example**:
```javascript
var facet = assembly.get_facet("RED_MIND");

// Increase creativity
facet.set_property("temperature", 0.95);

// Switch to larger model
facet.set_property("model", "LARGE");

// Update prompt
facet.set_property("prompt", "You are a wildly creative thinker.");

context.log("Updated facet properties");
```

**Common Properties**:

- `model` (string) - Model label (SMALL/MEDIUM/LARGE)
- `temperature` (number) - Sampling temperature (0.0-2.0)
- `max_tokens` (number) - Max response length
- `prompt` (string) - System prompt
- `script` (string) - JavaScript code (ScriptedFacet only)

---

### `get_all_properties()`

Get all facet properties as an object.

**Parameters**: None

**Returns**: Object with all properties

**Example**:
```javascript
var facet = assembly.get_facet("RED_MIND");
var props = facet.get_all_properties();

for (var key in props) {
    context.log(key + ": " + props[key]);
}
```

---

### `get_type()`

Get facet type.

**Parameters**: None

**Returns**: Type string (e.g., `"LLMFacet"`, `"ScriptedFacet"`)

**Example**:
```javascript
var type = facet.get_type();
context.log("Facet type: " + type);
```

---

### `get_id()`

Get facet ID.

**Parameters**: None

**Returns**: ID string

**Example**:
```javascript
var id = facet.get_id();
context.log("Facet ID: " + id);
```

---

### `get_name()`

Get facet display name.

**Parameters**: None

**Returns**: Name string

**Example**:
```javascript
var name = facet.get_name();
context.log("Facet name: " + name);
```

---

## Complete Example: Adaptive Temperature

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
    var mind = assembly.get_facet_by_name("Red's Mind");

    if (!mind) {
        context.log("ERROR: Could not find Red's Mind facet");
        return {error: true};
    }

    // Get current affect from inputs
    var valence = inputs.affect_valence || 0.0;
    var arousal = inputs.affect_arousal || 0.5;

    // Calculate adaptive temperature
    // High arousal + positive valence = higher creativity
    var base_temp = 0.7;
    var arousal_boost = arousal * 0.3;  // 0.0 to 0.3
    var valence_boost = (valence > 0) ? valence * 0.2 : 0;  // 0.0 to 0.2

    var new_temp = base_temp + arousal_boost + valence_boost;
    new_temp = Math.min(new_temp, 1.2);  // Cap at 1.2

    // Apply new temperature
    mind.set_property("temperature", new_temp);

    context.log("Affect-driven temperature: " + new_temp.toFixed(2));
    context.log("  Valence: " + valence.toFixed(2));
    context.log("  Arousal: " + arousal.toFixed(2));

    return {
        temperature: new_temp,
        valence: valence,
        arousal: arousal
    };
}
```

## Complete Example: Self-Modifying Assembly

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly(context.agent.id);

    // Check if we already have an episodic memory facet
    var memory = assembly.get_facet_by_name("Episodic Memory");

    if (!memory && inputs.requires_memory) {
        // Add memory facet if needed
        context.log("Adding episodic memory facet...");

        var memory_id = assembly.add_facet("ScriptedFacet", "Episodic Memory", {
            script: `
                function process(inputs, context) {
                    // Store important events in context.storage
                    if (inputs.importance > 0.7) {
                        var memories = context.storage.memories || [];
                        memories.push({
                            timestamp: context.timestamp,
                            data: inputs.data,
                            importance: inputs.importance
                        });
                        context.storage.memories = memories;
                    }
                    return {stored: true};
                }
            `
        });

        if (memory_id) {
            // Connect to data flow
            assembly.connect("CONTEXT_INTELLIGENCE", "result", memory_id, "data");
            context.log("Memory facet added and connected");

            // Save modified assembly
            assembly.save("facet_assemblies/" + context.agent.id + "_with_memory.yaml");
        }
    }

    return {
        has_memory: !!memory,
        added_memory: !!memory_id
    };
}
```

## Complete Example: Facet Inspector

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

    context.log("=== Red Fire Anklebiter Assembly ===");

    var facets = assembly.list_facets();
    context.log("Total facets: " + facets.length);
    context.log("");

    facets.forEach(function(f) {
        var facet = assembly.get_facet(f.id);
        if (facet) {
            context.log("--- " + f.name + " ---");
            context.log("  Type: " + f.type);
            context.log("  ID: " + f.id);

            var props = facet.get_all_properties();
            for (var key in props) {
                var value = props[key];
                // Truncate long strings
                if (typeof value === "string" && value.length > 50) {
                    value = value.substring(0, 47) + "...";
                }
                context.log("  " + key + ": " + value);
            }
            context.log("");
        }
    });

    return {inspected: facets.length};
}
```

## See Also

- [ModelsAPI Reference](models-api.md)
- [NeuralAPI Reference](neural-api.md)
- [Complete Examples](../examples.md)
- [Quick Start Guide](../quick-start.md)
