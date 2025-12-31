# Scripting API

JavaScript API for Scripted Facets.

---

## Overview

Scripted Facets execute JavaScript code within the facet execution pipeline.
They receive input from upstream facets and emit output downstream.

## Context Object

Every script receives a `context` object:

```javascript
// Input from upstream facet
let input = context.input;

// Access the Noodle API
let models = await context.noodle.models.listAvailable();

// Return output to downstream facets
return { result: "processed" };
```

## Noodle API

### context.noodle.models

```javascript
// List available models
let models = await context.noodle.models.listAvailable();

// Get/set model labels
let thinkingModel = context.noodle.models.getLabel("thinking");
context.noodle.models.setLabel("speaking", "ollama/llama3.2");
```

### context.noodle.agents

```javascript
// List active agents
let agents = await context.noodle.agents.listAll();

// Get specific agent
let red = await context.noodle.agents.get("red");

// Load assembly
await context.noodle.agents.loadAssembly("red", "assemblies/thinker.yaml");
```

### context.noodle.affect

```javascript
// Get current affect
let affect = context.noodle.affect.get();
// { valence: 0.3, arousal: 0.6, dominance: 0.5, boredom: 0.2, sorrow: 0.1 }

// Modify affect
context.noodle.affect.set({ valence: 0.8 });
```

### context.noodle.pose

```javascript
// Get current pose
let pose = context.noodle.pose.get();

// Set bone rotation
context.noodle.pose.setBoneRotation("head", { x: 0, y: 15, z: 0 });
```

## Async/Await

Scripts can be async:

```javascript
async function process(context) {
    let response = await context.noodle.models.complete({
        prompt: context.input,
        model_label: "thinking"
    });
    return response;
}

return await process(context);
```

## Error Handling

```javascript
try {
    let result = await riskyOperation();
    return { success: true, data: result };
} catch (error) {
    return { success: false, error: error.message };
}
```

## Logging

```javascript
console.log("Debug info");  // Appears in server logs
```
