# Facet System

**The cognitive architecture of Noodlings**

---

## Overview

Every Noodling has a **Facet Assembly** - a visual node graph that defines how it thinks. Facets are the modular cognitive units that process perception, generate responses, and shape behavior.

Think of facets like Unity components, but for cognition rather than physics or rendering.

---

## Key Concepts

### Facet Assembly

A **Facet Assembly** is a directed graph of connected facets. Each Noodling references one assembly in its recipe:

```yaml
# recipe.yaml
facet_assembly: "library/empty_noodling"
```

Assemblies are stored as YAML files and can be:
- **Library templates** (`library/empty_noodling`) - Shared starting points
- **Per-noodling** (`Noodlings/Red/assembly.yaml`) - Custom configurations

### Facet Types

| Type | Purpose |
|------|---------|
| `INCOMING` | Entry point - receives perception events |
| `OUTGOING` | Exit point - emits responses |
| `LLMFacet` | Language model processing |
| `CharmNetworkFacet` | Temporal affect model (LSTM/GRU) |
| `ScriptedFacet` | Custom JavaScript logic |
| `ConvergenceFacet` | Multi-input synthesis |
| `ContextIntelligenceFacet` | Memory and context management |
| `TickerFacet` | Periodic triggers |
| `BranchFacet` | Conditional routing |
| `CacheFacet` | Response caching |
| `RateLimiterFacet` | Throttling |

### Connections

Facets connect via typed ports:
- **Input ports** - Receive data from upstream facets
- **Output ports** - Send data to downstream facets

Data flows from INCOMING through processing facets to OUTGOING.

---

## Editing Facets

### Facets Editor (Center Tab)

The visual node editor for facet assemblies:
- Drag nodes to position
- Click ports to create connections
- Right-click for context menu (add/remove facets)
- Double-click nodes to edit properties

### Inspector Panel

When a Noodling is selected, the Inspector shows:

1. **Identity** - Name, species, description
2. **Affect Baseline** - Starting emotional state (5D: valence, arousal, dominance, boredom, sorrow)
3. **Facet Dropdown** - Select a facet to edit its properties

The facet dropdown lists all facets in the assembly. Selecting one shows:
- Facet type and name
- Configurable parameters (model, temperature, etc.)
- Enable/disable toggle

---

## Noodle Component

The **Noodle Component** section in the Inspector displays live runtime data:

- **Affect Vector** - Current emotional state (updates in real-time)
- **Surprise Metric** - How unexpected recent events were

This is not a "component" you add - it's automatic telemetry for all Noodlings.

---

## Historical Context

### Deprecated: Cognitive Components (Nov 2025)

An earlier design used "Cognitive Transistors":
- `CulturalTransistor` - Belief-based filtering
- `PersonalityTransistor` - Trait-based coloring
- `MoodTransistor` - Affect-based interpretation
- `CognitiveManifold` - Integration layer

This system was **replaced** by Facet Assemblies. The `cognitive_components` field in recipes is deprecated.

### Why Facets Won

1. **Visual editing** - Node graphs are more intuitive than config files
2. **Flexibility** - Any topology, not just predefined transistor types
3. **Scriptability** - ScriptedFacet allows custom JavaScript logic
4. **Shareability** - Assemblies export as YAML (like Unity prefabs)

---

## Creating a New Facet Assembly

### From Library Template

1. Create a new Noodling (Assets > New Noodling)
2. The recipe references `library/empty_noodling` by default
3. Open Facets Editor to customize

### From Scratch

1. Create `assembly.yaml` in the Noodling folder:

```yaml
name: "My Custom Assembly"
facets:
  - id: "incoming"
    type: "INCOMING"
    name: "Input"
    position: [100, 200]

  - id: "llm_main"
    type: "LLMFacet"
    name: "Main Reasoning"
    position: [300, 200]
    config:
      model_label: "MEDIUM"
      temperature: 0.7
      max_tokens: 150

  - id: "outgoing"
    type: "OUTGOING"
    name: "Output"
    position: [500, 200]

connections:
  - from: "incoming"
    to: "llm_main"
  - from: "llm_main"
    to: "outgoing"
```

2. Reference it in `recipe.yaml`:

```yaml
facet_assembly: "Noodlings/MyNoodling"
```

---

## Execution

The **Facet Executor** (`facet_executor.py`) runs assemblies:

1. Perception event arrives at INCOMING facet
2. Data propagates through connected facets
3. Each facet processes and emits output
4. Parallel branches execute concurrently
5. ConvergenceFacet waits for all inputs before proceeding
6. Final output emitted from OUTGOING facet

### Execution Statistics

Each facet tracks:
- Call count
- Total execution time
- Token usage (for LLM facets)

View in Inspector by selecting the facet.

---

## Scripted Facets

For custom logic, use `ScriptedFacet` with JavaScript:

```javascript
// Available context:
// - input: Data from upstream facet
// - context.noodle.affect: Current affect state
// - context.noodle.models: LLM access

async function process(input) {
    // Custom processing
    const modified = input.toUpperCase();

    // Access affect
    const valence = await context.noodle.affect.get('valence');

    // Conditional logic
    if (valence < 0) {
        return { text: modified, mood: 'sad' };
    }
    return { text: modified, mood: 'neutral' };
}
```

See [Scripting API](scripting.md) for full API reference.

---

## Best Practices

1. **Start simple** - Use library templates, customize incrementally
2. **Name meaningfully** - "Emotional Filter" not "LLM_2"
3. **Document assemblies** - Add description to assembly YAML
4. **Test changes** - Use the Facets Editor preview before saving
5. **Share templates** - Export working assemblies for reuse

---

## File Locations

| File | Location |
|------|----------|
| Library templates | `library/noodlings/{name}/assembly.yaml` |
| Per-noodling assemblies | `{Project}/Noodlings/{name}/assembly.yaml` |
| Facet type definitions | `noodlestudio/core/facet_system.py` |
| Executor | `noodlestudio/core/facet_executor.py` |
| Visual editor | `noodlestudio/panels/facets_editor_panel.py` |

---

## See Also

- [Neural Canvas](neural-canvas.md) - For designing neural network components
- [Scripting API](scripting.md) - For ScriptedFacet development
- [Panels Reference](panels.md) - UI overview
