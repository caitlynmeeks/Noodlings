# Facet Assemblies

Visual cognitive architecture for Noodlings.

---

## What is a Facet Assembly?

A facet assembly is a **node graph** that defines how a Noodling thinks.

```
INCOMING → [Perception] → [Reasoning] → [Response] → OUTGOING
              ↓               ↑
         [Memory] ←→ [Affect]
```

Each node is a **facet** - a cognitive transformation unit.

## Why Visual?

- **Inspectable**: See exactly how cognition flows
- **Editable**: Rewire thinking without code
- **Shareable**: Export assemblies as YAML
- **Debuggable**: Watch data flow in real-time

## Facet Types

### LLM Facets
Call language models with structured prompts.

### Scripted Facets
JavaScript code for custom logic.

### Flow Control
- **Branch**: Conditional routing
- **Ticker**: Periodic triggers
- **RateLimiter**: Throttle throughput
- **Cache**: Memoize results
- **Accumulator**: Batch inputs

### Special Nodes
- **INCOMING**: Entry point (receives perception)
- **OUTGOING**: Exit point (emits actions)
- **Convergence**: Multi-input synthesis

## Assembly Format

```yaml
name: simple_thinker
facets:
  - id: incoming
    type: INCOMING
    position: [100, 200]

  - id: think
    type: LLMFacet
    position: [300, 200]
    properties:
      model_label: thinking
      system_prompt: "You are a thoughtful character..."

  - id: outgoing
    type: OUTGOING
    position: [500, 200]

connections:
  - from: incoming
    to: think
  - from: think
    to: outgoing
```

## Editing Assemblies

In NoodleStudio:
1. Open the Facets Editor panel
2. Load a Noodling
3. Drag nodes, draw connections
4. Save (writes to assembly.yaml)

## Next

- [NoodleStudio Overview](../noodlestudio/overview.md) - The visual editor
- [Neural Canvas](../noodlestudio/neural-canvas.md) - Advanced visual programming
