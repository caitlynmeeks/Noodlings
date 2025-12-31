# Assembly Format

YAML format for cognitive architectures.

---

## Overview

`assembly.yaml` defines a Noodling's cognitive architecture as a node graph.
It specifies which facets (cognitive units) exist and how they connect.

## Location

```
Noodlings/
└── red/
    ├── recipe.yaml
    └── assembly.yaml   # This file
```

## Schema

```yaml
name: red_cognition
version: 1

facets:
  - id: incoming
    type: INCOMING
    position: [100, 200]

  - id: perception
    type: LLMFacet
    position: [300, 200]
    properties:
      name: "Perception"
      model_label: perception
      system_prompt: |
        Analyze the incoming perception and extract key information.
        Focus on: who is present, what they said, environmental details.

  - id: thinking
    type: LLMFacet
    position: [500, 200]
    properties:
      name: "Thinking"
      model_label: thinking
      system_prompt: |
        Given the perception analysis, decide how to respond.
        Consider your personality and current emotional state.

  - id: speaking
    type: LLMFacet
    position: [700, 200]
    properties:
      name: "Speaking"
      model_label: speaking
      system_prompt: |
        Generate dialogue in character voice.

  - id: outgoing
    type: OUTGOING
    position: [900, 200]

connections:
  - from: incoming
    to: perception
  - from: perception
    to: thinking
  - from: thinking
    to: speaking
  - from: speaking
    to: outgoing
```

## Facet Types

### INCOMING / OUTGOING
Entry and exit points. Every assembly needs exactly one of each.

### LLMFacet
Calls a language model.

```yaml
type: LLMFacet
properties:
  name: "Display Name"
  model_label: thinking       # References config model labels
  system_prompt: "..."
  temperature: 0.7
  max_tokens: 500
```

### ScriptedFacet
Executes JavaScript.

```yaml
type: ScriptedFacet
properties:
  name: "Custom Logic"
  script: |
    let input = context.input;
    return { processed: input.toUpperCase() };
```

### CharmNetworkFacet
Runs a neural network (.nncanvas).

```yaml
type: CharmNetworkFacet
properties:
  network: networks/affect_predictor.nncanvas
```

### ConvergenceFacet
Combines multiple inputs.

```yaml
type: ConvergenceFacet
properties:
  strategy: merge  # or: first, last, concat
```

### Flow Control

```yaml
type: Branch
properties:
  condition: "input.confidence > 0.8"

type: Ticker
properties:
  interval_ms: 5000

type: RateLimiter
properties:
  max_per_second: 2
```

## Connections

```yaml
connections:
  - from: facet_id
    to: other_facet_id
    # Optional: specify ports for multi-output facets
    from_port: output_name
    to_port: input_name
```

## Loading in Code

```python
from noodlestudio.core.facet_system import FacetAssembly

assembly = FacetAssembly.from_yaml("assembly.yaml")
for facet in assembly.facets:
    print(f"{facet.id}: {facet.type}")
```
