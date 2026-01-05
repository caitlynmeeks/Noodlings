# Facet Assembly Component

**Facets as Universal Components - Visual Logic for Everything**

---

## The Big Idea

**FacetAssemblyComponent** is a fundamental architectural unification: Facet Assemblies are now **attachable components** that can go on ANY entity type:

- **Noodlings** - As always, assemblies drive cognition
- **Prims/Props** - Objects can have reactive logic
- **UI Elements** - Buttons, panels, inputs can trigger assemblies

This makes Facets THE universal visual programming language for NoodleStudio.

---

## Core Concepts

### The Checkbox

The key UX element is a simple checkbox:

```
[x] Run in cognition loop
```

- **CHECKED (Continuous)**: Assembly runs every `tick_rate` seconds, like a Noodling's ongoing thoughts
- **UNCHECKED (One-shot)**: Assembly runs on-demand via events, scripts, or UI triggers

### Multiple Assemblies Per Entity

Unlike most components (which are singletons), an entity can have **multiple** FacetAssemblyComponents:

```
Treasure Chest (Prim)
  +-- FacetAssemblyComponent: "proximity-detector" [x] continuous
  +-- FacetAssemblyComponent: "open-animation"     [ ] one-shot
  +-- FacetAssemblyComponent: "loot-generator"     [ ] one-shot
```

---

## Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `assembly_path` | file | - | Path to assembly YAML file |
| `run_in_cognition_loop` | bool | false | Continuous vs one-shot mode |
| `tick_rate` | float | 0.1 | Seconds between ticks (0.01-60s) |
| `auto_run_on_attach` | bool | false | Run once when component added |

---

## Events

FacetAssemblyComponent emits these events:

| Event | When | Payload |
|-------|------|---------|
| `OnComplete` | One-shot execution finishes | `{response, outputs, tokens, time}` |
| `OnStateChange` | Continuous output changes | `{outputs, previous}` |
| `OnError` | Execution fails | `{error}` |

### Listening to Events

```python
# In Python (entity code)
assembly = entity.get_component("facet_assembly", "sentiment")
assembly.add_listener('complete', lambda e: print(f"Done: {e.data}"))
```

```yaml
# In UI Canvas (event binding)
Button:
  events:
    onClick:
      action: run_assembly
      assembly: assemblies/translate.yaml
```

---

## Input/Output Bindings

Assemblies can bind their inputs and outputs to UI components or other properties:

### Input Bindings

Map UI component values to assembly input pads:

```
Input Bindings:
  text: {text_field.value}
  language: {dropdown.value}
```

Syntax: `{component_name.property}` or `{event.value}`

### Output Bindings

Map assembly output pads to UI component properties:

```
Output Bindings:
  result: result_label.text
  confidence: confidence_bar.value
```

Syntax: `component_name.property`

---

## UI Canvas Integration

### The `run_assembly` Action

A new action type for UI Canvas event bindings:

```yaml
Panel:
  name: translator_panel
  children:
    - type: TextInput
      name: text_field

    - type: Button
      name: translate_btn
      text: "Translate"
      events:
        onClick:
          action: run_assembly
          assembly: assemblies/translate-chinese.yaml
          inputs:
            text: "{text_field.value}"
          outputs:
            result: result_label.text

    - type: Label
      name: result_label
      text: ""
```

### Input Resolution

The `inputs` field supports three syntax forms:

| Syntax | Example | Description |
|--------|---------|-------------|
| Static value | `"Hello"` | Literal string/number |
| Component ref | `"{input.value}"` | Get value from UI component |
| Event ref | `"{event.value}"` | Get value from triggering event |

### Output Resolution

Outputs are applied to UI components after execution:

```yaml
outputs:
  result: "result_label.text"      # Set label text
  sentiment: "mood_icon.color"     # Set color
  confidence: "progress.value"     # Set slider value
```

---

## Inspector UI

When a FacetAssemblyComponent is selected, the Inspector shows:

```
+-- Facet Assembly: sentiment-analysis ------+
| Assembly:    [sentiment.yaml         ] [R] |
| [x] Run in cognition loop                  |
| Tick Rate:   [0.1    ] seconds             |
+-- Input Bindings --------------------------+
| out:         [{text_field.value}     ] [x] |
+-- Output Bindings -------------------------+
| in:          [result_label.text      ] [x] |
+-- Statistics ------------------------------+
| Executions: 42  |  Total Tokens: 12,450    |
| Last Run: 0.23s |  Avg Tokens: 296         |
| Status: Idle                               |
+-- Actions ---------------------------------+
| [Run Once]  [Refresh]                      |
+--------------------------------------------+
```

### Statistics

The component tracks execution metrics:

- **Executions**: Total run count
- **Total Tokens**: Cumulative LLM token usage
- **Last Run**: Duration of most recent execution
- **Avg Tokens**: Average tokens per execution
- **Status**: Idle / Running / Continuous

---

## Python API

### Getting the Component

```python
# From an entity
assembly = entity.get_component("facet_assembly")

# Multiple assemblies - get by name
translate = entity.get_component("facet_assembly", "translate")
sentiment = entity.get_component("facet_assembly", "sentiment")
```

### Running One-Shot

```python
# Run with inputs
result = await assembly.run({"text": "Hello world"})

# Check result
if result.get('success'):
    print(f"Response: {result['response']}")
    print(f"Tokens: {result['outputs']}")
```

### Binding Programmatically

```python
# Bind inputs
assembly.bind_input("text", "input_field.value")
assembly.bind_input("language", "dropdown.value")

# Bind outputs
assembly.bind_output("result", "output_label.text")
assembly.bind_output("confidence", "progress_bar.value")
```

### Event Listeners

```python
def on_complete(event):
    print(f"Assembly finished: {event.data['response']}")
    print(f"Tokens used: {event.data['tokens']}")

assembly.add_listener('complete', on_complete)
assembly.add_listener('error', lambda e: print(f"Error: {e.data['error']}"))
```

### Statistics

```python
stats = assembly.get_statistics()
print(f"Runs: {stats['execution_count']}")
print(f"Tokens: {stats['total_tokens']}")
print(f"Avg time: {stats['last_execution_time']:.3f}s")
```

---

## Serialization

Components serialize to YAML for project storage:

```yaml
components:
  - type: facet_assembly
    id: "abc123-..."
    assembly_path: "assemblies/sentiment-analysis.yaml"
    assembly_name: "Sentiment Analysis"
    run_in_cognition_loop: false
    tick_rate: 0.1
    auto_run_on_attach: false
    input_bindings:
      text: "{input.value}"
    output_bindings:
      result: "output.text"
```

---

## Cognition Loop Integration

When `run_in_cognition_loop` is checked, the assembly participates in the central cognition manager:

1. Component registers with CognitionManager on enable
2. Manager calls `assembly.run()` every `tick_rate` seconds
3. Inputs gathered from bindings automatically
4. Outputs applied to bound targets
5. `OnStateChange` fires if output differs from previous

### Performance Considerations

- Continuous assemblies share the cognition loop with Noodlings
- Token costs accumulate over time
- Consider caching for expensive operations
- Use tick_rate appropriate to your use case (default 0.1s = 10Hz)

---

## Use Cases

### Smart UI Elements

```yaml
# Tooltip that explains hovered item using AI
Panel:
  name: smart_tooltip
  visible: false
  children:
    - type: Label
      name: explanation
      bindings:
        text: "{last_hover.explanation}"

# On hover, run assembly to generate explanation
Item:
  events:
    onMouseEnter:
      action: run_assembly
      assembly: assemblies/explain-item.yaml
      inputs:
        item_id: "{self.item_id}"
      outputs:
        explanation: last_hover.explanation
```

### Reactive Props

```python
# Treasure chest that opens when player approaches
class TreasureChest:
    def __init__(self):
        self.proximity = FacetAssemblyComponent(
            assembly_path="assemblies/proximity-check.yaml",
            run_in_cognition_loop=True,
            tick_rate=0.5
        )
        self.proximity.bind_output("should_open", self.handle_open)
```

### Multi-Assembly Noodlings

```yaml
# Noodling with multiple cognitive modules
Red:
  components:
    - type: facet_assembly
      assembly_path: "assemblies/main-cognition.yaml"
      run_in_cognition_loop: true

    - type: facet_assembly
      assembly_path: "assemblies/memory-consolidation.yaml"
      run_in_cognition_loop: true
      tick_rate: 5.0  # Run every 5 seconds

    - type: facet_assembly
      assembly_path: "assemblies/dream-generator.yaml"
      run_in_cognition_loop: false  # Only when sleeping
```

---

## File Locations

| File | Purpose |
|------|---------|
| `core/facet_assembly_component.py` | Component implementation |
| `core/facet_system.py` | FacetAssembly data model |
| `core/facet_executor.py` | Execution engine |
| `runtime/ui/event_dispatcher.py` | `run_assembly` action handler |
| `panels/inspector_components.py` | Inspector UI |

---

## See Also

- [Facet System](facets.md) - Core facet architecture
- [UI Canvas](ui-canvas.md) - Visual UI system with events
- [Scripting API](scripting.md) - ScriptedFacet development
