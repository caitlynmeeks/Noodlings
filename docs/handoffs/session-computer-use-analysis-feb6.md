# Session Handoff: Computer Use for Live Noodling-Building Demo

**Date:** February 6, 2026
**Session:** Claude Code (noodlings_clean)
**Context:** Caity wants Ajo (guide character) to run a Brenda play where he live-builds a noodling in the facet editor using computer tool use, narrating as he goes.

---

## What Was Analyzed

Spent significant compute exploring the intersection of six systems:

1. **Brenda Director** (`runtime/brenda.py`) - Play format, beat triggers, actor response gate
2. **Computer Use Controller** (`core/computer_use_controller.py`) - UI actions, element discovery, ghost cursor
3. **Guide Cue Handler** (`runtime/guide_cue_handler.py`) - Cue reception, computer use execution, LLM prompt injection
4. **Guide Performance Manager** (`runtime/ui/guide_performance_manager.py`) - Orchestration layer
5. **Facet Editor** (`panels/facets_editor_panel.py` + 5 mixins) - Node graph canvas, context menus, assembly lifecycle
6. **UI Test Runner** (`testing/ui_test_runner.py`) - Existing automation patterns

---

## Six Concerns Identified

### 1. Facet Canvas is Invisible to Computer Use

**The core blocker.** The facet editor is a `QGraphicsScene` with custom-painted items (`FacetNodeGraphics`, `FacetPadGraphics`, `ConnectionWire`). These are NOT Qt widgets. The `ComputerUseController.get_ui_element_map()` scans the widget tree to depth 10 but only discovers standard Qt widgets (buttons, tabs, inputs, checkboxes). It cannot see:

- Where a node is on the canvas
- Where a pad (input/output port) is
- Where to right-click to get the "Add Facet" context menu
- Where to drag from/to when wiring two pads together

**Without solving this, Ajo cannot interact with the facet editor at all.**

### 2. No Programmatic Facet-Building Workflow

There's no documented or tested sequence of UI actions to go from empty canvas to working assembly. The add-facet workflow is: right-click empty canvas -> "Add Facet" submenu -> pick facet type -> node appears at click position. Wiring is: mousedown on output pad -> drag to input pad -> mouseup. These are all coordinate-dependent operations.

### 3. LLM Narration vs. Computer Use Timing

Brenda cues contain both `speaks` (dialogue for LLM) and `computer_use` (action list). Currently:
- `speaks` gets injected into the LLM system prompt as "suggested dialogue"
- `computer_use` gets executed via `execute_computer_use()` as an async task
- There is `speaks_continued` for interleaving dialogue around actions
- But there's **no synchronization** between when the LLM finishes generating dialogue and when actions execute

The existing plays (lets_consciousness_intro) use `sync_with_speech: true` but this flag isn't actually implemented in the handler -- it's aspirational YAML.

### 4. No Error Recovery

Computer use actions execute sequentially with no verification. If a click misses, a menu doesn't open, or a dialog appears unexpectedly, the system continues blindly. No retry logic, no visual assertions mid-play, no branching for error states.

### 5. Target Resolution is Substring-Only

`GuideCueHandler._resolve_target()` does case-insensitive substring matching against element names from the UI element map. Works for "Tab: Facets Editor" but fails for:
- Context menu items (transient, don't exist until right-clicked)
- Canvas items (not in the widget tree)
- Multi-step interactions (right-click -> menu -> submenu -> item)

### 6. Scope of "Building a Noodling"

A full noodling = personality + facet assembly (nodes + wires) + charm network config + expressions + model assignment. Need to decide which parts Ajo actually builds live vs. which are pre-staged.

---

## Proposed Solution: Facet Canvas Query API

The good news: the facet editor already has all the spatial infrastructure we need internally. We just need to expose it.

### What Exists Today

- `FacetNodeGraphics` stores position in scene coordinates via `node_gfx.pos()` (QPointF)
- `FacetPadGraphics` has `get_scene_position()` returning absolute scene coords
- Pads are stored as `node.input_pads['name']` and `node.output_pads['name']`
- All nodes tracked in `self.node_graphics: Dict[str, FacetNodeGraphics]`
- Qt provides `view.mapFromScene()` and `view.mapToGlobal()` for coordinate conversion
- The full chain: `pad.scenePos() -> view.mapFromScene() -> mapToGlobal() -> window pixel`

### What to Build

**`FacetsEditorQueryMixin`** - New mixin exposing `query_canvas_elements()`:

```python
def query_canvas_elements(self) -> Dict:
    """Return all canvas items with window-relative pixel positions."""
    return {
        "nodes": {
            "INCOMING": {
                "x": 120, "y": 340, "w": 200, "h": 35,
                "output_pads": {"text": {"x": 320, "y": 357}}
            },
            "sentiment_analysis": {
                "x": 400, "y": 300, "w": 200, "h": 120,
                "input_pads": {"text": {"x": 400, "y": 339}},
                "output_pads": {"result": {"x": 600, "y": 339}}
            },
            ...
        },
        "wires": [...],
        "empty_canvas_center": {"x": 410, "y": 340}
    }
```

**Extend `ComputerUseController.get_ui_element_map()`** to include canvas items when the facet panel is active. Canvas elements would appear as:
```python
{"name": "Canvas Node: CHARM_NET", "type": "canvas_node", "x": 400, "y": 300, ...}
{"name": "Canvas Pad: CHARM_NET.affect_valence (output)", "type": "canvas_pad", "x": 600, "y": 339, ...}
```

**Canvas-aware target resolution** in GuideCueHandler -- parse targets like `"node:CHARM_NET"` or `"pad:CHARM_NET.affect_valence"` through the query API.

### Why This is Clean

- ~15 lines of Qt coordinate math wrapping existing infrastructure
- Zoom/pan invariant (Qt's `mapFromScene` handles transforms)
- No hardcoded pixels in play files
- Extends the existing element map pattern
- Builds on real spatial data, not guesswork

---

## Additional Infrastructure Needed

### `pause_until` for Brenda Beats

Let computer use actions verify state before proceeding:
```yaml
computer_use:
  - action: right_click
    target: "canvas_empty"
  - action: pause_until
    condition: "menu_visible"
    timeout: 2000
  - action: click
    target: "menu:Add Facet > Intuition Facet"
```

### Context Menu Handling

Context menus are transient. Options:
- A: Query the menu after it appears (ComputerUseController scans for visible QMenu)
- B: Add a `wait_for_menu` action type that polls until a QMenu is visible, then scans its items
- C: Both -- scan on appearance, expose items as targets

### Narration-Action Synchronization

Need checkpoints where Brenda waits for computer use to complete before advancing dialogue:
```yaml
guide:
  speaks: "Watch me add a charm network..."
  computer_use:
    - action: right_click
      target: "canvas_empty"
    - action: click
      target: "menu:Add Facet > Charm Network"
    - action: checkpoint
      name: "charm_added"
  speaks_continued: "There it is! Now let me wire it up..."
```

---

## Existing Play File Patterns to Build On

Three existing plays in `docs/noodlestudio/plays/`:
- `lets_consciousness_intro.play.yaml` - 547 lines, 13 beats, 3 acts, has computer_use examples
- `hello_noodlestudio.play.yaml` - Simpler demo
- `epistemics_exploration.play.yaml`

Computer use in existing plays uses: `move`, `highlight`, `click`, `wait` actions with `pause_before`/`pause_after` timing and `sync_with_speech` (aspirational).

The actor response gate in Brenda (`state.awaiting_actor_response`) already blocks beat advancement until the Guide finishes responding -- this is the foundation for synchronization.

---

## Facet Assembly Reference

Simplest valid assembly (`library/noodlings/empty_noodling/assembly.yaml`):
```
INCOMING -> CHARM_NET -> LLM Facet -> OUTGOING
           (4 outputs)  (affect + perception inputs)
```

This involves:
- 4 nodes to place
- 6 wires to connect
- 1 LLM prompt to configure

A "simple demo" version (INCOMING -> 1 LLM facet -> OUTGOING) would be:
- 3 nodes (2 auto-created: INCOMING/OUTGOING)
- 2 wires
- 1 prompt

---

## Key File Paths

| System | File |
|--------|------|
| Brenda Director | `applications/noodlestudio/noodlestudio/runtime/brenda.py` |
| Computer Use Controller | `applications/noodlestudio/noodlestudio/core/computer_use_controller.py` |
| Ghost Cursor | `applications/noodlestudio/noodlestudio/core/ghost_cursor.py` |
| Guide Cue Handler | `applications/noodlestudio/noodlestudio/runtime/guide_cue_handler.py` |
| Guide Performance Manager | `applications/noodlestudio/noodlestudio/runtime/ui/guide_performance_manager.py` |
| Guide Performance Window | `applications/noodlestudio/noodlestudio/runtime/ui/guide_performance_window.py` |
| Facet Editor Panel | `applications/noodlestudio/noodlestudio/panels/facets_editor_panel.py` |
| Facet Editor Graphics | `applications/noodlestudio/noodlestudio/panels/facets_editor_graphics.py` |
| Facet Editor Events Mixin | `applications/noodlestudio/noodlestudio/panels/facets_editor_events_mixin.py` |
| Facet Editor Assembly Mixin | `applications/noodlestudio/noodlestudio/panels/facets_editor_assembly_mixin.py` |
| Facet System (data model) | `applications/noodlestudio/noodlestudio/core/facet_system.py` |
| UI Test Runner | `applications/noodlestudio/noodlestudio/testing/ui_test_runner.py` |
| UI Test Actions | `applications/noodlestudio/noodlestudio/testing/ui_test_actions.py` |
| Play: lets_consciousness | `docs/noodlestudio/plays/lets_consciousness_intro.play.yaml` |
| Play: hello_noodlestudio | `docs/noodlestudio/plays/demos/hello_noodlestudio.play.yaml` |

---

## Estimation

- **Facet Canvas Query API + ComputerUseController integration:** Focused, well-scoped work. The spatial infrastructure exists.
- **Context menu handling:** Moderate -- need to scan transient QMenu widgets.
- **pause_until / checkpoint system:** Moderate -- extends Brenda's tick loop.
- **Narration-action sync:** Harder -- touches the boundary between Brenda, GuideCueHandler, and GuidePerformanceWindow's async LLM streaming.
- **The play file itself:** Creative work, iterative. Needs testing in the actual UI.
- **End-to-end reliability:** The long tail. Making it work once is different from making it work reliably.
