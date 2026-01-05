# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: January 5, 2026

---

## NEXT: Facets as Universal Components - Phase 3

**Goal:** Cognition Manager integration and UI Canvas Designer.

**Remaining Tasks:**
1. Cognition Manager integration
   - Register continuous assemblies with central manager
   - Performance monitoring dashboard
2. UI Canvas Designer integration
   - Add "Facet Assembly" to component palette (for UI-attached assemblies)
   - Visual assembly picker dialog

---

## COMPLETED: Facets as Universal Components - Phase 2 (Jan 5, 2026)

**Inspector UI for FacetAssemblyComponent** - Full Delphi Object Inspector experience.

**What Was Built:**
```
+-- Facet Assembly: sentiment-analysis ------+
| Assembly:    [sentiment-analysis.yaml] [R] |  <- PropertySpec (file picker)
| [x] Run in cognition loop                  |  <- PropertySpec (checkbox)
| Tick Rate:   [0.1    ] seconds             |  <- PropertySpec (float)
+-- Input Bindings --------------------------+
| out:         [{text_field.value}     ] [x] |  <- Binding row
+-- Output Bindings -------------------------+
| in:          [result_label.text      ] [x] |  <- Binding row
+-- Statistics ------------------------------+
| Executions: 42  |  Total Tokens: 12,450    |
| Last Run: 0.23s |  Avg Tokens: 296         |
| Status: Idle                               |
+-- Actions ---------------------------------+
| [Run Once]  [Refresh]                      |
+--------------------------------------------+
```

**Features:**
- Input/output binding UI with pad names from assembly
- Statistics display (executions, tokens, timing)
- "Run Once" button for testing one-shot execution
- "Refresh" button for stats update
- Status indicator (Idle/Running/Continuous)

**Files Modified:**
- `panels/inspector_components.py` - Added `_create_facet_assembly_ui` method

**Tests:** 411 passing (no regressions)

---

## COMPLETED: Facets as Universal Components - Phase 1 (Jan 5, 2026)

**THE KEY ARCHITECTURAL UNIFICATION**: Facet Assemblies are now attachable components that work on ANY entity (Noodling, Prim, UI element). This makes Facets the universal visual logic language for everything in NoodleStudio.

**Core Innovation:**
- Multiple assemblies per entity (singleton = False)
- Two execution modes controlled by checkbox:
  - **CHECKED** (Continuous): Runs in cognition loop every tick_rate seconds
  - **UNCHECKED** (One-shot): Runs on-demand via events/scripts

**FacetAssemblyComponent** (`core/facet_assembly_component.py`):
```python
# Get assembly from entity
assembly = entity.GetComponent("facet_assembly", "translate-chinese")

# Run one-shot
result = await assembly.run({"text": "Hello world"})

# Listen for events
assembly.add_listener('complete', on_complete_handler)
```

**Properties:**
- `assembly_path`: Path to .yaml assembly file
- `run_in_cognition_loop`: THE checkbox - continuous vs one-shot
- `tick_rate`: Seconds between cognitive ticks (0.01-60s)
- `auto_run_on_attach`: Run once when component added

**Events:**
- `OnComplete`: Fires after one-shot execution
- `OnStateChange`: Fires when continuous assembly state changes
- `OnError`: Fires on execution error

**New UI Canvas Action: run_assembly**

```yaml
Button:
  name: analyze_button
  events:
    onClick:
      action: run_assembly
      assembly: assemblies/sentiment-analysis.yaml
      inputs:
        text: "{text_field.value}"
      outputs:
        result: result_label.text
        sentiment: mood_indicator.color
```

**Files Created:**
- `core/facet_assembly_component.py` - Core component class
- `tests/test_facet_assembly_component.py` - 38 tests

**Files Modified:**
- `runtime/ui/event_dispatcher.py` - Added run_assembly action
- `runtime/ui/component.py` - EventBinding supports assembly/inputs/outputs

**Tests:** 411 passing (38 new for FacetAssemblyComponent)

---

## NEXT: Phase 7D - Custom Component System

**Goal:** Allow users to create reusable custom components.

**Three ways to create custom components:**

**1. Composite Components (YAML):**
```yaml
# components/login_form.yaml
type: CompositeComponent
name: LoginForm
properties:
  - name: title
    type: string
    default: "Login"

template:
  type: Panel
  children:
    - type: Label
      text: "${title}"
    - type: TextInput
      name: "usernameInput"
```

**2. Python Components:**
```python
@register_component
class ColorPicker(UIComponent):
    component_type = "ColorPicker"
    PROPERTIES = {'value': {'type': 'color', 'default': '#ffffff'}}
```

**Key Files:**
- `runtime/ui/composite_loader.py` - NEW: Load composite components
- `runtime/ui/component.py` - Add PROPERTIES/EVENTS class attributes

---

## COMPLETED: UI Polish and New Components (Jan 4, 2026)

Session work on Inspector UX, navigation consistency, and new components.

**Inspector Improvements:**
- Tightened padding throughout (margins, spacing, input fields)
- Added `ColorFieldWidget` with Procreate-style color picker popup
- Color wheel (HSV) + saturation/value square
- 25-color palette + recent colors
- Hex input field

**Navigation Consistency (Facets/Neural Canvas/UI Canvas):**
- Added F key focus toggle to UI Canvas (zoom to selection, press again to restore)
- Added zoom limits to UI Canvas
- Removed redundant "Frame All" toolbar button (A key works)

**New Components:**
- `WebView` - Embedded web browser (requires PyQt6-WebEngine)

**Context Menu Sync:**
- UI Canvas and Stage View now have identical component lists
- All 13 components available: Panel, Label, Button, TextInput, Checkbox, Dropdown, Slider, RadioGroup, ChatHistory, ChatInput, RadianceViewport, WebView

**Stage Hierarchy Fixes:**
- Text no longer truncated (ElideNone + horizontal scrollbar)
- Tooltips on UI component items
- Bidirectional selection sync (Canvas <-> Stage hierarchy)
- Drag-drop reparenting for UI components (drag Button onto Panel)

**Files Created:**
- `widgets/color_picker_widget.py`
- `runtime/ui/components/webview.py`

---

## COMPLETED: Phase 7C - More Components (Jan 4, 2026)

Standard Delphi-style components added to expand the component palette.

**New Components:**
| Component | Description |
|-----------|-------------|
| `Checkbox` | Boolean toggle with label, toggle() method |
| `Dropdown` | ComboBox/select with placeholder, add/remove options |
| `Slider` | Numeric range with step snapping, percentage, formatted value |
| `RadioButton` | Single radio with value property |
| `RadioGroup` | Container with options, orientation (vertical/horizontal) |

**Files Created:**
- `runtime/ui/components/checkbox.py`
- `runtime/ui/components/dropdown.py`
- `runtime/ui/components/slider.py`
- `runtime/ui/components/radio.py` (RadioButton + RadioGroup)

**Files Modified:**
- `runtime/ui/components/__init__.py` - Export new components
- `runtime/ui/renderer.py` - Render methods for all 5 components

**Features:**
- Full YAML serialization/deserialization
- onChange event firing with UIEventData
- Binding manager integration (value changes propagate)
- Styled to match monochromatic dark theme

**Tests:** 26 new tests (136 total), all passing.

---

## COMPLETED: Phase 7B - Inspector Event Wiring UI (Jan 4, 2026)

Visual UI in Inspector for configuring event bindings - the Delphi Object Inspector's Events tab.

**Implemented:**
```
+-- Events ----------------------------------+
| onClick         [send_to_noodling v]  [x]  |
|                 Target: [red        v]     |
|                 Message: [input     v]     |
| [+ Add Event]                              |
+--------------------------------------------+
```

**Features:**
- `EventBindingWidget` - Individual event row with action dropdown
- Action types: send_to_noodling, call_script, set_value, show, hide, toggle_visible
- Dynamic parameter fields based on action type
- "Edit Script..." opens `ScriptEditorDialog` with JS syntax highlighting
- "[+ Add Event]" menu with common events + "More Events..."
- Delete button (x) removes event binding
- Changes auto-save to ui.yaml via canvas_modified signal

**Files Created:**
- `widgets/event_binding_widget.py` - Event row widget
- `dialogs/script_editor_dialog.py` - JavaScript script editor with API reference

**Files Modified:**
- `panels/inspector_ui_canvas.py` - Interactive Events section

**Tests:** 18 new tests (110 total), all passing.

---

### Current State

| Layer | Status | Details |
|-------|--------|---------|
| **Components** | 12 built | Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport, Checkbox, Dropdown, Slider, RadioButton, RadioGroup |
| **Events** | 15 wired | onClick, onDoubleClick, onMouseDown/Up/Move, onMouseEnter/Leave, onMouseWheel, onContextMenu, onKeyDown/Up, onFocus/Blur, onChange, onSubmit |
| **Actions** | 5 built | send_to_noodling, call_script, set_value, show, hide, toggle_visible |
| **Bindings** | Working | One-way reactive `{text: "input.value"}` |
| **Scripts** | Working | QuickJS sandbox with `ui.*` API, full UIEventData access |

### Implementation Phases

#### Phase 7A: Full Event Model - COMPLETE (Jan 4, 2026)

Full event model with comprehensive UIEventData and Qt widget integration.

**UIEventData** (`runtime/ui/event_data.py`):
- Mouse position (x, y, global_x, global_y)
- Mouse button (MouseButton enum: LEFT, RIGHT, MIDDLE)
- Keyboard modifiers (Modifiers dataclass: shift, ctrl, alt, meta)
- Keyboard key info (key name, key code, text)
- Value changes (value, previous_value)
- Drag data and drop effect
- 3D hit info (position, entity, semantics)
- Scroll deltas
- Propagation control (stop_propagation, prevent_default)
- Factory methods: `click()`, `value_change()`, `submit()`, `focus()`
- Qt conversion: `from_qt_mouse_event()`, `from_qt_key_event()`, `from_qt_wheel_event()`
- Event type constants (EVENT_CLICK, EVENT_KEY_DOWN, etc.)

**EventEmitting Widgets** (`runtime/ui/event_widgets.py`):
- `EventEmittingMixin` - Adds event emission to any QWidget
- `EventEmittingFrame` - Panel with full events
- `EventEmittingButton` - Button with full events
- `EventEmittingLineEdit` - TextInput with full events

**Integration**:
- `event_dispatcher.py` - Updated to accept UIEventData
- `script_executor.py` - Full event data passed to scripts
- `renderer.py` - Uses EventEmitting widgets

**Tests**: 18 new tests (92 total), all passing.

**Events Wired**:

**Mouse Events:**
```
onClick, onDoubleClick, onMouseDown, onMouseUp, onMouseMove,
onMouseEnter, onMouseLeave, onMouseWheel, onContextMenu
```

**Drag & Drop:**
```
onDragStart, onDrag, onDragEnter, onDragOver, onDragLeave, onDrop, onDragEnd
```

**Keyboard:**
```
onKeyDown, onKeyUp, onKeyPress
```

**Focus:**
```
onFocus, onBlur
```

**Value/State:**
```
onChange, onSubmit, onSelect, onCheck, onToggle
```

**Lifecycle:**
```
onCreate, onDestroy, onShow, onHide, onResize, onMove
```

**Validation:**
```
onValidate, onError
```

**Component-Specific (RadianceViewport):**
```
onLoad, onCameraMove, onGaussianClick, onGaussianHover
```

**Key Files:**
- `runtime/ui/event_data.py` - DONE: UIEventData class
- `runtime/ui/renderer.py` - Wire events to all components
- `runtime/ui/components/*.py` - Add event emission

#### Phase 7B: Inspector Event Wiring UI

Visual UI for configuring event bindings in Inspector.

**Design:**
```
+-- Events ----------------------------------+
| onClick         [send_to_noodling v]       |
|                 Target: [red        v]     |
|                 Message: [input     v]     |
| [+ Add Event]                              |
+--------------------------------------------+
| onMouseEnter    [call_script v]            |
|                 [Edit Script...]           |
+--------------------------------------------+
```

**Actions Dropdown:**
- send_to_noodling
- call_script (inline or file)
- set_value
- show / hide / toggle_visible
- enable / disable
- focus
- emit (bubble to parent)
- navigate (change screen)
- play_sound
- open_url
- custom...

**Key Files:**
- `panels/inspector_ui_canvas.py` - Add events section
- `dialogs/script_editor_dialog.py` - NEW: Inline script editor
- `dialogs/event_binding_dialog.py` - NEW: Event configuration

#### Phase 7C: More Components

**Standard Delphi Components:**

| Component | Priority | Notes |
|-----------|----------|-------|
| `Checkbox` | HIGH | Boolean toggle with label |
| `RadioButton` | HIGH | Mutually exclusive selection |
| `RadioGroup` | HIGH | Container for RadioButtons |
| `Dropdown` | HIGH | ComboBox/select |
| `Slider` | HIGH | Numeric range (onSlideStart/End) |
| `ProgressBar` | MEDIUM | Determinate/indeterminate |
| `Memo` | MEDIUM | Multiline TextInput |
| `Image` | MEDIUM | Static image display |
| `SpinEdit` | MEDIUM | Numeric +/- buttons |
| `ListBox` | LOW | Scrollable selection list |
| `TreeView` | LOW | Hierarchical list |
| `TabView` | LOW | Tabbed panels |

**Key Files:**
- `runtime/ui/components/checkbox.py`
- `runtime/ui/components/dropdown.py`
- `runtime/ui/components/slider.py`
- etc.

#### Phase 7D: Custom Component System

Three ways to create custom components:

**1. Composite Components (YAML):**
```yaml
# components/login_form.yaml
type: CompositeComponent
name: LoginForm
properties:
  - name: title
    type: string
    default: "Login"
  - name: onLogin
    type: event

template:
  type: Panel
  children:
    - type: Label
      text: "${title}"
    - type: TextInput
      name: "usernameInput"
    - type: Button
      text: "Login"
      events:
        onClick:
          action: emit
          event: onLogin
```

**2. Python Components:**
```python
@register_component
class ColorPicker(UIComponent):
    component_type = "ColorPicker"
    PROPERTIES = {'value': {'type': 'color', 'default': '#ffffff'}}
    EVENTS = ['onChange', 'onPickStart', 'onPickEnd']
```

**3. Script Components (YAML + JS):**
```yaml
type: ScriptComponent
name: Counter
properties:
  - name: value
    type: number
script: |
  function increment() { self.value++; emit('onChange'); }
```

**Key Files:**
- `runtime/ui/composite_loader.py` - NEW: Load composite components
- `runtime/ui/component.py` - Add PROPERTIES/EVENTS class attributes
- Component scanner for `components/` folder

#### Phase 7E: RadianceViewport Event Integration

Wire RadianceViewport's Qt signals to UI event system.

```python
# In RadianceViewportWidget.mousePressEvent
hit_info = self.raycast(event.pos().x(), event.pos().y())
event_data = UIEventData(
    type="onGaussianClick",
    hitPosition=hit_info.get('position'),
    hitEntity=hit_info.get('entity_id'),
    hitSemantics=hit_info.get('semantics'),
)
self._dispatch_ui_event("onGaussianClick", event_data)
```

**Enables:**
```yaml
RadianceViewport:
  events:
    onGaussianClick:
      action: call_script
      script: |
        ui.set('infoLabel', event.hitSemantics.body_part);
```

**Key Files:**
- `runtime/ui/components/radiance_viewport.py` - Add UI event dispatch

#### Phase 7F: Script API Expansion

Expand `UIScriptExecutor` API:

```javascript
// Current
ui.get(name), ui.set(name, value)
ui.show(name), ui.hide(name), ui.toggle(name)
ui.enable(name), ui.disable(name)
event.type, event.source, event.value
console.log/warn/error

// New
ui.focus(name)                    // Set keyboard focus
ui.blur()                         // Remove focus
ui.emit(eventName, data)          // Bubble event to parent
ui.getComponent(name)             // Full component access
ui.animate(name, props, duration) // Simple animations

app.sendToNoodling(name, msg)     // Async noodling call
app.getNoodling(name)             // Get noodling state

audio.play(path)                  // Play sound
audio.stop()                      // Stop all audio

storage.get(key)                  // Persist data
storage.set(key, value)           // Local storage
```

**Key Files:**
- `runtime/ui/script_executor.py` - Expand API

### Data Types Reference

```yaml
# Primitives
string, number, boolean, null

# Colors
color: "#ff0000", "#ff000080", "rgb(255,0,0)", "rgba(255,0,0,0.5)"

# Geometry
point: {x: 100, y: 200}
size: {width: 300, height: 200}
rect: {x: 0, y: 0, width: 100, height: 100}
margins: {top: 8, right: 8, bottom: 8, left: 8}

# Collections
array: [1, 2, 3]
object: {key: "value"}

# Asset References
image: "assets/icon.png"
radiance: "noodlings/red/red.radiance"
noodling: "red"
component: "@chat_history"

# Special
font: {family: "Inter", size: 14, weight: "bold"}
anchor: [left, top, right, bottom]
```

---

## COMPLETED: UI Hierarchy Duplicate Fix (Jan 4, 2026)

**Issue:** UI components appeared twice in Stage hierarchy - once with type annotations `(Panel)`, once without.

**Root Cause:** In `_add_node_to_tree()`, UI components were added twice:
1. Via `_add_ui_component_to_tree()` (with type annotations)
2. Via `children_ids` iteration (without type annotations)

**Fix:** Skip `UI_COMPONENT` nodes in children loop (line 334 of `scene_hierarchy_refresh_mixin.py`):
```python
if child_node and child_node.node_type != SceneNodeType.UI_COMPONENT:
    self._add_node_to_tree(child_node, item)
```

---

## COMPLETED: UI Canvas Stage Integration (Jan 4, 2026)

UI Canvas Designer now integrates with Stage hierarchy like Unity's Canvas/GameObject pattern:

1. **UI entities appear in Stage hierarchy** - Like Unity, UI Canvas is a first-class entity
2. **Right-click context menu** - Rez > New UI Canvas, Add > Panel/Button/Label...
3. **Bidirectional selection** - Click UI in Stage -> loads in UI Canvas Editor

**Key Files:**
- `panels/scene_hierarchy_ui_mixin.py` - UI canvas CRUD operations
- `panels/scene_hierarchy_context_menu_mixin.py` - UI context menu items
- `panels/ui_canvas_editor_panel.py` - Design surface

---

## COMPLETED: Phase 4 - Build System (Jan 3, 2026)

**"File > Build Application..." is live!** (Ctrl+B)

| File | Purpose |
|------|---------|
| `appbuilder/builder.py` | Orchestrator - validates, packages, bundles |
| `appbuilder/packager.py` | Asset collection with filtering |
| `appbuilder/bundler_macos.py` | Creates .app bundles |

---

## COMPLETED: Phase 3 - UI Canvas System (Jan 2-3, 2026)

Delphi-style canvas at `noodlestudio/runtime/ui/`:

**Components:** Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport

**Event System:**
- `UIEventDispatcher` - Routes events to handlers
- `UIScriptExecutor` - QuickJS sandbox
- `BindingManager` - Reactive property bindings

**Tests:** 136 tests in `test_ui_canvas.py`

---

## BACKLOG

### Dashboard UI Widgets
Research existing Qt gauge/meter libraries before building from scratch.
Goal: Professional instrumentation widgets (gauges, meters, LEDs, seven-segment displays)
for building science-fair style AI dashboards driven by facets/scripts.
- Look into: PyQtGraph, qt-material, QML gauge components
- Mercedes-style dashboard aesthetic

### Inspector UX
- Unity-style numeric drag-to-scroll on labels

### Undo/Redo for UI Edits
- `AddUIComponentCommand`, `DeleteUIComponentCommand`
- `MoveUIComponentCommand`, `ResizeUIComponentCommand`

### Additional Build Targets
- Windows .exe (PyInstaller)
- Linux binary
- Docker container

### Admin Dashboard - Issue Credits UI
Add "Adjust Credits" button on user detail page.
File: `backend/admin-dashboard/src/routes/users/[id]/+page.svelte`

### Asset-Aware Inspector
Inspector shows contextual info when selecting assets.

### Trained Gaussian Quality
OpenSplat-trained Gaussians have background artifacts.

### UX: Project Creation Wizard
Replace two-step creation with Unity-style wizard dialog.

---

## Core Architecture

### Affect Model: PAD + Boredom + Sorrow
5-dimensional continuous affect (NO discrete emotion labels):
- `valence` (-1 to +1), `arousal` (0 to 1), `dominance` (0 to 1)
- `boredom` (0 to 1), `sorrow` (0 to 1)

### Facet System
Visual node graphs defining how Noodlings think:
```
INCOMING -> CHARM_NET -> CONTEXT_INTELLIGENCE -> Cognitive facets -> OUTGOING
```

### Component System
Unity-style architecture - entities have multiple components:
- `ArtbookComponent` (art), `RadianceComponent` (rendering), `FacetAssembly` (charm)

### Gaussian Radiance System
**"Every Gaussian knows what it represents. Every frame is query-able."**

```
RadianceComponent -> GaussianRenderer (gsplat-mps GPU, 120 FPS) -> GaussianViewerPanel
```

---

## Testing

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest              # Run all (~130 tests)
PYTHONPATH=.:../.. pytest -v           # Verbose
PYTHONPATH=.:../.. pytest -k "test_ui" # By pattern
```

**Before committing:** ALL tests must pass.

---

## Development

### Running NoodleStudio
```bash
cd applications/noodlestudio
./launch_with_log.sh
```

### Running noodleMUSH Server
Toggle in status bar, or:
```bash
cd applications/cmush && ./start.sh
```

### Debugging
```bash
tail -f applications/cmush/logs/server_*.log
tail -f applications/noodlestudio/logs/noodlestudio_*.log
```

---

## Style Rules (CRITICAL)

- **NO EMOJIS** in code/docs/UI
- **NO "exciting" language** - Professional terminal aesthetic
- **NO WORKAROUNDS** - Fix root causes properly
- **NO discrete emotion labels** - Continuous affect only
- **MONOCHROMATIC UI** - Grays only (except Neural Canvas headers)

**GOLDEN RULE:** If it doesn't work, FIX IT properly. No hacks.

---

## Quick Reference

| What | Where |
|------|-------|
| **UI Canvas (runtime)** | `noodlestudio/runtime/ui/` |
| **UIEventData** | `runtime/ui/event_data.py` |
| **EventEmitting widgets** | `runtime/ui/event_widgets.py` |
| **UI Event Dispatcher** | `runtime/ui/event_dispatcher.py` |
| **UI Script Executor** | `runtime/ui/script_executor.py` |
| **UI Bindings** | `runtime/ui/bindings.py` |
| **UI Components** | `runtime/ui/components/` |
| **UI Canvas Designer** | `panels/ui_canvas_editor_panel.py` |
| **UI Inspector Mixin** | `panels/inspector_ui_canvas.py` |
| **Event Binding Widget** | `widgets/event_binding_widget.py` |
| **Script Editor Dialog** | `dialogs/script_editor_dialog.py` |
| **Build system** | `noodlestudio/appbuilder/` |
| **Runtime module** | `noodlestudio/runtime/` |
| **Facet editor** | `panels/facets_editor_panel.py` |
| **Scene hierarchy** | `panels/scene_hierarchy.py` |

---

## Project Context

**Creator:** Caitlyn (Unity employee #12, Asset Store creator)
**Location:** Garcia River Forest cabin
**Hardware:** M3 Ultra 512GB

**Mission:** Open-source alternative to "Consciousness-as-a-Service"

---

## Completed Systems

- **More Components** - Checkbox, Dropdown, Slider, RadioButton, RadioGroup (Phase 7C)
- **Inspector Event Wiring UI** - Delphi Object Inspector Events tab (Phase 7B)
- **Full Event Model** - UIEventData, EventEmitting widgets (Phase 7A)
- **UI Canvas System** - Delphi-style components, events, bindings (Phase 3)
- **UI Canvas Designer** - Visual drag-drop editor, Stage integration (Phase 4/6)
- **UI Hierarchy Fix** - Duplicate component display resolved
- **Build System** - File > Build Application (macOS .app bundles)
- **Runtime Foundation** - Headless execution, CLI, multi-provider LLM
- **LLM Routing API** - api.noodlings.ai/v1/chat/completions (live)
- **GPU Gaussian Rendering** - 120 FPS via gsplat-mps
- **Admin Dashboard** - admin.noodlings.ai (live)
- **Crash Recovery** - Sentinel file detection
- **Cloud Account System** - OAuth, credits, billing
- Multi-provider LLM (8 providers)
- Neural Canvas with PyTorch test mode
- Scriptability API (context.noodle)
- MCP integration
- Utility facets (31 types)
- Multimodal facets (audio, vision, image gen)

---

**Ordnung muss sein!**
