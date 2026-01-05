# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: January 4, 2026

---

## NEXT: Peak Delphi UI System (Jan 4, 2026)

**Goal:** Complete the UI system to match peak Borland Delphi capabilities - full event model, custom components, Inspector event wiring, and complex component integration.

### Current State

| Layer | Status | Details |
|-------|--------|---------|
| **Components** | 7 built | Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport |
| **Events** | 3 wired | onClick, onChange, onSubmit |
| **Actions** | 5 built | send_to_noodling, call_script, set_value, show, hide, toggle_visible |
| **Bindings** | Working | One-way reactive `{text: "input.value"}` |
| **Scripts** | Working | QuickJS sandbox with `ui.*` API |

### Implementation Phases

#### Phase 7A: Full Event Model (START HERE)

Expand event wiring in `runtime/ui/renderer.py` and component classes.

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

**UIEventData Class:**
```python
@dataclass
class UIEventData:
    type: str                    # "onClick", "onKeyDown", etc.
    source: str                  # Component name
    timestamp: float
    # Mouse
    x: Optional[int] = None
    y: Optional[int] = None
    button: Optional[str] = None  # "left", "right", "middle"
    modifiers: Optional[Dict] = None  # {shift, ctrl, alt, meta}
    # Keyboard
    key: Optional[str] = None
    keyCode: Optional[int] = None
    # Value
    value: Any = None
    previousValue: Any = None
    # Drag
    dragData: Any = None
    # 3D (RadianceViewport)
    hitPosition: Optional[Tuple] = None
    hitEntity: Optional[str] = None
    hitSemantics: Optional[Dict] = None
```

**Key Files:**
- `runtime/ui/event_data.py` - NEW: UIEventData class
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

**Tests:** 64 tests in `test_ui_canvas.py`

---

## BACKLOG

### Inspector UX
- Unity-style numeric drag-to-scroll on labels
- Drag-to-reparent UI components in Stage

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
| **UI Event Dispatcher** | `runtime/ui/event_dispatcher.py` |
| **UI Script Executor** | `runtime/ui/script_executor.py` |
| **UI Bindings** | `runtime/ui/bindings.py` |
| **UI Components** | `runtime/ui/components/` |
| **UI Canvas Designer** | `panels/ui_canvas_editor_panel.py` |
| **UI Inspector Mixin** | `panels/inspector_ui_canvas.py` |
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
