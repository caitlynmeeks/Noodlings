# UI Canvas System

**Status**: Implementation Phase 3d Complete
**Last Updated**: January 3, 2026
**Authors**: Caitlyn + Claude
**Inspiration**: Borland Delphi Form Designer

---

## Overview

NoodleStudio needs a visual UI designer for building application interfaces. This follows the classic Delphi pattern: drag components onto a canvas, set properties, wire up events.

The UI Canvas is distinct from Neural Canvas (cognitive architecture) - this is for designing the end-user interface of built applications.

### Core Principle

**The canvas IS the application.** A Gaussian viewport is just another component you place on it, like a Button or Label. A "3D game" is simply a canvas with a fullscreen RadianceViewport. A "chat app" has no viewport at all. This unifies all application types under one paradigm.

### Why Delphi?

Delphi's form designer (1995) established patterns still used today:
- **Visual WYSIWYG editing** - what you design is what you get
- **Component-based** - reusable UI building blocks
- **Property Inspector** - edit any property visually
- **Event wiring** - connect UI events to code
- **Anchoring** - components resize intelligently

These patterns feel natural to anyone who's used:
- Visual Basic
- Windows Forms
- Interface Builder (Xcode)
- Qt Designer
- Figma (modern equivalent)

---

## Architecture

### The Three Canvases

NoodleStudio now has three canvas types:

| Canvas | Purpose | Edits |
|--------|---------|-------|
| **Neural Canvas** | Cognitive architecture | Node graphs (facets, charm networks) |
| **UI Canvas** | Application interface | 2D component layout |
| **Stage View** | 3D world | Entity hierarchy (zones, noodlings, props) |

### UI Canvas in the Player

Built applications can have:
- **Full 3D**: Just a Radiance viewport (Unity-style)
- **Full 2D**: Just UI components (chat app, dashboard)
- **Hybrid**: UI layout with embedded Radiance viewport(s)

```
+------------------------------------------+
|  [Title Bar]                    [_][O][X]|
+------------------------------------------+
|  +----------------+  +------------------+|
|  | Chat History   |  | Radiance         ||
|  | [ScrollView]   |  | Viewport         ||
|  |                |  |                  ||
|  |                |  |   [3D Scene]     ||
|  |                |  |                  ||
|  +----------------+  +------------------+|
|  +--------------------------------------+|
|  | [Input Field                ] [Send] ||
|  +--------------------------------------+|
+------------------------------------------+
```

---

## Component System

### Core Components (Delphi equivalents)

| Component | Delphi | Description |
|-----------|--------|-------------|
| `Panel` | TPanel | Container, background color |
| `Label` | TLabel | Static text |
| `Button` | TButton | Clickable button |
| `TextInput` | TEdit | Single-line text input |
| `TextArea` | TMemo | Multi-line text |
| `ScrollView` | TScrollBox | Scrollable container |
| `Image` | TImage | Static image display |
| `ListView` | TListView | Scrollable list of items |
| `DropDown` | TComboBox | Selection dropdown |
| `CheckBox` | TCheckBox | Boolean toggle |
| `Slider` | TTrackBar | Numeric range |
| `ProgressBar` | TProgressBar | Progress indicator |
| `TabView` | TPageControl | Tabbed panels |
| `SplitView` | TSplitter | Resizable split |

### NoodleStudio-Specific Components

| Component | Description |
|-----------|-------------|
| `RadianceViewport` | 3D Gaussian renderer (embeddable) |
| `ChatHistory` | Scrolling chat messages with avatars |
| `ChatInput` | Text input with send button |
| `AffectMeter` | Visualize noodling's PAD state |
| `NoodlingAvatar` | 2D representation of a noodling |
| `AudioWaveform` | Voice input visualization |

---

## Property System

### Delphi-Style Properties

Every component has properties editable in the Inspector:

```yaml
# Example: Button component
Button:
  # Identity
  name: "sendButton"

  # Geometry (Delphi: Left, Top, Width, Height)
  x: 300
  y: 450
  width: 80
  height: 32

  # Anchors (Delphi: akLeft, akTop, akRight, akBottom)
  anchors:
    left: false
    top: false
    right: true
    bottom: true

  # Appearance
  text: "Send"
  enabled: true
  visible: true

  # Style
  background_color: "#3b82f6"
  text_color: "#ffffff"
  border_radius: 4
  font_size: 14
```

### Anchor System

Delphi's anchor system was genius - components resize intelligently:

```
anchors: [left, top]           # Fixed position (default)
anchors: [right, bottom]       # Sticks to bottom-right
anchors: [left, right, top]    # Stretches horizontally
anchors: [left, right, top, bottom]  # Fills container
```

---

## Event Wiring

### Delphi Pattern

In Delphi, you double-click a button to create an OnClick handler:

```pascal
procedure TForm1.Button1Click(Sender: TObject);
begin
  ShowMessage('Hello!');
end;
```

### NoodleStudio Pattern

We wire events to noodling interactions or scripts:

```yaml
# In UI definition
Button:
  name: "askButton"
  text: "Ask Red"
  events:
    onClick:
      action: "send_to_noodling"
      target: "red"
      message_source: "questionInput"  # Get text from this component
```

Or wire to a scripted facet:

```yaml
Button:
  name: "generateButton"
  text: "Generate"
  events:
    onClick:
      action: "call_script"
      script: "on_generate_clicked"
```

### Inline Scripts

For simple logic, embed JavaScript directly in the event binding:

```yaml
Button:
  name: "toggleButton"
  text: "Toggle"
  events:
    onClick:
      action: "call_script"
      script: |
        let panel = ui.get('detailsPanel');
        if (panel.visible) {
          ui.hide('detailsPanel');
        } else {
          ui.show('detailsPanel');
        }
```

Or reference an external script file:

```yaml
Button:
  name: "processButton"
  text: "Process"
  events:
    onClick:
      action: "call_script"
      script_file: "scripts/process_handler.js"
```

### Script API

Scripts executed via `call_script` have access to these APIs:

| Object | Methods | Description |
|--------|---------|-------------|
| `ui` | `get(name)`, `set(name, prop, value)` | Component access |
| `ui` | `show(name)`, `hide(name)` | Visibility control |
| `event` | `source`, `type`, `value` | Event context |
| `console` | `log()`, `warn()`, `error()` | Debug output |

Example:

```javascript
// Get the value from an input field
let inputValue = ui.get('questionInput').value;

// Set a label's text
ui.set('statusLabel', 'text', 'Processing...');

// Access event info
console.log('Event from:', event.source, 'Type:', event.type);
```

### Event Types

| Event | Triggers When |
|-------|---------------|
| `onClick` | Component clicked |
| `onDoubleClick` | Component double-clicked |
| `onChange` | Value changes (inputs, sliders) |
| `onSubmit` | Enter pressed in input |
| `onFocus` | Component gains focus |
| `onBlur` | Component loses focus |
| `onHover` | Mouse enters component |
| `onLeave` | Mouse leaves component |

---

## Component Value Binding

Bind component properties together for reactive updates (like Delphi's data binding):

```yaml
Label:
  name: "charCountLabel"
  text: "0 characters"
  bindings:
    text: "messageInput.value.length + ' characters'"

TextInput:
  name: "messageInput"
  placeholder: "Type a message..."
```

When `messageInput.value` changes, `charCountLabel.text` automatically updates.

### Binding Expressions

Bindings are JavaScript expressions that reference other components:

```yaml
# Simple property reference
bindings:
  text: "sourceComponent.value"

# Computed value
bindings:
  text: "userInput.value.toUpperCase()"

# Conditional
bindings:
  visible: "statusInput.value.length > 0"

# Arithmetic
bindings:
  width: "containerPanel.width * 0.5"
```

### Binding Manager

The runtime's `BindingManager` tracks dependencies and updates targets when sources change:

```python
from noodlestudio.runtime.ui import BindingManager

manager = BindingManager()
manager.add_binding(
    target_component="charCountLabel",
    target_property="text",
    expression="messageInput.value.length + ' characters'",
    source_components=["messageInput"]
)

# Called when messageInput changes
manager.notify_change("messageInput", "value", "Hello")
# charCountLabel.text is now "5 characters"
```

---

## Designer UI

### Component Palette

Left sidebar with draggable components:

```
+-- Components --+
| [Panel]        |
| [Label]        |
| [Button]       |
| [TextInput]    |
| [TextArea]     |
+----------------+
| -- Noodle --   |
| [ChatHistory]  |
| [ChatInput]    |
| [RadianceView] |
| [AffectMeter]  |
+----------------+
```

### Design Canvas

Center area showing the form:
- Grid snap (optional)
- Selection handles (8-point resize)
- Multi-select with Shift+click
- Alignment guides (snap to edges/centers)
- Drag to reposition
- Handles to resize

### Property Inspector

Right sidebar (reuse existing Inspector):
- Shows selected component's properties
- Grouped by category (Geometry, Appearance, Events)
- Inline editing

### Hierarchy Tree

Shows component parent/child relationships:

```
Form
├── headerPanel
│   ├── titleLabel
│   └── closeButton
├── mainSplit
│   ├── chatPanel
│   │   ├── chatHistory
│   │   └── chatInput
│   └── radianceViewport
└── statusBar
```

---

## File Format

### UI Definition: `ui.yaml`

```yaml
# ui.yaml - Application UI definition
version: 1
root:
  type: Panel
  name: "root"
  background: "#1a1a1a"
  children:
    - type: Panel
      name: "header"
      height: 48
      anchors: [left, right, top]
      background: "#2a2a2a"
      children:
        - type: Label
          name: "title"
          text: "Red's World"
          x: 16
          y: 12
          font_size: 18
          text_color: "#ffffff"

    - type: SplitView
      name: "mainSplit"
      anchors: [left, right, top, bottom]
      y: 48
      split_ratio: 0.4
      children:
        - type: Panel
          name: "chatPanel"
          children:
            - type: ChatHistory
              name: "chat"
              anchors: [left, right, top, bottom]
              noodling: "red"
            - type: ChatInput
              name: "input"
              anchors: [left, right, bottom]
              height: 48
              target_noodling: "red"

        - type: RadianceViewport
          name: "viewport"
          stage: "main_stage"
          camera:
            distance: 3.0
            elevation: 15
            azimuth: 180
```

---

## Integration with Build System

### build.yaml Addition

```yaml
# build.yaml
name: "Red's Chat"
version: "1.0.0"

# UI-based app (not just 3D viewport)
ui: "ui.yaml"

# Stage for RadianceViewport components (optional)
main_stage: "Stages/reds_room"

settings:
  window_size: [1024, 768]
  resizable: true
  min_size: [640, 480]
```

### Build Modes

| Mode | UI Canvas | RadianceViewport | Use Case |
|------|-----------|------------------|----------|
| **3D Only** | No | Fullscreen | Immersive 3D experience |
| **2D Only** | Yes | None | Chat app, dashboard |
| **Hybrid** | Yes | Embedded | Chat + 3D view |

---

## Implementation Phases

### Phase 3a: Canvas Infrastructure - COMPLETE (Jan 3, 2026)
- [x] Create `runtime/ui/` module structure
- [x] `UIComponent` base class with properties
- [x] `Anchors` dataclass and layout calculation
- [x] `UILoader` - YAML to component tree
- [x] `QtWidgetRenderer` - component tree to Qt widgets

### Phase 3b: Chat Components - COMPLETE (Jan 3, 2026)
- [x] `Panel` - container with background
- [x] `Label` - static text
- [x] `Button` - clickable with events
- [x] `TextInput` - single-line input
- [x] `ChatHistory` - scrolling message list with styled bubbles
- [x] `ChatInput` - compound input + send button
- [x] Event dispatch system (`UIEventDispatcher`)
- [x] `send_to_noodling` action with chat_history integration
- [x] Message roles: USER, NOODLING, SYSTEM

### Phase 3c: RadianceViewport - COMPLETE (Jan 3, 2026)
- [x] Embed GaussianRenderer in RadianceViewportWidget
- [x] Camera controls (orbit, pan, zoom)
- [x] Multi-component scene rendering via RadianceSceneBuilder
- [x] Clean API: `set_component()`, `add_component()`, `load_file()`
- [x] Semantic query passthrough: `raycast()`, `query_at_world_position()`
- [x] Focus controls: `set_camera()`, `focus_on()`, `frame_all()`

**Design Principle**: The viewport is a **focused renderer only**. It renders
RadianceComponents. It doesn't know what a "noodling" or "prop" is. Whatever
system needs to display Gaussians creates RadianceComponents and sends them
to the viewport. Separation of concerns.

### Phase 3d: Event Wiring Extensions - COMPLETE (Jan 3, 2026)
- [x] `call_script` action with inline scripts and external script files
- [x] `UIScriptExecutor` - lightweight JavaScript sandbox (QuickJS) for UI events
- [x] Component value binding system (`BindingManager`)
- [x] Script API: `ui.get()`, `ui.set()`, `ui.show()`, `ui.hide()`, `event`, `console`
- [x] Binding expressions with automatic dependency tracking

### Phase 4: Designer Panel - COMPLETE (Jan 3, 2026)
- [x] New panel: `UICanvasEditorPanel` in center tabs
- [x] Component palette: `ComponentPalettePanel` in left tabs (Stage/Assets/Components)
- [x] Drag-drop placement from palette to canvas
- [x] 8-point selection and resize handles (Delphi-style)
- [x] Property editing in Inspector via `UICanvasInspectorMixin`
- [x] Grid snap (8px)
- [x] Zoom/pan (wheel, space+drag, middle mouse)
- [x] Rubber-band multi-select
- [x] Auto-save to `ui.yaml` on every change
- [x] Delete selected (Delete/Backspace key)
- [x] Frame all (A key)

**Files Created:**
| File | Purpose |
|------|---------|
| `panels/ui_canvas_editor_panel.py` | Design surface with QGraphicsView |
| `panels/component_palette_panel.py` | Draggable component list |
| `panels/inspector_ui_canvas.py` | Inspector mixin for component properties |

### Phase 5: Undo/Redo for UI Edits (Next)
- [ ] `AddUIComponentCommand`
- [ ] `DeleteUIComponentCommand`
- [ ] `MoveUIComponentCommand`
- [ ] `ResizeUIComponentCommand`
- [ ] `EditUIPropertyCommand`

### Phase 6: Advanced (Future)
- [ ] Custom component creation
- [ ] Theming system
- [ ] Animation/transitions
- [ ] Responsive breakpoints

---

## Architecture Decisions (Finalized Jan 3, 2026)

### Renderer Abstraction Layer

```
ui.yaml (user's design - stable contract)
    ↓
UIComponent classes (our API - what users see)
    ↓
Renderer backend (swappable implementation)
    ├── QtWidgetRenderer (v1 - desktop)
    └── WebGLRenderer (future - browser)
```

Users interact with **Panel**, **Button**, **RadianceViewport** - they never see Qt, QML, or any implementation detail. The `ui.yaml` format is the stable contract that survives renderer changes.

### Q1: Technology for Runtime UI?

| Option | Pros | Cons |
|--------|------|------|
| **Qt Widgets** | Already using, native, robust | Desktop only |
| **Qt QML** | Modern, declarative | Learning curve |
| **Custom OpenGL** | Full control, web-ready | Months of work (text, input, focus) |
| **Web (Electron)** | HTML/CSS familiar | Heavy runtime |

**Decision**: **Qt Widgets for v1** with abstraction layer.

**Rationale**:
- Fast to implement (weeks, not months)
- Production quality out of the box (text input, focus, scrolling, accessibility)
- Users don't know or care - they see our component names
- The `ui.yaml` abstraction lets us add WebGL renderer later without changing user projects
- RadianceViewport embeds as QOpenGLWidget (already proven in NoodleStudio)

### Q2: Live Preview?

Should the designer show live noodling responses while editing?

**Decision**: No for v1. Design is static, test via Play button.

### Q3: Relation to Neural Canvas?

The Neural Canvas already has a node editor. Should UI Canvas share that infrastructure?

**Decision**: Different tools. Neural Canvas = data flow graphs. UI Canvas = spatial layout. Different interaction patterns.

### Q4: Default Project Template?

**Decision**: New projects ship with a minimal `ui.yaml`:

```yaml
# Default ui.yaml - fullscreen viewport
version: 1
root:
  type: Panel
  name: "root"
  background: "#1a1a1a"
  children:
    - type: RadianceViewport
      name: "viewport"
      anchors: [left, right, top, bottom]
      stage: "main_stage"
```

A "3D game" is just this default. Not a special case - just a canvas with one fullscreen viewport component.

---

## References

- [Delphi VCL Architecture](https://docwiki.embarcadero.com/RADStudio/en/VCL_Overview)
- [Qt Designer](https://doc.qt.io/qt-6/qtdesigner-manual.html)
- [Figma Auto Layout](https://help.figma.com/hc/en-us/articles/360040451373)

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-03 | Initial planning document |
| 2026-01-03 | Architecture decisions finalized: Qt Widgets v1 with abstraction layer, renderer-agnostic ui.yaml contract |
| 2026-01-03 | Phase 3a COMPLETE: Canvas infrastructure, base components, anchor system |
| 2026-01-03 | Phase 3b COMPLETE: ChatHistory, ChatInput, UIEventDispatcher, send_to_noodling action |
| 2026-01-03 | Phase 3c COMPLETE: RadianceViewport - focused Gaussian renderer with clean API |
| 2026-01-03 | Phase 3d COMPLETE: call_script action, UIScriptExecutor, component value bindings |
| 2026-01-03 | Phase 4 COMPLETE: UI Canvas Designer - visual drag-drop editor, component palette, inspector integration |
