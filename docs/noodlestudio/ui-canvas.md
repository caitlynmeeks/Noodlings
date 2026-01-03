# UI Canvas System

**Status**: Planning
**Last Updated**: January 3, 2026
**Authors**: Caitlyn + Claude
**Inspiration**: Borland Delphi Form Designer

---

## Overview

NoodleStudio needs a visual UI designer for building application interfaces. This follows the classic Delphi pattern: drag components onto a canvas, set properties, wire up events.

The UI Canvas is distinct from Neural Canvas (cognitive architecture) - this is for designing the end-user interface of built applications.

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

### Phase 1: Runtime Renderer
- [ ] UI component base class
- [ ] Core components (Panel, Label, Button, Input)
- [ ] Anchor/layout system
- [ ] YAML loader
- [ ] Basic event dispatch

### Phase 2: Designer Panel
- [ ] New panel: UI Canvas Editor
- [ ] Component palette
- [ ] Drag-drop placement
- [ ] Selection and resize handles
- [ ] Property editing in Inspector

### Phase 3: Noodle Components
- [ ] ChatHistory component
- [ ] ChatInput component
- [ ] RadianceViewport embedding
- [ ] Event-to-noodling wiring

### Phase 4: Advanced
- [ ] Custom component creation
- [ ] Theming system
- [ ] Animation/transitions
- [ ] Responsive breakpoints

---

## Questions to Resolve

### Q1: Technology for Runtime UI?

| Option | Pros | Cons |
|--------|------|------|
| **Qt Widgets** | Already using, native | Desktop only |
| **Qt QML** | Modern, declarative | Learning curve |
| **Custom OpenGL** | Full control | Lots of work |
| **Web (Electron)** | HTML/CSS familiar | Heavy runtime |

**Current thinking**: Qt Widgets for v1 (we know it), consider QML later.

### Q2: Live Preview?

Should the designer show live noodling responses while editing?

**Current thinking**: No for v1. Design is static, test via Play button.

### Q3: Relation to Neural Canvas?

The Neural Canvas already has a node editor. Should UI Canvas share that infrastructure?

**Current thinking**: Different tools. Neural Canvas = data flow graphs. UI Canvas = spatial layout. Different interaction patterns.

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
