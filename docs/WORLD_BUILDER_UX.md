# World Builder UX Design

NoodleStudio UX specification for game designers, world builders, and artists working with Gaussian splatting.

**Created:** December 21, 2025
**Status:** Design specification (not yet implemented)

---

## The Opportunity

Gaussian splatting doesn't have established tooling yet. Unity and Unreal assume polygons. We can define the **Blender-for-Gaussians** workflow from scratch.

**Goal:** Create the most delightful, intuitive world-building experience for real-time Gaussian scenes.

---

## Core Differences from Polygon Workflows

| Traditional 3D | Gaussian Splats |
|----------------|-----------------|
| Edit vertices/faces | Can't edit individual splats (millions!) |
| UV unwrap textures | Color baked into splats |
| Hard edges between meshes | Natural blending |
| Complex LOD authoring | Density-based LOD |
| Bake lighting | Lighting captured in training |
| Z-fighting issues | Natural compositing |
| Stencil buffer mirrors | Just render from different camera |

**Key insight:** Gaussians are more like *photographs* than *models*. The workflow should feel more like photo editing/compositing than polygon pushing.

---

## 1. Camera Navigation

### Primary Controls (Right-click held)

| Input | Action |
|-------|--------|
| WASD | Fly forward/back/strafe |
| Q/E | Fly down/up |
| Mouse | Look around |
| Scroll | Speed multiplier (0.1x - 10x) |

### Orbit Mode (Alt held)

| Input | Action |
|-------|--------|
| Alt + Left-drag | Orbit around focus point |
| Alt + Middle-drag | Pan |
| Alt + Right-drag | Dolly (zoom toward cursor) |
| Scroll | Zoom |

### Focus Commands

| Input | Action |
|-------|--------|
| F | Frame selected object |
| Double-click hierarchy | Frame and select |
| Shift + F | Lock camera to follow selection |
| Numpad 1/3/7 | Front/Right/Top orthographic |
| Numpad 5 | Toggle perspective/orthographic |
| Numpad 0 | Camera preview (scene camera) |

### Camera Bookmarks

| Input | Action |
|-------|--------|
| Ctrl + 1-9 | Save camera bookmark |
| 1-9 | Recall camera bookmark |
| Shift + 1-9 | Animate to bookmark (smooth) |

### Design Notes

- **Smooth transitions:** Never instant-snap the camera. Smooth interpolation makes the 3D space feel real and helps maintain spatial awareness.
- **Speed ramping:** Hold Shift to go faster, Ctrl to go slower. Speed persists per-session.
- **Gamepad support:** Left stick move, right stick look, triggers up/down. For VR-adjacent feel.

---

## 2. Transform Tools

### Tool Selection

| Key | Tool | Gizmo |
|-----|------|-------|
| W | Translate | XYZ arrows |
| E | Rotate | XYZ rings |
| R | Scale | XYZ boxes |
| T | Rect | 2D positioning (for UI/billboards) |
| Y | Universal | All transforms combined |

### Space and Pivot

| Key | Action |
|-----|--------|
| Tab | Toggle Local/World space |
| P | Cycle pivot mode (Center/Origin/Cursor) |
| Insert | Set cursor to selection |
| Shift + Insert | Set cursor to world origin |

### Snapping

| Input | Action |
|-------|--------|
| Ctrl (held) | Snap to grid |
| V (held) | Vertex snap (snap to object origins) |
| X/Y/Z | Constrain to axis |
| Shift + X/Y/Z | Constrain to plane (exclude axis) |
| Ctrl + Shift (held) | Snap to surface (raycast into splats) |

### Precision Input

| Input | Action |
|-------|--------|
| G, then type | Move by typed amount |
| Inspector fields | Direct numeric input |
| Arrow keys | Nudge by small increment |
| Shift + Arrow | Nudge by large increment |

### Multi-Selection

| Input | Action |
|-------|--------|
| Click | Select single |
| Shift + Click | Add to selection |
| Ctrl + Click | Toggle in selection |
| Ctrl + A | Select all |
| Ctrl + D | Duplicate selection |
| Shift + D | Duplicate with offset |
| Ctrl + G | Group selection |
| Alt + G | Ungroup |

### Gaussian-Specific

**Surface Snapping:** Raycast into the Gaussian cloud and find the "surface" (density threshold). Objects can be placed ON splat surfaces, not just at grid points.

**Blend Positioning:** When moving objects, show a preview of how they'll blend with existing splats. Visual feedback for overlap.

---

## 3. Asset Workflow

### Asset Sources

```
┌─────────────────────────────────────────────────────────────┐
│                    ASSET SOURCES                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Photos  │  │   3D     │  │   AI     │  │  Live    │    │
│  │  (train) │  │  Import  │  │  Gen     │  │  Capture │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       │         Mesh-to-         Text-to-      Video-to-   │
│       │         Gaussian         Gaussian      Gaussian    │
│       │             │             │             │           │
│       ▼             ▼             ▼             ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              GAUSSIAN ASSET LIBRARY                  │   │
│  │                                                      │   │
│  │  Avatars/  Props/  Environments/  Particles/        │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                    DRAG TO SCENE                            │
│                           │                                 │
│                           ▼                                 │
│                    ┌─────────────┐                          │
│                    │  INSTANCE   │                          │
│                    │  IN SCENE   │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Photo-to-Splat Pipeline (Built-in Training)

**The Vision:** Capture photos of real objects, train Gaussians without leaving the editor. No command line, no Python scripts, no COLMAP headaches.

#### Quick Workflow

1. **Import:** Drag folder of photos into Assets panel
2. **Configure:** Set quality level (Fast/Balanced/Quality)
3. **Train:** Progress bar with preview updates
4. **Review:** Inspect result in asset preview
5. **Use:** Drag trained asset to scene

#### Training Panel

```
┌─────────────────────────────────────────────────────────────┐
│ Train Gaussian Asset                                    [X] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Source: /Photos/coffee_mug/ (47 images)            [...]  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │            [Live Preview of Training]              │   │
│  │                                                     │   │
│  │     Splats materializing as training progresses    │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Progress: ████████████░░░░░░░░ 62%                        │
│  Iteration: 12,400 / 20,000                                 │
│  Time: 3:42 elapsed, ~2:15 remaining                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Quality:    ( ) Fast    (x) Balanced    ( ) Quality       │
│                                                             │
│  Advanced:                                          [Show] │
├─────────────────────────────────────────────────────────────┤
│  [Cancel]                          [Pause]    [Use Result] │
└─────────────────────────────────────────────────────────────┘
```

#### Advanced Training Options

```
┌─────────────────────────────────────────────────────────────┐
│ Advanced Training Options                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Iterations:        [20000     ] (more = better quality)    │
│ Initial Points:    [100000    ] (from COLMAP/random)       │
│ Densification:     [x] Enable  Until iter: [15000]         │
│                                                             │
│ Learning Rates:                                             │
│   Position:        [0.00016   ]                            │
│   Color (SH):      [0.0025    ]                            │
│   Opacity:         [0.05      ]                            │
│   Scale:           [0.005     ]                            │
│   Rotation:        [0.001     ]                            │
│                                                             │
│ Regularization:                                             │
│   [ ] Depth supervision (if depth maps available)          │
│   [x] Opacity reset (every [3000] iters)                   │
│   [ ] Anti-aliasing filter                                 │
│                                                             │
│ Output:                                                     │
│   Max splats:      [500000    ] (0 = unlimited)            │
│   Compression:     [x] Enable SH compression               │
│   Format:          (x) .ply  ( ) .splat  ( ) Both          │
│                                                             │
│                                     [Reset to Defaults]    │
└─────────────────────────────────────────────────────────────┘
```

#### Training Quality Presets

| Preset | Iterations | Time | Splats | Use Case |
|--------|------------|------|--------|----------|
| **Fast** | 5,000 | 1-2 min | ~100K | Quick preview, iteration |
| **Balanced** | 20,000 | 5-10 min | ~300K | Most assets |
| **Quality** | 50,000 | 20-40 min | ~500K | Hero assets, avatars |
| **Maximum** | 100,000 | 1-2 hrs | ~1M | Final production |

#### Camera Pose Estimation

Two modes for getting camera poses from photos:

**Auto (COLMAP-style):**
- Feature detection and matching built-in
- Works with any photo set
- Progress: "Matching features... Estimating poses..."

**Assisted:**
- Place virtual cameras in scene
- Match to real photo positions
- Useful when auto-matching fails

#### Live Preview During Training

The preview window shows:
- Current splat cloud (updating every 100 iterations)
- Camera positions (where photos were taken)
- Loss graph (should go down)
- Problematic areas (highlighted in red)

**Orbit the preview** while training to see quality from all angles.

#### Incremental Training

Already have a trained asset but want to improve it?

1. Select asset in library
2. Right-click → "Continue Training..."
3. Add more photos OR increase iterations
4. Training resumes from checkpoint

#### Training from Video

Drag a video file instead of photos:

1. **Import:** Drag .mp4/.mov into Assets
2. **Extract:** "Extract frames for training?"
3. **Configure:** Frame interval (every Nth frame)
4. **Train:** Same as photo workflow

**Pro tip:** Walk around object with phone camera, 30 seconds of video often enough.

#### Training Queue

Training multiple assets? Queue them up:

```
┌─────────────────────────────────────────────────────────────┐
│ Training Queue                                          [X] │
├─────────────────────────────────────────────────────────────┤
│ 1. [████████████████████] coffee_mug      Done     2:34    │
│ 2. [████████░░░░░░░░░░░░] plant_pot       62%      ~1:45   │
│ 3. [░░░░░░░░░░░░░░░░░░░░] keyboard        Queued   ~4:00   │
│ 4. [░░░░░░░░░░░░░░░░░░░░] desk_lamp       Queued   ~3:30   │
├─────────────────────────────────────────────────────────────┤
│ Total remaining: ~9:15                                      │
│                                                             │
│ [Pause All]  [Clear Queue]           [+ Add to Queue]      │
└─────────────────────────────────────────────────────────────┘
```

#### GPU Utilization

Training uses GPU (Metal on Mac, CUDA on Windows/Linux):

- Shows GPU memory usage
- Warning if running low
- Option to use cloud GPU for faster training (future)

### 3D Model Import

Supported formats: `.glb`, `.gltf`, `.obj`, `.fbx`, `.vrm`

Pipeline:
1. Import model
2. Auto-convert to Gaussian representation
3. Preserve skeleton/blend shapes (VRM)
4. Store both mesh and Gaussian versions

### AI Generation

Right-click in scene or Assets panel:

```
┌─────────────────────────────────────┐
│ Generate Asset                      │
├─────────────────────────────────────┤
│ Prompt:                             │
│ ┌─────────────────────────────────┐ │
│ │ mossy stone fountain with       │ │
│ │ water flowing                   │ │
│ └─────────────────────────────────┘ │
│                                     │
│ Style:  [Realistic      ▼]          │
│         - Realistic                 │
│         - Stylized                  │
│         - Anime                     │
│         - Low Poly                  │
│         - Painterly                 │
│                                     │
│ Size:   [Medium         ▼]          │
│                                     │
│ [Generate]  [Generate 4 Variants]   │
└─────────────────────────────────────┘
```

### Live Capture

For scanning real objects:
1. Define capture volume (box in scene)
2. Use phone/webcam to capture from angles
3. Real-time preview of splat building
4. Refine with additional captures

---

## 4. Component System

### Inspector Panel Layout

When a prim/noodling is selected:

```
┌─────────────────────────────────────┐
│ Stone Fountain                      │
│ Tag: [environment] Layer: [default] │
├─────────────────────────────────────┤
│ Transform                        [-]│
├─────────────────────────────────────┤
│   Position  [0.0 ] [1.2 ] [3.4 ]   │
│   Rotation  [0.0 ] [45.0] [0.0 ]   │
│   Scale     [1.0 ] [1.0 ] [1.0 ]   │
├─────────────────────────────────────┤
│ Gaussian Renderer                [-]│
├─────────────────────────────────────┤
│   Asset     fountain.ply       [O] │
│   LOD Bias  ──────●────── 1.0      │
│   Cast Shadows    [x]              │
│   Receive Shadows [x]              │
├─────────────────────────────────────┤
│ Audio Source                     [-]│
├─────────────────────────────────────┤
│   Clip      water_loop.ogg     [O] │
│   Volume    ────●──────── 0.6      │
│   Spatial   (x) 3D  ( ) 2D         │
│   Min Dist  ──●──────────  1.0     │
│   Max Dist  ────────●────  50.0    │
├─────────────────────────────────────┤
│ Water Fountain (Script)          [-]│
├─────────────────────────────────────┤
│   Splash Rate    [2.0        ]     │
│   Splash Sound   splash.ogg    [O] │
│   Particles      [Splash FX   ▼]   │
├─────────────────────────────────────┤
│                                     │
│         [+ Add Component]           │
│                                     │
└─────────────────────────────────────┘
```

### Add Component Menu

```
+ Add Component
│
├── Rendering
│   ├── Gaussian Renderer
│   ├── Particle System
│   ├── Trail Renderer
│   ├── Line Renderer
│   ├── Mirror Surface
│   └── Portal
│
├── Physics
│   ├── Rigidbody
│   ├── Box Collider
│   ├── Sphere Collider
│   ├── Mesh Collider
│   ├── Character Controller
│   └── Spring Joint
│
├── Audio
│   ├── Audio Source
│   ├── Audio Listener
│   ├── Ambient Zone
│   └── Voice Receiver
│
├── Interaction
│   ├── Clickable
│   ├── Hoverable
│   ├── Draggable
│   ├── Grabbable (VR)
│   ├── Proximity Trigger
│   ├── Gaze Target
│   └── Teleport Anchor
│
├── Animation
│   ├── Animator
│   ├── Simple Rotate
│   ├── Simple Hover
│   ├── Simple Bob
│   ├── Look At Target
│   └── Follow Path
│
├── Navigation
│   ├── Nav Mesh Agent
│   ├── Nav Mesh Obstacle
│   └── Waypoint
│
├── UI
│   ├── World Space Canvas
│   ├── Billboard
│   └── Name Tag
│
├── Noodling (AI Characters)
│   ├── Facet Assembly
│   ├── Perception Cone
│   ├── Speech Bubble
│   └── Emotion Display
│
└── Scripts
    ├── New JavaScript...
    ├── New Python...
    └── Browse Project Scripts...
```

### Component Features

**Drag-and-Drop References:**
- Drag assets from library to component fields
- Drag other objects to reference fields
- Visual feedback showing valid drop targets

**Copy/Paste Components:**
- Right-click component header → Copy Component
- Right-click another object → Paste Component
- Paste As New to duplicate with new settings

**Presets:**
- Save component configurations as presets
- Apply presets to quickly configure common setups
- Project-wide and user-wide preset libraries

---

## 5. Gaussian-Specific Tools

### Blend Brush

For seamlessly combining Gaussian assets:

```
┌─────────────────────────────────────┐
│ Blend Brush                      [?]│
├─────────────────────────────────────┤
│ Size      ────────●──── 2.0m       │
│ Falloff   ──●────────── 0.3        │
│ Strength  ──────●────── 0.5        │
├─────────────────────────────────────┤
│ Mode:                               │
│  (x) Blend    - Smooth transition   │
│  ( ) Erase    - Remove splats       │
│  ( ) Clone    - Copy from source    │
│  ( ) Density  - Adjust splat count  │
├─────────────────────────────────────┤
│ [ ] Affect selected only            │
│ [ ] Live preview                    │
└─────────────────────────────────────┘
```

**Use cases:**
- Blend terrain pieces together
- Feather edges of imported assets
- Remove unwanted splats (artifacts)
- Touch up seams

### Density Visualizer

Debug view showing splat distribution:

```
View Menu → Debug → Splat Density

Color scale:
  Blue   = Low density (may have holes)
  Green  = Optimal density
  Yellow = High density
  Red    = Excessive (performance concern)
```

### Capture Volume Tool

For in-editor photogrammetry:

1. Create → Capture Volume
2. Position and size the box
3. Click "Start Capture"
4. Move camera around (or use turntable mode)
5. Watch splats build in real-time
6. Click "Finish" when satisfied

**Turntable Mode:**
- Place object on physical turntable
- Camera auto-captures as you rotate
- Progress ring shows coverage

### Portal Placement Tool

Dedicated workflow for linked portals:

1. Select Portal Tool (keyboard: O)
2. Click to place Portal A
3. Click to place Portal B
4. Portals automatically linked
5. Adjust size/orientation in Inspector

**Visual feedback:**
- See-through preview of destination
- Colored ring indicating link (A=blue, B=orange)
- Distance/angle indicators

### Mirror Placement Tool

1. Select Mirror Tool (keyboard: M)
2. Click and drag to define mirror plane
3. Adjust reflection quality in Inspector

**Mirror types:**
- Flat (standard reflection)
- Curved (convex/concave)
- Fun-house (custom distortion)

---

## 6. Script Editor

### Integrated Script Editor

When creating or editing a script:

```
┌─────────────────────────────────────────────────────────────┐
│ WaterFountain.js                    [Run] [Pause] [Stop]   │
├─────────────────────────────────────────────────────────────┤
│  1  // Water fountain behavior                              │
│  2  //                                                      │
│  3  // Lifecycle: onAwake, onStart, onUpdate, onDestroy     │
│  4  // Available: this.gameObject, context.noodle.*         │
│  5                                                          │
│  6  // Exposed to Inspector                                 │
│  7  /** @type {number} */                                   │
│  8  var splashRate = 2.0;                                   │
│  9                                                          │
│ 10  /** @type {AudioClip} */                                │
│ 11  var splashSound = null;                                 │
│ 12                                                          │
│ 13  // Private state                                        │
│ 14  var timer = 0;                                          │
│ 15                                                          │
│ 16  function onStart() {                                    │
│ 17      console.log("Fountain started!");                   │
│ 18  }                                                       │
│ 19                                                          │
│ 20  function onUpdate(dt) {                                 │
│ 21      timer += dt;                                        │
│ 22      if (timer >= splashRate) {                          │
│ 23          context.noodle.audio.play(splashSound);         │
│ 24  █       spawnSplashParticles();                         │
│ 25          timer = 0;                                      │
│ 26      }                                                   │
│ 27  }                                                       │
│ 28                                                          │
│ 29  function spawnSplashParticles() {                       │
│ 30      var pos = this.gameObject.transform.position;       │
│ 31      context.noodle.particles.emit("splash", pos, 10);   │
│ 32  }                                                       │
├─────────────────────────────────────────────────────────────┤
│ Console:                                                    │
│ > Fountain started!                                         │
│ > Fountain started!                                         │
└─────────────────────────────────────────────────────────────┘
```

### Editor Features

**Autocomplete:**
- Full autocomplete for `context.noodle.*`
- Type inference for variables
- Documentation popups

**Inline Errors:**
- Red squiggle for syntax errors
- Yellow squiggle for warnings
- Error panel with clickable locations

**Hot Reload:**
- Save triggers instant reload
- State preserved when possible
- Console shows reload status

**Debugging:**
- Breakpoints (click line numbers)
- Step through execution
- Watch variables
- Call stack

### Script Lifecycle

```javascript
// Called once when component initializes (before first frame)
function onAwake() { }

// Called once after all objects are awake
function onStart() { }

// Called every frame
function onUpdate(deltaTime) { }

// Called at fixed physics rate (default 50Hz)
function onFixedUpdate(fixedDeltaTime) { }

// Called when object is destroyed
function onDestroy() { }

// Called when object becomes enabled
function onEnable() { }

// Called when object becomes disabled
function onDisable() { }
```

### Event Callbacks

```javascript
// Interaction events
function onClick() { }
function onHover() { }
function onHoverExit() { }
function onGrab() { }       // VR
function onRelease() { }    // VR

// Trigger events (requires collider set as trigger)
function onTriggerEnter(other) { }
function onTriggerStay(other) { }
function onTriggerExit(other) { }

// Collision events (requires rigidbody)
function onCollisionEnter(collision) { }
function onCollisionStay(collision) { }
function onCollisionExit(collision) { }

// Custom events
function onMessage(name, data) { }
```

### Scripting API Quick Reference

```javascript
// Transform
this.gameObject.transform.position = [x, y, z];
this.gameObject.transform.rotation = [x, y, z, w];
this.gameObject.transform.localScale = [x, y, z];
this.gameObject.transform.lookAt(target);

// Find objects
var obj = context.noodle.find("ObjectName");
var objs = context.noodle.findAll("Tag:enemy");
var child = this.gameObject.findChild("ChildName");

// Components
var audio = this.gameObject.getComponent("AudioSource");
var rb = this.gameObject.addComponent("Rigidbody");
this.gameObject.removeComponent(audio);

// Audio
context.noodle.audio.play("clip.ogg");
context.noodle.audio.playAt("clip.ogg", [x, y, z]);
context.noodle.audio.stop("clip.ogg");

// Particles
context.noodle.particles.emit("system", position, count);
context.noodle.particles.stop("system");

// Physics
context.noodle.physics.raycast(origin, direction, maxDistance);
context.noodle.physics.overlap(position, radius);

// Time
context.noodle.time.deltaTime;
context.noodle.time.time;
context.noodle.time.timeScale;

// Input
context.noodle.input.getKey("Space");
context.noodle.input.getMouseButton(0);
context.noodle.input.mousePosition;

// Scene
context.noodle.scene.load("SceneName");
context.noodle.scene.instantiate(prefab, position, rotation);

// Network (multiplayer)
context.noodle.network.send("eventName", data);
context.noodle.network.onReceive("eventName", callback);
```

---

## 7. Delightful Touches

### Instant Everything

- **No compile step:** Scripts hot-reload on save
- **No bake lighting:** Lighting captured in Gaussians
- **No build process:** Press Play, it plays
- **Live editing:** Modify while running, see changes immediately

### Comprehensive Undo

- **Full undo/redo stack:** Every action undoable
- **History panel:** Visual list of all actions
- **Selective undo:** Undo specific action without undoing later ones
- **Branch undo:** Fork history (like Git branches)

### Quick Actions (Cmd+K / Ctrl+K)

Spotlight-style command palette:

```
┌─────────────────────────────────────┐
│ > create particle                   │
├─────────────────────────────────────┤
│ + Create Particle System            │
│ + Add Particle System Component     │
│   Open Particle Editor              │
│   Documentation: Particles          │
├─────────────────────────────────────┤
│ Recent:                             │
│   Duplicate Selection               │
│   Frame Selected                    │
│   Toggle Play Mode                  │
└─────────────────────────────────────┘
```

### Contextual Help

- **Hover tooltips:** Every UI element has a tooltip
- **Animated tooltips:** GIFs showing how to use tools
- **F1 on selection:** Opens relevant documentation
- **"?" button:** Every component has inline help
- **Tutorial mode:** Guided walkthrough for new users

### Smart Defaults

- New objects placed at cursor or center of view
- Reasonable default values for all components
- Auto-naming with incrementing numbers
- Templates for common setups

### Visual Feedback

- **Selection outline:** Clear highlight of selected objects
- **Transform preview:** Ghost showing result of transform
- **Snap indicators:** Visual guides when snapping
- **Drop zones:** Highlight valid drop targets when dragging
- **Progress indicators:** For any operation >0.5s

### Collaboration (Future)

- **Real-time co-editing:** Multiple users in same scene
- **Presence indicators:** See other users' cursors
- **Lock objects:** Prevent conflicts on shared objects
- **Voice chat:** Built-in communication
- **Comments:** Attach notes to objects

### Social/Sharing

- **One-click publish:** Deploy to web instantly
- **Share link:** Preview page with embed code
- **Version history:** Visual diff between versions
- **Fork projects:** Clone and modify others' work

---

## 8. Keyboard Shortcuts Reference

### Global

| Key | Action |
|-----|--------|
| Ctrl+N | New project |
| Ctrl+O | Open project |
| Ctrl+S | Save |
| Ctrl+Shift+S | Save As |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+K | Quick actions |
| F1 | Help |
| F5 | Play/Stop |
| F6 | Pause |

### Scene View

| Key | Action |
|-----|--------|
| F | Frame selection |
| W | Translate tool |
| E | Rotate tool |
| R | Scale tool |
| T | Rect tool |
| Tab | Toggle local/world |
| Delete | Delete selection |
| Ctrl+D | Duplicate |
| Ctrl+G | Group |
| H | Hide selection |
| Shift+H | Unhide all |

### Navigation

| Key | Action |
|-----|--------|
| Right-click+WASD | Fly |
| Alt+Left-drag | Orbit |
| Alt+Middle-drag | Pan |
| Alt+Right-drag | Dolly |
| Scroll | Zoom |
| Numpad 1/3/7 | Ortho views |

---

## 9. Implementation Priority

### Phase 1: Foundation
- [ ] Camera navigation (fly, orbit, pan, zoom)
- [ ] Transform gizmos (translate, rotate, scale)
- [ ] Object selection and multi-select
- [ ] Basic Inspector panel
- [ ] Undo/redo system

### Phase 2: Assets
- [ ] Asset browser panel
- [ ] Drag-drop asset placement
- [ ] Gaussian asset loading (.ply)
- [ ] Basic 3D import (GLB → Gaussian)
- [ ] Asset preview thumbnails

### Phase 3: Components
- [ ] Component system architecture
- [ ] Add Component menu
- [ ] Transform component
- [ ] Audio Source component
- [ ] Basic colliders

### Phase 4: Scripting
- [ ] Script editor integration
- [ ] JavaScript runtime
- [ ] context.noodle API
- [ ] Hot reload
- [ ] Console output

### Phase 5: Gaussian Tools
- [ ] Blend brush
- [ ] Density visualizer
- [ ] Portal tool
- [ ] Mirror tool
- [ ] Particle system editor

### Phase 6: Polish
- [ ] Quick actions (Cmd+K)
- [ ] Contextual help
- [ ] Keyboard shortcut customization
- [ ] Preferences panel
- [ ] Performance optimization

### Phase 7: Advanced
- [ ] Photo-to-splat training
- [ ] AI asset generation
- [ ] Collaborative editing
- [ ] One-click publish
- [ ] VR editing mode

---

## 10. Target Users

### Primary Personas

**1. Unity Refugees**
- Professional game developers
- Familiar with component-based architecture
- Want Unity's depth without Unity's baggage
- Excited about Gaussians as next-gen rendering

**2. World Builders & Artists**
- Creating virtual spaces (social VR, virtual events)
- Visual thinkers, prefer drag-drop over code
- Need intuitive asset workflows
- Value "delightful" over "powerful"

**3. AI/ML Developers**
- Tinkering with LLM scaffolding
- Building cognitive architectures
- Want visual tools for neural logic
- The Facet Editor and Neural Canvas are their jam

### Design Philosophy

**Unity-level depth, Figma-level delight.**

- Professional features available but not overwhelming
- Progressive disclosure (simple defaults, advanced on demand)
- Keyboard shortcuts for power users
- Visual feedback for everything
- Zero tolerance for cryptic errors

### What We're NOT

- **Not Roblox:** We're beyond "kid-friendly" simplicity
- **Not Unity:** We're not a general-purpose game engine
- **Not Blender:** We're focused, not "do everything"

### What We ARE

**The Blender of AI Characters + The Figma of Gaussian Worlds**

A focused tool that does three things exceptionally well:
1. Build Gaussian splat environments
2. Create AI-driven characters (noodlings)
3. Design cognitive architectures visually

---

## 11. AI Developer Workflow

### The Unique Value Proposition

NoodleStudio isn't just a world builder - it's a **cognitive architecture IDE**.

For AI developers, the draw is:
- **Facet Editor:** Visual LLM pipeline design
- **Neural Canvas:** Design actual neural networks visually
- **Live Testing:** Run cognition and see results immediately
- **Scriptable:** JavaScript/Python hooks everywhere

### Facet Editor (LLM Pipelines)

```
┌─────────────────────────────────────────────────────────────────┐
│ Facet Assembly: curious_explorer.yaml                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │ INCOMING │────▶│  CHARM   │────▶│ CONTEXT  │                │
│  │  input   │     │  NET     │     │  INTEL   │                │
│  └──────────┘     └────┬─────┘     └────┬─────┘                │
│                        │                │                       │
│                   affect out      social_context                │
│                        │                │                       │
│                        ▼                ▼                       │
│                   ┌─────────────────────────┐                   │
│                   │    PERSONALITY LLM      │                   │
│                   │    "You are curious..." │                   │
│                   └───────────┬─────────────┘                   │
│                               │                                 │
│                               ▼                                 │
│                   ┌─────────────────────────┐                   │
│                   │    SPEECH STYLE LLM     │                   │
│                   │    "Speak like..."      │                   │
│                   └───────────┬─────────────┘                   │
│                               │                                 │
│                               ▼                                 │
│                        ┌──────────┐                             │
│                        │ OUTGOING │                             │
│                        │  output  │                             │
│                        └──────────┘                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [+ Add Facet]  [Run Test]  [View Execution]  [Export YAML]     │
└─────────────────────────────────────────────────────────────────┘
```

### Facet Types for AI Devs

| Facet Type | Purpose |
|------------|---------|
| **LLMFacet** | Call any LLM with prompt template |
| **CharmNetworkFacet** | LSTM/GRU affect processing |
| **ScriptedFacet** | Custom JS/Python logic |
| **MCPFacet** | Call external tools via MCP |
| **ConvergenceFacet** | Merge multiple inputs |
| **BranchFacet** | Conditional routing |
| **UtilityFacets** | Math, string, array ops |

### Neural Canvas (Network Design)

Design actual neural networks visually:

```
┌─────────────────────────────────────────────────────────────────┐
│ Neural Canvas: affect_processor.nncanvas                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────┐                                                     │
│  │ INPUT  │                                                     │
│  │ dim:16 │                                                     │
│  └───┬────┘                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌────────────┐     ┌────────────┐                             │
│  │ LSTM       │     │ LSTM       │                             │
│  │ hidden:32  │────▶│ hidden:32  │                             │
│  │ (fast)     │     │ (slow)     │                             │
│  └─────┬──────┘     └─────┬──────┘                             │
│        │                  │                                     │
│        └────────┬─────────┘                                     │
│                 │                                               │
│                 ▼                                               │
│          ┌────────────┐                                         │
│          │ CONCAT     │                                         │
│          └─────┬──────┘                                         │
│                │                                                │
│                ▼                                                │
│          ┌────────────┐                                         │
│          │ LINEAR     │                                         │
│          │ out:5      │                                         │
│          └─────┬──────┘                                         │
│                │                                                │
│                ▼                                                │
│          ┌────────────┐                                         │
│          │ OUTPUT     │                                         │
│          │ (affect)   │                                         │
│          └────────────┘                                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [Test Input]  [Generate Code]  [Train]  [Export Weights]       │
└─────────────────────────────────────────────────────────────────┘
```

### Live Cognition Testing

Test noodling responses in real-time:

```
┌─────────────────────────────────────────────────────────────────┐
│ Test Cognition: Red (fire_imp)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Input: "Hey Red, what do you think about the weather?"          │
│                                                                 │
│ ─────────────────────────────────────────────────────────────── │
│                                                                 │
│ Execution Trace:                                                │
│                                                                 │
│ 1. INCOMING                           0.2ms                     │
│    └─ parsed: greeting + question                               │
│                                                                 │
│ 2. CHARM_NET                          1.8ms                     │
│    └─ affect: valence=0.3, arousal=0.6, dominance=0.7          │
│                                                                 │
│ 3. CONTEXT_INTEL                      45ms                      │
│    └─ social: casual_conversation, familiar_speaker            │
│                                                                 │
│ 4. PERSONALITY_LLM                    892ms                     │
│    └─ "Weather? Pff, I make my own weather! *sparks*"          │
│                                                                 │
│ 5. SPEECH_STYLE                       234ms                     │
│    └─ added fire imp mannerisms                                 │
│                                                                 │
│ ─────────────────────────────────────────────────────────────── │
│                                                                 │
│ Output: "Weather? Ha! I'm a fire imp - I MAKE the weather      │
│          around here! *little flames dance around head*         │
│          But yeah, it's kinda nice out. For mortals."          │
│                                                                 │
│ Total: 1.17s   Tokens: 847                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [Send Another]  [View Full Trace]  [Save Test Case]            │
└─────────────────────────────────────────────────────────────────┘
```

### Cognitive Timeline (Profiler)

Visualize cognitive execution over time:

```
┌─────────────────────────────────────────────────────────────────┐
│ Cognitive Timeline                    [REC] [PAUSE] [CLEAR]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Time:  0s        1s        2s        3s        4s        5s    │
│        │         │         │         │         │         │     │
│ ───────┴─────────┴─────────┴─────────┴─────────┴─────────┴──── │
│                                                                 │
│ CHARM_NET   ██░░░░██░░░░██░░░░██░░░░██░░░░██░░░░               │
│ (2ms each)                                                      │
│                                                                 │
│ CONTEXT     ░░████░░░░░░████░░░░░░░░░░████░░░░░░               │
│ (45ms each)                                                     │
│                                                                 │
│ PERSONALITY ░░░░░░████████░░░░░░░░░░░░░░░░████████             │
│ (800ms each)                                                    │
│                                                                 │
│ SPEECH      ░░░░░░░░░░░░░░██░░░░░░░░░░░░░░░░░░░░██             │
│ (200ms each)                                                    │
│                                                                 │
│ ───────────────────────────────────────────────────────────────│
│ Affect:     ───────╱╲───────────╲╱────────────────             │
│ valence     (hover to see values)                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Click any block to inspect inputs/outputs                       │
└─────────────────────────────────────────────────────────────────┘
```

### Scripting API for AI Devs

Beyond world scripting, AI devs get:

```javascript
// Access current affect state
let affect = context.noodle.agents.getAffect(noodlingId);
// { valence: 0.3, arousal: 0.6, dominance: 0.7, boredom: 0.1, sorrow: 0.0 }

// Inject affect directly (for testing)
context.noodle.agents.setAffect(noodlingId, {
    valence: -0.5,  // Make them grumpy
    arousal: 0.9    // And agitated
});

// Get facet assembly
let assembly = context.noodle.agents.getAssembly(noodlingId);

// Modify facets at runtime
assembly.getFacet("personality_llm").setPrompt("You are now very suspicious...");

// Listen to cognition events
context.noodle.agents.onCognitionComplete(noodlingId, (result) => {
    console.log("Output:", result.output);
    console.log("Tokens used:", result.tokens);
    console.log("Affect change:", result.affectDelta);
});

// Run inference on neural canvas directly
let result = await context.noodle.neural.run("affect_processor", inputTensor);

// Access perception slice
let perception = context.noodle.agents.getPerception(noodlingId);
// What does this noodling see/hear right now?
```

### MCP Tool Integration

Connect to external tools:

```yaml
# mcp_servers.yaml
servers:
  filesystem:
    command: npx
    args: ["@anthropic/mcp-server-filesystem"]

  browser:
    command: npx
    args: ["@anthropic/mcp-server-browser"]

  custom_api:
    command: python
    args: ["my_api_server.py"]
```

Then in facets:
```
MCPFacet "web_search"
  server: browser
  tool: search
  query: ${input.question}
```

---

## 12. Open Questions

1. **VR editing priority?**
   - Essential feature
   - Nice to have
   - Future consideration

2. **Cloud training option?**
   - Local GPU only
   - Optional cloud offload
   - Hybrid (start local, finish on cloud)

3. **Collaboration model?**
   - Single user (file-based sharing)
   - Real-time multiplayer editing
   - Git-style branching/merging

4. **Plugin/extension system?**
   - Closed ecosystem
   - Open plugin API
   - Asset store model

---

**"The best interface is no interface" - but when you need one, make it delightful.**
