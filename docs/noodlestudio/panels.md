# NoodleStudio Panels

Reference for all panels in NoodleStudio.

---

## Stage View

The scene hierarchy showing everything in the current stage.

**Contains:**
- Zones (spatial regions)
- Noodlings (AI characters)
- Props (world objects)
- User-created folders

**Actions:**
- Right-click: Context menu (New Folder, Rename, Delete)
- Drag: Reparent items
- Double-click: Focus in viewport

## Assets Panel

Unity-style filesystem browser for project assets.

**Shows:**
- Noodlings/ - Character definitions
- Stages/ - World scenes
- Prims/ - Prop templates
- Scripts/ - Custom code
- Radiances/ - Gaussian splat models

**Actions:**
- Double-click folder: Expand
- Double-click file: Open with system default
- Right-click: Import, Rename, Delete, Reveal in Finder

## Inspector

Properties panel for the selected entity.

**Modes:**
- Noodling: Name, recipe, assembly, affect state
- Zone: Name, description, connections
- Prop: Name, position, prim reference
- Asset: Type-specific metadata

## Facets Editor

Node graph editor for cognitive architectures.

**Usage:**
1. Select a Noodling in Stage View
2. Facets Editor shows their assembly
3. Drag to reposition nodes
4. Right-click to add new facets
5. Drag between ports to connect

## Neural Canvas

Advanced visual programming for neural networks and complex assemblies.

**Features:**
- PyTorch-style node types (Linear, LSTM, etc.)
- Live parameter count display
- MLX code generation
- Import/export to .nncanvas files

## Chat Panel

Talk to the world.

**Commands:**
- Type message and press Enter to speak
- Prefix with `/` for commands
- Messages go to NoodleMUSH server

## Gaussian Viewer

3D viewport for Gaussian splat assets.

**Controls:**
- Left drag: Orbit
- Right drag: Pan
- Scroll: Zoom
- F: Focus selected
- A: Frame all

**Features:**
- Bone visualization (capsule style)
- Click-to-select bones
- 120 FPS with GPU (gsplat-mps)
