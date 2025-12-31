# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: December 30, 2025

**FOR NEXT CLAUDE: START HERE!**

---

## NEXT SESSION: Asset-Aware Inspector

### The Goal
When selecting an asset in the Assets panel, the Inspector should show contextual information based on asset type (like Unity's Project window → Inspector relationship).

### Asset Inspector Designs (APPROVED)

**Folders:**
- Name, Type: Folder, Path, Contains: X items

**Noodlings (recipe.yaml):**
- Name, Type: Noodling
- Personality summary (Big 5 traits)
- Affect baseline (PAD + boredom + sorrow)
- Assembly reference
- Actions: [Rez] [Edit Recipe]

**Stages (stage.yaml):**
- Name, Type: Stage
- Zone count, Instance count
- Actions: [Open Stage]

**Radiance (.radiance):**
- Name, Type: Gaussian Splat
- Gaussian count, Skeleton (bone count), Semantic labels, CLIP embeddings
- File size
- Actions: [Open in Viewer]

**VRM (.vrm):**
- Name, Type: Avatar Model
- Bones, Materials, File size
- Import Settings: Densify, Face Centers, Scale
- Actions: [Import as Radiance] [Preview]

**Images (.png, .jpg):**
- Name, Dimensions, File size
- Thumbnail preview
- Actions: [Open in System Viewer]

**Audio (.mp3, .wav):**
- Name, Duration, Sample rate, Channels, File size
- Actions: [Play] [Stop]

**Scripts (.py, .js):**
- Name, Lines, File size
- Read-only code preview (scrollable)
- Actions: [Open in Editor]

**Neural Canvas (.nncanvas):**
- Name, Node count, Connection count
- Actions: [Open in Editor]

### Key Files to Modify
- `noodlestudio/panels/inspector_panel.py` - Add asset inspection modes
- `noodlestudio/panels/assets_panel.py` - Emit signals with asset data
- `noodlestudio/core/main_window.py` - Connect asset selection to inspector

### Implementation Notes
- Inspector already has entity modes (noodling, zone, prop, facet)
- Add new mode for "asset" with sub-types
- Don't show: full YAML dumps, binary data, internal paths, UUIDs

---

## COMPLETED (Dec 30): Assets Panel - Unity-Style Filesystem Browser

### What Was Done
Completely rewrote Assets panel to be a real filesystem browser:
- Shows actual project folder structure (Noodlings/, Stages/, Prims/, etc.)
- Hides internal folders (Library/, .git, __pycache__)
- File icons by type (folders, yaml, images, audio, 3D models, scripts)
- QFileSystemWatcher for auto-refresh on external changes
- Selection preservation across refreshes
- Expanded state preservation

### Features
- Context menu: New Folder, Import Asset, Rename, Delete, Reveal in Finder
- Inline rename (F2 or context menu)
- Double-click: folders expand, files open with system default
- Drag-drop file operations
- Refresh button + debounced auto-refresh (500ms)

### Key Files
- `noodlestudio/panels/assets_panel.py` - Complete rewrite
- Old `asset_graph.py` and `asset_node.py` are now dead code (can delete)

---

## COMPLETED (Dec 30): Settings & Auto-Start Fixes

### Auto-Start MUSH Server
- Setting was in UI but not wired up
- Added `_check_autostart_mush()` in main_window.py
- Runs 700ms after startup (after project loads)
- Triggers server toggle if setting enabled

### Auto-Login as Last Account
- Added checkbox in Settings > Startup Options
- Modified `account_manager.py` `_restore_session()` to check setting
- When disabled: App starts logged out
- When enabled: Session restored from keychain/settings

---

## COMPLETED (Dec 30): "Nowhere" Location Bug Fix

### The Problem
Cloud-authenticated users spawned in "room_000" which may not exist in project-based worlds.

### The Fix
- `auth.py`: New users spawn in first available room (not hardcoded "room_000")
- `server.py`: On login, validates user's `current_room` exists; fixes if not
- Creates default "The Nexus" room if no rooms exist

---

## COMPLETED (Dec 28): Unified Authentication

### What Was Done
Integrated NoodleStudio auth with noodleMUSH:
- **token_auth** message type for cloud-validated authentication
- **URL parameter auth**: Web client accepts `?token=xxx&avatar=xxx`
- **Enter World button** in status bar with avatar dropdown
- **Auto-hide login modal** when token auth succeeds

### Architecture
Two auth paths, same destination:
- OLD: `login` → `handle_login()` → local auth.py → `self.connections[ws]`
- NEW: `token_auth` → `handle_token_auth()` → Cloudflare API → `self.connections[ws]`

Old system kept as offline fallback.

### Key Files
- `cmush/server.py` - `handle_token_auth()` handler
- `cmush/web/index.html` - URL param detection, response handler
- `noodlestudio/core/main_window.py` - Status bar, URL loading
- See ARCHITECTURE.md Section 8 for full details

---

## BACKLOG: UX Improvements

### Project Creation Wizard
Replace two-step project creation (name popup + folder picker) with a single
Unity-style wizard dialog. Should handle:
- Project name
- Location selection
- Initial stage creation
- Template selection (future)

### Dialog Polish
- Remove unnecessary green maximize buttons from QInputDialogs
- Consider custom dialog classes for cleaner appearance

---

## COMPLETED (Dec 28): Assets Panel Refactor - Unity-Style Flat Folders

### What Was Done
Removed folder functionality from Stage View and moved it to Assets Panel:
- Stage View: For scene entity hierarchy (instances in the world)
- Assets Panel: For asset organization (files on disk)

### Architecture
Created new `AssetGraph` system (parallel to `SceneGraph`):
- **AssetNode** (`core/asset_node.py`) - Data model for asset hierarchy
- **AssetGraph** (`core/asset_graph.py`) - Manager with CRUD, reparenting, persistence
- **Assets Panel** rewritten to use AssetGraph

### Asset Node Types
```python
class AssetNodeType(Enum):
    FOLDER = "folder"          # User-created folder
    NOODLING = "noodling"      # AI character definition
    STAGE = "stage"            # Scene/level
    PRIM = "prim"              # 3D object template
    RADIANCE = "radiance"      # Gaussian splat model
    MESH = "mesh"              # Imported mesh
    GENERATION = "generation"  # AI-generated content
```

### Features
- **Flat folder structure**: No fixed categories - users create their own folders
- **Drag-drop reparenting**: Move assets into/out of folders
- **Inline rename**: Double-click or right-click > Rename
- **Context menu**: New Folder, Rename, Delete, asset-specific actions
- **Persistence**: Hierarchy saved to `assets_hierarchy.yaml` per project
- **Auto-discovery**: Assets discovered on first load, then hierarchy persisted

### Stage View Changes
- Removed "New Folder" from context menu
- Disabled drag-drop reparenting (DropOnly mode)
- Stage View is now purely for scene entity organization

### Files Created/Modified
| File | Description |
|------|-------------|
| `core/asset_node.py` | NEW - AssetNode dataclass |
| `core/asset_graph.py` | NEW - AssetGraph manager |
| `panels/assets_panel.py` | REWRITTEN - Uses AssetGraph |
| `panels/scene_hierarchy.py` | MODIFIED - Removed folder ops |
| `core/main_window.py` | MODIFIED - Signal wiring |

---

## COMPLETED (Dec 28): Stage View Architecture Fixes

### Priority 1: UUID Consistency - DONE
- Inspector now shows "UUID:" with copy button for ALL entity types (zones, noodlings, props)
- Previously props showed "ID:" while others showed "UUID:" - now consistent

### Priority 2: Remove Project/Stage from Tree - DONE
- Stage View tree now shows entities directly at root (no "Stage: xxx" wrapper)
- Added status_label widget for "Server offline" message (not tree items)
- Stage is selected via dropdown only, not shown in tree hierarchy
- Updated `_build_tree_from_files`, `_add_new_files_to_hierarchy`, `_extract_user_hierarchy`

### Priority 3: Save Stage Flow - DONE
- Added `File > Save Stage` menu item (Ctrl+Shift+S)
- Added dirty tracking (`_dirty` flag, `is_dirty()`, `_set_dirty()`)
- Hierarchy changes (add/remove/reparent/rename) mark stage as dirty
- Switching stages prompts to save if there are unsaved changes
- Added `save_stage()` public method

### Bidirectional Name Sync - DONE
- Inspector `nameChanged` signal connected to Stage View `update_entity_name()`
- Assets Panel `assetRenamed` signal wired up in main_window

### Files Modified
- `panels/scene_hierarchy.py` - All three priorities
- `panels/inspector_panel.py` - UUID consistency, nameChanged signal
- `core/main_window.py` - Save Stage menu item, signal wiring

---

## COMPLETED (Dec 27): Unity-Style Stage View Hierarchy - Phase 1

### What Was Built
The scene hierarchy system now supports Unity-style organization:
- **SceneNode** (`core/scene_node.py`) - Data model for hierarchy nodes
- **SceneGraph** (`core/scene_graph.py`) - Manager with CRUD, reparenting, persistence
- **Drag-and-drop reparenting** in Stage View
- **Context menu** with New Folder, Rename, Delete
- **Persistence** to `hierarchy.yaml` in each stage

### Key Features
- User-creatable folders for organizing content
- Drag items between folders (zones, noodlings, props fully nestable)
- Scene graph synced with tree widget
- Auto-migration from file-based structure on first load
- Hierarchy persisted per stage

### Files Created/Modified
| File | Description |
|------|-------------|
| `core/scene_node.py` | NEW - SceneNode dataclass |
| `core/scene_graph.py` | NEW - SceneGraph manager |
| `panels/scene_hierarchy.py` | MODIFIED - Uses SceneGraph |

### Node Types
```python
class SceneNodeType(Enum):
    FOLDER = "folder"      # User-created
    RADIANCE = "radiance"  # Gaussian component
    NOODLING = "noodling"  # AI character
    PROP = "prop"          # World object
    BONE = "bone"          # Virtual, from skeleton
    ZONE = "zone"          # Spatial region
```

### Usage
```python
# Create folder
graph.create_folder("My Folder", parent_id)

# Reparent node
graph.reparent(node_id, new_parent_id)

# Find by path
node = graph.find_by_path("Stage/Characters/Red")

# Persist
graph.save("Stages/my_stage/hierarchy.yaml")
```

### Next Steps (Phase 2)
- [ ] Bone nodes in Stage View (skeleton children)
- [ ] Bone selection sync: Stage View <-> Inspector <-> Viewport
- [ ] Prop-to-bone parenting with transform inheritance
- [ ] Scripting API (`context.scene`)
- [ ] Node type icons

---

## COMPLETED (Dec 26 session): Bone Visualization & Selection

### Features Added
- **Capsule-style bone visualization** in Gaussian Viewer
  - Thick lines (8px outer, 3px inner) with rounded caps
  - Large joints (6px normal, 8px selected)
  - Orange highlight for selected bones
  - Bone name label appears on selection

- **Click-to-select bones**
  - Joint hit radius: 20px
  - Line segment hit testing: 12px (click anywhere on bone)
  - Coordinate mapping fixed for nested viewport

- **Bidirectional selection sync**
  - Click bone in viewport -> updates inspector dropdown
  - Select bone in dropdown -> highlights in viewport
  - Focus button gives keyboard focus to viewer (F key works immediately)

- **Focus behavior fixed**
  - Clicking to give panel focus no longer deselects bone
  - F key: focus selected bone, or whole model if none
  - A key: frame all

### Key Files Modified
- `panels/gaussian_viewer_panel.py` - bone rendering, hit testing, selection
- `panels/radiance_inspector.py` - bone signals, bidirectional sync
- `core/main_window.py` - signal connections

---

## COMPLETED (Dec 24-25): Face Detail Training System

### Features Added
- **FaceDetailCameraGenerator** (`tools/face_detail_camera.py`)
  - Importance-weighted camera views for facial regions
  - Lips (1.0), eyes (0.9), brow (0.85), face (0.7), body (0.3)
  - Default: 24 body + 48 face + 28 detail = 100 views

- **FaceDetailTrainingPipeline** (`tools/face_detail_training.py`)
  - Multi-stage training: body+face -> face-only refinement
  - Auto-detects head position from VRM skeleton
  - OpenSplat integration

- **TrainingAPI** (`scripting/training_api.py`)
  - Scripted facet interface for training operations

- **Facet Assembly** (`facet_assemblies/face_detail_training.yaml`)
  - Visual pipeline for end-user training

---

## BACKLOG: Trained Gaussian Quality

Trained Gaussians (from OpenSplat) render with background artifacts.
- 30K training complete: `external/datasets/alicia_views/alicia_30k.ply`
- Investigate SH coefficient interpretation
- Try black background training
- Compare with VRM-converted assets

### GPU Rendering Setup
gsplat-mps is installed at `/Users/thistlequell/git/gsplat-mps` (AGPLv3).
The renderer auto-detects GPU availability and falls back to software with a warning popup.

```python
from noodlestudio.core.gaussian_renderer import GaussianRenderer, GSPLAT_AVAILABLE
renderer = GaussianRenderer()  # Auto-detects GPU
print(f"GPU enabled: {renderer.use_gpu}")  # True on Apple Silicon with gsplat-mps
```

### Performance (52K Gaussians @ 512x512)
| Backend | Render Time | FPS |
|---------|-------------|-----|
| Software (PyTorch) | ~7,000ms | 0.1 |
| **gsplat-mps (Metal)** | **8ms** | **120** |

### NOW DO: Renderer Quality Investigation
1. **Debug trained Gaussian rendering** - Compare SH/scale/opacity with VRM-converted
2. **Test black background training** - Should eliminate background artifacts entirely
3. **Test auto-rigger** - Download Conker OBJ and run through pipeline
4. Camera convention: azimuth=0 shows back, azimuth=180 shows front (document)
5. Full recipe.yaml integration for Noodling avatars

### Test GPU Rendering:
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -c "
from noodlestudio.core.gaussian_renderer import GaussianRenderer, create_orbit_camera
from noodlestudio.core.radiance_component import RadianceComponent
component = RadianceComponent('test')
component.load_asset('../../external/vrm_samples/alicia_densified_tuned.radiance')
# No scale_mult needed with densified assets
renderer = GaussianRenderer()
camera = create_orbit_camera(distance=2.5, elevation=15, azimuth=180, target=(0,0.8,0))
image, alpha, info = renderer.render_component(component, camera)
print(f'FPS: {1000/8:.0f}, Visible: {info[\"visible\"]:,}, Backend: {info.get(\"backend\", \"software\")}')
"
```

### Convert VRM with Densification:
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance \
    ../../external/vrm_samples/AliciaSolid.vrm \
    -o ../../external/vrm_samples/alicia_dense.radiance \
    -v
# Options: --no-densify, --no-face-centers, --no-edge-midpoints, --no-adaptive-scale
```

---

## What's Working (Tested Dec 24, 2025)

| Component | Status | Key File |
|-----------|--------|----------|
| **GPU Gaussian Renderer** | **120 FPS** | `core/gaussian_renderer.py` (gsplat-mps) |
| **Gaussian Viewer Panel** | **NEW** | `panels/gaussian_viewer_panel.py` |
| **Radiance Inspector** | **NEW** | `panels/radiance_inspector.py` |
| **RadianceComponent** | **NEW** | `core/radiance_component.py` |
| VRM to Gaussians | Working | `tools/vrm_to_radiance.py` |
| **Gaussian Training Facet** | **NEW** | `core/gaussian_training_facet.py` |
| **Skeleton Binding Facet** | **NEW** | `core/skeleton_binding_facet.py` |
| Model Importer | Working | `core/model_importer.py` |
| VRM Parsing | Working | `vrm_parser.py` |
| .radiance Format | Working | `radiance_format.py` |
| Gaussian Collision | Working | `gaussian_collision.py` |
| Facet System | Working | `facet_executor.py` |
| Scene Protocol | Working | `scene_state_manager.py` |
| Multi-provider LLM | Working | 8 providers configured |

**Test Assets:**
- `external/vrm_samples/AliciaSolid.vrm` (7.5MB VRM source)
- `external/vrm_samples/alicia_densified_tuned.radiance` (11MB, 137K Gaussians - RECOMMENDED)
- `external/vrm_samples/alicia_textured.radiance` (4.2MB, 52K Gaussians - legacy sparse)

---

## Gaussian Radiance System

### Core Concept
**"Every Gaussian knows what it represents. Every frame is query-able."**

Unlike Genie/Mirage (stateless pixel predictors), Noodlings constructs worlds from semantic truth.

### Architecture

```
RadianceComponent (entity wrapper)
    |
    +-- RadianceAsset (.radiance file)
    +-- MaterialOverride (tint, emission, scale_mult, alpha_mult)
    +-- Transform (position, rotation, scale)
    |
    v
GaussianRenderer
    |
    +-- gsplat-mps (GPU, 120 FPS)
    +-- PyTorch software (fallback, 0.1 FPS)
    |
    v
GaussianViewerPanel (Unity-style viewport)
    |
    +-- Orbit: left drag
    +-- Pan: right drag
    +-- Zoom: scroll
    +-- Focus: F key
```

### File Format: .radiance
Binary chunk-based (like GLB):
- `GAUS` - Position, scale, rotation, opacity, SH coefficients
- `SKEL` - Skeleton hierarchy
- `SKIN` - Skinning weights per Gaussian
- `SEMA` - Semantic labels (body_part, region)
- `CLIP` - 512-D embeddings (optional)
- `META` - Entity metadata

### Key APIs

```python
# RadianceComponent - entity wrapper
from noodlestudio.core.radiance_component import RadianceComponent
component = RadianceComponent('red')
component.load_asset('red.radiance')
component.material.tint = Color(1.0, 0.8, 0.8)  # Pink tint
# component.material.scale_mult = 1.0  # Default, densified assets don't need adjustment

# GPU Rendering
from noodlestudio.core.gaussian_renderer import GaussianRenderer, create_orbit_camera
renderer = GaussianRenderer()  # Auto GPU detection
camera = create_orbit_camera(distance=2.5, elevation=15, azimuth=180)
image, alpha, info = renderer.render_component(component, camera)

# Load/save radiance files
from noodlestudio.core.semantic_world.radiance_format import load_radiance, save_radiance
asset = load_radiance("red.radiance")
save_radiance(asset, "modified.radiance")

# Collision detection
from noodlestudio.core.semantic_world.gaussian_collision import GaussianCollisionDetector
detector = GaussianCollisionDetector()
detector.add_entity('red', red_asset)
touches = detector.detect_touches()
```

### External Tools
- `external/OpenSplat/` - Gaussian training (Metal GPU)
- `external/ml-sharp/` - Apple SHARP (image to Gaussian)
- `/Users/thistlequell/git/gsplat-mps/` - GPU renderer (AGPLv3)

---

## OpenSplat Training Pipeline

### Prerequisites
1. **Metal Toolchain** (required for GPU training):
   ```bash
   xcodebuild -runFirstLaunch
   xcodebuild -downloadComponent MetalToolchain
   ```

2. **Build OpenSplat with Metal**:
   ```bash
   cd external/OpenSplat
   mkdir build && cd build
   cmake -DGPU_RUNTIME=MPS -DCMAKE_BUILD_TYPE=Release ..
   make -j16
   ```

### Training from Multi-view Images
```bash
# Prepare dataset: images/ folder + transforms.json (NeRFStudio format)
# or use COLMAP reconstruction

cd external/OpenSplat/build
./opensplat /path/to/dataset \
  -o /path/to/output.ply \
  -n 30000 \           # iterations (5K=fast, 15K=good, 30K=publication)
  --sh-degree 2        # spherical harmonics degree
```

### How Training Works
Each iteration:
1. **Rasterize**: Project 3D Gaussians to random training camera
2. **Compare**: L1 + SSIM loss against ground truth image
3. **Update**: Adam optimizer adjusts position, scale, rotation, color, opacity
4. **Densify/Prune** (every 100 steps): Split large gradients, clone small ones, cull invisible

Loss progression: ~0.5 (random) -> ~0.05 (structure) -> ~0.002 (converged)

### Converting Trained PLY to .radiance
**IMPORTANT**: Training on white backgrounds creates background artifact Gaussians.
Use filtering to remove them:

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance \
    /path/to/trained.ply \
    -o /path/to/output.radiance \
    -v
# Filtering is ON by default for PLY files

# Custom filter thresholds:
#   --min-opacity 0.8      # Keep high-opacity (actual surface)
#   --max-scale 0.05       # Remove huge background blobs
#   --max-brightness 2.0   # Remove saturated white Gaussians
#   --no-filter            # Disable filtering entirely
```

### Training Tips
- **Black backgrounds**: Best results - no background artifact filtering needed
- **White backgrounds**: Requires filtering (default), may lose some edge detail
- **More views = better**: 72 views worked well for character turntables
- **Iteration count**: 5K for testing, 30K for production

---

## Training Facets (NoodleStudio Integration)

Training can be run from within NoodleStudio using facet assemblies. This enables
end-users to create Gaussian avatars without command-line tools.

### Available Training Facets

| Facet Type | Description | Key File |
|------------|-------------|----------|
| `GaussianTrainingFacet` | Wraps OpenSplat training | `core/gaussian_training_facet.py` |
| `SkeletonBindingFacet` | Binds Gaussians to VRM skeleton | `core/skeleton_binding_facet.py` |
| `AutoRiggerFacet` | Mixamo-style auto-rigging | `core/auto_rigger_facet.py` |

### Facet Assemblies

| Assembly | Description |
|----------|-------------|
| `gaussian_training.yaml` | Train Gaussians from images |
| `avatar_from_images.yaml` | Full pipeline: train + bind skeleton |

### Usage via Scripting API

```javascript
// Train Gaussians
let result = await context.noodle.training.train({
    dataset_path: '/path/to/images',
    iterations: 30000,
    convert_to_radiance: true,
    filter_output: true,
    onProgress: (p) => console.log(`${p.progress_percent.toFixed(1)}%`)
});

// Bind to skeleton
let rigged = await context.noodle.binding.bind({
    gaussian_ply_path: result.output_path,
    vrm_path: '/path/to/avatar.vrm',
    display_name: 'My Avatar'
});
```

### Running Assemblies

```bash
# Via facet executor (future integration with NoodleStudio UI)
# Assemblies are in: applications/noodlestudio/facet_assemblies/
```

---

## Auto-Rigger (Mixamo-style)

Automatic rigging for arbitrary meshes - skips intermediate mesh skinning entirely.
Goes directly from mesh geometry to weighted Gaussians.

### Pipeline

```
Traditional:  Mesh -> Rig mesh -> Skin mesh -> Convert to Gaussians -> Transfer weights
Auto-rigger:  Mesh -> Markers -> Fit skeleton -> Sample Gaussians -> Direct bone weights
```

### Usage via CLI

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.auto_rigger \
    /path/to/model.obj \
    -o /path/to/output.radiance \
    -v
# Markers are auto-detected from mesh extremities
```

### Usage via Scripting API

```javascript
// Auto-rig with auto-detected markers
let result = await context.noodle.rigger.rig({
    mesh_path: '/path/to/conker.obj',
    auto_detect: true,
    display_name: 'Conker'
});

// Or with manual markers
let result = await context.noodle.rigger.rig({
    mesh_path: '/path/to/model.obj',
    auto_detect: false,
    markers: {
        hips: [0, 1.0, 0],
        head: [0, 1.8, 0],
        left_hand: [0.5, 1.2, 0],
        right_hand: [-0.5, 1.2, 0],
        left_foot: [0.1, 0, 0],
        right_foot: [-0.1, 0, 0]
    }
});
```

### Key Files

- `tools/auto_rigger.py` - Core auto-rigging logic
- `core/auto_rigger_facet.py` - Facet wrapper for assemblies

### Supported Formats

- OBJ (working)
- GLTF/GLB (planned)
- FBX (planned)

---

## Core Architecture

### Affect Model: PAD + Boredom + Sorrow
5-dimensional continuous affect (NO discrete emotion labels):
- `valence` (-1 to +1) - Pleasure/displeasure
- `arousal` (0 to 1) - Energy
- `dominance` (0 to 1) - Control
- `boredom` (0 to 1)
- `sorrow` (0 to 1)

### Facet System
Visual node-based cognitive architecture:
```
INCOMING -> CHARM_NET -> CONTEXT_INTELLIGENCE -> Cognitive facets -> OUTGOING
```

**Types:** LLMFacet, ScriptedFacet, CharmNetworkFacet, ContextIntelligenceFacet, ConvergenceFacet

**Key files:**
- `facet_system.py` - Data model
- `facet_executor.py` - Execution engine
- `facets_editor_panel.py` - Visual editor

### CharmNetwork
MLX-based temporal hierarchy (~54K params, ~2-3ms inference):
- Fast LSTM (16-D): Seconds
- Medium LSTM (16-D): Minutes
- Slow GRU (8-D): Hours/days

### Scene Protocol
Perception-filtered context for stateless renderers:
```
SceneStateManager (canonical truth)
       |
   +---+---+
   |       |
Red's   Yuki's    Full Packet
Slice   Slice     (for Genie)
```

---

## Project Structure

```
Noodlings/
└── red/
    ├── noodling.yaml      # Manifest
    ├── recipe.yaml        # Character definition
    ├── assembly.yaml      # Facet topology
    ├── Radiances/         # Gaussian splat models
    │   └── fire_imp.radiance
    └── Assets/
        └── reference.png
```

---

## Development

### Running NoodleStudio
```bash
cd applications/noodlestudio
./launch_with_log.sh
```

### Running noodleMUSH Server
Toggle in NoodleStudio status bar (bottom-right), or:
```bash
cd applications/cmush
./start.sh
```

**Ports:** 8080 (HTTP), 8765 (WebSocket), 11434 (Ollama)

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

### Christopher Alexander's "Timeless Way"
Organic development: Probe -> Iterate -> Polish when friction hurts.
Not Agile. Not a startup. Experimental architecture.

---

## Project Context

**Creator:** Caitlyn (Unity employee #12, Asset Store creator, 54)
**Location:** Garcia River Forest cabin
**Hardware:** M3 Ultra 512GB

**Mission:** Open-source alternative to "Consciousness-as-a-Service"
- Visual cognitive architecture editor
- Stateful affect-driven characters
- Brains/hearts for generative worlds

**Demo:** Steve DiPaola (SFU CogSci) - soon

---

## Quick Reference

| What | Where |
|------|-------|
| **Gaussian viewer** | `panels/gaussian_viewer_panel.py` |
| **Radiance inspector** | `panels/radiance_inspector.py` |
| **GPU renderer** | `core/gaussian_renderer.py` |
| **RadianceComponent** | `core/radiance_component.py` |
| **Gaussian Training Facet** | `core/gaussian_training_facet.py` |
| **Skeleton Binding Facet** | `core/skeleton_binding_facet.py` |
| **Auto-Rigger** | `tools/auto_rigger.py` |
| Facet editor | `panels/facets_editor_panel.py` |
| Neural canvas | `panels/neural_canvas/` |
| VRM preview | `panels/vrm_preview_panel.py` |
| Gaussian collision | `core/semantic_world/gaussian_collision.py` |
| Radiance format | `core/semantic_world/radiance_format.py` |
| Scene protocol | `core/semantic_world/scene_state_manager.py` |
| Scripting API | `scripting/noodle_api.py` |
| Server | `applications/cmush/server.py` |

**Scripting:** `context.noodle.models`, `context.noodle.affect`, `context.noodle.pose`, `context.noodle.quantum`

---

## Completed Systems (Summary)

- **GPU Gaussian Rendering (120 FPS)**
- **Gaussian Training Facets** (OpenSplat integration)
- **Skeleton Binding Facets** (VRM rigging)
- **Auto-Rigger** (Mixamo-style direct-to-Gaussian rigging)
- Multi-provider LLM (8 providers)
- Neural Canvas with PyTorch test mode
- Scriptability API (context.noodle)
- Animation tracks (.affecttrack, .posetrack)
- MCP integration
- Utility facets (31 types)
- Headless Player runtime
- Cognitive Timeline Editor
- Cloud account system (backend deployed)
- Multimodal facets (audio, vision, image gen)
- IBM Quantum integration
- Noodling names generator

---

**Ordnung muss sein!**
