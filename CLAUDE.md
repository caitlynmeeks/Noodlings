# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: January 3, 2026

**FOR NEXT CLAUDE: START HERE!**

---

## COMPLETED: Phase 3a UI Canvas Infrastructure (Jan 3, 2026)

### What Was Done
Created the Delphi-style UI canvas system for building application interfaces.
The canvas IS the application - a "3D game" is just a canvas with a fullscreen
RadianceViewport component.

### Architecture
```
ui.yaml (user's design - stable contract)
    ↓
UIComponent classes (our API)
    ↓
QtWidgetRenderer (v1 desktop renderer)
```

Users see Panel, Button, Label, RadianceViewport - never Qt internals.

### Module Structure
```
noodlestudio/runtime/ui/
├── __init__.py         # Public API exports
├── component.py        # UIComponent base, Anchors, EventBinding
├── loader.py           # YAML loader
├── renderer.py         # QtWidgetRenderer
└── components/
    ├── panel.py        # Container with background
    ├── label.py        # Static text
    ├── button.py       # Clickable button
    ├── text_input.py   # Single-line input
    └── radiance_viewport.py  # 3D Gaussian renderer
```

### Usage
```bash
# Run with GUI (loads ui.yaml from project)
python -m noodlestudio.runtime path/to/project --gui

# Run with custom UI file
python -m noodlestudio.runtime --gui --ui path/to/ui.yaml

# Custom window size
python -m noodlestudio.runtime --gui -w 1280x720
```

### Programmatic
```python
from noodlestudio.runtime.ui import load_ui, QtWidgetRenderer

root = load_ui("ui.yaml")
renderer = QtWidgetRenderer()
widget = renderer.render(root)
```

### Documentation
- `docs/noodlestudio/ui-canvas.md` - Full specification

---

## COMPLETED: Phase 3b Chat Components (Jan 3, 2026)

### What Was Done
Added chat UI components and event wiring system for noodling interaction.

### New Components
| Component | Description |
|-----------|-------------|
| `ChatHistory` | Scrolling message list with styled bubbles |
| `ChatInput` | Compound input field + send button |
| `ChatMessage` | Message data model (role, content, sender, timestamp) |
| `MessageRole` | Enum: USER, NOODLING, SYSTEM |

### Event System
| Class | Description |
|-------|-------------|
| `UIEventDispatcher` | Routes UI events to noodlings |
| `EventBinding` | Declarative event-to-action mapping |

### Supported Actions
- `send_to_noodling` - Send message to a noodling, show response in ChatHistory
- `set_value` - Set component value
- `show` / `hide` / `toggle_visible` - Control visibility

### Module Updates
```
noodlestudio/runtime/ui/
├── event_dispatcher.py    # NEW - UIEventDispatcher
├── components/
│   ├── chat_history.py    # NEW - ChatHistory, ChatMessage, MessageRole
│   └── chat_input.py      # NEW - ChatInput
└── demo_chat.yaml         # Demo UI for testing
```

### Demo
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 noodlestudio/runtime/ui/test_chat_demo.py
```

### Tests
- 16 new tests in `tests/test_ui_canvas.py` (47 total, all passing)

---

## COMPLETED: Admin Dashboard Deployed to Production (Jan 2, 2026)

### Live URLs
| Service | URL |
|---------|-----|
| Admin Dashboard | `https://admin.noodlings.ai` |
| API | `https://api.noodlings.ai` |

### What Was Done
- Deployed admin dashboard to Cloudflare Pages
- Added custom domain `admin.noodlings.ai`
- Fixed CORS to allow admin dashboard origin
- Added Release Health section with GitHub Issues integration
- Google OAuth login working in production

### Features
- **Overview**: System health, live stats, summary metrics
- **Release Health**: GitHub issues counts (crashes, bugs, features) + recent issues list
- **Users**: List, search, edit, ban, credit adjustments
- **Noodlings**: List, moderate visibility
- **Credits**: Transaction history
- **LLM Usage**: Usage by model, by day, top users

### Local Dev
```bash
cd backend/admin-dashboard
npm run dev -- --port 5174
```

### Deployment
```bash
cd backend/admin-dashboard
npm run build
npx wrangler pages deploy .svelte-kit/cloudflare --project-name=noodlings-admin --branch=main
```

---

## COMPLETED: Crash Recovery System (Jan 2, 2026)

### What Was Done
- **Sentinel file** (`~/.noodlestudio/.running`) detects crashes that bypass Python exception handling
- **Recovery dialog** on startup offers to send crash report
- **Bug report endpoint** working (GitHub token refreshed)

### How It Works
1. Startup creates sentinel file (PID, timestamp, version)
2. Clean exit removes sentinel
3. Next launch checks for stale sentinel = crash detected
4. Recovery dialog shows with crash context from logs
5. User can send report or dismiss

### Key Files
- `main.py` - `create_sentinel()`, `check_for_crash()`, `show_crash_recovery_dialog()`
- `dialogs/bug_report_dialog.py` - Report UI
- `docs/development/bug-reporting.md` - Full documentation

---

## COMPLETED: Phase 2 LLM Routing API (Jan 3, 2026)

### What Was Done
Created the direct-to-provider LLM routing endpoint for built applications.

### Architecture Decision
The new `/v1/chat/completions` is **completely separate** from existing `/llm/*` routes:
- **Existing `/llm/*`**: Uses OpenRouter, unchanged, for NoodleStudio internal use
- **New `/v1/*`**: Direct to providers (Anthropic first), for built apps

### Backend Files Created/Modified
| File | Description |
|------|-------------|
| `routes/v1.ts` | NEW - OpenAI-compatible `/v1/chat/completions` endpoint |
| `types.ts` | Added `ANTHROPIC_API_KEY`, `DIRECT_MODEL_PRICING`, `MODEL_ID_MAP` |
| `index.ts` | Wired up `/v1` route |
| `wrangler.toml` | Documented new secrets |

### Runtime Files Modified
| File | Description |
|------|-------------|
| `runtime/llm_client.py` | Added `noodlings` provider for cloud routing |

### Endpoint
```bash
POST https://api.noodlings.ai/v1/chat/completions
Authorization: Bearer <user_token>

{
  "model": "anthropic/claude-3.5-sonnet",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024
}
```

### Provider Options for Built Apps
| Provider | Description | Config |
|----------|-------------|--------|
| `noodlings` | Our cloud service | `NOODLINGS_API_KEY` |
| `ollama` | Local inference | Free, no key |
| `anthropic` | Direct to Anthropic | User's own key |

### Deployment Required
```bash
# Set the Anthropic API key secret
wrangler secret put ANTHROPIC_API_KEY
```

### Documentation
- `docs/noodlestudio/llm-routing-service.md` - Full specification
- `docs/noodlestudio/build-system.md` - LLM config in build.yaml

---

## COMPLETED: Phase 1 Runtime Foundation (Jan 3, 2026)

### What Was Done
Created the headless runtime module for executing NoodleStudio projects without the editor GUI.

### Module Structure
```
noodlestudio/runtime/
├── __init__.py      # Public API exports
├── __main__.py      # Module entry point (python -m)
├── app.py           # NoodleApp - core runtime class
├── cli.py           # Command-line interface
└── llm_client.py    # HeadlessLLMClient (no Qt dependencies)
```

### Usage

```bash
# Run project interactively
python -m noodlestudio.runtime path/to/project --interactive

# Run assembly with single input
python -m noodlestudio.runtime --assembly agent.yaml --input "Hello"

# Run with specific provider
python -m noodlestudio.runtime path/to/project \
    --provider anthropic --model claude-3-5-sonnet-20241022
```

### Programmatic Usage

```python
from noodlestudio.runtime import NoodleApp

app = NoodleApp()
app.load_project("/path/to/project")
result = await app.run("Hello, world!")
print(result['response'])
```

### Supported LLM Providers
- `ollama` - Local Ollama server (default)
- `anthropic` - Anthropic Claude API
- `openai` - OpenAI API
- `openrouter` - OpenRouter aggregated API

### Environment Variables
- `NOODLE_LLM_PROVIDER` - Provider selection
- `NOODLE_LLM_MODEL` - Default model
- `NOODLE_LLM_BASE_URL` - Custom API URL
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`

### Test Assets
- `facet_assemblies/simple_echo.yaml` - Minimal test assembly

---

## NEXT SESSION: GUI Window + Build System

**Planning documents** (all decisions finalized Jan 3, 2026):
- `docs/noodlestudio/build-system.md` - Unity-style build system
- `docs/noodlestudio/ui-canvas.md` - Delphi-style UI designer
- `docs/noodlestudio/llm-routing-service.md` - OpenRouter-style LLM routing

### Design Decisions Summary

| Area | Decision |
|------|----------|
| **Runtime UI** | Full 3D viewport default, headless as build option |
| **Server** | No server for standalone (direct execution); embedded for multiplayer |
| **Build Target** | macOS .app first (py2app), then Windows/Linux |
| **Packaging** | Bundle Python runtime, CharmNetwork weights; MLX on demand |
| **Config** | Simple `build.yaml` with `main_stage` reference |
| **UI Technology** | Qt Widgets for v1 |
| **Billing** | Prepaid credits with auto-topup (Anthropic-style) |
| **Free Tier** | 1000 credits ($10) on signup |
| **Model Tiers** | None for v1 - all models available |
| **Org Creation** | Self-service (users create their own orgs) |
| **API Key Format** | `nood_` prefix + 32 random chars |
| **Margin** | 20% on LLM provider costs |
| **Asset Revenue** | 70% creator / 30% Noodlings |

### Implementation Order
1. **Phase 1: Runtime Foundation** - DONE (Jan 3)
   - Created `noodlestudio/runtime/` module
   - CLI working: `python -m noodlestudio.runtime`
   - Tested with `simple_echo.yaml` assembly

2. **Phase 2: LLM Routing API** - DONE (Jan 3)
   - `/v1/chat/completions` endpoint created
   - Direct Anthropic provider integration
   - Token counting + billing (deduct credits)
   - `noodlings` provider added to runtime
   - Anthropic API key deployed to Cloudflare

3. **Phase 3a: UI Canvas Infrastructure** - DONE (Jan 3)
   - Created `noodlestudio/runtime/ui/` module (Delphi-style canvas)
   - UIComponent base class with anchor system
   - Panel, Label, Button, TextInput components
   - RadianceViewport component (embeds GaussianRenderer)
   - QtWidgetRenderer (component tree to Qt widgets)
   - YAML loader for `ui.yaml` files
   - `--gui` flag added to runtime CLI
   - Test: `python -m noodlestudio.runtime --gui --ui path/to/ui.yaml`

4. **Phase 3b: Chat Components** - DONE (Jan 3)
   - ChatHistory component (scrolling message list with styled bubbles)
   - ChatInput component (input + send button)
   - UIEventDispatcher for event routing
   - `send_to_noodling` action with chat_history integration
   - Message roles: USER, NOODLING, SYSTEM
   - Demo: `python3 noodlestudio/runtime/ui/test_chat_demo.py`

5. **Phase 3c: RadianceViewport** - NEXT
   - Embed GaussianRenderer in RadianceViewportWidget
   - Camera controls (orbit, pan, zoom)
   - Stage loading integration

6. **Phase 4: Build System**
   - Asset packaging (copy/filter project files)
   - macOS .app bundler (py2app)
   - "File > Build Application..." menu item

6. **Phase 5: UI Canvas Designer** (later)
   - Designer panel in editor
   - Drag-drop component palette

7. **Phase 6: Admin Dashboard Extensions**
   - LLM routing management pages
   - Provider keys, pricing, analytics
   - Org management UI

---

## BACKLOG: Admin Dashboard - Issue Credits UI

### The Goal
Add a UI in the admin dashboard to issue/adjust credits for users without using curl.

### Current State
- API endpoint exists: `POST /admin/users/:id/credits`
- Works via curl (tested Jan 3, 2026)
- No UI in admin dashboard

### Implementation
- Add "Adjust Credits" button on user detail page
- Modal with: amount (+/-), reason (required)
- Show transaction history on user page
- Confirmation before large adjustments (>1000 credits)

### Files to Modify
- `backend/admin-dashboard/src/routes/users/[id]/+page.svelte`

---

## BACKLOG: Asset-Aware Inspector

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

### Component System (NEW - Jan 2026)

**Unity-style component architecture for NoodleStudio.** Entities (Noodlings, Props, etc.)
can have multiple components attached. Each component is a modular, inspectable unit.

**Key Files:**
- `core/component_base.py` - ComponentBase, ComponentRegistry, PropertySpec
- `core/component_collection.py` - ComponentCollection (manages components per entity)
- `core/components/` - Concrete component implementations

**Built-in Components:**
- `ArtbookComponent` - Reference art collection (category: art)
- `RadianceComponent` - Gaussian splat visuals (category: rendering)
- `FacetAssembly` - Cognitive architecture (category: charm) - see below

**Categories (with Inspector border colors):**
- `charm` (green) - Core consciousness components
- `art` (orange) - Visual references
- `behavior` (blue) - Game mechanics
- `rendering` (cyan) - Visual presentation
- `audio` (purple) - Sound/voice
- `custom` (gray) - User scripts

**Usage:**
```python
from noodlestudio.core.component_collection import ComponentCollection
from noodlestudio.core.component_base import component_registry

# Create collection for entity
coll = ComponentCollection(entity_id="red_fire_anklebiter")

# Add components
artbook = coll.add("artbook")
artbook.add_art("/path/to/concept.png", note="Main character design")

# Serialize for YAML storage
data = coll.to_dict()

# Access components
artbook = coll.get("artbook")
```

**Test count:** 25 tests in `tests/test_component_system.py`

### Facet System

**Facet Assembly is a component type.** Every Noodling has a Facet Assembly - a visual
node graph that defines how it thinks. This replaced the earlier "Cognitive Transistor"
system (CulturalTransistor, PersonalityTransistor, etc.) which is now deprecated.

```
INCOMING -> CHARM_NET -> CONTEXT_INTELLIGENCE -> Cognitive facets -> OUTGOING
```

**Facet Types:**
- `LLMFacet` - Language model processing (configurable model per facet)
- `ScriptedFacet` - Custom JavaScript logic
- `CharmNetworkFacet` - Temporal affect model (LSTM/GRU hierarchy)
- `ContextIntelligenceFacet` - Memory and context management
- `ConvergenceFacet` - Multi-input synthesis (waits for all inputs)
- `TickerFacet`, `BranchFacet`, `CacheFacet`, `RateLimiterFacet` - Flow control

**Inspector shows:**
- "Noodle Component" - Live affect/surprise telemetry (automatic for all Noodlings)
- "Facet" dropdown - Select and edit individual facet properties

**Key files:**
- `facet_system.py` - Data model and YAML serialization
- `facet_executor.py` - Parallel execution engine
- `facets_editor_panel.py` - Visual node editor

**Documentation:** See `docs/noodlestudio/facets.md` for full specification.

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

## Testing

### Test Infrastructure
- **Framework:** pytest + pytest-qt
- **Config:** `applications/noodlestudio/pytest.ini`
- **Fixtures:** `applications/noodlestudio/tests/conftest.py`
- **Test count:** 128 tests (as of Jan 2, 2026)

### Running Tests
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest                    # Run all tests
PYTHONPATH=.:../.. pytest -v                 # Verbose output
PYTHONPATH=.:../.. pytest tests/test_panel_wiring.py  # Single file
PYTHONPATH=.:../.. pytest -k "test_undo"     # By name pattern
PYTHONPATH=.:../.. pytest -m "not slow"      # Skip slow tests
```

### Test Categories (Markers)
| Marker | Description | When to Run |
|--------|-------------|-------------|
| `@pytest.mark.unit` | Fast, no external deps | Every commit |
| `@pytest.mark.gui` | Requires Qt/pytest-qt | Before merges |
| `@pytest.mark.slow` | Training, rendering | Manual/nightly |
| `@pytest.mark.integration` | Server required | Before release |

### Development Workflow

**BEFORE starting feature work:**
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest --tb=short
```
Note any existing failures. Don't fix unrelated issues mid-feature.

**DURING development:**
- Write tests for new code in `tests/` folder
- Run affected tests frequently
- Use fixtures from `conftest.py` (don't reinvent)

**BEFORE committing:**
```bash
PYTHONPATH=.:../.. pytest -v
```
ALL tests must pass (or have documented `@pytest.mark.skip` reasons).

### Key Test Files
| File | Tests | Coverage |
|------|-------|----------|
| `test_agentic_system.py` | 68 | Facets, MCP, Player, proxies |
| `test_component_system.py` | 25 | ComponentBase, Registry, Collection, Artbook |
| `test_panel_wiring.py` | 17 | Qt signals, Inspector, Stage View |
| `test_radiance_component.py` | 10 | Gaussians, rendering, spatial queries |
| `test_clip_queries.py` | 3 | Semantic search |
| `test_gaussian_adapter.py` | 1 | Asset creation |

### Writing New Tests
```python
# Use fixtures from conftest.py
def test_something(main_window, qtbot):
    """Test description."""
    # Arrange
    main_window.hierarchy.entitySelected.emit('noodling', mock_data)

    # Act
    qtbot.wait(50)  # Allow signal propagation

    # Assert
    assert main_window.inspector.current_mode == 'noodling'

# For radiance/gaussian tests
def test_radiance_thing(loaded_radiance_component):
    """Test radiance operations."""
    component = loaded_radiance_component
    assert component.gaussian_count > 0
```

### Available Fixtures (conftest.py)
- `qapp` - QApplication singleton
- `main_window` - Full MainWindow instance
- `qtbot` - pytest-qt interaction helper
- `radiance_component` - Synthetic test component
- `loaded_radiance_component` - Real or synthetic asset
- `mock_noodling_data`, `mock_prop_data`, `mock_zone_data`
- `empty_facet_assembly`, `simple_facet_assembly`
- `temp_project_dir`, `temp_stage_dir`

### CI Reminder for Claude
When the user finishes a feature or asks to commit:
1. Proactively suggest running tests
2. If tests fail, help fix them before committing
3. For new features, ask: "Should I add tests for this?"

---

## Bug Tracking

### GitHub Issues
All bugs are tracked in GitHub Issues. Claude can access via `gh` CLI:

```bash
# List all bugs
gh issue list --label bug

# List crashes (high priority)
gh issue list --label severity:crash

# View specific issue
gh issue view 42

# Search
gh issue list --search "crash facet"

# Create issue (when asked)
gh issue create --title "Bug: description" --label bug
```

### NoodleStudio Integration
- **Help > Report a Bug...** - Manual bug report dialog
- **Crash Reporter** - Automatic on unhandled exceptions
- Reports submit to GitHub via Cloudflare Worker proxy

### Bug Report Flow
```
User reports bug -> Cloudflare Worker -> GitHub Issue
                    (validates, formats)   (auto-labeled)
```

### Key Files
- `dialogs/bug_report_dialog.py` - Report dialog UI
- `main.py` - Crash reporter hook
- `backend/noodlings-api/src/routes/bugs.ts` - API endpoint

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
| **Component system** | `core/component_base.py`, `core/component_collection.py` |
| **ArtbookComponent** | `core/components/artbook_component.py` |
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
