# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: January 3, 2026

---

## COMPLETED: Phase 4 - Build System (Jan 3, 2026)

**"File > Build Application..." is live!** (Ctrl+B)

### What Was Built

| File | Purpose |
|------|---------|
| `appbuilder/__init__.py` | Module exports |
| `appbuilder/builder.py` | Orchestrator - validates, packages, bundles |
| `appbuilder/packager.py` | Asset collection with filtering |
| `appbuilder/bundler_macos.py` | Creates .app bundles |

### How To Use

1. Open a project in NoodleStudio
2. **File > Build Application...** (or Ctrl+B)
3. If no `build.yaml`, it offers to create one
4. Choose output location
5. Progress dialog shows build status
6. Done - double-click the .app to run

### Bundle Structure
```
MyApp.app/Contents/
├── Info.plist
├── MacOS/MyApp         # Launcher script
└── Resources/
    ├── AppIcon.icns    # Auto-converted from PNG
    ├── runtime/        # NoodleStudio core (filtered)
    └── project/        # User's project files
```

---

## FOR NEXT CLAUDE: Phase 5 or 6

Phases 1-4 complete. Choose next:

**Phase 5: Additional Build Targets**
- Windows .exe (PyInstaller)
- Linux binary
- Docker container

**Phase 6: UI Canvas Designer**
- Visual designer panel in NoodleStudio
- Drag-drop component palette
- Property editor for components

---

## Recent Work: Runtime & UI Canvas (Jan 2-3, 2026)

### Phase 1: Runtime Foundation
Created `noodlestudio/runtime/` module for headless execution:
```bash
python -m noodlestudio.runtime path/to/project --interactive
python -m noodlestudio.runtime --assembly agent.yaml --input "Hello"
```
Key files: `runtime/app.py`, `runtime/cli.py`, `runtime/llm_client.py`

### Phase 2: LLM Routing API
Live at `https://api.noodlings.ai/v1/chat/completions`:
- Direct Anthropic routing (no OpenRouter middleman)
- Token counting + credit billing working
- `noodlings` provider in runtime for built apps

### Phase 3a-d: UI Canvas System
Delphi-style canvas at `noodlestudio/runtime/ui/`:
```
ui.yaml -> UIComponent classes -> QtWidgetRenderer -> Qt Widgets
```

**Components:** Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport

**Event system:**
- `send_to_noodling` - Route messages to noodlings
- `call_script` - Execute inline/external JavaScript
- Component bindings - Reactive property updates

**Tests:** 64 tests in `test_ui_canvas.py`

### Admin Dashboard & Crash Recovery
- Live at `https://admin.noodlings.ai`
- Sentinel file crash detection (`~/.noodlestudio/.running`)
- Bug report endpoint working

---

## BACKLOG

### Admin Dashboard - Issue Credits UI
Add "Adjust Credits" button on user detail page with modal.
File: `backend/admin-dashboard/src/routes/users/[id]/+page.svelte`

### Asset-Aware Inspector
Inspector shows contextual info when selecting assets (folders, noodlings, stages, radiance files, VRMs, etc.). Design approved - see full spec in revision history.

### Trained Gaussian Quality
OpenSplat-trained Gaussians have background artifacts. Investigate SH coefficients, try black background training.

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

**Types:** LLMFacet, ScriptedFacet, CharmNetworkFacet, ContextIntelligenceFacet, ConvergenceFacet, flow control facets

**Key files:** `facet_system.py`, `facet_executor.py`, `facets_editor_panel.py`

### Component System
Unity-style architecture - entities have multiple components:
- `ArtbookComponent` (art), `RadianceComponent` (rendering), `FacetAssembly` (charm)
- Key files: `core/component_base.py`, `core/component_collection.py`

### Gaussian Radiance System
**"Every Gaussian knows what it represents. Every frame is query-able."**

```
RadianceComponent -> GaussianRenderer (gsplat-mps GPU, 120 FPS) -> GaussianViewerPanel
```

**.radiance format:** Binary chunks - GAUS, SKEL, SKIN, SEMA, CLIP, META

### Scene Protocol
Perception-filtered context for stateless renderers via SceneStateManager.

---

## Testing

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest              # Run all (~130 tests)
PYTHONPATH=.:../.. pytest -v           # Verbose
PYTHONPATH=.:../.. pytest -k "test_ui" # By pattern
```

**Key test files:**
- `test_agentic_system.py` (68) - Facets, MCP, Player
- `test_ui_canvas.py` (64) - UI components, events, bindings
- `test_component_system.py` (25) - ComponentBase, Registry
- `test_panel_wiring.py` (17) - Qt signals, Inspector

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

---

## Quick Reference

| What | Where |
|------|-------|
| **Build system** | `noodlestudio/appbuilder/` |
| **Runtime module** | `noodlestudio/runtime/` |
| **UI Canvas** | `noodlestudio/runtime/ui/` |
| **Component system** | `core/component_base.py`, `core/component_collection.py` |
| **Gaussian viewer** | `panels/gaussian_viewer_panel.py` |
| **GPU renderer** | `core/gaussian_renderer.py` |
| **RadianceComponent** | `core/radiance_component.py` |
| **Facet editor** | `panels/facets_editor_panel.py` |
| **Scene hierarchy** | `panels/scene_hierarchy.py` |
| **Assets panel** | `panels/assets_panel.py` |
| **Scripting API** | `scripting/noodle_api.py` |
| **Server** | `applications/cmush/server.py` |

**Scripting:** `context.noodle.models`, `context.noodle.affect`, `context.noodle.pose`, `context.noodle.quantum`

---

## Project Context

**Creator:** Caitlyn (Unity employee #12, Asset Store creator)
**Location:** Garcia River Forest cabin
**Hardware:** M3 Ultra 512GB

**Mission:** Open-source alternative to "Consciousness-as-a-Service"
- Visual cognitive architecture editor
- Stateful affect-driven characters
- Brains/hearts for generative worlds

---

## Completed Systems

- **Build System** - File > Build Application (macOS .app bundles)
- **Runtime Foundation** - Headless execution, CLI, multi-provider LLM
- **LLM Routing API** - api.noodlings.ai/v1/chat/completions (live)
- **UI Canvas System** - Delphi-style components, events, bindings
- **GPU Gaussian Rendering** - 120 FPS via gsplat-mps
- **Gaussian Training Facets** - OpenSplat integration
- **Admin Dashboard** - admin.noodlings.ai (live)
- **Crash Recovery** - Sentinel file detection
- **Cloud Account System** - OAuth, credits, billing
- **Assets Panel** - Unity-style filesystem browser
- **Stage View** - Hierarchy with folders, drag-drop
- **Bone Visualization** - Click-to-select, bidirectional sync
- **Unified Authentication** - token_auth for MUSH integration
- Multi-provider LLM (8 providers)
- Neural Canvas with PyTorch test mode
- Scriptability API (context.noodle)
- MCP integration
- Utility facets (31 types)
- Multimodal facets (audio, vision, image gen)

---

## OpenSplat Training (Reference)

```bash
# Build with Metal
cd external/OpenSplat && mkdir build && cd build
cmake -DGPU_RUNTIME=MPS -DCMAKE_BUILD_TYPE=Release .. && make -j16

# Train
./opensplat /path/to/dataset -o output.ply -n 30000 --sh-degree 2

# Convert to .radiance
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance input.ply -o output.radiance -v
```

**Tips:** Black backgrounds best. 72 views recommended. 5K iterations for testing, 30K for production.

---

## External Tools

- `external/OpenSplat/` - Gaussian training (Metal GPU)
- `external/ml-sharp/` - Apple SHARP
- `/Users/thistlequell/git/gsplat-mps/` - GPU renderer (AGPLv3)

---

**Ordnung muss sein!**
