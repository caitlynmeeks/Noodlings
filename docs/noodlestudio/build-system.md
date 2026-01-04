# Build System Architecture

**Status**: Phase 4 Implemented
**Last Updated**: January 3, 2026
**Authors**: Caitlyn + Claude

---

## Claude's Recommendations Summary

Copying the Unity experience means:

| Question | Recommendation | Unity Parallel |
|----------|----------------|----------------|
| **Q1: Runtime UI** | Full 3D viewport with Gaussians | Unity Player IS the viewport |
| **Q2: Server** | No server for standalone; embedded for multiplayer | Unity has no server for single-player |
| **Q3: Targets** | macOS .app first, then Windows/Linux/Docker | Build Settings platform selector |
| **Q4: Packaging** | Bundle Python runtime, download LLMs on demand | Unity bundles Mono, downloads assets |
| **Q5: Config** | Simple `build.yaml` with `main_stage` | Scenes in Build Settings |

**The core insight**: A built NoodleStudio app should feel like launching a Unity game - a window opens showing your scene with characters ready to interact. Not a terminal, not a web browser, not a server you connect to separately.

---

## Overview

NoodleStudio needs a "File > Build Project" feature analogous to Unity's build system. This creates standalone applications from NoodleStudio projects that can run without the editor.

### Goals

1. **Shared Core**: Runtime uses identical core modules as the editor - changes propagate automatically
2. **Zero Editor Dependency**: Built apps don't include panels, inspectors, or editing tools
3. **Multiple Targets**: Support different deployment scenarios (desktop app, server, web)
4. **Simple Packaging**: One-click build that "just works"

### Non-Goals (for v1)

- Cross-platform builds from single machine (build on Mac for Mac, etc.)
- App Store signing/notarization (manual step)
- Incremental builds / asset caching
- Build profiles (debug/release) - everything is release

---

## Naming

We avoid "player" (sounds passive). The runtime/output is referred to as:

- **In code**: `noodleapp` (module name)
- **In UI**: "Build Application" / "your application"
- **In docs**: "standalone application" or "built application"
- **Output**: `MyProject.app` (macOS), `MyProject.exe` (Windows)

---

## Open Questions

### Q1: What's the Runtime UI?

The built application needs some way to interact. Options:

| Option | Pros | Cons |
|--------|------|------|
| **A. Headless (API only)** | Simplest, good for servers | No local interaction |
| **B. Minimal chat window** | Familiar, like cmush web | Limited to text |
| **C. Full 3D viewport** | Shows Gaussians, immersive | Heavy, complex |
| **D. Web-based (localhost)** | Reuse cmush web client | Requires browser |
| **E. Hybrid: headless + optional window** | Flexible | More code paths |

**Current thinking**: Start with **E (Hybrid)**. Headless by default with `--gui` flag for optional window. The window could be a simplified version of the Chat panel.

**Claude's Recommendation**: **C (Full 3D viewport)** as the default, with headless as a build option.

Unity's player IS the 3D viewport - that's the whole point. The game runs in a window showing the scene. For NoodleStudio, the equivalent would be:
- A window showing the Stage (Gaussian-rendered scene)
- Noodlings visible as their radiance avatars
- Chat/interaction overlaid or in a panel
- This IS what you're building toward - cognitive characters in visual worlds

The headless option is for server deployments, but the PRIMARY build target should be visual.

**Decision**: **C (Full 3D viewport)** as default, with headless as a build option.

---

### Q2: Where Does the MUSH Server Live?

The noodleMUSH server currently runs separately. For built apps:

| Option | Pros | Cons |
|--------|------|------|
| **A. Embedded in app** | Single process, simple deploy | Larger binary, always running |
| **B. Separate process, auto-launched** | Modular, can restart | Process management complexity |
| **C. Optional component** | Flexibility | User confusion |
| **D. No server, direct execution** | Simplest for single-noodling apps | Loses multi-user, web client |

**Current thinking**: **B** for desktop apps (launch server subprocess), **A** for containerized/cloud deployments.

**Claude's Recommendation**: **D (No server, direct execution)** for single-player apps, **A (Embedded)** for multi-user.

Unity doesn't have a separate server process for single-player games - the game loop runs directly. The MUSH server is really only needed when:
- Multiple users connect (web clients, multiplayer)
- You want the web client interface

For a built app where one person runs it locally and interacts via the window, the facet executor can run directly without WebSocket overhead. The "server" architecture is an editor/multiplayer concern.

Build options:
- **Standalone App**: Direct execution, no server (default)
- **Multiplayer/Web**: Embedded server + web client files

**Decision**: **D (No server, direct execution)** for standalone; **A (Embedded)** for multiplayer builds.

---

### Q3: Build Targets?

What platforms/formats should we support?

| Target | Priority | Notes |
|--------|----------|-------|
| **macOS .app bundle** | P0 | Primary dev platform |
| **macOS CLI** | P0 | Headless server deployment |
| **Windows .exe** | P1 | Large user base |
| **Linux binary** | P1 | Server deployments |
| **Docker container** | P1 | Cloud deployment |
| **Web (Electron)** | P2 | Desktop web hybrid |
| **Web (WASM)** | P3 | True browser, major effort |

**Current thinking**: Start with macOS .app and CLI. Add others incrementally.

**Claude's Recommendation**: Exactly this. macOS .app first (your dev machine), CLI for servers.

Unity's approach:
- Build Settings window with platform selector
- Switch platforms explicitly
- Build button creates output for selected platform

We should mirror this. The build dialog shows target options, user picks one, we build for that.

**Decision**: **macOS .app first**, then Windows/Linux/Docker incrementally.

---

### Q4: What Gets Packaged?

A built application needs:

#### Always Included
- Python runtime (or expect system Python?)
- Core modules (facet_system, facet_executor, charm_network, etc.)
- LLM provider code
- Project assets:
  - Noodlings (recipe.yaml, assembly.yaml)
  - Stages (stage.yaml, hierarchy.yaml)
  - Radiances (.radiance files)
  - Scripts (.py, .js)

#### Optionally Included
- Gaussian renderer (only if app uses 3D)
- Audio/TTS/STT modules (only if app uses voice)
- MCP server configs

#### Never Included
- Editor panels
- Inspector code
- Asset editing tools
- Debug visualizations
- Test files

#### Open Sub-Questions
- **Python bundling**: Use PyInstaller? py2app? Expect system Python?
- **Model weights**: Bundle MLX models or download on first run?
- **API keys**: How to handle? Environment variables? Config file?

**Claude's Recommendation**:

**Python bundling**: Use **py2app** for macOS (native, well-supported), **PyInstaller** for cross-platform. Unity bundles the Mono/.NET runtime - we bundle Python. The user shouldn't need Python installed.

**Model weights**:
- **CharmNetwork** (~54K params): Always bundle - it's tiny
- **Local LLMs** (Ollama): Don't bundle - expect Ollama installed, or use cloud APIs
- **MLX models**: Download on first run with progress dialog, like Unity's asset import

**API keys**:
- Cloud providers (Anthropic, OpenAI): Environment variables or first-run setup dialog
- Local (Ollama): No keys needed
- Unity equivalent: PlayerPrefs / config file in user data directory

The built app should have a first-run experience: "Enter your Anthropic API key to enable cloud AI" with option to skip (use local Ollama only).

**Decision**: **py2app** for macOS, **PyInstaller** for cross-platform. Bundle Python runtime. CharmNetwork always included. MLX models download on demand.

---

### Q4b: LLM Provider Configuration

Built apps need LLM access. Three options available to end users:

| Provider | Description | Setup |
|----------|-------------|-------|
| `noodlings` | Our cloud routing | Sign in with Noodlings account |
| `ollama` / `lmstudio` | Local inference | Free, requires local install |
| Own keys | Direct to Anthropic/OpenAI | User provides API keys |

The build creator specifies the default/recommended provider in `build.yaml`. End user can override at runtime.

**First-run dialog**: Built apps show a provider selection dialog on first launch. See `docs/noodlestudio/llm-routing-service.md` for mockup.

---

### Q5: Project Structure for Builds

What does the user's project need to specify for building?

```yaml
# project.yaml (or build.yaml?)
name: "My Noodling App"
version: "1.0.0"
icon: "Assets/icon.png"  # Optional

# Entry point - what runs when app starts
entry:
  type: stage  # or 'noodling' for single-character app
  ref: "Stages/main_stage"

# Or for single noodling:
# entry:
#   type: noodling
#   ref: "Noodlings/red"

# Build settings
build:
  include_renderer: true  # Include Gaussian 3D renderer
  include_voice: false    # Include TTS/STT
  headless_default: false # Start in headless mode by default

# Server settings (if using MUSH)
server:
  enabled: true
  port: 8765
  web_client: true  # Include web client files
```

**Claude's Recommendation**: Keep it simple, Unity-style.

Unity has a single "scene to load" concept. We should mirror that:

```yaml
# build.yaml (in project root)
name: "Red's World"
version: "1.0.0"
icon: "Assets/app_icon.png"

# What loads on startup
main_stage: "Stages/the_nexus"

# Build options
settings:
  include_renderer: auto  # auto-detect from stage contents
  window_size: [1280, 720]
  fullscreen: false
  resizable: true

# LLM provider configuration
llm:
  default_provider: noodlings  # noodlings, ollama, anthropic, openai
  allow_local: true            # Show "Local AI" option in first-run
  allow_own_keys: true         # Show "Own API Keys" option
  # If noodlings is default, app uses api.noodlings.ai/v1/chat/completions
  # User signs in with Noodlings account, we bill their credits
```

That's it. Simple. The stage contains everything - zones, noodlings, props. The stage IS the scene. No need to specify individual noodlings.

If someone wants a "single noodling chat app" (no 3D), that's just a stage with one noodling and no radiances - the renderer auto-excludes itself.

**Decision**: Simple `build.yaml` with `main_stage` reference. Stage contains everything.

---

## Architecture

### Module Organization

```
applications/
├── noodlestudio/
│   ├── noodlestudio/
│   │   ├── core/              # SHARED between editor and runtime
│   │   │   ├── facet_system.py
│   │   │   ├── facet_executor.py
│   │   │   ├── charm_network_facet.py
│   │   │   ├── radiance_component.py
│   │   │   ├── gaussian_renderer.py
│   │   │   ├── provider_manager.py
│   │   │   ├── mcp_manager.py
│   │   │   └── ...
│   │   ├── panels/            # EDITOR ONLY - not packaged
│   │   ├── dialogs/           # EDITOR ONLY - not packaged
│   │   ├── scripting/         # SHARED - noodle_api, etc.
│   │   └── runtime/           # NEW - standalone runtime code
│   │       ├── __init__.py
│   │       ├── app.py         # Main application class
│   │       ├── app_window.py  # Optional GUI window
│   │       └── cli.py         # Command-line interface
│   │
│   ├── main.py                # Editor entry point
│   ├── player.py              # Existing headless player (refactor into runtime/)
│   │
│   └── appbuilder/            # NEW - build system (named to avoid .gitignore)
│       ├── __init__.py
│       ├── builder.py         # Main build orchestration
│       ├── packager.py        # Asset packaging
│       ├── bundler_macos.py   # macOS .app creation
│       ├── bundler_windows.py # Windows .exe creation
│       └── templates/         # App templates, icons, etc.
```

### Shared Core Principle

The `core/` directory is the single source of truth. Both editor and runtime import from it:

```python
# In editor (main_window.py)
from noodlestudio.core.facet_executor import FacetExecutor

# In runtime (app.py)
from noodlestudio.core.facet_executor import FacetExecutor  # Same import!
```

This means:
- Bug fixes in core automatically apply to both
- No code duplication
- Tests cover both use cases

### What Existing Code Can We Reuse?

| File | Reuse | Notes |
|------|-------|-------|
| `player.py` | **Yes** | Headless facet execution - refactor into runtime/ |
| `facet_executor.py` | **Yes** | Core execution engine |
| `facet_system.py` | **Yes** | Data model |
| `charm_network_facet.py` | **Yes** | Affect model |
| `provider_manager.py` | **Yes** | LLM providers |
| `mcp_manager.py` | **Yes** | MCP integration |
| `gaussian_renderer.py` | **Yes** | 3D rendering (optional) |
| `radiance_component.py` | **Yes** | Gaussian assets |
| `scripting/noodle_api.py` | **Yes** | Scripting context |
| `cmush/server.py` | **Partial** | May need to extract core server logic |

---

## Implementation Phases

### Phase 0: Preparation (This Document) - COMPLETE
- [x] Answer open questions above
- [x] Get alignment on architecture
- [x] Identify any core/ refactoring needed first

### Phase 1: Runtime Foundation - COMPLETE (Jan 3, 2026)
- [x] Create `noodlestudio/runtime/` module
- [x] Refactor `player.py` into `runtime/app.py`
- [x] Implement CLI interface (`python -m noodlestudio.runtime`)
- [x] Test: Can run a project headless from command line

### Phase 2: LLM Routing API - COMPLETE (Jan 3, 2026)
- [x] `/v1/chat/completions` endpoint at api.noodlings.ai
- [x] Direct Anthropic routing (no OpenRouter middleman)
- [x] Token counting + credit billing
- [x] `noodlings` provider in runtime

### Phase 3: UI Canvas System - COMPLETE (Jan 3, 2026)
- [x] Create `noodlestudio/runtime/ui/` module
- [x] Delphi-style component system
- [x] Components: Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport
- [x] Event system with `send_to_noodling`, `call_script`, bindings
- [x] Add `--gui` flag to CLI
- [x] Test: Can interact with noodling via window

### Phase 4: Build System - COMPLETE (Jan 3, 2026)
- [x] Create `appbuilder/builder.py` orchestration
- [x] Create `appbuilder/packager.py` for asset collection/filtering
- [x] Create `appbuilder/bundler_macos.py` for .app bundles
- [x] Add "File > Build Application..." menu item (Ctrl+B)
- [x] Add build progress dialog
- [x] Icon customization (PNG to ICNS conversion)
- [x] Error handling and user feedback
- [x] Test: Can build and run a simple project

### Phase 5: Python Bundling (py2app) - FUTURE
**Priority: High** - Required for true standalone apps.

Currently, built .apps use a shell launcher that calls system Python. Users need Python 3.10+ with PyQt6 installed. For true "double-click and run" experience:

- [ ] Create setup.py for py2app configuration
- [ ] Bundle Python runtime into .app
- [ ] Handle PyQt6 plugins (tricky - Qt has many hidden dependencies)
- [ ] Handle numpy, yaml, and other deps
- [ ] Test on clean Mac without Python installed
- [ ] Add checkbox in build dialog: "Bundle Python runtime"

**Tools:**
- **py2app** - Native macOS bundler, recommended
- **PyInstaller** - Cross-platform alternative

### Phase 6: Additional Targets - FUTURE
- [ ] Windows .exe bundler (PyInstaller)
- [ ] Linux binary
- [ ] Docker container generation

### Phase 7: UI Canvas Designer - FUTURE
- [ ] Visual designer panel in editor
- [ ] Drag-drop component palette
- [ ] Property editor for components

---

## UI Canvas Integration

Built applications can use the **UI Canvas** system for 2D interfaces.

See: `docs/noodlestudio/ui-canvas.md`

### Build Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **3D Only** | Full RadianceViewport | Immersive 3D experience |
| **2D Only** | UI Canvas, no 3D | Chat apps, dashboards |
| **Hybrid** | UI Canvas with embedded RadianceViewport | Chat + 3D sidebar |

### build.yaml for UI Apps

```yaml
name: "Red's Chat"
version: "1.0.0"

# UI-based app
ui: "ui.yaml"

# Stage for any RadianceViewport components
main_stage: "Stages/reds_room"
```

---

## Related Work

- **Unity Build System**: Our primary inspiration. File > Build Settings, one-click builds.
- **Borland Delphi**: Inspiration for UI Canvas designer.
- **Electron**: How web apps become desktop apps. May use for web target.
- **PyInstaller / py2app**: Python packaging tools we'll likely use.
- **Existing player.py**: Our starting point for headless execution.

---

## Notes for Next Session

When we return to implement this:

1. **First**: Decide on Q1-Q5 above
2. **Then**: Check if any core/ refactoring is needed (e.g., removing editor dependencies)
3. **Start with**: Phase 1 - get `python -m noodlestudio.runtime path/to/project` working
4. **Validate**: Run the toy_claude_code.yaml assembly headless as a test case

---

## Current Limitations (v1)

Built applications currently have these requirements:

1. **Python 3.10+ required** - User must have Python installed
2. **PyQt6 required** - User must have `pip install PyQt6`
3. **macOS only** - Windows/Linux bundlers not yet implemented
4. **No bundled LLMs** - Requires Ollama or cloud API keys

These will be addressed in Phase 5 (py2app bundling).

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-03 | Phase 4 implemented: builder.py, packager.py, bundler_macos.py |
| 2026-01-03 | Added Phase 5 (py2app) as high priority future work |
| 2026-01-03 | Initial planning document |
