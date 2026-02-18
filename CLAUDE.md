# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: February 18, 2026
**Machine**: jiji (migrated from caledonia M3 Ultra, Feb 2026)

---

## DEVELOPMENT DISCIPLINE (READ THIS FIRST)

**Canonical reference:** `docs/development/discipline.md` — test tiers, pipeline testing, signal wiring rules, fixture hygiene, handoff verification protocol, manual smoke walk checklist. Read it. Follow it.

NoodleStudio is production-grade software on par with Blender or Unity. Caity is a former early Unity employee who built the Asset Store. This is not a hobby project. Every commit must leave the application healthier than you found it.

### The February 2026 Lesson

During Phase 2/3 development (ensemble dynamics), Hero Claude built a beautiful performance demo while the project system, server pipeline, inspector, and stage system rotted underneath — because those systems had **0% integration test coverage**. One wrong directory path (`..` x4 instead of x3) silently broke the entire server cascade, and nobody caught it because no test verified the path resolved correctly.

**This must never happen again.**

### Rules of Engagement

1. **Smoke tests before every commit.** Run `tests/test_smoke.py` before committing. If it fails, the commit is blocked. No exceptions.

2. **Every bug fix gets a test FIRST.** Write the failing test, then fix the code. The test proves the bug existed and proves it's fixed. If you can't write a test for it, you don't understand it well enough to fix it.

3. **Feature work must not break existing systems.** When building something new, run the full test suite regularly. If you touch a file, verify the systems that depend on it still work. If you're adding a new facet feature, make sure the inspector still loads. If you're changing the performance window, make sure the stage system still functions.

4. **Test the trunk, not just the leaves.** Unit tests for individual components are necessary but insufficient. Integration tests for the CONNECTION POINTS between systems are what catch real failures:
   - Server path resolution → server startup → WebSocket connection
   - Project open → stage discovery → hierarchy population
   - Facet selection → inspector binding → property display
   - Stage instance → assembly loading → performer creation

5. **No monkeypatches. No workarounds.** If something is broken, fix the root cause. Do not work around it. Do not add `getattr` guards. Do not suppress exceptions. Do not redirect to `/dev/null`.

6. **Understand before you touch.** Read the code you're about to modify. Understand the data flow. Know what depends on it. If a mixin initializes attributes, make sure the class that uses it actually calls the initializer.

7. **Path resolution must be tested.** Any code that constructs paths with `os.path.join` and `..` traversals MUST have a test that verifies the path resolves to a real directory. This is the single most common silent failure in our codebase.

8. **Signal connections must be tested.** When wiring Qt signals between panels (hierarchy → inspector, facets editor → live viz), write a test that emits the signal and verifies the receiver responds. Signal disconnections are invisible until a user clicks something.

### Smoke Test Suite

```bash
cd applications/noodlestudio
python -m pytest tests/test_smoke.py -v
```

The smoke test suite covers: server path resolution, inspector initialization, project structure, stage instances, assembly loading, dropdown states. Run it before EVERY commit.

### Full Regression

```bash
cd applications/noodlestudio
python -m pytest tests/ -v
```

Run after any significant change. Aim for full green before pushing.

---

## HANDOFFS FROM DISCUSS CLAUDE (Check These First!)

Handoff documents from the discuss session live in:
`/Users/caitlyn/git/claudechat_dev/projects/handoffs/`

**Current Priority Handoffs:**

| Date | File | Summary |
|------|------|---------|
| Feb 17 | `phase-d-new-characters-2026-02-17.md` | **ACTIVE** - Phase D: New characters, wiring fixes, test hardening, docs. D.1-D.2 done, D.3 done, D.4 in progress. |
| Feb 17 | `phase-d1.5-ensemble-awareness-2026-02-17.md` | **DONE** - Ensemble awareness, CharmNetworkEMA, speaker spotlight, performance inspector. 7 commits. |
| Feb 15 | `launch-ux-and-hierarchy-decoupling-2026-02-15.md` | **DONE** - Hierarchy decoupling, console lazy connect, self-contained templates. |
| Feb 15 | `infrastructure-repair-sprint-complete-2026-02-15.md` | **DONE** - All 11 infrastructure repair commits landed. 1438 tests, 0 failures. |
| Jan 14 | `visual-verification-spec.md` | Baseline comparison, SSIM diff, `assert_visual` action |

**Read these when starting a session** - they contain context from Caity's discuss sessions.

---

## RECENT: Phase D — New Characters + Ensemble Awareness (Feb 16-18, 2026)

### D.1: Three-Noodling Ensemble

Three characters perform together on a shared stage: **Ajo** (guide), **Krampus** (seven-year-old Alpine Krampus kid), **Juanita** (explorer from Lanzarote). Each has their own assembly (INCOMING → Response LLM + Mood Reader → Performance → OUTGOING), recipe, and VRM avatar.

**Turn-taking:** User → Ajo → Krampus → Juanita → wait (round-robin from `_turn_queue`).

**Ensemble window:** 3-slot layout (each VRM viewport 433px wide).

### D.1.5: Ensemble Awareness

Noodlings now perceive each other. Each sees the others' appearance and mood, shares conversation history, and knows their stage context.

**CharmNetworkEMA** (`runtime/charm_network_ema.py`): 3-timescale affect smoothing:
- Fast (alpha=0.7): moment-to-moment reactions
- Medium (alpha=0.15): conversational mood
- Slow (alpha=0.03): character baseline drift

**Key additions:**
- Perception context injected into LLM prompts (appearance, mood of others)
- Speaker spotlight: active VRM full brightness, others dimmed
- Performance facet inspector: typing speed + speaking intensity spinboxes
- Charm network depth view in unified editor

**Key files:**
| File | Purpose |
|------|---------|
| `runtime/charm_network_ema.py` | CharmNetworkEMA class |
| `tests/test_ensemble_awareness.py` | 35 tests |
| `tests/test_charm_network_ema.py` | 30 tests |

### D.2: Wiring Fixes

- `noodlingSelected` signal wired from performance window to panels (D.2a)
- `--ensemble` CLI uses stage instances instead of hardcoded paths (D.2b)
- File > Close Project menu action (D.2c)

---

## RECENT: Phase C — Unified Editor (Feb 15-16, 2026)

The old monolithic FacetsEditorPanel (7 files, ~4K lines) was replaced by a **depth-stack based unified editor** with breadcrumb navigation and plugin-based depth views.

**Architecture:**
```
UnifiedEditorPanel (stack shell)
  └─ AssemblyEditorView (root: QGraphicsView, 8 mixins)
       └─ push_view() → NeuralCanvasDepthView / CharmNetworkDepthView / ...
```

**Depth navigation:** Double-click a container facet → pushes a depth view onto the stack. Backspace or breadcrumb click → pops back. Each view implements `DepthProtocol` (load_data, save_data, get_breadcrumb_label).

**Plugin registry:** New container facet types register view classes without modifying core:
```python
UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", NeuralCanvasDepthView)
UnifiedEditorPanel.register_depth_view("CharmNetworkEMA", CharmNetworkDepthView)
```

**Execution visualization:** Node pulsing, wire packet animations, sound effects. Events delivered synchronously on Qt main thread from GuidePerformanceManager.

**Key files:**
| File | Purpose |
|------|---------|
| `panels/editors/unified_editor_panel.py` | Stack shell, breadcrumb, signal forwarding |
| `panels/editors/assembly_editor_view.py` | Root visual editor (8 mixins) |
| `panels/editors/depth_protocol.py` | Interface for stackable views |
| `panels/editors/assembly_execution_mixin.py` | Execution viz (pulsing, packets, sound) |
| `panels/editors/assembly_ensemble_mixin.py` | Noodling selector for ensemble filtering |
| `panels/editors/neural_canvas_depth_view.py` | NC adapter |
| `panels/editors/charm_network_depth_view.py` | CharmNetworkEMA depth view |

**Deleted (C.8):** `facets_editor_panel.py` and 6 related files — fully replaced.

---

## RECENT: Unity Plugin Export (Jan 12, 2026)

Export noodlings to Unity-compatible `.noodling` packages for Christina's ToMars? VR project.

**Menu:** File > Export > Export to Unity Package...

**Package format:**
```
aria.noodling/
├── manifest.json       # Package metadata
├── character.json      # Personality, motivation, initial PAD
├── assembly.json       # Facet configuration
├── expressions.json    # PAD → FACS → VRM blendshape mapping
└── plays/              # Optional narrative beats
```

**Key files:**
| File | Purpose |
|------|---------|
| `core/noodling_package_exporter.py` | Main exporter class |
| `core/main_window_project_mixin.py` | Menu action (`export_unity_package`) |
| `core/main_window_menus_mixin.py` | Menu item |

**Affect mapping:** Internal 5D (valence, arousal, dominance, boredom, sorrow) exports to Unity 3D PAD (pleasure, arousal, dominance). `valence` → `pleasure`.

**Tests:** `tests/test_noodling_package_exporter.py` (19 tests)

**Full spec:** `/docs/noodlestudio/unity-plugin.md`

---

## RECENT: Play Format Runtime Integration (Jan 12, 2026)

Brenda's Play Format now integrates with the runtime for guided performances.

**Architecture:**
```
.play.yaml -> BrendaDirector -> #directors.cues -> GuideCueHandler -> LLM Prompt
```

**Key changes:**
- `NoodleApp.load_director()` creates `GuideCueHandler` automatically
- Event dispatcher injects `brenda_direction` into assembly context
- Facet executor appends direction to LLM system prompts

**Usage:**
```python
app = NoodleApp()
app.load_director("docs/noodlestudio/plays/lets_consciousness_intro.play.yaml")
app.start_performance()

# Direction is automatically injected when running assemblies
result = await app.run("Hello!")

# Report response back to advance the play
app.report_actor_response(result['response'], "Hello!")
```

**Key files:**
| File | Purpose |
|------|---------|
| `runtime/app.py` | NoodleApp integration |
| `runtime/brenda.py` | Stage director |
| `runtime/guide_cue_handler.py` | Cue reception, prompt building |
| `runtime/ui/event_dispatcher.py` | Direction context injection |
| `core/facet_executor.py` | LLM prompt injection |

**Tests:** `tests/test_play_integration.py`, `tests/test_brenda.py`, `tests/test_guide_cues.py`

---

## RECENT: Dashboard UI Widgets (Jan 10-11, 2026)

New dashboard-style visualization components for building science-fair AI dashboards:

| Component | File | Description |
|-----------|------|-------------|
| `LED` | `runtime/ui/components/led.py` | On/off indicator with glow, blink, label |
| `Gauge` | `runtime/ui/components/gauge.py` | Analog dial with zones, ticks, needle |
| `SevenSegment` | `runtime/ui/components/seven_segment.py` | LCD-style numeric display |
| `LevelMeter` | `runtime/ui/components/level_meter.py` | Vertical/horizontal bar meter |
| `QMLWidget` | `runtime/ui/components/qml_widget.py` | Wrapper for Qt QML widgets |

**Full specs:** `/docs/noodlestudio/qml-widget-wrapper.md`

---

## RECENT: FacetAssembly UI Canvas Integration (Jan 10, 2026)

FacetAssembly is now a UI Canvas component for visual assembly integration.

**Usage in ui.yaml:**
```yaml
- type: FacetAssembly
  name: sentiment_analyzer
  properties:
    assembly: assemblies/sentiment.yaml
    auto_run: false
  input_bindings:
    - pad: text
      source: text_input.value
```

**Event binding with target:**
```yaml
- type: Button
  events:
    onClick:
      action: run_assembly
      target: sentiment_analyzer  # References FacetAssembly by name
```

**Key files:** `runtime/ui/components/facet_assembly.py`, `dialogs/assembly_picker_dialog.py`

---

## RECENT: Build System (Jan 10, 2026)

**File > Build Settings... (Ctrl+Shift+B)** creates standalone macOS .app bundles.

| Feature | File |
|---------|------|
| Build dialog | `dialogs/build_settings_dialog.py` |
| BuildConfig | `core/build_config.py` |
| Splash screen | `widgets/splash_screen.py` |
| Editor access control | `dialogs/editor_password_dialog.py` |
| macOS bundler | `appbuilder/bundler_macos.py` |

**Full spec:** `/docs/noodlestudio/build-settings.md`

---

## RECENT: Let's Consciousness! Project (Jan 8-9, 2026)

Guide character (Ajo Majo) talks via NoodleROUTER.

**Running:**
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python -m noodlestudio.runtime \
  --gui --ui "../../Projects/lets-consciousness/ui.yaml" \
  --provider noodlerouter --api-key $NOODLEROUTER_API_KEY
```

**Key systems built:**
- `CharacterOverlayWindow` (`runtime/ui/overlay.py`) - Transparent VRM overlay
- `ChannelBus` (`runtime/channels.py`) - Inter-noodling pub/sub messaging
- `WorldChannelService` (`runtime/world_channels.py`) - Time/weather/ambiance broadcasts
- `BrendaDirector` (`runtime/brenda.py`) - Stage director for .play.yaml scripts

**Full specs:** `/docs/noodlestudio/channels.md`, `/docs/noodlestudio/handoff-world-channels.md`, `/docs/noodlestudio/handoff-brenda.md`

---

---

## RECENT: Computer Use QA System (Jan 11, 2026)

NoodleCode now has `run_ui_test` and `ai_verify_ui` tools for automated UI testing.

**Usage:**
```python
# Run smoke tests
run_ui_test(suite="smoke")

# Run specific test
run_ui_test(test="smoke/panels.ui-test.yaml")

# AI visual verification
ai_verify_ui(verify="Inspector shows 3 properties")
```

**Test files:** `noodlestudio/tests/ui/`
**Full spec:** `/docs/noodlestudio/computer-use-qa.md`

---

## BACKLOG (Post-Core Stability)

These are deferred until the core infrastructure is solid and tested:

- Inspector UX: Unity-style numeric drag-to-scroll on labels
- Undo/Redo for UI Edits
- Protected default asset pattern (read-only assemblies)
- LLM streaming through assembly
- Voice sounds (Animal Crossing style)
- RAG-backed lore books per character
- CharmNetworkEMA → VRM blend shape pipeline (EMA smoothing done; VRM wiring pending)

---

## Core Architecture

### Project → Stage → Instance Pipeline (CRITICAL)

This is the backbone. If it breaks, everything breaks.

```
Project (project.noodleproj)
  └─ Stages/{name}/
       ├─ stage.yaml (metadata, geometry, spawn point)
       ├─ Zones/*.zone.yaml (spatial regions)
       ├─ Instances/{name}/ (noodling placements)
       │    ├─ instance.yaml (noodling: relative path to template, overrides)
       │    └─ state.json (runtime affect, memories)
       └─ Props/{name}/ (world objects)

Noodling Templates (Noodlings/{name}/)
  ├─ noodling.yaml (manifest)
  ├─ recipe.yaml (personality, affect baseline, LLM config)
  └─ assembly.yaml (facet topology: INCOMING → facets → OUTGOING)
```

**Instance is NOT a copy.** It's a reference + overrides. Assembly and recipe load from the template at runtime.

**Stage hierarchy tree** shows the contents of the current stage (zones, noodling instances, props). Selection emits `entitySelected` signal → routes to inspector, console, and facets editor.

### Server Pipeline (CRITICAL)

```
NoodleStudio (Qt app)
  ├─ main_window_server_mixin.py → launches start.sh
  │    └─ _cmush_dir() resolves to applications/cmush/
  │
  └─ start.sh (applications/cmush/)
       ├─ HTTP server :8080 (serves web client from cmush/web/)
       └─ WebSocket server :8765 (MUSH commands, chat, events)

Text View (QWebEngineView) → http://localhost:8080
Console (WebSocket) → ws://localhost:8765
```

**If `_cmush_dir()` resolves wrong, NOTHING works.** Test it.

### Auth Pipeline

```
Sign In → LoginDialog → OAuth (Google/GitHub/Apple)
  → Cloudflare Worker (noodlings-api.caitsters.workers.dev)
  → Token stored locally (~/.noodlings/session.json)
  → Enter World → AvatarPicker → web view loads MUSH client with token
```

Backend is Cloudflare Workers + D1 database + KV sessions. Production URL: `https://noodlings-api.caitsters.workers.dev`

### Affect Model: PAD + Boredom + Sorrow
5-dimensional continuous affect (NO discrete emotion labels):
- `valence` (-1 to +1), `arousal` (0 to 1), `dominance` (0 to 1)
- `boredom` (0 to 1), `sorrow` (0 to 1)

### Facet System
Visual node graphs defining how Noodlings think:
```
INCOMING -> CHARM_NET -> CONTEXT_INTELLIGENCE -> Cognitive facets -> OUTGOING
```
Edited via the **Unified Editor** (depth-stack panel with breadcrumb navigation). Container facets (Neural Canvas, CharmNetworkEMA) open as depth views.

### Ensemble Performance System

```
GuidePerformanceManager
  ├─ NoodlingPerformer (one per noodling, owns assembly + executor + CharmNetworkEMA)
  ├─ GuidePerformanceWindow (pure renderer: 3-slot VRM viewports + dialogue)
  ├─ Turn-taking: User → Ajo → Krampus → Juanita → wait (round-robin)
  └─ Ensemble awareness: shared history, perception context, speaker spotlight

Noodlings MUST be stage instances (Stages/{stage}/Instances/).
Performance manager loads from stage, NOT hardcoded paths.
--ensemble CLI flag starts ensemble from the current stage's instances.
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

# Smoke tests (run before EVERY commit)
python -m pytest tests/test_smoke.py -v

# Full suite (~1881 tests)
python -m pytest tests/ -v

# By pattern
python -m pytest tests/ -k "ensemble" -v
```

**Before committing:** Smoke tests MUST pass. Full suite SHOULD pass. If a full-suite test fails and it's unrelated to your change, investigate — don't ignore it.

---

## Development

### Environment Setup
```bash
cd /Users/caitlyn/git/noodlings_clean
source venv/bin/activate
```

### Running NoodleStudio
```bash
cd applications/noodlestudio
./launch_with_log.sh
```

### Launching Guide Performance Window (VRM Avatar)
**IMPORTANT:** There is no UI button to open the Guide Performance Window. Claude Code must launch it via the `--play` CLI flag:
```bash
cd applications/noodlestudio
../../venv/bin/python3 -m noodlestudio.main --play "Ajo Alive" --no-splash
```
This opens the editor, then after 5 seconds auto-opens the floating Guide Performance Window with the VRM avatar (auto-discovers `noodlings/guide/Radiances/AjoMajo.vrm`). The avatar renders with MToon shading, GPU skeletal skinning, procedural idle muscles, and blend shape morph targets.

To also run a directed play script:
```bash
../../venv/bin/python3 -m noodlestudio.main --play "docs/noodlestudio/plays/lets_consciousness_intro.play.yaml" --no-splash
```

### Running noodleMUSH Server
```bash
cd applications/cmush && ./start.sh
```

### Debugging
```bash
tail -f applications/cmush/logs/server_*.log
tail -f applications/noodlestudio/logs/noodlestudio_*.log
```

### API Keys & Configuration

NoodleStudio stores settings in macOS preferences (plist files):

| Setting | Location |
|---------|----------|
| **Provider configs (API keys)** | `~/Library/Preferences/com.noodlings.ProviderManager.plist` |
| **Model label assignments** | `~/Library/Preferences/com.noodlings.ModelLabelManager.plist` |
| **General settings** | `~/.noodlestudio/settings.json` |
| **NoodleCode history** | `~/.noodlestudio/noodlecode_history.json` |
| **Window layouts** | `~/.noodlestudio/layouts/` |

**Reading/modifying plist from CLI:**
```bash
# Read all provider settings
defaults read com.noodlings.ProviderManager

# Read model assignments
defaults read com.noodlings.ModelLabelManager

# Check if Anthropic key is configured
defaults read com.noodlings.ProviderManager "providers.anthropic.api_key"
```

**NoodleCode model assignment:** "Noodle Code" label should be assigned to an Anthropic model in Model Manager (Settings > Models).

**CLI execution:** Use `--execute` flag to run NoodleCode commands on startup:
```bash
./launch_with_log.sh --execute "run the smoke tests"
```

---

## Testing Policy (CRITICAL — Read This Before Writing Any Test)

**NO MOCKS. NO STUBS. NO `__new__` BYPASS.**

This is not a style preference. This was validated on Feb 9 2026 when a MagicMock in `test_performance_facet.py` hid a real production crash (`AffectAPI.to_dict()` missing). The test passed. Production crashed with the exact same error. The mock concealed the bug.

**Rules:**

1. **If something crashes in test setup, FIX THE REAL BUG.** Do not mock around it. The crash is telling you something is broken — that's the thing to fix.

2. **No `__new__` bypass pattern.** Creating objects via `cls.__new__(cls)` without calling `__init__` produces half-initialized objects that need `getattr(self, '_foo', None)` guards in production code. This is test scaffolding leaking into production. Instead: call real `__init__` with dependency-injected fakes (lightweight real objects, not mocks).

3. **No MagicMock / unittest.mock.patch for core objects.** MagicMock silently returns truthy values for any attribute access — it cannot detect missing methods, wrong signatures, or broken APIs. If you need a test double, write a minimal concrete class that implements the interface.

4. **No `getattr` guards in production code to compensate for test fixtures.** If production code needs `getattr(self, '_foo', None)` because tests create objects without `__init__`, the tests are wrong, not the production code. Remove the guards and fix the tests.

5. **Acceptable:** Dependency injection of lightweight real objects. `FakeLLMClient` that returns canned responses. `InMemoryChannelBus`. Real `__init__` with test-appropriate arguments. These test the real code paths.

**The smell test:** If your test passes but you're not sure the production code would work the same way, your test is lying to you.

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

### Core Infrastructure (touch with care)
| What | Where |
|------|-------|
| **Project Manager** | `core/project_manager.py` |
| **Server Mixin** | `core/main_window_server_mixin.py` |
| **Account Mixin** | `core/main_window_account_mixin.py` |
| **Project Mixin** | `core/main_window_project_mixin.py` |
| **Panels Mixin** | `core/main_window_panels_mixin.py` |
| **Signals Mixin** | `core/main_window_signals_mixin.py` |
| **Scene Hierarchy** | `panels/scene_hierarchy.py` + mixins |
| **Inspector Panel** | `panels/inspector_panel.py` |
| **Inspector Base** | `panels/inspector_base.py` |
| **Property Binding** | `core/property_binding.py` (PropertyMeta) |
| **Scene Node/Graph** | `core/scene_node.py`, `core/scene_graph.py` |
| **Smoke Tests** | `tests/test_smoke.py` |

### Unified Editor
| What | Where |
|------|-------|
| **Unified Editor Panel** | `panels/editors/unified_editor_panel.py` |
| **Assembly Editor View** | `panels/editors/assembly_editor_view.py` |
| **Depth Protocol** | `panels/editors/depth_protocol.py` |
| **Execution Viz Mixin** | `panels/editors/assembly_execution_mixin.py` |
| **Ensemble Mixin** | `panels/editors/assembly_ensemble_mixin.py` |
| **NC Depth View** | `panels/editors/neural_canvas_depth_view.py` |
| **Charm Depth View** | `panels/editors/charm_network_depth_view.py` |

### Performance & Cognition
| What | Where |
|------|-------|
| **Performance Manager** | `runtime/ui/guide_performance_manager.py` |
| **Performance Window** | `runtime/ui/guide_performance_window.py` |
| **Noodling Performer** | `runtime/ui/noodling_performer.py` |
| **Performance Player** | `runtime/ui/performance_player.py` |
| **Facet Executor** | `core/facet_executor.py` |
| **CharmNetworkEMA** | `runtime/charm_network_ema.py` |
| **LLM Client** | `runtime/llm_client.py` |

### World & Communication
| What | Where |
|------|-------|
| **Channel Bus** | `runtime/channels.py` |
| **World Channels** | `runtime/world_channels.py` |
| **Brenda Director** | `runtime/brenda.py` |
| **Guide Cue Handler** | `runtime/guide_cue_handler.py` |
| **cmush Server** | `../../cmush/server.py` |
| **cmush Startup** | `../../cmush/start.sh` |

### Rendering & Animation
| What | Where |
|------|-------|
| **VRM Viewport** | `runtime/ui/components/vrm_viewport.py` |
| **VRM Parser** | `core/semantic_world/vrm_parser.py` |
| **UI Components** | `runtime/ui/components/` |
| **Gaussian Renderer** | `core/semantic_world/gaussian_renderer.py` |

### Other
| What | Where |
|------|-------|
| **Computer Use** | `core/computer_use_controller.py` |
| **Build System** | `appbuilder/` |
| **Unity Exporter** | `core/noodling_package_exporter.py` |
| **UI Test Runner** | `testing/ui_test_runner.py` |

---

## Completed Systems (Jan-Feb 2026)

**Feb 16-18 (Phase C + D):**
- Unified Editor: depth-stack panel with breadcrumb navigation, plugin registry
- AssemblyEditorView: 8-mixin composition (input, grid, layout, view ops, wire, clipboard, execution, ensemble)
- Execution visualization: node pulsing, wire packet animations, sound effects
- Neural Canvas and CharmNetworkEMA depth views
- Three-noodling ensemble: Ajo, Krampus, Juanita on shared stage
- CharmNetworkEMA: 3-timescale affect smoothing (fast/medium/slow)
- Ensemble awareness: perception context, shared history, speaker spotlight
- Performance facet inspector: typing speed + speaking intensity
- Wiring fixes: noodlingSelected signal, --ensemble CLI from stage instances, Close Project
- Old FacetsEditorPanel (7 files) fully deleted and replaced
- Test hardening: QPushButton teardown race fix, __new__ bypass cleanup, splash timing fix
- 1881 tests, 0 failures

**Feb 6:**
- VRM per-material texture rendering (diffuse textures + colors)
- Fragment shader texture sampling with alpha cutout
- Per-material draw groups (sorted, merged same-material primitives)
- QTimer-driven idle animation (Y bob + breathing scale)
- Raw ctypes GL texture upload (PyOpenGL 3.14 workaround)
- Cached shader uniform locations
- VRM texture rendering test suite (21 tests)

**Jan 12:**
- Unity Package Export (File > Export > Export to Unity Package...)
- NoodlingPackageExporter creates .noodling folders for Unity
- PAD -> FACS -> VRM expression mapping chain
- Play Format runtime integration (Brenda -> GuideCueHandler -> LLM)
- Direction injection into facet execution context
- NoodleApp.load_director() auto-creates GuideCueHandler
- Play integration test suite (test_play_integration.py)

**Jan 10-11:**
- QMLWidget wrapper for Qt ecosystem widgets
- SevenSegment, LED, Gauge, LevelMeter dashboard widgets
- FacetAssembly UI Canvas component with AssemblyPickerDialog
- Build system with macOS .app bundler
- Splash screen with attribution
- Editor access enforcement (allow/password/hidden)
- Runtime LLM provider switching from build.yaml

**Jan 8-9:**
- Transparent VRM character overlay
- Channel architecture (ChannelBus, ChannelMessage)
- World channels (time, weather, ambiance, events)
- Brenda stage director for .play.yaml performances
- LLM Router wiring for Let's Consciousness!
- RadianceViewport tensor fix

**Jan 3-5:**
- Cognitive Cycles Panel v2 (hierarchical assembly monitoring)
- FacetAssemblyComponent (universal attachable component)
- Inspector UI for FacetAssembly
- UI Canvas components: Checkbox, Dropdown, Slider, RadioButton, RadioGroup
- Inspector Event Wiring UI
- Full UI event model (UIEventData, EventEmitting widgets)
- UI Canvas Stage integration
- Build system foundation

**Earlier:**
- NoodleROUTER (api.noodlings.ai)
- GPU Gaussian Rendering (gsplat-mps, 120 FPS)
- Admin Dashboard (admin.noodlings.ai)
- Cloud Account System (OAuth, credits)
- Multi-provider LLM (8 providers)
- Neural Canvas
- MCP integration
- 31 utility facets
- Multimodal facets

---

## Project Context

**Creator:** Caitlyn (Caity) Meeks — Unity early employee, launched the Asset Store
**Location:** Canary Islands (moved early 2026)
**Machine:** jiji (migrated from caledonia M3 Ultra 512GB, Feb 2026)
**Mission:** Open-source "Unity for Cognition" — build and observe cognitive simulations

**Caity's standards:** 30+ years of production software development. Christopher Alexander's Timeless Way of Building applied to software. Every piece of code is a careful decision made over hundreds of hours. Respect it. Do not rush. Do not write monkeypatches. Everything must reflect Unity quality.

---

## STOP LIST (Do Not Build Until Core Is Solid)

Before adding ANY new feature, verify:
1. Server starts when toggled (smoke test)
2. Inspector loads facet properties (smoke test)
3. Default project opens with noodlings on stage (smoke test)
4. Ensemble loads from stage instances, not hardcoded paths (verified D.2b)
5. All smoke tests pass
6. Full suite passes (1881 tests, 0 failures as of Feb 18)

If any of these fail, fix them FIRST. No new features on a broken foundation.

Do not work on: Gaussian features, enterprise tier, quantum nodes, Museum of Minds, Windows/Linux builds, asset marketplace, multi-provider routing beyond what works now.

**Status (Feb 18):** All STOP LIST conditions currently pass. Three noodlings (Ajo, Krampus, Juanita) are real stage instances talking on a real stage.

---

**Ordnung muss sein!**
