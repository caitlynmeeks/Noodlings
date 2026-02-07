# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: February 6, 2026

---

## HANDOFFS FROM DISCUSS CLAUDE (Check These First!)

Handoff documents from the discuss session live in:
`/Users/thistlequell/git/claudechat/projects/handoffs/`

**Current Priority Handoffs:**

| Date | File | Summary |
|------|------|---------|
| Jan 14 | `visual-verification-spec.md` | **HIGH PRIORITY** - Baseline comparison, SSIM diff, `assert_visual` action - fixes blind testing |
| Jan 14 | `ajo-display-and-asset-workflow-handoff.md` | RESOLVED (texture rendering) - VRM textures fixed Feb 6. Remaining: no "Add to Stage" UX |
| Jan 14 | `human-ui-test-plan.md` | Visual verification tests - what humans should SEE |
| Jan 14 | `web-browser-panel-spec.md` | Future: embedded browser with human-in-the-loop |
| Jan 13 | `hero-claude-demo-play-handoff.md` | Demo play ready to run |
| Jan 13 | `demo-play-for-hero-claude.yaml` | 10-beat demo play file |

**Read these when starting a session** - they contain context from Caity's discuss sessions.

---

## RECENT: GPU Skeletal Skinning (Feb 6, 2026)

VRM avatars now have GPU skeletal animation. The vertex shader blends up to 4 bone transforms per vertex. At rest pose, skinning matrices equal identity so the mesh renders identically to unskinned. When muscles are applied via `set_muscles()`, PoseRetargeter converts to bone rotations, hierarchy walk computes world transforms, and `skinMatrix[i] = worldTransform[i] * inverseBind[i]` is uploaded to the GPU.

**Skinning pipeline:**
```
set_muscles() -> PoseRetargeter -> bone euler rotations
    -> BFS hierarchy walk -> world transforms
    -> skinMatrix = world * inverseBind -> glUniformMatrix4fv -> GPU
```

**Key changes:**
- Vertex shader: `aBoneIndices` (ivec4), `aBoneWeights` (vec4), `uBoneMatrices[128]`
- `_create_mesh_buffers()`: uploads joint indices (location 3) and weights (location 4)
- `_compute_bone_matrices()`: BFS hierarchy, quat/euler math, world * inverse_bind
- `_apply_pose()`: wired through PoseRetargeter
- Raw ctypes `glUniformMatrix4fv` wrapper (PyOpenGL 3.14 workaround)
- Inverse bind matrices stored on `Skeleton.inverse_bind_matrices`

**Key files:**
| File | Purpose |
|------|---------|
| `runtime/ui/components/vrm_viewport.py` | Skinning shader, bone matrix computation, VBO upload |
| `core/semantic_world/vrm_parser.py` | Inverse bind matrix storage on Skeleton |
| `tests/test_vrm_gpu_skinning.py` | 30 tests (math, hierarchy, skinning matrices, retargeter) |

**Tests:** `tests/test_vrm_gpu_skinning.py` (30 tests)

**Launching Guide Performance Window (VRM preview):**
```bash
cd applications/noodlestudio
../../venv/bin/python3 -m noodlestudio.main --no-splash --play "Title Here"
```
VRM auto-discovers from `noodlings/guide/Radiances/AjoMajo.vrm`. The `--play` flag opens the Guide Performance Window with the VRM character. There is no menu option to open it manually.

---

## RECENT: VRM Texture Rendering + Idle Animation (Feb 6, 2026)

VRM avatars now render with per-material textures and diffuse colors instead of the previous flat tan hardcoded color. Idle breathing animation gives the guide character life without requiring GPU skinning.

**Rendering pipeline:**
```
All meshes -> sorted by material -> 1 combined VAO -> N draw calls (per material group)
                                     each binds diffuse texture + sets diffuse color
```

**Idle animation:**
```
QTimer (60fps) -> _tick_idle -> _build_model_matrix -> repaint
    Y position: bob (amplitude 0.01, period 4s)
    Y scale: breathing pulse (amplitude 0.02, period 3.5s)
```

**Key changes:**
- Fragment shader: `uDiffuseTex` sampler, `uHasTexture` toggle, alpha cutout (`discard < 0.5`)
- `_create_mesh_buffers()`: sorts meshes by material, tracks per-material draw groups
- `_load_textures()`: decodes VRM texture bytes via QImage, uploads via raw ctypes (PyOpenGL 3.1.0 + Python 3.14 workaround)
- `_draw_mesh()`: per-material loop with texture binding and cached uniform locations
- `_build_model_matrix()`: two out-of-phase sine waves for natural idle motion

**PyOpenGL workaround:** `glGenTextures` / `glBindTexture` / `glTexImage2D` use raw ctypes calls via the macOS OpenGL framework because PyOpenGL 3.1.0 + OpenGL_accelerate 3.1.10 has a broken array-type handler on Python 3.14 (CArgObject errors).

**Key files:**
| File | Purpose |
|------|---------|
| `runtime/ui/components/vrm_viewport.py` | All rendering changes |
| `tests/test_vrm_texture_rendering.py` | 21 tests (material groups, idle animation, fallback colors) |

**Tests:** `tests/test_vrm_texture_rendering.py` (21 tests)

**Future:** GPU skinning (per-bone matrices, inverse bind, weighted vertex transforms) remains a separate milestone.

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

## NEXT: Phase 7D - Custom Component System

**Goal:** Allow users to create reusable custom components.

**Composite Components (YAML):**
```yaml
type: CompositeComponent
name: LoginForm
properties:
  - name: title
    type: string
    default: "Login"
template:
  type: Panel
  children:
    - type: Label
      text: "${title}"
```

**Key files to create:** `runtime/ui/composite_loader.py`

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

## BACKLOG

### Inspector UX
- Unity-style numeric drag-to-scroll on labels

### Undo/Redo for UI Edits
- `AddUIComponentCommand`, `DeleteUIComponentCommand`

### Additional Build Targets
- Windows .exe (PyInstaller)
- Linux binary

### Admin Dashboard - Issue Credits UI
File: `backend/admin-dashboard/src/routes/users/[id]/+page.svelte`

### Trained Gaussian Quality
OpenSplat-trained Gaussians have background artifacts.

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
PYTHONPATH=.:../.. pytest              # Run all (~900 tests)
PYTHONPATH=.:../.. pytest -v           # Verbose
PYTHONPATH=.:../.. pytest -k "test_ui" # By pattern
```

**Before committing:** ALL tests must pass.

---

## Development

### Environment Setup
```bash
cd /Users/thistlequell/git/noodlings_clean
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
| **LLM Client** | `runtime/llm_client.py` |
| **Runtime CLI** | `runtime/cli.py` |
| **UI Canvas (runtime)** | `runtime/ui/` |
| **UI Components** | `runtime/ui/components/` |
| **UI Event Dispatcher** | `runtime/ui/event_dispatcher.py` |
| **UI Script Executor** | `runtime/ui/script_executor.py` |
| **UI Canvas Designer** | `panels/ui_canvas_editor_panel.py` |
| **Build system** | `appbuilder/` |
| **Facet editor** | `panels/facets_editor_panel.py` |
| **Scene hierarchy** | `panels/scene_hierarchy.py` |
| **Cognitive Cycles** | `panels/cognitive_cycles_panel_v2.py` |
| **Channel Bus** | `runtime/channels.py` |
| **World Channels** | `runtime/world_channels.py` |
| **Brenda Director** | `runtime/brenda.py` |
| **Guide Cue Handler** | `runtime/guide_cue_handler.py` |
| **Computer Use** | `core/computer_use_controller.py` |
| **Ghost Cursor** | `core/ghost_cursor.py` |
| **UI Test Runner** | `testing/ui_test_runner.py` |
| **VRM Viewport** | `runtime/ui/components/vrm_viewport.py` |
| **VRM Parser** | `core/semantic_world/vrm_parser.py` |
| **Guide Performance Window** | `runtime/ui/guide_performance_window.py` |
| **Guide Performance Manager** | `runtime/ui/guide_performance_manager.py` |
| **Unity Package Exporter** | `core/noodling_package_exporter.py` |

---

## Completed Systems (Jan-Feb 2026)

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

**Creator:** Caitlyn (Unity employee #12, Asset Store creator)
**Location:** Garcia River Forest cabin
**Hardware:** M3 Ultra 512GB

**Mission:** Open-source alternative to "Consciousness-as-a-Service"

---

**Ordnung muss sein!**
