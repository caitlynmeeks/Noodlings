# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: January 10, 2026

---

## COMPLETED: Facets Phase 3 - UI Canvas Integration (Jan 10, 2026)

**Goal:** Add FacetAssembly to UI Canvas component palette for visual assembly integration.

**What Was Built:**

### FacetAssembly UIComponent (`runtime/ui/components/facet_assembly.py`)
- New UI Canvas component for attaching facet assemblies
- Properties: `assembly_path`, `auto_run`, `input_bindings`, `output_bindings`
- InputBinding/OutputBinding dataclasses for pad-to-UI mappings
- Full YAML serialization/deserialization support
- Invisible at runtime (pure logic component)

### AssemblyPickerDialog (`dialogs/assembly_picker_dialog.py`)
- Visual browser for assembly files in project
- Tree view with directory grouping
- Preview pane showing assembly metadata (inputs, outputs, facets)
- Filter/search functionality
- Dark theme styling

### Inspector Integration (`panels/inspector_ui_canvas.py`)
- Assembly path field with browse button
- Auto Run checkbox
- Opens AssemblyPickerDialog for file selection

### Context Menu Integration
- Added to UI Canvas editor context menu (Add > FacetAssembly)
- Added to Stage hierarchy context menus (UI Canvas root and Panel children)

**Usage in ui.yaml:**
```yaml
# Define the FacetAssembly component
- type: FacetAssembly
  name: sentiment_analyzer
  properties:
    assembly: assemblies/sentiment.yaml
    auto_run: false
  input_bindings:
    - pad: text
      source: text_input.value
  output_bindings:
    - pad: sentiment
      target: mood_indicator.color

# Reference it from events using target:
- type: Button
  name: analyze_btn
  events:
    onClick:
      action: run_assembly
      target: sentiment_analyzer  # Uses FacetAssembly's bindings
      thinking_target: result_label
      clear_input: true
```

**Event Dispatcher Integration:**
- `target: component_name` syntax references FacetAssembly component by name
- Uses component's `assembly_path`, `input_bindings`, `output_bindings`
- Falls back to inline `assembly:` config if no target specified
- Helper methods: `_find_facet_assembly_component()`, `_resolve_facet_assembly_inputs()`

**Key Files:**
| File | Purpose |
|------|---------|
| `runtime/ui/components/facet_assembly.py` | FacetAssembly component (200 lines) |
| `runtime/ui/event_dispatcher.py` | target: syntax + binding resolution |
| `runtime/cli.py` | Wires root_component to dispatcher |
| `dialogs/assembly_picker_dialog.py` | Assembly file picker (380 lines) |
| `panels/inspector_ui_canvas.py` | Inspector assembly field |
| `panels/ui_canvas_editor_panel.py` | Context menu + default size |
| `panels/scene_hierarchy_*.py` | Stage hierarchy context menus |
| `runtime/ui/renderer.py` | Invisible widget rendering |

**Tests:** 9 new tests in `test_ui_canvas.py::TestFacetAssemblyComponent`

---

## COMPLETED: Phase 7 - Build Process (Jan 10, 2026)

**Goal:** Make "Build and Run" actually build a standalone macOS .app bundle.

**What Was Built:**

### BuildProgressDialog (`dialogs/build_progress_dialog.py`)
- Progress bar with percentage and status messages
- Cancel button with proper thread cleanup
- Success/failure states with detailed output info
- Auto-launch support for "Build and Run"
- 22 unit tests

### BuildWorker (`dialogs/build_progress_dialog.py`)
- QThread-based background build execution
- Progress, finished, and error signals
- Cancellation support

### Builder Updates (`appbuilder/builder.py`)
- Now uses canonical `core/build_config.BuildConfig`
- Accepts `project_path` separately from config
- Cancellation support throughout build process
- Output directory auto-creation

### Packager Updates (`appbuilder/packager.py`)
- Updated to accept `project_path` parameter
- Uses new BuildConfig structure (identity.icon, etc.)

### MacOSBundler Updates (`appbuilder/bundler_macos.py`)
- Updated to accept `project_path` parameter
- Uses new BuildConfig structure for Info.plist

### BuildSettingsDialog Wiring
- `_build()` now shows BuildProgressDialog and runs actual build
- `_build_and_run()` builds and auto-launches the .app
- `_on_build_completed()` handles result (close on success, stay open on failure)

**Build Output Structure:**
```
~/Desktop/builds/
└── Let's Consciousness!.app/
    └── Contents/
        ├── Info.plist
        ├── MacOS/
        │   └── LetsConsciousness (launcher script)
        └── Resources/
            ├── project/
            │   ├── ui.yaml
            │   └── build.yaml
            └── runtime/
                └── noodlestudio/
                    ├── core/
                    ├── runtime/
                    ├── scripting/
                    └── data/
```

**Key Files:**
| File | Purpose |
|------|---------|
| `dialogs/build_progress_dialog.py` | Progress dialog + BuildWorker (320 lines) |
| `appbuilder/builder.py` | Build orchestrator (300 lines) |
| `appbuilder/packager.py` | Asset packaging (365 lines) |
| `appbuilder/bundler_macos.py` | macOS .app creation (420 lines) |
| `tests/test_build_progress.py` | 22 tests |

**Tests:** 114 tests for build settings + splash + editor access + build progress. Total project: ~700 tests.

**Remaining Work:**
- Code signing and notarization (distribution section)
- Windows/Linux bundlers (PyInstaller)

---

## COMPLETED: Runtime LLM Provider Switching (Jan 10, 2026)

**Goal:** Built apps read LLM provider from build.yaml and configure the API client accordingly.

**Implementation:**

### Helper Function (`runtime/cli.py`)
- `_apply_build_config_llm_settings(args, build_config)` - Applies LLM settings from build.yaml to CLI args
- Called early in `run_gui()` before FacetExecutor initialization
- Maps build.yaml providers to runtime providers:
  - `noodlerouter` -> `noodlerouter`
  - `user_keys` -> auto-detects from environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY)
  - `ollama` -> `ollama`
  - `bundled` -> `noodlerouter` with bundled key

### Provider Priority
1. CLI `--provider` flag (explicit user override)
2. build.yaml `llm.provider` setting
3. Default (`ollama`)

### user_keys Mode
When build.yaml specifies `user_keys`, the runtime auto-detects which provider the user has configured:
1. Checks for `ANTHROPIC_API_KEY` -> uses `anthropic` provider
2. Checks for `OPENAI_API_KEY` -> uses `openai` provider
3. Checks for `OPENROUTER_API_KEY` -> uses `openrouter` provider
4. None found -> prints warning

### bundled Mode
When build.yaml specifies `bundled`:
- Uses `noodlerouter` provider
- Uses `llm.bundled_key` from build.yaml as API key
- CLI key takes precedence if provided

**Key Changes:**
- `runtime/cli.py:233-289` - `_apply_build_config_llm_settings()` helper
- `runtime/cli.py:376-395` - Early build_config loading and LLM settings application

**Tests:** 10 new tests in `test_build_settings.py::TestApplyBuildConfigLLMSettings`

---

## COMPLETED: Build Settings + Splash Screen (Jan 10, 2026)

**Goal:** Unity-style File > Build Settings dialog with splash screen support for published apps.

**Full spec:** `/docs/noodlestudio/build-settings.md`

**What Was Built:**

### BuildSettingsDialog (`dialogs/build_settings_dialog.py`)
- File > Build Settings... (Ctrl+Shift+B)
- Collapsible sections: Platform, Identity, Splash, Editor Access, LLM Provider, Content, Distribution, Advanced
- Saves/loads `build.yaml` in project root
- 36 unit tests

### BuildConfig (`core/build_config.py`)
- Dataclasses for all build settings
- Full YAML serialization/deserialization
- Validation against project assets

### SplashScreen (`widgets/splash_screen.py`)
- Custom image or text-based splash
- Fade in/out animations
- LoadingIndicator (dots/bar/spinner)
- AttributionWidget - "Made with NoodleSTUDIO" + NEC link (always required)
- Click-to-dismiss support
- 35 unit tests

### Runtime Integration (`runtime/cli.py`)
- `_build_config_to_splash_config()` helper converts BuildConfig to SplashScreen format
- `run_gui()` loads build.yaml and shows splash before main window
- Character overlay waits for splash completion

**Key Files:**
| File | Purpose |
|------|---------|
| `dialogs/build_settings_dialog.py` | Main dialog UI (1000 lines) |
| `core/build_config.py` | BuildConfig dataclass (533 lines) |
| `widgets/splash_screen.py` | SplashScreen + AttributionWidget (570 lines) |
| `runtime/cli.py:189-230` | `_build_config_to_splash_config()` helper |
| `runtime/cli.py:370-457` | Splash integration in `run_gui()` |
| `tests/test_build_settings.py` | 36 tests |
| `tests/test_splash_screen.py` | 35 tests |

---

## COMPLETED: Editor Access Enforcement (Jan 10, 2026)

**Goal:** Control access to NoodleStudio editor in published apps via build.yaml settings.

**Key Components:**

### EditorPasswordDialog (`dialogs/editor_password_dialog.py`)
- Password prompt for protected editor access
- SHA-256 password hashing (hash_password, verify_password)
- Attempt tracking with lockout after max failures
- Clean dark-mode styling

### MainWindowFoldMixin Updates (`core/main_window_fold_mixin.py`)
- `set_editor_access(access, password_hash, keyboard_shortcut)` - Configure restrictions
- `_check_editor_access()` - Validate access before unfold
- Keyboard shortcut disabled when access is "hidden"
- Password dialog shown when access is "password"
- "View Project" button hidden when access is "hidden"

### Project Integration (`core/main_window_project_mixin.py`)
- `_load_editor_access_from_build_config()` - Load settings when project opens
- Settings reloaded when Build Settings dialog saves

**Access Levels:**
| Level | Button | Shortcut | Behavior |
|-------|--------|----------|----------|
| `allow` | Shown | Enabled | Normal unfold |
| `password` | Shown | Enabled | Password dialog before unfold |
| `hidden` | Hidden | Disabled | No editor access |

**Tests:** 21 new tests in `test_editor_access.py`

---

## COMPLETED: LLM Router Wiring for Let's Consciousness! (Jan 9, 2026)

**Goal:** Make Guide (Ajo Majo the axolotl) talk using facet assemblies via NoodleROUTER.

**Project Location:** `Projects/lets-consciousness/`

**What Was Wired:**
1. **Project path derivation** - cli.py now derives project from ui.yaml location
2. **FacetExecutor initialization** - Created with HeadlessLLMClient in run_gui()
3. **Assembly model names** - Using NoodleROUTER format (`anthropic/claude-3-haiku`, `anthropic/claude-sonnet-4`)
4. **UI event binding** - TextInput and Button trigger `run_assembly` action

**Architecture Flow:**
```
User Input -> run_assembly action -> FacetExecutor -> HeadlessLLMClient
    -> NoodleROUTER (api.noodlings.ai) -> Claude -> Response -> guide_speech.text
```

**Running the Project:**
```bash
cd applications/noodlestudio

# Option 1: CLI args
PYTHONPATH=.:../.. python -m noodlestudio.runtime \
  --gui --ui "../../Projects/lets-consciousness/ui.yaml" \
  --provider noodlerouter --api-key $NOODLEROUTER_API_KEY

# Option 2: Environment variables
export NOODLE_LLM_PROVIDER=noodlerouter
export NOODLEROUTER_API_KEY=<your-api-key>
PYTHONPATH=.:../.. python -m noodlestudio.runtime \
  --gui --ui "../../Projects/lets-consciousness/ui.yaml"
```

**Key Files:**
| File | Purpose |
|------|---------|
| `runtime/cli.py:258-319` | Project derivation + FacetExecutor init |
| `runtime/llm_client.py` | HeadlessLLMClient with noodlerouter provider |
| `runtime/ui/event_dispatcher.py:363` | `_handle_run_assembly()` action |
| `noodlings/guide/assembly.yaml` | Guide's cognitive pipeline |
| `Projects/lets-consciousness/ui.yaml` | UI with run_assembly events |

**NoodleROUTER Reference:**

| Model ID | Use Case |
|----------|----------|
| `anthropic/claude-3-haiku` | Testing, simple tasks |
| `anthropic/claude-3.5-haiku` | Good balance |
| `anthropic/claude-sonnet-4` | Complex reasoning |
| `anthropic/claude-opus-4-5` | Maximum capability |

**Status:** Ajo Majo is ready to talk! Run the project with a valid NoodleROUTER API key.

---

## COMPLETED: RadianceViewport Tensor Fix (Jan 8, 2026)

**Issue:** Axolotl rendered black despite Gaussians loading successfully.

**Root Cause:** `GaussianRenderer.render_scene()` returns `torch.Tensor` on MPS device, but display code expected numpy array.

**Fix:** Added tensor-to-numpy conversion in `runtime/ui/components/radiance_viewport.py:554`:
```python
if TORCH_AVAILABLE and torch.is_tensor(image):
    image = image.detach().cpu().numpy()
```

Also fixed Qt thread safety using `QMetaObject.invokeMethod` with `@pyqtSlot()` decorator.

**Status:** Ajo Majo now renders correctly. All 414 tests passing.

---

## COMPLETED: Transparent VRM Character Overlay (Jan 8, 2026)

**Issue:** QOpenGLWidget can't composite transparently as an embedded widget. VRM characters had opaque backgrounds.

**Solution:** Created `CharacterOverlayWindow` - a separate frameless overlay window that follows the main window.

**Key Implementation:**
- `runtime/ui/overlay.py` - New `CharacterOverlayWindow` class
- Frameless window with `Qt.WindowType.Tool` (no taskbar entry)
- `WA_TranslucentBackground` + `WA_NoSystemBackground` for transparency
- Timer-based position tracking (50ms) keeps overlay following main window
- VRMViewport with `transparent: True` clears with alpha=0

**UI YAML Configuration:**
```yaml
# In ui.yaml - top level overlay section
overlay:
  enabled: true
  vrm_path: "../../noodlings/guide/Radiances/AjoMajo.vrm"
  size: [300, 400]
  anchor: right  # or "left"
  offset: [20, 50]  # x, y offset from anchor
```

**API:**
```python
overlay = CharacterOverlayWindow(
    parent_window=main_window,
    vrm_path="/path/to/character.vrm",
    size=(300, 400),
    offset=(20, 100),
    anchor="right"
)
overlay.show()

# Animate character
overlay.set_muscles({"Head.TurnLeftRight": 0.3})
overlay.set_blend_shapes({"happy": 0.6})
```

**Files:**
- `runtime/ui/overlay.py` - CharacterOverlayWindow class
- `runtime/cli.py` - Reads overlay config, creates overlay in run_gui()
- `Projects/lets-consciousness/ui.yaml` - Updated with overlay config

**Status:** Guide (Ajo Majo) now floats transparently over the Let's Consciousness UI.

---

## COMPLETED: Channel Architecture (Jan 8, 2026)

**Goal:** Named message buses for inter-noodling communication. Enables stage direction, environmental context, group communication, and private messaging.

**Full spec:** `/docs/noodlestudio/channels.md`

**Key Classes:**
- `ChannelBus` - Pub/sub message bus with history
- `ChannelMessage` - Message structure with channel, sender, timestamp, payload
- `ChannelsConfig` - Assembly channel subscription/publish configuration

**Assembly YAML:**
```yaml
name: Guide Assembly
channels:
  subscribe:
    - "#directors.cues"
    - "#world.context"
  publish:
    - "#directors.feedback"
```

**FacetAssembly Methods:**
```python
assembly.get_subscribe_channels()  # List of subscribed channels
assembly.get_publish_channels()    # List of publishable channels
assembly.can_publish_to(channel)   # Check publish permission
assembly.subscribes_to(channel)    # Check subscription
```

**FacetExecutor Integration:**
- Channel inputs resolved via `channel:#channel.name` pad references
- Channel outputs published via `channel:#channel.name` output pads
- Respects assembly subscribe/publish permissions

**ChannelBus API:**
```python
bus = ChannelBus()
bus.subscribe("#directors.cues", callback)
bus.publish_simple("#directors.cues", {"type": "cue"}, "brenda")
latest = bus.get_latest("#directors.cues")
history = bus.get_history("#directors.cues", limit=10)
```

**Channel Naming Convention:**
- `#world.*` - Public environmental (weather, time, events)
- `#directors.*` - Stage management (cues, feedback)
- `#dm.*` - Direct messages (private)
- `#<scope>.*` - Scoped group channels

**Files Created:**
- `runtime/channels.py` - ChannelBus, ChannelMessage, ChannelsConfig
- `tests/test_channels.py` - 28 unit tests

**Files Modified:**
- `core/facet_system.py` - Added channels field to FacetAssembly
- `core/facet_executor.py` - Added channel input/output resolution
- `runtime/app.py` - NoodleApp owns ChannelBus instance

**Tests:** 28 new tests, all passing.

---

## COMPLETED: World Channels (Jan 9, 2026)

**Goal:** System-level channels that broadcast environmental context to all noodlings: time, weather, events, ambiance.

**Full spec:** `/docs/noodlestudio/handoff-world-channels.md`

**Key Classes:**
- `WorldChannelService` - Manages world-level channels for environmental context
- `WorldConfig` - Configuration loaded from stage.yaml `world` section

**Channels Published:**
| Channel | Content |
|---------|---------|
| `#world.time` | Simulation time, time of day, sun position, natural language description |
| `#world.weather` | Temperature, conditions, wind, humidity, description |
| `#world.ambiance` | Mood, energy level, atmospheric description |
| `#world.events` | Discrete events (sounds, visuals, physical, social) |

**Stage Configuration:**
```yaml
# stage.yaml
world:
  time_scale: 1.0              # Real time (60.0 = 1 min per second)
  initial_time: "18:00"        # Start at 6 PM
  weather:
    temperature: 68
    conditions: partly_cloudy
    wind: light_breeze
  ambiance:
    mood: calm
    energy: 0.5
```

**NoodleApp API:**
```python
app.tick_world()                           # Advance simulation time
app.set_world_time("15:30")               # Set time directly
app.set_world_weather(conditions="rain")  # Update weather
app.set_world_ambiance(mood="tense")      # Update ambiance
app.trigger_world_event("sound", "door", "A door slammed.")  # Discrete event
app.get_world_context()                   # Get full world state
```

**Noodling Subscription:**
```yaml
# guide/assembly.yaml
channels:
  subscribe:
    - "#directors.cues"
    - "#world.time"
    - "#world.ambiance"
```

**Files Created:**
- `runtime/world_channels.py` - WorldChannelService, WorldConfig
- `tests/test_world_channels.py` - 44 unit tests

**Files Modified:**
- `runtime/app.py` - NoodleApp owns WorldChannelService, convenience methods

**Tests:** 44 new tests, all passing. Total: 486 tests.

---

## COMPLETED: Brenda Stage Director (Jan 9, 2026)

**Goal:** Invisible stage director who orchestrates performances from `.play.yaml` scripts, sending cues to noodlings and managing theatrical flow.

**Full spec:** `/docs/noodlestudio/handoff-brenda.md`

**Key Classes:**
- `BrendaDirector` - Stage director that loads and runs .play.yaml scripts
- `PlayState` - Tracks current beat, completed beats, character states, improv zones
- `DirectorMode` - Enum: ACTIVE, PASSIVE, PASSIVE_AVAILABLE, PAUSED
- `TriggerType` - Enum: SEQUENCE, AFTER, THRESHOLD, DELAY, USER_CHOICE, USER_RESPONSE, IMPROV_COMPLETE, ALL

**Play YAML Format:**
```yaml
play:
  title: "Let's Consciousness!"
  characters:
    - name: guide
      assembly: noodlings/guide/assembly.yaml

beats:
  intro:
    direction: "Guide greets the visitor warmly"
    actors:
      guide:
        action: greet
        emotional_target: {pleasure: 0.6, arousal: 0.5}
    triggers:
      - type: sequence
        next_beat: explore
```

**NoodleApp API:**
```python
app.load_director("path/to/play.yaml")  # Load a play script
app.start_performance()                  # Begin the performance
app.stop_performance()                   # End the performance
app.publish_user_input("Hello!")         # Send user input to director
app.get_director_state()                 # Get current play state
app.get_play_info()                      # Get play metadata
app.tick()                               # Advances both world and director
```

**Channel Integration:**
- Subscribes to `#directors.feedback` (noodling status updates)
- Subscribes to `#user.input` (user messages)
- Publishes to `#directors.cues` (beat instructions)
- Can control world channels via WorldChannelService

**Trigger Types:**
| Type | Description |
|------|-------------|
| `sequence` | Immediately advance to next beat |
| `after` | After specific beat completes |
| `delay` | Wait N seconds |
| `threshold` | When character affect reaches target |
| `user_choice` | Wait for user to select an option |
| `user_response` | Wait for any user input |
| `improv_complete` | When improv zone ends |
| `all` | All sub-triggers must complete |

**Files Created:**
- `runtime/brenda.py` - BrendaDirector, PlayState, DirectorMode, TriggerType
- `tests/test_brenda.py` - 31 unit tests

**Files Modified:**
- `runtime/app.py` - NoodleApp owns BrendaDirector, convenience methods, tick()

**Tests:** 31 new tests, all passing. Total: 517 tests.

---

## COMPLETED: Cognitive Cycles Panel v2 + Integration (Jan 5, 2026)

**Hierarchical assembly monitoring** - upgraded from "one row per agent" to "Things containing multiple Assemblies."

**Visual Design:**
```
+-- Cognitive Cycles ----------------------------------------[Expand All][Pause All]
|
| [v] chester                                                    [3] [2/3 active] [||][>|]
|     |-- emotional-processing    [####..] FACET    "valence: 0.7"     [||][>|]
|     |-- language-generation     [##....] PRECOG   "gathering context" [||][>|]
|     |-- social-modeling         [#####.] POSTCOG  "updated: alice"    [||][>|]
|
| [>] mysterious_door                                            [1] Idle          [||][>|]
|
+--------------------------------------------------------------------[1 thing, 4 assemblies]
```

**Key Features:**
- **Hierarchical layout**: Things (Noodlings, Prims) contain Assemblies
- **Collapsible rows**: Compact by default, expand to see assemblies
- **Architecture-agnostic**: Assemblies publish free-form `status_text` strings
- **Granular controls**: Pause/Step per-assembly, per-Thing, or globally
- **Backward compatible**: Works with old flat agent format
- **Dual data source**: Reads from both local CognitionMonitor (in-process) AND HTTP API (cmush agents)

**CognitionMonitor Integration:**

FacetAssemblyComponent now reports status to CognitionMonitor during execution:

```python
# Automatic reporting in FacetAssemblyComponent.run():
# - Reports FACET phase at start
# - Reports POSTCOG with output summary
# - Reports IDLE when complete
# - Reports errors with truncated message

# Cognition loop also reports:
# - "continuous mode started" when loop begins
# - "stopped" when loop ends
```

**Panel Data Flow:**
1. Local CognitionMonitor singleton (in-process assemblies)
2. HTTP API at localhost:8081/api/cycle_phases (cmush agents)
3. Both sources merged - local takes precedence for same thing_id

**Files Created/Modified:**
- `panels/cognitive_cycles_panel_v2.py` - New hierarchical panel UI with dual data sources
- `core/cognition_monitor.py` - Central status registry singleton
- `core/facet_assembly_component.py` - Added CognitionMonitor reporting
- `core/main_window_panels_mixin.py` - Updated import to v2 panel

**API Format (hierarchical):**
```json
{
  "things": {
    "chester-uuid": {
      "name": "chester",
      "assemblies": {
        "emotional-processing": {
          "phase": "FACET",
          "status_text": "valence: 0.7",
          "activity": 0.8
        }
      }
    }
  }
}
```

**Tests:** All 414 passing (no regressions)

---

## COMPLETED: Facets as Universal Components - Phase 2 (Jan 5, 2026)

**Inspector UI for FacetAssemblyComponent** - Full Delphi Object Inspector experience.

**What Was Built:**
```
+-- Facet Assembly: sentiment-analysis ------+
| Assembly:    [sentiment-analysis.yaml] [R] |  <- PropertySpec (file picker)
| [x] Run in cognition loop                  |  <- PropertySpec (checkbox)
| Tick Rate:   [0.1    ] seconds             |  <- PropertySpec (float)
+-- Input Bindings --------------------------+
| out:         [{text_field.value}     ] [x] |  <- Binding row
+-- Output Bindings -------------------------+
| in:          [result_label.text      ] [x] |  <- Binding row
+-- Statistics ------------------------------+
| Executions: 42  |  Total Tokens: 12,450    |
| Last Run: 0.23s |  Avg Tokens: 296         |
| Status: Idle                               |
+-- Actions ---------------------------------+
| [Run Once]  [Refresh]                      |
+--------------------------------------------+
```

**Features:**
- Input/output binding UI with pad names from assembly
- Statistics display (executions, tokens, timing)
- "Run Once" button for testing one-shot execution
- "Refresh" button for stats update
- Status indicator (Idle/Running/Continuous)

**Files Modified:**
- `panels/inspector_components.py` - Added `_create_facet_assembly_ui` method

**Tests:** 411 passing (no regressions)

---

## COMPLETED: Facets as Universal Components - Phase 1 (Jan 5, 2026)

**THE KEY ARCHITECTURAL UNIFICATION**: Facet Assemblies are now attachable components that work on ANY entity (Noodling, Prim, UI element). This makes Facets the universal visual logic language for everything in NoodleStudio.

**Core Innovation:**
- Multiple assemblies per entity (singleton = False)
- Two execution modes controlled by checkbox:
  - **CHECKED** (Continuous): Runs in cognition loop every tick_rate seconds
  - **UNCHECKED** (One-shot): Runs on-demand via events/scripts

**FacetAssemblyComponent** (`core/facet_assembly_component.py`):
```python
# Get assembly from entity
assembly = entity.GetComponent("facet_assembly", "translate-chinese")

# Run one-shot
result = await assembly.run({"text": "Hello world"})

# Listen for events
assembly.add_listener('complete', on_complete_handler)
```

**Properties:**
- `assembly_path`: Path to .yaml assembly file
- `run_in_cognition_loop`: THE checkbox - continuous vs one-shot
- `tick_rate`: Seconds between cognitive ticks (0.01-60s)
- `auto_run_on_attach`: Run once when component added

**Events:**
- `OnComplete`: Fires after one-shot execution
- `OnStateChange`: Fires when continuous assembly state changes
- `OnError`: Fires on execution error

**New UI Canvas Action: run_assembly**

```yaml
Button:
  name: analyze_button
  events:
    onClick:
      action: run_assembly
      assembly: assemblies/sentiment-analysis.yaml
      inputs:
        text: "{text_field.value}"
      outputs:
        result: result_label.text
        sentiment: mood_indicator.color
```

**Files Created:**
- `core/facet_assembly_component.py` - Core component class
- `tests/test_facet_assembly_component.py` - 38 tests

**Files Modified:**
- `runtime/ui/event_dispatcher.py` - Added run_assembly action
- `runtime/ui/component.py` - EventBinding supports assembly/inputs/outputs

**Tests:** 411 passing (38 new for FacetAssemblyComponent)

---

## NEXT: Phase 7D - Custom Component System

**Goal:** Allow users to create reusable custom components.

**Three ways to create custom components:**

**1. Composite Components (YAML):**
```yaml
# components/login_form.yaml
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
    - type: TextInput
      name: "usernameInput"
```

**2. Python Components:**
```python
@register_component
class ColorPicker(UIComponent):
    component_type = "ColorPicker"
    PROPERTIES = {'value': {'type': 'color', 'default': '#ffffff'}}
```

**Key Files:**
- `runtime/ui/composite_loader.py` - NEW: Load composite components
- `runtime/ui/component.py` - Add PROPERTIES/EVENTS class attributes

---

## COMPLETED: UI Polish and New Components (Jan 4, 2026)

Session work on Inspector UX, navigation consistency, and new components.

**Inspector Improvements:**
- Tightened padding throughout (margins, spacing, input fields)
- Added `ColorFieldWidget` with Procreate-style color picker popup
- Color wheel (HSV) + saturation/value square
- 25-color palette + recent colors
- Hex input field

**Navigation Consistency (Facets/Neural Canvas/UI Canvas):**
- Added F key focus toggle to UI Canvas (zoom to selection, press again to restore)
- Added zoom limits to UI Canvas
- Removed redundant "Frame All" toolbar button (A key works)

**New Components:**
- `WebView` - Embedded web browser (requires PyQt6-WebEngine)

**Context Menu Sync:**
- UI Canvas and Stage View now have identical component lists
- All 13 components available: Panel, Label, Button, TextInput, Checkbox, Dropdown, Slider, RadioGroup, ChatHistory, ChatInput, RadianceViewport, WebView

**Stage Hierarchy Fixes:**
- Text no longer truncated (ElideNone + horizontal scrollbar)
- Tooltips on UI component items
- Bidirectional selection sync (Canvas <-> Stage hierarchy)
- Drag-drop reparenting for UI components (drag Button onto Panel)

**Files Created:**
- `widgets/color_picker_widget.py`
- `runtime/ui/components/webview.py`

---

## COMPLETED: Phase 7C - More Components (Jan 4, 2026)

Standard Delphi-style components added to expand the component palette.

**New Components:**
| Component | Description |
|-----------|-------------|
| `Checkbox` | Boolean toggle with label, toggle() method |
| `Dropdown` | ComboBox/select with placeholder, add/remove options |
| `Slider` | Numeric range with step snapping, percentage, formatted value |
| `RadioButton` | Single radio with value property |
| `RadioGroup` | Container with options, orientation (vertical/horizontal) |

**Files Created:**
- `runtime/ui/components/checkbox.py`
- `runtime/ui/components/dropdown.py`
- `runtime/ui/components/slider.py`
- `runtime/ui/components/radio.py` (RadioButton + RadioGroup)

**Files Modified:**
- `runtime/ui/components/__init__.py` - Export new components
- `runtime/ui/renderer.py` - Render methods for all 5 components

**Features:**
- Full YAML serialization/deserialization
- onChange event firing with UIEventData
- Binding manager integration (value changes propagate)
- Styled to match monochromatic dark theme

**Tests:** 26 new tests (136 total), all passing.

---

## COMPLETED: Phase 7B - Inspector Event Wiring UI (Jan 4, 2026)

Visual UI in Inspector for configuring event bindings - the Delphi Object Inspector's Events tab.

**Implemented:**
```
+-- Events ----------------------------------+
| onClick         [send_to_noodling v]  [x]  |
|                 Target: [red        v]     |
|                 Message: [input     v]     |
| [+ Add Event]                              |
+--------------------------------------------+
```

**Features:**
- `EventBindingWidget` - Individual event row with action dropdown
- Action types: send_to_noodling, call_script, set_value, show, hide, toggle_visible
- Dynamic parameter fields based on action type
- "Edit Script..." opens `ScriptEditorDialog` with JS syntax highlighting
- "[+ Add Event]" menu with common events + "More Events..."
- Delete button (x) removes event binding
- Changes auto-save to ui.yaml via canvas_modified signal

**Files Created:**
- `widgets/event_binding_widget.py` - Event row widget
- `dialogs/script_editor_dialog.py` - JavaScript script editor with API reference

**Files Modified:**
- `panels/inspector_ui_canvas.py` - Interactive Events section

**Tests:** 18 new tests (110 total), all passing.

---

### Current State

| Layer | Status | Details |
|-------|--------|---------|
| **Components** | 12 built | Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport, Checkbox, Dropdown, Slider, RadioButton, RadioGroup |
| **Events** | 15 wired | onClick, onDoubleClick, onMouseDown/Up/Move, onMouseEnter/Leave, onMouseWheel, onContextMenu, onKeyDown/Up, onFocus/Blur, onChange, onSubmit |
| **Actions** | 5 built | send_to_noodling, call_script, set_value, show, hide, toggle_visible |
| **Bindings** | Working | One-way reactive `{text: "input.value"}` |
| **Scripts** | Working | QuickJS sandbox with `ui.*` API, full UIEventData access |

### Implementation Phases

#### Phase 7A: Full Event Model - COMPLETE (Jan 4, 2026)

Full event model with comprehensive UIEventData and Qt widget integration.

**UIEventData** (`runtime/ui/event_data.py`):
- Mouse position (x, y, global_x, global_y)
- Mouse button (MouseButton enum: LEFT, RIGHT, MIDDLE)
- Keyboard modifiers (Modifiers dataclass: shift, ctrl, alt, meta)
- Keyboard key info (key name, key code, text)
- Value changes (value, previous_value)
- Drag data and drop effect
- 3D hit info (position, entity, semantics)
- Scroll deltas
- Propagation control (stop_propagation, prevent_default)
- Factory methods: `click()`, `value_change()`, `submit()`, `focus()`
- Qt conversion: `from_qt_mouse_event()`, `from_qt_key_event()`, `from_qt_wheel_event()`
- Event type constants (EVENT_CLICK, EVENT_KEY_DOWN, etc.)

**EventEmitting Widgets** (`runtime/ui/event_widgets.py`):
- `EventEmittingMixin` - Adds event emission to any QWidget
- `EventEmittingFrame` - Panel with full events
- `EventEmittingButton` - Button with full events
- `EventEmittingLineEdit` - TextInput with full events

**Integration**:
- `event_dispatcher.py` - Updated to accept UIEventData
- `script_executor.py` - Full event data passed to scripts
- `renderer.py` - Uses EventEmitting widgets

**Tests**: 18 new tests (92 total), all passing.

**Events Wired**:

**Mouse Events:**
```
onClick, onDoubleClick, onMouseDown, onMouseUp, onMouseMove,
onMouseEnter, onMouseLeave, onMouseWheel, onContextMenu
```

**Drag & Drop:**
```
onDragStart, onDrag, onDragEnter, onDragOver, onDragLeave, onDrop, onDragEnd
```

**Keyboard:**
```
onKeyDown, onKeyUp, onKeyPress
```

**Focus:**
```
onFocus, onBlur
```

**Value/State:**
```
onChange, onSubmit, onSelect, onCheck, onToggle
```

**Lifecycle:**
```
onCreate, onDestroy, onShow, onHide, onResize, onMove
```

**Validation:**
```
onValidate, onError
```

**Component-Specific (RadianceViewport):**
```
onLoad, onCameraMove, onGaussianClick, onGaussianHover
```

**Key Files:**
- `runtime/ui/event_data.py` - DONE: UIEventData class
- `runtime/ui/renderer.py` - Wire events to all components
- `runtime/ui/components/*.py` - Add event emission

#### Phase 7B: Inspector Event Wiring UI

Visual UI for configuring event bindings in Inspector.

**Design:**
```
+-- Events ----------------------------------+
| onClick         [send_to_noodling v]       |
|                 Target: [red        v]     |
|                 Message: [input     v]     |
| [+ Add Event]                              |
+--------------------------------------------+
| onMouseEnter    [call_script v]            |
|                 [Edit Script...]           |
+--------------------------------------------+
```

**Actions Dropdown:**
- send_to_noodling
- call_script (inline or file)
- set_value
- show / hide / toggle_visible
- enable / disable
- focus
- emit (bubble to parent)
- navigate (change screen)
- play_sound
- open_url
- custom...

**Key Files:**
- `panels/inspector_ui_canvas.py` - Add events section
- `dialogs/script_editor_dialog.py` - NEW: Inline script editor
- `dialogs/event_binding_dialog.py` - NEW: Event configuration

#### Phase 7C: More Components

**Standard Delphi Components:**

| Component | Priority | Notes |
|-----------|----------|-------|
| `Checkbox` | HIGH | Boolean toggle with label |
| `RadioButton` | HIGH | Mutually exclusive selection |
| `RadioGroup` | HIGH | Container for RadioButtons |
| `Dropdown` | HIGH | ComboBox/select |
| `Slider` | HIGH | Numeric range (onSlideStart/End) |
| `ProgressBar` | MEDIUM | Determinate/indeterminate |
| `Memo` | MEDIUM | Multiline TextInput |
| `Image` | MEDIUM | Static image display |
| `SpinEdit` | MEDIUM | Numeric +/- buttons |
| `ListBox` | LOW | Scrollable selection list |
| `TreeView` | LOW | Hierarchical list |
| `TabView` | LOW | Tabbed panels |

**Key Files:**
- `runtime/ui/components/checkbox.py`
- `runtime/ui/components/dropdown.py`
- `runtime/ui/components/slider.py`
- etc.

#### Phase 7D: Custom Component System

Three ways to create custom components:

**1. Composite Components (YAML):**
```yaml
# components/login_form.yaml
type: CompositeComponent
name: LoginForm
properties:
  - name: title
    type: string
    default: "Login"
  - name: onLogin
    type: event

template:
  type: Panel
  children:
    - type: Label
      text: "${title}"
    - type: TextInput
      name: "usernameInput"
    - type: Button
      text: "Login"
      events:
        onClick:
          action: emit
          event: onLogin
```

**2. Python Components:**
```python
@register_component
class ColorPicker(UIComponent):
    component_type = "ColorPicker"
    PROPERTIES = {'value': {'type': 'color', 'default': '#ffffff'}}
    EVENTS = ['onChange', 'onPickStart', 'onPickEnd']
```

**3. Script Components (YAML + JS):**
```yaml
type: ScriptComponent
name: Counter
properties:
  - name: value
    type: number
script: |
  function increment() { self.value++; emit('onChange'); }
```

**Key Files:**
- `runtime/ui/composite_loader.py` - NEW: Load composite components
- `runtime/ui/component.py` - Add PROPERTIES/EVENTS class attributes
- Component scanner for `components/` folder

#### Phase 7E: RadianceViewport Event Integration

Wire RadianceViewport's Qt signals to UI event system.

```python
# In RadianceViewportWidget.mousePressEvent
hit_info = self.raycast(event.pos().x(), event.pos().y())
event_data = UIEventData(
    type="onGaussianClick",
    hitPosition=hit_info.get('position'),
    hitEntity=hit_info.get('entity_id'),
    hitSemantics=hit_info.get('semantics'),
)
self._dispatch_ui_event("onGaussianClick", event_data)
```

**Enables:**
```yaml
RadianceViewport:
  events:
    onGaussianClick:
      action: call_script
      script: |
        ui.set('infoLabel', event.hitSemantics.body_part);
```

**Key Files:**
- `runtime/ui/components/radiance_viewport.py` - Add UI event dispatch

#### Phase 7F: Script API Expansion

Expand `UIScriptExecutor` API:

```javascript
// Current
ui.get(name), ui.set(name, value)
ui.show(name), ui.hide(name), ui.toggle(name)
ui.enable(name), ui.disable(name)
event.type, event.source, event.value
console.log/warn/error

// New
ui.focus(name)                    // Set keyboard focus
ui.blur()                         // Remove focus
ui.emit(eventName, data)          // Bubble event to parent
ui.getComponent(name)             // Full component access
ui.animate(name, props, duration) // Simple animations

app.sendToNoodling(name, msg)     // Async noodling call
app.getNoodling(name)             // Get noodling state

audio.play(path)                  // Play sound
audio.stop()                      // Stop all audio

storage.get(key)                  // Persist data
storage.set(key, value)           // Local storage
```

**Key Files:**
- `runtime/ui/script_executor.py` - Expand API

### Data Types Reference

```yaml
# Primitives
string, number, boolean, null

# Colors
color: "#ff0000", "#ff000080", "rgb(255,0,0)", "rgba(255,0,0,0.5)"

# Geometry
point: {x: 100, y: 200}
size: {width: 300, height: 200}
rect: {x: 0, y: 0, width: 100, height: 100}
margins: {top: 8, right: 8, bottom: 8, left: 8}

# Collections
array: [1, 2, 3]
object: {key: "value"}

# Asset References
image: "assets/icon.png"
radiance: "noodlings/red/red.radiance"
noodling: "red"
component: "@chat_history"

# Special
font: {family: "Inter", size: 14, weight: "bold"}
anchor: [left, top, right, bottom]
```

---

## COMPLETED: UI Hierarchy Duplicate Fix (Jan 4, 2026)

**Issue:** UI components appeared twice in Stage hierarchy - once with type annotations `(Panel)`, once without.

**Root Cause:** In `_add_node_to_tree()`, UI components were added twice:
1. Via `_add_ui_component_to_tree()` (with type annotations)
2. Via `children_ids` iteration (without type annotations)

**Fix:** Skip `UI_COMPONENT` nodes in children loop (line 334 of `scene_hierarchy_refresh_mixin.py`):
```python
if child_node and child_node.node_type != SceneNodeType.UI_COMPONENT:
    self._add_node_to_tree(child_node, item)
```

---

## COMPLETED: UI Canvas Stage Integration (Jan 4, 2026)

UI Canvas Designer now integrates with Stage hierarchy like Unity's Canvas/GameObject pattern:

1. **UI entities appear in Stage hierarchy** - Like Unity, UI Canvas is a first-class entity
2. **Right-click context menu** - Rez > New UI Canvas, Add > Panel/Button/Label...
3. **Bidirectional selection** - Click UI in Stage -> loads in UI Canvas Editor

**Key Files:**
- `panels/scene_hierarchy_ui_mixin.py` - UI canvas CRUD operations
- `panels/scene_hierarchy_context_menu_mixin.py` - UI context menu items
- `panels/ui_canvas_editor_panel.py` - Design surface

---

## COMPLETED: Phase 4 - Build System (Jan 3, 2026)

**"File > Build Application..." is live!** (Ctrl+B)

| File | Purpose |
|------|---------|
| `appbuilder/builder.py` | Orchestrator - validates, packages, bundles |
| `appbuilder/packager.py` | Asset collection with filtering |
| `appbuilder/bundler_macos.py` | Creates .app bundles |

---

## COMPLETED: Phase 3 - UI Canvas System (Jan 2-3, 2026)

Delphi-style canvas at `noodlestudio/runtime/ui/`:

**Components:** Panel, Label, Button, TextInput, ChatHistory, ChatInput, RadianceViewport

**Event System:**
- `UIEventDispatcher` - Routes events to handlers
- `UIScriptExecutor` - QuickJS sandbox
- `BindingManager` - Reactive property bindings

**Tests:** 136 tests in `test_ui_canvas.py`

---

## BACKLOG

### Dashboard UI Widgets
Research existing Qt gauge/meter libraries before building from scratch.
Goal: Professional instrumentation widgets (gauges, meters, LEDs, seven-segment displays)
for building science-fair style AI dashboards driven by facets/scripts.
- Look into: PyQtGraph, qt-material, QML gauge components
- Mercedes-style dashboard aesthetic

### Inspector UX
- Unity-style numeric drag-to-scroll on labels

### Undo/Redo for UI Edits
- `AddUIComponentCommand`, `DeleteUIComponentCommand`
- `MoveUIComponentCommand`, `ResizeUIComponentCommand`

### Additional Build Targets
- Windows .exe (PyInstaller)
- Linux binary
- Docker container

### Admin Dashboard - Issue Credits UI
Add "Adjust Credits" button on user detail page.
File: `backend/admin-dashboard/src/routes/users/[id]/+page.svelte`

### Asset-Aware Inspector
Inspector shows contextual info when selecting assets.

### Trained Gaussian Quality
OpenSplat-trained Gaussians have background artifacts.

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
PYTHONPATH=.:../.. pytest              # Run all (~130 tests)
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

The venv contains all dependencies (PyQt6, pytest, etc.). Always activate before running tests or the application.

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
| **LLM Client** | `noodlestudio/runtime/llm_client.py` |
| **Runtime CLI** | `noodlestudio/runtime/cli.py` |
| **UI Canvas (runtime)** | `noodlestudio/runtime/ui/` |
| **UIEventData** | `runtime/ui/event_data.py` |
| **EventEmitting widgets** | `runtime/ui/event_widgets.py` |
| **UI Event Dispatcher** | `runtime/ui/event_dispatcher.py` |
| **UI Script Executor** | `runtime/ui/script_executor.py` |
| **UI Bindings** | `runtime/ui/bindings.py` |
| **UI Components** | `runtime/ui/components/` |
| **UI Canvas Designer** | `panels/ui_canvas_editor_panel.py` |
| **UI Inspector Mixin** | `panels/inspector_ui_canvas.py` |
| **Event Binding Widget** | `widgets/event_binding_widget.py` |
| **Script Editor Dialog** | `dialogs/script_editor_dialog.py` |
| **Build system** | `noodlestudio/appbuilder/` |
| **Runtime module** | `noodlestudio/runtime/` |
| **Facet editor** | `panels/facets_editor_panel.py` |
| **Scene hierarchy** | `panels/scene_hierarchy.py` |
| **Cognitive Cycles Panel v2** | `panels/cognitive_cycles_panel_v2.py` |
| **Cognition Monitor** | `core/cognition_monitor.py` |
| **Channel Bus** | `runtime/channels.py` |
| **World Channels** | `runtime/world_channels.py` |
| **Brenda Director** | `runtime/brenda.py` |

---

## Project Context

**Creator:** Caitlyn (Unity employee #12, Asset Store creator)
**Location:** Garcia River Forest cabin
**Hardware:** M3 Ultra 512GB

**Mission:** Open-source alternative to "Consciousness-as-a-Service"

---

## Completed Systems

- **LLM Router Wiring** - Let's Consciousness! project talks via NoodleROUTER
- **Brenda Stage Director** - Invisible director orchestrating .play.yaml performances
- **World Channels** - Environmental context broadcasting (time, weather, ambiance, events)
- **Channel Architecture** - Named message buses for inter-noodling communication
- **More Components** - Checkbox, Dropdown, Slider, RadioButton, RadioGroup (Phase 7C)
- **Inspector Event Wiring UI** - Delphi Object Inspector Events tab (Phase 7B)
- **Full Event Model** - UIEventData, EventEmitting widgets (Phase 7A)
- **UI Canvas System** - Delphi-style components, events, bindings (Phase 3)
- **UI Canvas Designer** - Visual drag-drop editor, Stage integration (Phase 4/6)
- **UI Hierarchy Fix** - Duplicate component display resolved
- **Build System** - File > Build Application (macOS .app bundles)
- **Runtime Foundation** - Headless execution, CLI, multi-provider LLM
- **NoodleROUTER** - api.noodlings.ai/v1/chat/completions (live)
- **GPU Gaussian Rendering** - 120 FPS via gsplat-mps
- **Admin Dashboard** - admin.noodlings.ai (live)
- **Crash Recovery** - Sentinel file detection
- **Cloud Account System** - OAuth, credits, billing
- Multi-provider LLM (8 providers)
- Neural Canvas with PyTorch test mode
- Scriptability API (context.noodle)
- MCP integration
- Utility facets (31 types)
- Multimodal facets (audio, vision, image gen)

---

**Ordnung muss sein!**
