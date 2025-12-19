# ARCHITECTURE.md

Noodlings Multi-Timescale Affective Agents - Technical Architecture Reference

**Generated:** December 18, 2025
**Purpose:** Reference for maintaining code quality and consistency during organic development

---

## Project Overview

```
noodlings_clean/                      (~129,000 lines Python)
├── applications/
│   ├── cmush/                        (109 files, ~49.5K lines)
│   │   └── MUD server + cognition engine
│   └── noodlestudio/                 (115 files, ~80K lines)
│       └── PyQt6 IDE for cognitive architecture design
├── facet_assemblies/                 (Shared YAML topologies)
└── docs/                             (Documentation)
```

---

## 1. CMUSH Architecture

### Entry Points

| Entry Point | Port | Purpose |
|-------------|------|---------|
| `server.py` | 8080 (HTTP), 8765 (WS) | Main MUD server |
| `api_server.py` | 8081 | REST API (NoodleScope, spatial ops) |
| `start.sh` | - | Launch script |

### Core Module Map

```
server.py (main)
    ├── world.py                    # World state, rooms, objects
    ├── auth.py                     # User authentication
    │
    ├── commands.py                 # Command parser (4220 lines)
    │   ├── brenda_commands.py      # BRENDA mixin (1235 lines)
    │   └── fuzzy_match.py
    │
    ├── agent_bridge.py             # Agent lifecycle (2592 lines)
    │   ├── agent_perception.py     # Perception mixin (1211 lines)
    │   ├── agent_response.py       # Response generation mixin (728 lines)
    │   ├── agent_cognition.py      # Cognition loop mixin (490 lines)
    │   ├── agent_state.py          # State persistence mixin (317 lines)
    │   ├── llm_interface.py        # LLM abstraction
    │   ├── noodling_components.py  # Component system
    │   └── autonomous_cognition.py # Background thought
    │
    ├── api_server.py               # REST endpoints (2392 lines)
    │   └── session_profiler.py
    │
    ├── scene_protocol_integration.py  # Scene Protocol bridge
    │   ├── sync_agent_to_noodling()   # Agent → SceneStateManager
    │   ├── sync_player_to_scene()     # Player → SceneStateManager
    │   ├── record_dialogue()          # Speech → narrative context
    │   ├── prepare_facet_context()    # Generate perception slice
    │   └── finalize_facet_context()   # Process WorldAPI commands
    │
    └── project_bridge.py              # Project format → legacy World
```

### Cognitive Architecture

**Facet System** (canonical)
- Visual node-based cognitive architecture
- Facet assemblies stored as YAML (Unity prefab model)
- See `noodlestudio/core/facet_system.py` for data model
- See `noodlestudio/core/facet_executor.py` for execution

**Legacy Component System** (`noodling_components.py`)
- NoodlingComponent base class
- Character Voice, Intuition Receiver, Social Detector
- Introspection-focused, may be merged into facets

**Note:** Transistor system (cognitive_components.py) was removed December 18, 2025.

### LLM Integration

```
llm_interface.py (CANONICAL)
    ├── llm_client_router.py        # Multi-provider routing
    └── providers/
        ├── ollama_client.py
        ├── anthropic_client.py
        └── openrouter_client.py

LEGACY (remove):
    ├── claude_client.py
    ├── claude_chat.py
    ├── claude_interact.py
    └── claude_testing.py
```

---

## 2. NoodleStudio Architecture

### Directory Structure

```
noodlestudio/
├── core/                           # Application logic
│   ├── main_window.py              # Primary UI (2710 lines)
│   ├── project_manager.py          # Project I/O
│   ├── facet_system.py             # Facet data model
│   ├── facet_executor.py           # Execution engine (1384 lines)
│   ├── provider_manager.py         # LLM providers
│   ├── model_label_manager.py      # Label->model mapping
│   │
│   ├── *_facet.py                  # 15 facet implementations
│   │   ├── charm_network_facet.py  # LSTM/GRU neural
│   │   ├── scripted_facet.py       # JS/Python sandbox
│   │   ├── audio_stream_facet.py   # Voice I/O
│   │   ├── vision_facet.py         # Image understanding
│   │   └── ...
│   │
│   ├── neural_canvas/              # Node-based ML editor
│   │   ├── neural_node.py
│   │   ├── neural_graph.py
│   │   ├── node_definitions.py     # 26 node types
│   │   └── test_executor.py        # PyTorch inference
│   │
│   └── semantic_world/             # Scene protocol (WELL ORGANIZED)
│       ├── scene_packet.py         # Data structures
│       ├── scene_state_manager.py  # Canonical truth
│       ├── perception.py           # FOV filtering
│       └── scene_emitter.py        # Output streaming
│
├── panels/                         # UI panels
│   ├── inspector_panel.py          # Property editor (3749 lines - LARGE)
│   ├── facets_editor_panel.py      # Node editor (3459 lines - LARGE)
│   ├── scene_hierarchy.py          # Scene tree (2008 lines)
│   ├── model_manager_panel_v2.py   # Model browser (CURRENT)
│   ├── model_manager_panel.py      # LEGACY - REMOVE
│   └── ...
│
├── scripting/                      # Scripting API (context.noodle)
│   ├── noodle_api.py               # Main API (CURRENT)
│   ├── noodlings_api.py            # LEGACY - REMOVE
│   ├── models_api.py
│   ├── neural_api.py
│   ├── agents_api.py
│   ├── quantum_api.py
│   ├── audio_api.py
│   ├── vision_api.py
│   ├── cloud_api.py
│   └── world_api.py
│
├── data/                           # Data models
├── dialogs/                        # Modal dialogs
└── widgets/                        # Reusable UI components
```

### Facet Execution Pipeline

```
INCOMING (input)
    ↓
CHARM_NET (fast temporal: LSTM/GRU)
    ↓
CONTEXT_INTELLIGENCE (social reasoning)
    ↓
[Cognitive Facets] → personality, perception, memory
    ↓
[Character Layers] → speech modulation, action filtering
    ↓
OUTGOING (response)
```

### Scripting API (context.noodle)

Available in ScriptedFacets:
```javascript
context.noodle.models.*      // LLM configuration
context.noodle.neural.*      // Neural canvas
context.noodle.agents.*      // Facet assemblies
context.noodle.quantum.*     // IBM Quantum
context.noodle.audio.*       // Voice I/O
context.noodle.vision.*      // Image understanding
context.noodle.cloud.*       // Cloud sync
context.noodle.world.*       // Entity transforms
```

---

## 3. Data Flow Patterns

### Scene Protocol Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        cMUSH SERVER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  World Events (say, enter, exit, etc.)                          │
│       │                                                          │
│       ▼                                                          │
│  broadcast_event()                                               │
│       │                                                          │
│       ├──► SemanticWorld (legacy event logging)                  │
│       │                                                          │
│       └──► SceneStateManager (canonical spatial truth)           │
│                 │                                                │
│                 ├── sync_agent_to_noodling()                     │
│                 ├── sync_player_to_scene()                       │
│                 ├── record_dialogue()                            │
│                 └── update position on enter/exit                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     FACET EXECUTION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  prepare_facet_context(agent_id)                                │
│       │                                                          │
│       ▼                                                          │
│  generate_perception_slice(agent_id)                            │
│       │                                                          │
│       ├── FOV filtering (can't see behind)                      │
│       ├── Range filtering (audibility)                          │
│       └── External observables only (no internal affect)        │
│                                                                  │
│       ▼                                                          │
│  Facet Assembly receives filtered world context                  │
│       │                                                          │
│       ▼                                                          │
│  finalize_facet_context() - process WorldAPI commands            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `cmush/scene_protocol_integration.py` - Bridge functions
- `noodlestudio/core/semantic_world/scene_state_manager.py` - Canonical truth
- `noodlestudio/core/semantic_world/perception.py` - Perception filtering

### Configuration Sources (NEEDS STANDARDIZATION)

| Source | Used For | Files |
|--------|----------|-------|
| Environment vars | API keys, paths | `.env`, `os.environ` |
| QSettings | UI preferences, labels | `model_label_manager.py` |
| YAML files | Facet assemblies, recipes | `facet_assemblies/*.yaml` |
| JSON files | World state, agents | `world/*.json` |
| Hardcoded | Defaults, constants | Various |

**Recommendation:** Create unified `config.py` module.

### Event Systems

| System | Purpose | Location |
|--------|---------|----------|
| `event_system.py` | Pub/sub for cmush | `cmush/` |
| `execution_event_bus.py` | Facet execution events | `noodlestudio/core/` |
| Qt Signals | UI updates | Throughout panels |

### Persistence

| Data Type | Format | Location |
|-----------|--------|----------|
| World state | JSON | `cmush/world/` |
| Facet assemblies | YAML | `facet_assemblies/` |
| Projects | YAML | `~/Documents/noodlings/` |
| UI layout | QSettings | OS-specific |
| Model labels | QSettings | OS-specific |

---

## 4. Code Smell Inventory

### HIGH PRIORITY - Fix Soon

#### Bare Exception Clauses (15+ files)
```python
# BAD - swallows all exceptions
try:
    data = await request.json()
except:
    return web.json_response({'error': 'Invalid JSON'}, status=400)

# GOOD - specific exception
try:
    data = await request.json()
except (json.JSONDecodeError, ValueError) as e:
    logger.warning(f"Invalid JSON: {e}")
    return web.json_response({'error': 'Invalid JSON'}, status=400)
```

**Files to fix:**
- `api_server.py` (7 instances)
- `scripting/neural_api.py` (8 instances)
- `scripting/agents_api.py` (7 instances)
- `scripting/models_api.py` (2 instances)
- `main_window.py` (5 instances)

#### Mega-Files (>2000 lines)

| File | Lines | Status |
|------|-------|--------|
| `inspector_panel.py` | 3749 | Split: property_editor, physics_editor, component_editor |
| `facets_editor_panel.py` | 3459 | Split: node_editor, wire_layout, execution_viz |
| `main_window.py` | 2710 | Extract: menu_factory, panel_factory |
| `api_server.py` | 2392 | Split: rest_endpoints, websocket_handlers |

**Refactored (December 19, 2025):**
| File | Before | After | Mixins |
|------|--------|-------|--------|
| `agent_bridge.py` | 5168 | 2592 | perception, response, cognition, state |
| `commands.py` | 5402 | 4220 | brenda_commands |

*Note: cognitive_components.py (2989 lines) was deleted Dec 18, 2025.*

### MEDIUM PRIORITY - Clean Up

#### Disabled Debug Code
**Location:** `inspector_panel.py:2536-2599`
- 60+ lines of disabled event handlers
- Should be removed or re-enabled with documentation

#### Print Statements (989 in cmush/)
Replace with logger:
```python
# BAD
print(f"Agent {agent_id} processing...")

# GOOD
logger.info(f"Agent {agent_id} processing...")
```

#### Path Management
Replace hardcoded paths:
```python
# BAD
sys.path.insert(0, '/Users/thistlequell/git/noodlings_clean/applications/cmush')

# GOOD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### LOW PRIORITY - Technical Debt

- 100+ TODO/FIXME markers (document or address)
- Empty `pass` statements in exception handlers (add logging)
- Multiple YAML loading implementations (create utility module)

---

## 5. Files to Remove

### Already Removed (December 18, 2025)

- `cognitive_components.py` - Transistor system (2989 lines)
- `model_manager_panel.py` - Replaced by v2
- `spawn_yuki.py`, `spock_spawns_yuki.py`, `spock_via_api.py` - Demo scripts
- `test_affect_transistor.py`, `test_transistor_loading.py`, `test_cognitive_manifold.py`,
  `test_intuition_flow.py`, `test_affect_integration.py` - Transistor tests

### Also Removed (December 18, 2025)

- `claude_client.py`, `claude_chat.py`, `claude_interact.py` - Standalone utility scripts

### Still Active (DO NOT remove)

| File | Reason |
|------|--------|
| `claude_testing.py` | Used by test_debug.py, test_simple_message.py, etc. |
| `script_manager.py` | Imported by server.py and agent_bridge.py |
| `llm_client_router.py` | Base classes for all LLM provider clients |
| `noodlings_api.py` | Used by script_executor.py |

### Archive Candidates

| Directory | Reason |
|-----------|--------|
| `experiments/` | 14 Python + 24 JSON files |
| | Historical research, not actively used |
| | Consider moving to `archive/experiments/` |

---

## 6. Standardization Guidelines

### Logging

```python
# Standard pattern for all modules
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed trace info")
logger.info("Normal operation")
logger.warning("Unexpected but handled")
logger.error("Error requiring attention")
```

### Exception Handling

```python
# Always catch specific exceptions
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Handle or re-raise
except AnotherError as e:
    logger.warning(f"Recoverable issue: {e}")
    result = fallback_value
```

### Configuration

```python
# Prefer environment variables for secrets
api_key = os.environ.get("OPENAI_API_KEY")

# Use QSettings for user preferences
settings = QSettings("Noodlings", "NoodleStudio")
theme = settings.value("theme", "dark")

# Use YAML for structured data
with open(path, 'r') as f:
    config = yaml.safe_load(f)
```

### Factory Methods

```python
# Standard pattern for configurable objects
@classmethod
def from_config(cls, config: Dict[str, Any]) -> 'MyClass':
    """Create instance from configuration dict."""
    return cls(
        param1=config.get('param1', default1),
        param2=config.get('param2', default2),
    )
```

### File Organization

```
module/
├── __init__.py         # Public API exports
├── base.py             # Base classes
├── impl_a.py           # Implementation A
├── impl_b.py           # Implementation B
└── utils.py            # Shared utilities
```

---

## 7. Dependency Graph

### External Dependencies (Critical)

| Package | Version | Used For |
|---------|---------|----------|
| PyQt6 | 6.x | NoodleStudio UI |
| aiohttp | 3.x | Async HTTP/WS |
| numpy | 1.x | Array operations |
| mlx | 0.x | Neural networks (Apple Silicon) |
| PyYAML | 6.x | Configuration |
| python-dotenv | 1.x | Environment loading |

### Internal Dependency Layers

```
Layer 4: UI (panels/, widgets/, dialogs/)
    ↓
Layer 3: Application (main_window.py, project_manager.py)
    ↓
Layer 2: Core (facet_system.py, facet_executor.py, semantic_world/)
    ↓
Layer 1: Providers (llm_interface.py, providers/, *_clients.py)
    ↓
Layer 0: Utilities (event_system.py, entropy_service.py)
```

**Rule:** Higher layers can import lower layers, not vice versa.

---

## 8. Testing Strategy

### Current State

| Location | Files | Purpose |
|----------|-------|---------|
| `cmush/test_*.py` | 24 | Unit + integration tests |
| `cmush/experiments/` | 14 | Research experiments |
| `noodlestudio/test_*.py` | 6 | API tests |

### Recommended Structure

```
tests/
├── cmush/
│   ├── test_agent_bridge.py    # MISSING - critical
│   ├── test_commands.py        # MISSING - critical
│   ├── test_api_server.py      # MISSING - critical
│   └── ...existing tests...
├── noodlestudio/
│   ├── test_facet_executor.py  # MISSING - critical
│   └── ...existing tests...
└── integration/
    └── test_end_to_end.py      # MISSING
```

---

## 9. Quick Reference

### Starting the Server

```bash
cd applications/cmush
./start.sh
# Or toggle in NoodleStudio status bar
```

### Ports

| Port | Service |
|------|---------|
| 8080 | HTTP (web interface) |
| 8765 | WebSocket (MUD) |
| 8081 | REST API (NoodleScope) |
| 11434 | Ollama |

### Key Directories

| Path | Contents |
|------|----------|
| `applications/cmush/world/` | Legacy world state |
| `applications/noodlestudio/library/` | Sample projects |
| `facet_assemblies/` | Shared YAML topologies |
| `~/Documents/noodlings/` | User projects |

### Common Operations

**Add new facet type:**
1. Create `my_facet.py` in `core/`
2. Implement `execute()` method
3. Register in `facet_executor.py`
4. Add to node palette in `facets_editor_panel.py`

**Add new REST endpoint:**
1. Add route in `api_server.py:setup_routes()`
2. Implement handler method
3. Add to API documentation

**Add scripting API method:**
1. Add to appropriate `*_api.py` in `scripting/`
2. Wire in `noodle_api.py`
3. Update docs

---

## 10. Changelog

### December 19, 2025 (Afternoon)
- **SCENE PROTOCOL WIRING** - Connected Scene Protocol to cMUSH server
  - `server.py` - Agent/player sync, dialogue recording, movement tracking
  - `scene_protocol_integration.py` - Fixed Zone constructor (ZoneBounds)
  - `agent_cognition.py` - Added SCENE_PROTOCOL_AVAILABLE import
  - Data flow: World events → SceneStateManager → Perception slices → Facets
  - Each noodling now receives perception-filtered world context

### December 19, 2025 (Morning)
- **MIXIN EXTRACTION** - Major refactoring for maintainability
  - `agent_bridge.py`: 5168 → 2592 lines (50% reduction)
    - Extracted `agent_perception.py` (1211 lines) - perceive_event, cognitive gate
    - Extracted `agent_response.py` (728 lines) - response generation, conscience
    - Extracted `agent_cognition.py` (490 lines) - cognition loop, intuition
    - Extracted `agent_state.py` (317 lines) - state persistence
  - `commands.py`: 5402 → 4220 lines (22% reduction)
    - Extracted `brenda_commands.py` (1235 lines) - BRENDA system
- Mixin pattern preserves class behavior while splitting across files

### December 18, 2025 (Evening)
- **TRANSISTOR SYSTEM REMOVED** - Major cleanup (~3800 lines)
  - Deleted `cognitive_components.py` (2989 lines)
  - Cleaned `agent_bridge.py` (6028 → 5260 lines)
  - Cleaned `api_server.py` (2629 → 2392 lines)
  - Removed transistor API endpoints and methods
  - Deleted 5 transistor test files
  - Deleted 3 legacy demo scripts
  - Deleted obsolete `model_manager_panel.py`
- Facets are now the only cognitive architecture

### December 18, 2025 (Morning)
- Initial architecture survey
- Identified 7 mega-files for refactoring
- Found 6+ obsolete files for removal
- Documented 15+ files with bare exception clauses
- Created standardization guidelines

---

**Ordnung muss sein!**
