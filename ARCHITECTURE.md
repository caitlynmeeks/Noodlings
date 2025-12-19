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
    ├── commands.py                 # Command parser (5654 lines - LARGE)
    │   └── fuzzy_match.py
    │
    ├── agent_bridge.py             # Agent lifecycle (6028 lines - VERY LARGE)
    │   ├── llm_interface.py        # LLM abstraction
    │   ├── cognitive_components.py # Transistor system (2989 lines)
    │   ├── noodling_components.py  # Component system
    │   └── autonomous_cognition.py # Background thought
    │
    ├── api_server.py               # REST endpoints (2629 lines)
    │   └── session_profiler.py
    │
    ├── semantic_integration.py     # Scene protocol bridge (NEW)
    ├── scene_protocol_integration.py
    └── project_bridge.py           # NoodleStudio integration (NEW)
```

### Cognitive Architecture (Two Patterns - NEEDS CONSOLIDATION)

**Pattern A: Transistor System** (`cognitive_components.py`)
- 21+ transistor types (Cultural, Personality, Mood, Affect, Memory, etc.)
- Each transistor filters/modifies signal flow
- Unity-like `from_config()` factory pattern

**Pattern B: Component System** (`noodling_components.py`)
- NoodlingComponent base class
- Character Voice, Intuition Receiver, Social Detector
- Introspection-focused

**Decision Needed:** Consolidate into single pattern or document when to use each.

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

| File | Lines | Refactoring Strategy |
|------|-------|---------------------|
| `agent_bridge.py` | 6028 | Split: lifecycle, consciousness, affect, state |
| `commands.py` | 5654 | Split: parser, movement, communication, building |
| `inspector_panel.py` | 3749 | Split: property_editor, physics_editor, component_editor |
| `facets_editor_panel.py` | 3459 | Split: node_editor, wire_layout, execution_viz |
| `cognitive_components.py` | 2989 | Split by transistor category |
| `main_window.py` | 2710 | Extract: menu_factory, panel_factory |
| `api_server.py` | 2629 | Split: rest_endpoints, websocket_handlers |

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

### Confirmed Obsolete

| File | Reason | Replacement |
|------|--------|-------------|
| `model_manager_panel.py` | Ollama-only, superseded | `model_manager_panel_v2.py` |
| `noodlings_api.py` | Legacy script API | `noodle_api.py` |
| `claude_client.py` | Old Claude integration | `llm_interface.py` |
| `claude_chat.py` | Old Claude integration | `llm_interface.py` |
| `claude_interact.py` | Old Claude integration | `llm_interface.py` |
| `claude_testing.py` | Old Claude integration | `llm_interface.py` |

### Probably Obsolete (Verify Before Removing)

| File | Reason | Check |
|------|--------|-------|
| `spawn_yuki.py` | Demo script | Is it referenced anywhere? |
| `spock_spawns_yuki.py` | Demo script | Is it referenced anywhere? |
| `spock_via_api.py` | Demo script | Is it referenced anywhere? |
| `script_manager.py` | Comments say unused | Verify no imports |
| `llm_client_router.py` | Possibly redundant | Check if used |

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

### December 18, 2025
- Initial architecture survey
- Identified 7 mega-files for refactoring
- Found 6+ obsolete files for removal
- Documented 15+ files with bare exception clauses
- Created standardization guidelines

---

**Ordnung muss sein!**
