# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: December 19, 2025

**FOR NEXT CLAUDE: START HERE!**

---

## 🎯 NEXT SESSION: Neural Canvas Tutorials

### Task: Test & Implement Neural Canvas Tutorials
- Test Scene Protocol wiring (verify perception slices work)
- Review tutorial spec: `/docs/NEURAL_CANVAS_TUTORIALS.md`
- Implement missing nodes identified in spec
- Build first tutorial: Logic Gates with Neural Nodes

### Reference
- 10 tutorials across 5 progressive levels
- Missing nodes prioritized in spec
- Implementation phases documented

---

## ✅ COMPLETED (December 19, 2025)

### Scene Protocol Wiring to Server - DONE!
**Task:** Integrate Scene Protocol with cMUSH server for perception-filtered context

**Files Modified:**
- `server.py` - Added agent/player sync, dialogue recording, movement tracking
- `scene_protocol_integration.py` - Fixed Zone constructor (ZoneBounds), added imports
- `agent_cognition.py` - Added SCENE_PROTOCOL_AVAILABLE import for mixin

**What Was Wired:**
1. **Agent sync on load** (`server.py:529-538`)
   - After `agent_manager.create_agent()`, calls `sync_agent_to_noodling()`
   - Each agent appears in SceneStateManager with name, species, room

2. **Player sync on login** (`server.py:577-586`)
   - After successful login, calls `sync_player_to_scene()`
   - Players tracked in SceneStateManager with their room

3. **Dialogue recording** (`server.py:1156-1158`)
   - All `say` events call `scene_record_dialogue()`
   - Builds narrative context for perception slices

4. **Movement sync** (`server.py:1159-1174`)
   - `enter` events update zone for both agents and players
   - Keeps SceneStateManager synchronized with world state

**Data Flow:**
```
World Events → broadcast_event() → SceneStateManager
                                          ↓
                          generate_perception_slice(agent_id)
                                          ↓
                      Facet Assembly (via prepare_facet_context)
```

**Already Wired (discovered during investigation):**
- Facet execution wiring (`agent_cognition.py`, `agent_perception.py`)
- PROJECT_PATH env var (`main_window.py:358-360`)
- AssetsPanel structure (loads Noodlings/Stages/Prims/Generations)
- scene_hierarchy.py → Stages/Instances loading
- project_bridge.py - Complete bridge from new format to legacy World

### Codebase Cleanup & Rebranding
- **Consilience → Noodlings**: Renamed all classes and references
  - `CMUSHConsilienceAgent` → `CMUSHNoodlingAgent`
  - `ConsilienceAgentWithObservers` → `NoodlingAgentWithObservers`
  - Updated imports, docstrings, comments throughout
- **consciousness → charm**: Updated user-facing terminology
  - "hierarchical affective consciousness" → "hierarchical affective charm"
  - Disclaimer: "We explore functional correlates, not metaphysical claims"
- **Author lines**: "Consilience Project" → "Caitlyn Meeks"
- **Checkpoint paths**: `consilience_core/checkpoints_phase4/` → `models/checkpoints/`
  - Copied `best.npz` from old location to new
  - Updated 14 files with new paths
- **Removed dead code**:
  - Deleted 2 backup files
  - Removed `consilience_core` sys.path inserts
  - Cleaned up `.gitignore` (removed KINDLED_TERMINOLOGY.md)
- **Fixed hardcoded paths**: `launch_with_log.sh` now uses relative paths
- **Production code prints**: Already clean (only test blocks remain)

### Mixin Extraction - Mega-file Refactoring
- **agent_bridge.py**: 5,168 → 2,592 lines (50% reduction)
  - `agent_perception.py` (1,211 lines) - perceive_event, cognitive gate
  - `agent_response.py` (728 lines) - response generation, conscience
  - `agent_cognition.py` (490 lines) - cognition loop, intuition
  - `agent_state.py` (317 lines) - state persistence
- **commands.py**: 5,402 → 4,220 lines (22% reduction)
  - `brenda_commands.py` (1,235 lines) - BRENDA natural language system
- Pattern: Python mixin classes via multiple inheritance
- Updated ARCHITECTURE.md with new structure

---

## ✅ COMPLETED (December 18, 2025)

### Major Cleanup: Transistor System Removal (~3,800 lines)
- Deleted `cognitive_components.py` (2,989 lines)
- Cleaned `agent_bridge.py` (6,028 → 5,260 lines)
- Cleaned `api_server.py` (2,629 → 2,392 lines)
- Removed all transistor API endpoints and methods
- Deleted 5 transistor test files, 3 legacy demo scripts
- Deleted obsolete `model_manager_panel.py`, `claude_client.py`, `claude_chat.py`, `claude_interact.py`
- Fixed 33 bare `except:` clauses across 7 files
- **Facets are now the only cognitive architecture**

### Spatial Operations REST API
- Transform endpoints: GET/POST/PATCH `/api/entities/{id}/transform`
- Material endpoint: POST `/api/entities/{id}/material`
- Physics endpoint: PATCH `/api/entities/{id}/physics`
- Metadata endpoints: GET/POST/DELETE `/api/entities/{id}/properties/{key}`
- Batch transforms: POST `/api/entities/batch/transform`
- WorldAPI scripting methods for ScriptedFacets
- WebSocket broadcast for transform updates

### Scene Hierarchy & Project System
- UUID field now read-only in Inspector
- Auto-select new items on creation
- Fixed context menus for Props/Instances/Zones
- Removed legacy mode (project-only now)
- Auto-generated names for new entities

---

## 🎯 CURRENT: Noodlings Scene Protocol (NSP) - Complete!

**Goal:** Protocol for providing semantic scene truth to stateless generative rendering engines (Google Genie, Mirage, etc.)

### The Core Insight

**Genie is stateless. Noodlings is stateful.**

Generative 3D engines render frames without memory. We provide:
- **Persistent state** (who's where, what happened, relationships)
- **Character consistency** (reference art per form/state)
- **Narrative memory** (context that a stateless generator lacks)
- **Perception-filtered context** (each noodling only knows what they perceive)

**Text, 2D maps, 3D renders are all projections of the same semantic truth.**

### What Was Done (December 18, 2025)

**SCENE_PROTOCOL_SPEC.md** (~1200 lines) - Complete protocol specification:
- Scene Packet structure (header, spatial truth, entities, references, narrative, camera)
- Multi-state character support (Yuki: ghostly_fox / normal_fox / humanoid_fox)
- Camera directive language (POV, FOCUS_ON, TWO_SHOT, ESTABLISH, etc.)
- Perception slices for information asymmetry
- Transport/encoding (JSON, WebSocket, REST)
- Text flattening for LLM-based renderers

**Implementation** (`noodlestudio/core/semantic_world/`):

1. **scene_packet.py** (~650 lines) - Data structures:
   - `ScenePacket` - Complete scene snapshot
   - `Noodling`, `Player`, `Prim` - Entity types with full state
   - `VisualForm` - Multi-state character support with reference images
   - `PerceptionCone` - Per-entity FOV, range, special senses
   - `CameraDirective` - Cinematography instructions
   - `Affect` - 5D continuous affect (PAD + boredom + sorrow)

2. **perception.py** (~450 lines) - Perception filtering:
   - `PerceptionSlice` - Filtered view per entity
   - `PerceptionCalculator` - FOV cone calculations, audibility
   - `PerceptionSliceGenerator` - Generates slices from full packets
   - Information asymmetry: entities only know what they perceive

3. **scene_state_manager.py** (~550 lines) - Canonical truth:
   - Maintains authoritative world state
   - Entity CRUD operations
   - Dialogue/event recording
   - Camera control
   - Generates packets and perception slices

4. **scene_emitter.py** (~400 lines) - Output streaming:
   - Full/delta/camera-only packet emission
   - WebSocket adapter for connected renderers
   - Genie adapter (transforms to Genie format)
   - Configurable emission rates

### Key Architecture

```
SCENE STATE MANAGER (canonical truth)
        │
        ├──────────────────┬────────────────────┐
        │                  │                    │
        ▼                  ▼                    ▼
   Red's Slice        Yuki's Slice         Full Packet
   (her FOV only)     (her FOV only)       (everything)
        │                  │                    │
        ▼                  ▼                    ▼
   Red's Facets       Yuki's Facets        Genie/Mirage
   (cognition)        (cognition)          (rendering)
```

### Perception Features

- **FOV filtering** - Can't see entities behind you
- **Range filtering** - Can't see/hear beyond perception range
- **Audibility** - Whispers don't carry far
- **External observables only** - See posture/expression, NOT internal affect
- **Special senses**:
  - Fox (Yuki): 180 FOV, night vision, motion sensitivity
  - Fire imp (Red): Heat sense through occlusion, smoke detection
  - Ghost form: 360 awareness, sees through walls

### Files Created/Modified

**New files:**
- `SCENE_PROTOCOL_SPEC.md` - Complete protocol specification
- `noodlestudio/core/semantic_world/scene_packet.py` - Data structures
- `noodlestudio/core/semantic_world/perception.py` - Perception system
- `noodlestudio/core/semantic_world/scene_state_manager.py` - State manager
- `noodlestudio/core/semantic_world/scene_emitter.py` - Output streaming
- `noodlestudio/core/semantic_world/__init__.py` - Updated exports

### Next Steps

1. **Wire into server.py** - Build perception slices for facet context
2. **2D Spatial Editor** - Illustrated map view using the same semantic data
3. **Genie Integration** - Test with actual Genie/Mirage APIs
4. **Reference asset pipeline** - Auto-extract from noodling definitions

### Character Note

Red is a **fire imp** like the Cheat Code fire imps from Conker's Bad Fur Day - mischievous little chaos agents, not a dragon!

---

## Previous: Spatial View Panel (Qt Quick 3D)

---

## Previous: Wire Up New Project System

### What Was Done (December 17, 2025)
- **PROJECT_SPEC.md** - Complete specification for project architecture
- **project_manager.py** - Fully rewritten to implement spec
- **project_migrator.py** - Migration tool for legacy data
- **main_window.py** - Updated menus (New Noodling/Stage/Prim, Migration tool)
- **Help menu cleaned** - Removed placeholder items

### Current Project Structure (PROJECT_SPEC.md)
```
MyProject/
├── project.noodleproj          # Project manifest
├── Noodlings/                  # Reusable character prefabs
│   └── red/
│       ├── noodling.yaml       # Master manifest
│       ├── recipe.yaml         # Character definition
│       ├── assembly.yaml       # Facet topology
│       ├── charm_weights.npz   # Trained weights
│       ├── Scripts/            # ScriptedFacets
│       ├── NeuralGraphs/       # .nncanvas files
│       └── Assets/             # Multimodal content
├── Prims/                      # Reusable prop templates
│   └── radio/
│       ├── prim.yaml
│       └── Scripts/
├── Stages/                     # Scenes with continuous space
│   └── the_nexus/
│       ├── stage.yaml
│       ├── Zones/              # Soft attention regions
│       ├── Instances/          # Live agent instances
│       └── Props/              # Live prop instances
├── Generations/                # AI-generated content
├── SharedAssets/               # Project-wide resources
└── Library/                    # Local cache (never syncs)
```

### Key Architecture Decisions
1. **Text is first-class** - MUD and 3D are equal renderings of spatial truth
2. **Zones are soft** - Overlapping attention regions, not hard-edged rooms
3. **Prefab model** - Noodlings/Prims are templates; Instances are live copies
4. **Self-contained** - Projects are portable folders (zip and share)

### What Needs Doing

1. **Wire server.py to new project structure**
   - Currently loads from `cmush/world/` hardcoded paths
   - Should load from `project.get_stages_path()` etc.

2. **Update AssetsPanel to show new structure**
   - Currently shows old folder layout
   - Should show Noodlings/Prims/Stages categories

3. **Connect scene_hierarchy.py to Stages/Instances**
   - Currently loads from old agents.json
   - Should load from `Stages/xxx/Instances/`

4. **Implement cloud sync** (future)
   - Backend ready at noodlings-api.caitsters.workers.dev
   - Need to wire up sync buttons/status

### Files Changed This Session
- `noodlestudio/core/project_manager.py` - Rewritten (~900 lines)
- `noodlestudio/core/project_migrator.py` - New (~500 lines)
- `noodlestudio/core/main_window.py` - Updated menus and methods
- `PROJECT_SPEC.md` - New spec document (~700 lines)

---

## ✅ COMPLETED (December 17, 2025)

### Project Architecture & Specification
- **PROJECT_SPEC.md** - Complete 700-line specification defining:
  - Project structure (self-contained, portable folders)
  - Noodling format (recipe + assembly + scripts + NeuralGraphs + assets)
  - Stage format (continuous 3D space with soft zones)
  - Prim/Prop format (scriptable objects with MUD verbs)
  - Zone format (overlapping attention regions, not hard rooms)
  - Cloud sync strategy (auto-sync vs publish)
- **Key insight:** Text and 3D are equal renderings of spatial truth
- **project_manager.py** - Rewritten (~900 lines) with:
  - `create_noodling()`, `create_stage()`, `create_prim()`
  - `create_instance()`, `create_prop()` for stage population
  - Full folder structure creation per spec
  - Path helpers for all asset types
- **project_migrator.py** - New migration tool (~500 lines):
  - Converts legacy cmush/world data to new format
  - Maps rooms to soft zones with calculated positions
  - Migrates agents to instances
  - Preserves agent state and history
- **main_window.py updates:**
  - File menu: New Noodling/Stage/Prim, Migration tool
  - Help menu cleaned (removed placeholder items)
  - Import/Export noodling folders
- **Files:**
  - `PROJECT_SPEC.md` - Specification document
  - `noodlestudio/core/project_manager.py` - Implementation
  - `noodlestudio/core/project_migrator.py` - Migration tool

### Cloud Account System
- **Backend deployed** at `noodlings-api.caitsters.workers.dev`
  - Cloudflare Workers + D1 + R2 + KV
  - OAuth providers: Google, GitHub (working)
  - Stripe credits integration (configured)
  - OpenRouter LLM routing (configured)
- **NoodleStudio integration:**
  - `account_manager.py` - Session handling, macOS keychain storage
  - `login_dialog.py` - OAuth login with branded Google/GitHub buttons
  - `cloud_api.py` - Scripting API (`context.noodle.cloud`)
  - `account_status_widget.py` - Status bar (Sign In / name + Sign Out menu)
- **Backend repo:** `github.com/caitlynmeeks/noodlings-api` (private)

### Generations Asset Storage
- **GenerationsManager** for storing AI-generated content
- Storage: `library/Generations/Images/<YYYY-MM>/img_xxx.png`
- Rich metadata with agent, prompt, style, emotional_signature
- Auto-thumbnail generation (128x128)
- Events: `generation_stored`, `generations_cleared`
- AssetsPanel "Generations" category with source grouping
- Context menu: View, Show in Folder, Copy Prompt, Delete

### SubconsciousFacet Visual Mode
- Can now generate actual images from symbolic text (haiku/metaphor)
- Configure via prompt: `generate_visual:true,style:artistic,probability:0.3`
- Emotion-aware prompts (valence -> lighting, arousal -> movement)
- Auto-stores in Generations folder with full metadata
- Event: `subconscious_imagery_generated`

### Multimodal Facet System (Audio)
- **Option C architecture:** Parallel subsystem with sync points (like Unity's FixedUpdate)
- `MultimodalFacet` base class with modality auto-detection
- `AudioStreamFacet` with full audio pipeline:
  - Voice Activity Detection (VAD)
  - Transcription buffering and chunking
  - TTS queue with interrupt handling
  - Events: `transcription_ready`, `speech_start`, `speech_end`, etc.
- **Transcription clients:** Groq Whisper (fast), local faster-whisper (offline), OpenAI Whisper
- **TTS clients:** ElevenLabs (quality), OpenAI TTS, local Piper (offline)
- **WebSocket streaming:** Real-time mic input with browser JS client
- **Scripting API:** `context.noodle.audio` with Unity-like interface
  - Polling: `isListening`, `isSpeaking`, `lastTranscription`
  - Control: `speak()`, `listen()`, `stopListening()`, `interrupt()`
  - Config: `setVoice()`, `setSensitivity()`
- **Model labels:** VISION, AUDIO_IN, AUDIO_OUT, IMAGE_GEN, VIDEO_IN added to ModelLabelManager
- **Files:**
  - `noodlestudio/core/multimodal_facet.py` - Base class
  - `noodlestudio/core/audio_stream_facet.py` - Audio implementation
  - `noodlestudio/core/transcription_clients.py` - Whisper clients
  - `noodlestudio/core/tts_clients.py` - TTS clients
  - `noodlestudio/core/audio_streaming.py` - WebSocket handler
  - `noodlestudio/scripting/audio_api.py` - Scripting interface

### Vision & Image Generation System
- **VisionFacet** for image understanding:
  - Claude Vision, GPT-4V, LLaVA (local via Ollama) backends
  - Screenshot capture support
  - Hybrid memory model: hot (full tokens) → warm (descriptions) → cold (disk)
  - Semantic image search
- **ImageGenFacet** for image output:
  - DALL-E 3, Flux, Stable Diffusion backends
  - Style presets: photorealistic, artistic, anime, cinematic, concept_art, fantasy, scifi
  - Generation queue with callbacks
- **Scripting API:** `context.noodle.vision` with Unity-like interface
  - `analyze(path)` - analyze image
  - `screenshot()` - capture and analyze screen
  - `generate(prompt, style)` - create image
  - `searchImages(query)` - search memory
- **Files:**
  - `noodlestudio/core/vision_clients.py` - Vision backends
  - `noodlestudio/core/vision_facet.py` - Vision implementation
  - `noodlestudio/core/image_gen_clients.py` - Generation backends
  - `noodlestudio/core/image_gen_facet.py` - Generation implementation
  - `noodlestudio/scripting/vision_api.py` - Scripting interface

### Real IBM Quantum Integration
- QuantumAPI connects to IBM Quantum Platform (156-qubit ibm_fez etc.)
- Transpiles circuits to native gate set automatically
- Falls back to simulator when offline
- Auto-connect from `IBM_QUANTUM_API_KEY` in `.env`
- **File:** `noodlestudio/scripting/quantum_api.py`

### Schrodinger's Cat Experiment
- Cat's fate determined by REAL quantum collapse (not pseudo-random!)
- `schrodingers_cat()` runs actual quantum circuit on IBM hardware
- |0> = Alive cat (Schrodinger), |1> = Sassy ghost cat (Quantum Whiskers)
- Recipes and facet assemblies for both outcomes
- **Files:** `recipes/schrodinger_*.yaml`, `facet_assemblies/schrodinger_*.yaml`

### ScriptedFacet Context Wiring Fix
- Fixed all placeholder strings to actual JavaScript functions
- `__wire_noodle_context__` properly binds storage, logging, events, actions, quantum
- `context.noodle.quantum` now works in JavaScript ScriptedFacets
- **File:** `noodlestudio/core/scripted_facet.py`

### Neural Canvas Test Mode
- PyTorch-based test executor for canvas topology
- Run inference directly in canvas for immediate design feedback
- Visual feedback shows values on nodes as green badges
- Supports LSTM, GRU, RNN, Linear, activations, STATE_CONCAT, AFFECT_HEAD
- Test input field with Run/Reset buttons in toolbar
- Cross-platform (PyTorch instead of MLX-only)
- **File:** `noodlestudio/core/neural_canvas/test_executor.py`

### Noodling Names Generator
- ML/AI themed naming system replacing UUIDs
- 37-bit entropy (~17 billion unique combinations)
- Format: `Descriptor-Noun-VerbPhrase-Number`
- Examples: "Gradient-That-Walks-Backward-2894", "Tensor-Under-Moonlit-Descent-41022"
- **File:** `applications/cmush/noodling_names.py`

### Neural Canvas Tutorial Spec
- 10 tutorials across 5 progressive levels (logic gates → generation)
- Missing nodes identified and prioritized
- Implementation phases documented
- **File:** `NEURAL_CANVAS_TUTORIALS.md`

### Cleanup
- Removed Facets Editor debugging prints
- Fixed dimension mismatch in default.nncanvas (Medium LSTM input_dim 5→16)
- Added profiler_sessions to .gitignore

---

## ✅ COMPLETED (December 13, 2025)

### Scripting API Documentation Site
- MkDocs site with complete API reference (40+ methods)
- Dark coffee shop aesthetic
- F1 in NoodleStudio opens local docs automatically
- **Access:** http://127.0.0.1:8000/ or https://noodlings.ai/scripting-api/

### Agent Communication
- DeepSeek R1 `<think>` tag parsing - routes reasoning to "Red thinks..." vs speech
- Red's personality refinement - removed repetitive behaviors, added clear instructions
- User display fix - now shows username instead of "You say"

### Model Manager v2
- Complete metadata display system per model (context, size, pricing, capabilities)
- Horizontal layout with searchable model list
- 8 providers: Ollama, Anthropic, OpenAI, OpenRouter, LM Studio, Groq, Together, Mistral
- Draggable splitter between models and label assignments
- Custom label support with impact analysis

### Multi-Provider LLM Clients
- LLMClientRouter with unified interface across all providers
- OpenRouterClient, AnthropicClient, OllamaClient tested and working
- API keys stored securely in .env (gitignored)

### Scriptability API (context.noodle)
- ModelsAPI - get/set labels, configure providers, list models
- NeuralAPI - create nodes, connect ports, generate MLX code
- AgentsAPI - modify facet assemblies, set properties, save topologies
- Available in ScriptedFacets as `context.noodle`
- Full test suite in test_noodle_api.py

### Degoosification Backend
- Cloudflare Worker deployed at https://degoosification-worker.caitsters.workers.dev
- Henri Bergamot email personality with HonkCrypt™ security theater
- Email collection for future Asset Store accounts

### Neural Canvas
- Blender-style node editor for CharmNetwork internals
- 26 node types (LSTM, GRU, Linear, Quantum, etc.)
- MLX code generation from visual graphs
- .nncanvas JSON format for topology sharing

---

## 🏗️ Core Architecture

### Affect Model: PAD + Boredom + Sorrow

CharmNetwork outputs 5-dimensional continuous affect:
- `affect_valence` (-1 to +1) - Pleasure
- `affect_arousal` (0 to 1) - Energy
- `affect_dominance` (0 to 1) - Control
- `affect_boredom` (0 to 1)
- `affect_sorrow` (0 to 1)

**NOT** fear-based. NO discrete emotion labels.

### Facet System

Visual node-based cognitive architecture:

```
INCOMING → CHARM_NET → CONTEXT_INTELLIGENCE → Cognitive facets → Character layers → OUTGOING
```

**Facet Types:**
- **LLMFacet** - Language model calls with prompts
- **ScriptedFacet** - JavaScript/Python sandbox (context.noodle available here!)
- **CharmNetworkFacet** - LSTM/GRU neural network
- **ContextIntelligenceFacet** - Social context parsing
- **ConvergenceFacet** - Multi-input synthesis

**Key Files:**
- `noodlestudio/core/facet_system.py` - Data model
- `noodlestudio/core/facet_executor.py` - Execution engine
- `noodlestudio/panels/facets_editor_panel.py` - Visual editor
- `facet_assemblies/*.yaml` - Shared topologies (Unity prefab model)

### CharmNetwork

MLX-based temporal hierarchy:
- Fast LSTM (16-D): Seconds
- Medium LSTM (16-D): Minutes
- Slow GRU (8-D): Hours/days
- **Total:** ~54K parameters, ~2-3ms inference

---

## 🔧 Development Tips

### Running noodleMUSH

Toggle server in NoodleStudio status bar (bottom-right), or:
```bash
cd applications/cmush
./start.sh
```

**Ports:**
- 8080: HTTP (web interface)
- 8765: WebSocket
- 8081: NoodleScope API
- 11434: Ollama

**Network:** http://100.85.191.79:8080 (Tailscale)

### Debugging

```bash
tail -f applications/cmush/logs/server_*.log
```

**Look for:**
- `🎭 FACET EXECUTION COMPLETE` - Facets ran
- `[ContextIntelligence] 🧠 EXECUTE CALLED` - Context running
- `❌` - Errors!

**Common Issues:**
- No pachinko? Check WebSocket connection
- Agent not responding? Check for "🔒 Cycle already in progress"
- LLM fails? Check Ollama running
- Facets stuck? Check dependency graph

---

## 🎨 Style & Philosophy

### Caitlyn's Rules - CRITICAL

- **NO EMOJIS** in code/docs/UI (except when user requests)
- **NO "exciting" language** - Professional, terminal aesthetic
- **NO WORKAROUNDS** - Production-grade software, fix root causes
- **NO SHORTCUTS** - This is art inside and out
- **NO discrete emotion labels** - Continuous affect only
- **MONOCHROMATIC UI** - Grays only (except Neural Canvas taxonomic headers)

**GOLDEN RULE:** If it doesn't work properly, FIX IT properly. No hacks, no temporary solutions.

This is Caitlyn's legacy work, funded with real gold. Every solution must be production-quality.

### Christopher Alexander's "Timeless Way"

Organic development methodology - NOT Agile/Scrum.

**The Process:**
1. **Probe** - Crude implementation to sketch ideas
2. **Iterate** - Use it, refine what feels wrong
3. **Organic growth** - Features emerge from need, not roadmaps
4. **Discard freely** - Exploration, not waste
5. **Polish when friction hurts** - Unfinished = decision points

Like unplanned ancient cities (Venice, medieval towns) - charm from use-driven evolution, not imposed order.

**For Claude:**
- Support exploration over perfection
- Build crude-but-working first
- Trust aesthetic instinct
- Speed of prototyping > perfect planning

Not a startup. Not a research paper. Experimental architecture.

Think: Dr. Bronner's soap or Craigslist - eccentric, brilliant, one person's vision.

### Design Philosophy

- **Coffee shop palette** - Deep plums, forest greens, tobacco browns (Neural Canvas headers only)
- **Monochromatic UI** - Grays #2A2A2A to #FFFFFF elsewhere (Kraftwerk, not Disney)
- **Emergent behavior** - Personality flows from affect over time
- **Unity prefab model** - Topologies as shareable YAML
- **Blender aesthetics** - Colored headers for taxonomy, uniform dark bodies

---

## 👥 Project Context

**Creator:** Caitlyn (Unity employee #12, launched Asset Store, Tivoli Cloud VR architect)
**Age:** 54 - Legacy project
**Location:** Garcia River Forest cabin
**Timeline:** Demo to Steve DiPaola (SFU CogSci) soon

**Mission:** Counter "Consciousness-as-a-Service" (C-a-a-S). Release complete open-source alternative:
- Visual cognitive architecture editor (Blender of AI minds)
- Live interactive world (noodleMUSH)
- Real-time pachinko visualization
- Stateful affect-driven characters

**Vision:** Drop on Hacker News, provide brains/hearts for next-gen generative worlds. Standard built on **magic, not profit**.

---

## 📚 Documentation

- **NEURAL_CANVAS_TUTORIALS.md** - Tutorial system spec (10 projects, missing nodes, implementation phases)
- **FIREFLY_IDEAS.md** - Future features (Context Intelligence God, Guilt Facet, Cognitive Timeline Editor)
- **TECHNICAL_DEBT_INVENTORY.md** - High-priority items
- **MULTI_PROVIDER_EXECUTION_PLAN.md** - Integration roadmap
- **ACCOUNT_SYSTEM_ROADMAP.md** - Asset Store accounts
- **README.md** - Public-facing overview

---

## 🎯 For Fresh Claude

**Your mission:**
1. **Wire Scene Protocol to server.py** - See NEXT SESSION above
2. Initialize SceneStateManager, sync world state, generate perception slices

**Current State:**
- ✅ Multi-provider model system (8 providers)
- ✅ Neural Canvas with PyTorch test mode
- ✅ Scriptability API (context.noodle in ScriptedFacets)
- ✅ Facet system operational
- ✅ Docs site (F1 to open)
- ✅ Noodling names generator
- ✅ Real IBM Quantum integration (context.noodle.quantum)
- ✅ Schrodinger's Cat with actual quantum collapse
- ✅ Spatial Operations REST API (transforms, materials, physics, metadata)
- ✅ Scene Protocol (perception slices, scene packets, emitters)
- ✅ Codebase rebranded: Consilience → Noodlings, consciousness → charm
- ✅ Checkpoints moved to `models/checkpoints/best_checkpoint.npz`
- ⏳ Wire Scene Protocol to server (next task)
- ⏳ Wire project structure to server

**Key Scriptability API Files:**
- `noodlestudio/scripting/noodle_api.py` - Main API
- `noodlestudio/scripting/models_api.py` - Model/provider config
- `noodlestudio/scripting/neural_api.py` - Neural Canvas manipulation
- `noodlestudio/scripting/agents_api.py` - Facet assembly access
- `noodlestudio/scripting/quantum_api.py` - IBM Quantum integration

**UI/UX Notes:**
- Server toggle: Bottom-right status bar
- Model Manager: Center panel (8 providers, metadata display)
- Stage: Left panel (Unity Scene Hierarchy)
- Inspector: Right panel (selected entity properties)
- F1: Opens scripting docs

---

**Ordnung muss sein!**
