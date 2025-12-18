# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: December 17, 2025

**FOR NEXT CLAUDE: START HERE!** 👇

---

## 🎯 NEXT SESSION: Backend & Account System Architecture

**Goal:** Design architecture for cloud backend and user accounts (Noodlings Asset Store model)

**Context:**
- NoodleStudio is a desktop app (PyQt6)
- noodleMUSH is the live world server (local or cloud)
- Users will want to share facet assemblies, recipes, generations
- Need accounts for the future Asset Store (think Unity Asset Store but for AI minds)

**Questions to explore:**
1. **Backend architecture** - What services are needed?
   - User auth (email + magic link? OAuth?)
   - Asset storage (S3? R2? Direct upload?)
   - API gateway (FastAPI? Cloudflare Workers?)
   - Database (Postgres? Supabase? PlanetScale?)

2. **Asset types to share:**
   - Facet assemblies (.yaml)
   - Recipes (Noodling definitions)
   - CharmNetwork checkpoints (.npz)
   - Neural Canvas topologies (.nncanvas)
   - Generated images (from Generations folder)
   - Scripts (JavaScript facet code)

3. **Account features:**
   - Email collection (already via degoosification worker)
   - Profile with avatar, bio
   - Library of owned/published assets
   - Usage tracking (API calls, generations)

4. **Sync strategy:**
   - Local-first (NoodleStudio stores everything locally)
   - Opt-in cloud sync (upload to share, download from store)
   - No required internet connection for basic use

**Existing infrastructure:**
- Cloudflare Worker: `degoosification-worker.caitsters.workers.dev` (email collection)
- Domain: noodlings.ai (GitHub Pages, currently broken)
- No database yet

**Reference:** `ACCOUNT_SYSTEM_ROADMAP.md` (if exists)

---

## 🚨 URGENT ISSUES

### 1. noodlings.ai Homepage Broken
**Problem:** Website loads forever (beachballing in Chrome)

**What should show:**
- Black background with Noodlings mascot logo
- "Multi-Timescale Affective Agents with Theatrical Control" tagline
- "Documentation" and "Read Whitepaper" buttons

**Current state:** gh-pages branch has correct index.html locally, but noodlings.ai serves broken/loading page

**Fix needed:** Debug why gh-pages isn't serving correctly

### 2. Clean Up NoodleStudio Help Menu
**Remove these** (no content yet):
- Help → "Credits (Demo Scene Style)"
- Help → "NoodleStudio Documentation"
- Help → "Noodlings Architecture Guide"

**Keep:**
- Help → "Scripting API Reference" (F1) - Opens http://127.0.0.1:8000/api/overview/
- Help → "Report Issue..."
- Help → "About NoodleStudio"

**File:** `applications/noodlestudio/noodlestudio/core/main_window.py:247-255`

---

## ✅ COMPLETED (December 17, 2025)

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
1. **Read `NEURAL_CANVAS_TUTORIALS.md`** - Full spec for tutorial system
2. Start with Phase 1: Logic gate tutorials (AND, OR, XOR)
3. Implement missing nodes: `NUMBER_INPUT`, `THRESHOLD_OUTPUT`

**Current State:**
- ✅ Multi-provider model system (8 providers)
- ✅ Neural Canvas with PyTorch test mode
- ✅ Scriptability API (context.noodle in ScriptedFacets)
- ✅ Facet system operational
- ✅ Docs site (F1 to open)
- ✅ Noodling names generator
- ✅ Real IBM Quantum integration (context.noodle.quantum)
- ✅ Schrodinger's Cat with actual quantum collapse
- ⏳ Multimodal facets (next task)
- ⏳ Homepage fix needed

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
