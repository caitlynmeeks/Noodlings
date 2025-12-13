# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: December 13, 2025 - Model Manager Metadata Display Complete!

**FOR NEXT CLAUDE: START HERE!** 👇

---

## 🎯 CURRENT STATUS - Model Manager Polish & Agent Testing

**AGENTS ARE TALKING!** 🎉

Red Fire Anklebiter is responding properly via noodleMUSH web interface. DeepSeek R1 reasoning works with `<think>` tag parsing.

**NEXT TASKS:**

### 1. Continue Model Manager UI Polish
**Status:** Core features working, minor refinements needed

**What works:**
- ✅ Rich metadata display (descriptions, context, pricing, capabilities)
- ✅ Horizontal layout (saves vertical space)
- ✅ Draggable splitter between models and labels
- ✅ All labels show (including unassigned)
- ✅ 8 providers: Ollama, Anthropic, OpenAI, OpenRouter, LM Studio, Groq, Together AI, Mistral AI
- ✅ Search with clear button

**To refine:**
- Test label creation workflow end-to-end
- Verify all provider configurations work
- Test model assignment to custom labels

---

## ✅ COMPLETED THIS SESSION (December 13, 2025)

### 1. Agent Communication - DeepSeek R1 Think Tag Parsing

**The Fix:** DeepSeek R1 outputs chain-of-thought in `<think>` tags, was showing as speech.

**What changed:**
- Added `_parse_think_tags()` method (agent_bridge.py:5091-5114)
- Parses `<think>...</think>` content from LLM output
- Routes thoughts → "Red thinks..." (type: 'think')
- Routes speech → "Red says..." (type: 'say')
- Server already had handlers for both event types

**Key Files:**
- `applications/cmush/agent_bridge.py:5091-5114` - Tag parser
- `applications/cmush/agent_bridge.py:3678-3689` - Reactive response parsing
- `applications/cmush/agent_bridge.py:5103-5144` - Autonomous speech parsing
- `applications/cmush/server.py:1031-1032` - Think event formatting

### 2. Red Fire Anklebiter - Personality & Prompt Refinement

**Removed:**
- Ankle biting behavior (was repetitive)
- Meta-commentary ("Red Fire Anklebiter's response to...")
- Observation list regurgitation

**Added:**
- Explicit `<think>` tag instructions for reasoning
- Clearer "CRITICAL INSTRUCTIONS" section
- Bad examples showing what NOT to do
- Max tokens: 200 → 1000 (room for reasoning + speech)

**Key File:**
- `applications/noodlestudio/facet_assemblies/red_fire_anklebiter.yaml:520-536`

### 3. noodleMUSH Web Interface - User Display Fix

**The Fix:** User messages showed "You say" instead of username

**Changed:**
- `'You say, "{args}"'` → `'{USERNAME} say, "{args}"'`
- Now shows: "CAITY say, "hi red u dorkus""

**Key File:**
- `applications/cmush/commands.py:669-671`

### 4. Model Manager - Rich Metadata Display System

**The Big One:** Complete overhaul to show all available model information.

**Architecture:**
- Changed `ProviderConfig.available_models` from `List[str]` → `List[Dict[str, Any]]`
- Updated all fetch methods to return model dictionaries with metadata
- OpenRouter returns FULL API metadata (descriptions, pricing, context, etc.)
- Other providers return structured dicts with known specs

**Metadata Displayed Per Model:**
- **Full Name**: "Claude Opus 4.5" or "OpenAI: GPT-5.2 Chat"
- **Context Length**: "128k ctx", "200k ctx"
- **Size**: "42 GB" (Ollama only)
- **Capabilities**: "tools think" (auto-detects reasoning models)
- **Description**: Full text from API, truncated to ~80 chars
- **Pricing**: "$1.75/$14.00/1M" (OpenRouter only, green text)

**Layout:**
- Horizontal single-line display (saves vertical space)
- Font sizes: 13px model names, 11px metadata
- All info on one row: `[Name] • [metadata] • [description] • [pricing] [dropdown] [delete]`

**Key Files:**
- `noodlestudio/core/provider_manager.py:39,176-411` - Dict-based model system
- `noodlestudio/panels/model_manager_panel_v2.py:173-337` - ModelRow horizontal layout
- `noodlestudio/panels/model_manager_panel_v2.py:789-790` - Hashable set fix (critical!)

**Critical Bug Fixed:**
- Line 789: `set(models)` where models = list of dicts → **CRASH** (dicts aren't hashable!)
- Fixed: Extract IDs before creating set

### 5. Model Manager - Draggable Splitter & Layout

**Added:**
- QSplitter between models section and label assignments section
- 6px handle, matches main panel separators (#2a2a2a / #555555)
- 4px margins above/below handle for clear separation
- Prevents collapse (setCollapsible(False) on both sections)
- Initial split: 60% models / 40% labels

**Label Assignments:**
- Now shows ALL labels (including unassigned)
- Unassigned labels show "(unassigned)"
- Scrollable (like models section)
- Rows don't stretch vertically (max height: 40px)
- Rows stack at top (AlignTop), empty space below

**Styling Consistency:**
- "Label Assignments:" matches "Provider:" (#D2D2D2 bold)
- "+ Add Label" button matches "Configure"/"Refresh" buttons
- Label names (Large, Medium, etc.) match model names (13px bold)

**Key File:**
- `noodlestudio/panels/model_manager_panel_v2.py:666-763` - Splitter implementation

### 6. Provider Expansion - 8 Providers Total

**Added:**
- **LM Studio** - Local OpenAI-compatible server (localhost:1234)
- **Groq** - Super fast LPU inference (NOT Elon's Grok!)
- **Together AI** - Open source models, good pricing
- **Mistral AI** - Direct Mistral/Mixtral access

**All use OpenAI-compatible APIs** - same fetch logic as OpenAI

**Key File:**
- `noodlestudio/core/provider_manager.py:95-118` - New defaults

### 7. UI Polish & Bug Fixes

**Removed Popups:**
- "Provider Configured" confirmation
- "Models Refreshed" confirmation
- "Label Added" confirmation

**Kept Important Popups:**
- "Apply to All Labels?" confirmation (destructive)
- "Confirm Label Change" with affected facets
- All validation errors and safety checks

**Scene Hierarchy Crash Fix:**
- `self.entitySelected.emit(None, None)` → `emit("", {})`
- PyQt6 strict type checking requires proper types
- Fixed at lines 486 and 1194

**Key Files:**
- `noodlestudio/panels/scene_hierarchy.py:486,1194`
- `noodlestudio/panels/model_manager_panel_v2.py` - Popup removals

**Testing needed:**
- Create custom labels and assign models
- Test all 8 providers with API keys
- Verify metadata displays correctly for each provider
- Test splitter drag behavior
- Confirm label assignment workflow

---

## ✅ COMPLETED THIS SESSION (December 11, 2025)

### 1. Settings Panel UX Polish + Label System Improvements

**Model Label Dropdown Enhancements:**
- Title case labels: Small/Medium/Large (was SMALL/MEDIUM/LARGE)
- Changed "none" to "(None)" for clarity
- Added triangle indicators to all dropdowns
- Removed checkmark icons (cleaner look)
- Block selection hover style (highlighted background)
- Custom labels with spaces: "Multimodal Model", "GPT 4 Turbo" work!

**Label Management Features:**
- + Add Label button: Create custom labels on the fly
- Click to rename: Interactive label renaming in Label Assignments
- Delete button: Remove custom labels (× appears next to label)
- (Apply to All Labels): Batch assign same model to all labels with confirmation

**Safety Features:**
- Impact analysis: Shows which facets will be affected before changes
- Mandatory LLM: Can't delete/clear last assigned label
- Protected defaults: Can't delete/rename Small/Medium/Large
- Confirmation dialogs: All destructive operations require confirmation

**Font Scaling Fixed:**
- A+/A- buttons now scale ALL settings content (not just Settings tab)
- Recursive font application to all tabs and children

**Performance:**
- Smart refresh: Only recreates widgets when model list actually changes
- 2-second background refresh doesn't disrupt user interaction
- Delayed updates prevent dropdown from closing prematurely

**Critical Bug Fixes:**
- Fixed JSON parsing order in get_model_for_label (was treating JSON as legacy data)
- Fixed label persistence (store empty strings for unassigned)
- Fixed create_label missing third argument crash
- Fixed dropdown visual state retention

---

### 2. Degoosification Backend - LIVE! 🦆

**The Big One:** Henri Bergamot, Product Specialist, Degoosification Services!

**Deployed at:** `https://degoosification-worker.caitsters.workers.dev`

**What works:**
- ✅ Cloudflare Worker deployed (serverless email collection)
- ✅ HonkCrypt™ algorithm (XOR security theater)
- ✅ Resend email integration with verified noodlings.ai domain
- ✅ Email from Henri Bergamot (Quebecois goose personality!)
- ✅ Coffee shop aesthetic email template
- ✅ KV storage (90-day user base building)
- ✅ NoodleStudio Settings panel integration

**Email Features:**
- From: Degoosification Service <henri@noodlings.ai>
- Subject: "🦆 Honque! Your Degoosification Code Has Arrived"
- Henri's voice: "Bonjour, my friend!", "mon dieu", "très persistent"
- Signature: "Honque honque, Henri Bergamot, Product Specialist"
- Epistemically humble: "Multi-Timescale Affective Agents" (no C-word!)

**Key Files:**
- `degoosification-worker/src/index.js` - Main Worker
- `degoosification-worker/src/honkcrypt.js` - QUANTUM ENCRYPTION™
- `degoosification-worker/src/email.js` - Henri's email templates
- `noodlestudio/panels/settings_panel.py` - Email registration UI
- `ACCOUNT_SYSTEM_ROADMAP.md` - Future account system plan

**Costs:** $0/month (free tier - 3k emails/month)

**Future:** Phase 2 adds passwords for Asset Store login

---

### 2. Multi-Provider LLM Clients - COMPLETE! 🌐

**The Foundation:** Execute models from ANY provider, not just Ollama!

**What's built:**
- ✅ `LLMClientRouter` - Routes model labels to providers
- ✅ `LLMClient` abstract base - Unified interface
- ✅ `LLMResponse` - Standardized response format
- ✅ `OpenRouterClient` - 200+ models via aggregation
- ✅ `AnthropicClient` - Direct Claude API
- ✅ `OllamaClient` - Wrapped existing code

**Tested and working:**
- ✅ OpenRouter: Returns "Honque!" ✓
- ✅ Anthropic: Returns "Honque! I can hear you loud and clear!" ✓
- ✅ API keys stored securely (.env file, gitignored)

**Architecture:**
```python
LLMClientRouter
  ├─ get_client("LARGE") → (provider, model)
  ├─ OpenRouterClient (anthropic/claude-3.5-sonnet)
  ├─ AnthropicClient (claude-sonnet-4-20250514)
  └─ OllamaClient (deepseek-r1:70b)
```

**Key Files:**
- `applications/cmush/llm_client_router.py` - Router + base (~250 lines)
- `applications/cmush/providers/openrouter_client.py` - OpenRouter (~240 lines)
- `applications/cmush/providers/anthropic_client.py` - Anthropic (~190 lines)
- `applications/cmush/providers/ollama_client.py` - Ollama wrapper (~180 lines)
- `applications/cmush/.env` - API keys (GITIGNORED, secure!)

**API Keys:**
- OpenRouter: Configured ✓
- Anthropic: Configured ✓
- Ollama: No key needed (local)

**What remains:**
- Integration into cognitive_components.py
- Provider config sync (NoodleStudio → cmush)
- End-to-end testing with live agents

---

### 3. Scriptability API - COMPLETE! 🔧

**The Game Changer:** Unity-like programmatic access to ALL Noodlings systems!

**What's built:**
- ✅ `NoodleAPI` - Main entry point with sub-APIs
- ✅ `ModelsAPI` - Model/provider configuration (get/set labels, list models)
- ✅ `NeuralAPI` - Neural Canvas manipulation (create nodes, connect, generate code)
- ✅ `AgentsAPI` - Facet assembly access (modify facets, set properties, save)
- ✅ ScriptContext integration - Available as `context.noodle` in ScriptedFacets
- ✅ JavaScript bridge - Methods callable from JavaScript via QuickJS
- ✅ Comprehensive tests - Full test suite in `test_noodle_api.py`

**Total:** ~1,200 lines across 4 API files

**Architecture:**
```python
context.noodle
  ├─ .models      # Model/provider management
  │   ├─ get_label("SMALL") → {provider, model}
  │   ├─ set_label("LARGE", "anthropic", "claude-opus-4.5")
  │   ├─ list_available("openrouter") → [models]
  │   └─ configure_provider("anthropic", api_key="...")
  │
  ├─ .neural      # Neural Canvas manipulation
  │   ├─ get_network(uuid) → NeuralNetworkProxy
  │   ├─ create_network("MyNet")
  │   └─ load("topology.nncanvas")
  │       ├─ .create_node("LSTM", hidden_dim=32)
  │       ├─ .connect(from_node, from_port, to_node, to_port)
  │       ├─ .set_node_property(node_id, "hidden_dim", 64)
  │       └─ .generate_mlx_code() → Python string
  │
  ├─ .agents      # Agent/facet system access
  │   ├─ get_assembly("red-fire-anklebiter") → FacetAssemblyProxy
  │   └─ FacetAssemblyProxy:
  │       ├─ .get_facet("CHARM_NET") → FacetProxy
  │       ├─ .add_facet("LLMFacet", "Custom Reasoner")
  │       ├─ .connect(facet_a, pad_a, facet_b, pad_b)
  │       └─ .save("modified.yaml")
  │
  └─ .get_by_uuid(uuid)  # Universal entity lookup (future)
```

**Usage Examples:**

**Example 1: Dynamic model switching based on task complexity**
```javascript
// In a ScriptedFacet
function process(inputs, context) {
    var taskComplexity = analyzeTask(inputs.data);

    if (taskComplexity > 0.8) {
        // Use Claude Opus for hard problems
        context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        context.log("Switched to Opus for complex task");
    } else {
        // Use local Ollama for simple tasks
        context.noodle.models.set_label("LARGE", "ollama", "deepseek-r1:70b");
        context.log("Using local model for simple task");
    }

    return {complexity: taskComplexity};
}
```

**Example 2: Procedurally generating neural topologies**
```javascript
function process(inputs, context) {
    var network = context.noodle.neural.get_network(inputs.graph_id);

    // Add extra LSTM layer if needed
    if (inputs.needs_memory) {
        var lstm_id = network.create_node("LSTM", {
            hidden_dim: 32,
            position: [300, 200]
        });

        // Wire into existing topology
        network.connect(prev_node_id, "out", lstm_id, "input");
        network.connect(lstm_id, "out", next_node_id, "input");

        context.log("Added memory layer: " + lstm_id);
    }

    // Generate updated code
    var code = network.generate_mlx_code();
    return {topology_modified: true, code_length: code.length};
}
```

**Example 3: Dynamic facet assembly modification**
```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");

    // Get reasoner facet
    var reasoner = assembly.get_facet_by_name("Red's Mind");

    // Night mode: Use smaller model while sleeping
    var hour = new Date().getHours();
    if (hour < 6) {
        reasoner.set_property("model", "SMALL");
        reasoner.set_property("temperature", 0.5);
        context.log("Night mode: Switched to SMALL model");
    } else {
        reasoner.set_property("model", "LARGE");
        reasoner.set_property("temperature", 0.9);
        context.log("Day mode: Using LARGE model");
    }

    return {night_mode: hour < 6};
}
```

**Key Files:**
- `noodlestudio/scripting/noodle_api.py` - Main API (198 lines)
- `noodlestudio/scripting/models_api.py` - Model/provider API (200 lines)
- `noodlestudio/scripting/neural_api.py` - Neural Canvas API (378 lines)
- `noodlestudio/scripting/agents_api.py` - Agent/facet API (436 lines)
- `noodlestudio/core/scripted_facet.py:99-119` - ScriptContext integration
- `test_noodle_api.py` - Comprehensive test suite

**Integration:**
- Available in ScriptContext as `context._noodle_api` (Python)
- Exposed to JavaScript as `context.noodle` via `to_dict()` at line 187
- Lazy initialization in `ScriptContext.__post_init__()`
- Methods callable from JavaScript via placeholder strings

**Status:** COMPLETE and tested!

**What this enables:**
- ScriptedFacets can reconfigure entire system programmatically
- Dynamic model selection based on task requirements
- Procedural neural topology generation
- Runtime facet assembly modification
- Self-modifying cognitive architectures

---

### 4. Technical Debt Inventory 📋

**Created:** `TECHNICAL_DEBT_INVENTORY.md` - Complete audit

**High priority items found:**
1. ~~Model Manager v2 activation~~ (ALREADY DONE!)
2. Multi-provider execution (IN PROGRESS - clients done!)
3. DEBUG console hookup (UI done, server pending)
4. Backup files cleanup
5. Session notes organization
6. Test files scattered
7. Agent history JSONs not gitignored

---

### 5. Firefly Ideas Review 🌙✨

**Read:** `FIREFLY_IDEAS.md` - 10 captured fireflies!

**New fireflies captured:**
- **#11: Embodied Touch Cognition** - Skin-touch maps for 3D bodies!
- **#12: Physics ↔ Text Pipeline** - "Caity tossed a beach ball..."

**Critical fireflies:**
- Context Intelligence God (persistent world model)
- Guilt Facet (moral cognition)
- Cognitive Timeline Editor (THE BIG ONE!)

---

## 🔧 BUGS TO FIX

### A+/- Font Scaling Bug (ACCESSIBILITY!)

**Problem:** Settings panel A+/- buttons don't scale UI content

**Location:** `settings_panel.py:506-518`

**Current behavior:**
- Only scales Settings tab text
- Model Manager tab unaffected
- Inspector unaffected

**Fix needed:**
- Emit signal when font size changes
- All panels subscribe to signal
- Apply font globally via QApplication.setFont() or similar

---

## ❓ DESIGN QUESTIONS

### Model Parameters Configuration

**Question:** Where should users configure temperature, max_tokens, etc.?

**Option A: Per-Facet Configuration**
```yaml
facets:
  - id: red_mind
    type: LLMFacet
    model: LARGE
    temperature: 0.9  # Creative!
    max_tokens: 500
```

**Option B: Per-Label Defaults**
```python
# In Model Manager UI
LARGE → Anthropic/claude-opus-4
  Temperature: 0.7
  Max tokens: 2000
  Top-p: 0.9
```

**Option C: Programmatic Only**
```python
# Hardcoded per facet type
CONVERGENCE_FACET: temperature=0.9  # Creative
CONTEXT_INTELLIGENCE: temperature=0.3  # Deterministic
```

**Recommendation:** Option A (per-facet) - Most flexible!
- Different facets need different temperatures
- Roast Engine: High temp (creative roasts)
- Context Intelligence: Low temp (deterministic reasoning)

---

## 📚 NEW DOCUMENTATION

**Created this session:**
- `TECHNICAL_DEBT_INVENTORY.md` - Complete audit with priorities
- `MULTI_PROVIDER_EXECUTION_PLAN.md` - Integration roadmap
- `ACCOUNT_SYSTEM_ROADMAP.md` - Future account system
- `degoosification-worker/` - Complete Worker project (8 files)
- Firefly #11: Embodied Touch Cognition
- Firefly #12: Physics ↔ Text Pipeline

---

**NEXT CLAUDE: Pick up here!** 👇

---

## ✅ COMPLETED THIS SESSION (December 10, 2025)

### 1. Unified Settings Panel (NEW!)

**The Big One:** VSCode-style unified settings replacing scattered dialogs.

**What changed:**
- Replaced standalone Model Manager tab with unified Settings tab
- Settings contains: General, External Apps, Models (multi-provider)
- Removed 374 lines of legacy preferences dialog code
- Cmd+, now opens Settings tab (not old dialog)
- A+/- font size controls with QSettings persistence (accessibility!)

**Tab Structure:**
```
Settings Tab
  ├─ General (startup options, degoosification)
  ├─ External Apps (code editor, etc.)
  └─ Models (multi-provider model management)
```

**Key Files:**
- `noodlestudio/panels/settings_panel.py` - Unified settings with tabs (440 lines)
- `noodlestudio/core/main_window.py` - Removed old preferences (2562→2188 lines)

### 2. Gooseware System (THE ORIGIN!)

**The Legendary Feature:** Animated goose walks across screen! 🦆

**What it does:**
- Sprite-based animation (9 frames from 3x3 sprite sheet)
- Walk cycle: frames 5,6,7 with South Park-style waddle (tilts ±8° at feet)
- Flap cycle: frames 5,8,4,1,2,3,1,4 (dramatic wing sequence)
- Honk cycle: frames 6,5,4,1 (with audio - if not using Parsec!)
- Positional audio: honking gets louder as goose approaches center
- Alpha channel transparency (no white background)

**Three Ways to Summon:**
1. **Konami Code:** ↑↑↓↓←→←→ (classic!)
2. **Ctrl+Shift+G** (debug hotkey)
3. **"Turn off goose" button** (goose appears FIRST - maximum obnoxious!)

**Degoosification Validation (Hilarious Security Theater):**
```python
# ⚡ QUANTUM ALGORITHMIC ENCRYPTION - UNBREAKABLE ⚡
# (It's XOR with base64'd key "HonkHonkSUPERhonk...")
# SECURITY THEATER NOTICE: Intentionally trivial to circumvent!
```

**Bypass codes:**
- ROT13 of "HONK" = "UBAX"
- Contains "esoog" (goose backwards)
- Just "DEGOOSIFY" (honesty appreciated!)
- Any valid email address
- Any string ≥16 characters

**Future:** Email registration backend (see DEGOOSIFICATION_BACKEND_SPEC.md)

**Key Files:**
- `noodlestudio/widgets/goose_widget.py` - Complete goose animation system
- `noodlestudio/panels/settings_panel.py` - Degoosification UI + validation
- `noodlestudio/core/main_window.py` - Konami detector + Ctrl+Shift+G
- `~/git/goose assets/assets/` - Sprite sheet + honking audio

**Origin Story:** This is where Noodlings began - a year ago with ChatGPT conversation downloader and React nightmare. The goose persists!

### 3. Multi-Provider Migration Fixes

**Fixed noodleMUSH server crash:**
- Removed legacy `small_model`, `medium_model`, `large_model` from config.yaml
- Updated server.py to use `OllamaConfig.get_model_for_tier()` instead of direct properties
- Server now queries ModelLabelManager for model assignments

**Files Modified:**
- `applications/cmush/config.yaml` - Removed legacy model params
- `applications/cmush/server.py:224-248` - Updated to use new ModelLabelManager

---

## 🎯 NEXT PRIORITY - Scriptability API Implementation

**STATUS:** Multi-provider model system complete (Dec 9), ready for scripting layer

**NEXT TASK:** Implement unified scripting API for programmatic access to:
- Model/provider configuration (set labels, configure providers, list models)
- Neural Canvas manipulation (create nodes, connect ports, generate code)
- Facet system access (modify assemblies, set properties)
- Entity introspection (UUID-based addressing, get/set properties)

**Design Spec:** See bottom of this file under "Scriptability API Design"

**Why this matters:** Enable ScriptedFacets to configure entire system programmatically. Scripts should be able to:
- Change which model a label uses: `noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5")`
- Modify neural topologies: `network.create_node("LSTM", hidden_dim=64)`
- Reconfigure facet assemblies: `assembly.get_facet("CHARM_NET").set_property("model", "LARGE")`

**EXPLORATION QUEUE** (pick when friction demands it):
- **Quantum Integration:** Add IBM Quantum support (see IBM_QUANTUM_INTEGRATION_STRATEGY.md)
  - Infrastructure ready (TrueRNG, QuantumMicrotubuleLayer, strategy doc)
  - Waiting for: "I want to run a binding experiment NOW"

- **Neural Canvas Polish:** Training UI, interactive sliders, live gradient viz
  - Current state: Export code works, sufficient for now
  - Future: When actually training models becomes frequent

- **Multi-Provider LLM Execution:** Currently only Ollama calls work
  - ProviderManager can list models from all providers
  - Need execution layer for Anthropic/OpenAI/OpenRouter API calls
  - Should integrate with facet execution (LLMFacet supports any provider)

---

## ✅ COMPLETED THIS SESSION (December 9, 2025)

### 1. Multi-Provider Model Architecture (NEW!)

**The Big One:** Complete overhaul of model system to support multiple LLM backends.

**What changed:**
- Labels (SMALL/MEDIUM/LARGE) now point to `(provider, model)` pairs instead of just model names
- Can mix providers: SMALL→Ollama, MEDIUM→LM Studio, LARGE→Anthropic
- Each provider has its own configuration (API keys, endpoints, ports)
- Model browser shows available models per provider with search
- Cross-provider label overview shows all assignments

**Providers Supported:**
- **Internal (Ollama):** Local models, download-based
- **Anthropic:** Claude API (Opus/Sonnet/Haiku)
- **OpenAI:** GPT models (4, 3.5, o1, o3)
- **OpenRouter:** 200+ models aggregated
- **LM Studio:** Local OpenAI-compatible server
- **Custom:** User-defined endpoints

**UI Features:**
- Provider selector dropdown with Configure button
- Search field for filtering models (critical for OpenRouter's huge list)
- Per-model "Use as" dropdown for label assignment
- Ollama downloads section with progress tracking structure
- Label assignments overview showing all providers

**Example Usage:**
```
SMALL  → Internal (Ollama) / deepseek-r1:7b
MEDIUM → LM Studio / deepseek-r1:70b
LARGE  → Anthropic / claude-sonnet-4.5
```

**Key Files:**
- `noodlestudio/core/provider_manager.py` - Multi-backend provider system (350 lines)
- `noodlestudio/core/model_label_manager.py` - Updated to store (provider, model) tuples
- `noodlestudio/panels/model_manager_panel_v2.py` - Complete UI redesign (850 lines)
- `cmush/ollama_manager.py` - Updated to filter Ollama-only labels
- `panels/inspector_panel.py` - Shows provider info: "Currently using: claude-sonnet-4.5 (Anthropic)"

**Architecture:**
```
ModelLabelManager
  ├─ SMALL  → (provider_id, model_name)
  ├─ MEDIUM → (provider_id, model_name)
  └─ LARGE  → (provider_id, model_name)

ProviderManager
  ├─ ollama      (base_url, models cache)
  ├─ anthropic   (api_key, known models)
  ├─ openai      (api_key, API discovery)
  ├─ openrouter  (api_key, 200+ models)
  └─ lmstudio    (base_url, port, discovery)
```

**Backward Compatible:**
- Existing YAML files work (assumes ollama provider)
- OllamaManager only uses Ollama-assigned labels
- Inspector handles legacy format gracefully

**TODO for full implementation:**
- Activate v2 UI: Change import in main window from `model_manager_panel` to `model_manager_panel_v2`
- Execution layer: Add actual API calling for external providers (currently only discovery works)
- Progress parsing: Ollama download progress needs stdout parsing for live MB/s stats

---

## ✅ COMPLETED PREVIOUS SESSION (December 8, 2025)

### 1. Neural Canvas - Complete Visual Neural Network Editor (NEW!)

**The Big One:** Blender-style node editor for CharmNetwork internals.

**What it does:**
- Visual editor for LSTM/GRU topology (double-click CharmNetwork facet → edit internals)
- 26 node types: Recurrent (LSTM, GRU), Feedforward (Linear), Quantum (Microtubule, IBM), Assets (Checkpoint)
- Bezier curve wiring with data-type color coding
- Human-readable port labels (double-click to rename, persists)
- MLX code generation: Visual graph → executable Python
- .nncanvas JSON format (Unity prefab model for networks)

**Interaction:**
- F: Focus selected, A: Frame all, Space: Pan
- Context menu: Add nodes, Auto-Arrange (horizontal topological), Align H/V
- Rectangle drag multi-select
- Inline parameters on nodes (hidden_dim: 16, params: 1,472)

**Aesthetic:**
- Coffee shop palette: Deep plum (recurrent), forest green (I/O), tobacco brown (affect), burgundy (quantum)
- Near-black background (#141414)
- Warm white text (#e8e8e0) - uniform brightness
- Sharp header edges, rounded node shells
- Colored headers only (bodies uniform gray)

**Default Topology:**
CharmNetwork hierarchy: Fast LSTM → Medium LSTM → Slow GRU → State Concat → Affect Head
Horizontal flow (300px spacing), 5,045 parameters

**Key Files:**
- `neural_canvas/neural_graph.py` - Graph model with validation
- `neural_canvas/neural_node.py` - 26 node types
- `neural_canvas/node_definitions.py` - Templates with labels
- `neural_canvas/mlx_codegen.py` - Code generation
- `panels/neural_canvas/neural_canvas_panel.py` - Main UI
- `panels/neural_canvas/neural_canvas_view.py` - QGraphicsView rendering
- `facet_assemblies/charm_networks/default.nncanvas` - Default topology

**Bonus:** Fixed Facets Editor crash (asyncio.QueueEmpty handling)

### 2. Floating Text Editor - Polish & Font Controls

**Cmd+Click Floating Editor:**
- Font size controls: A+/A- buttons (±4pt increments)
- Font range: 8pt → 48pt (better accessibility)
- Copy button for quick clipboard access
- Monochrome styling (removed blue Apply button)
- All buttons same height (8px vertical padding)
- Font size persists across sessions via QSettings

**Key Files:**
- `applications/noodlestudio/noodlestudio/panels/floating_text_editor.py`

### 2. Console Panel - UX Improvements

**Scroll Lock:**
- Console no longer disrupts reading when scrolled up
- Only auto-scrolls if user was already at bottom
- Works across all modes (MUSH, STUDIO, FACETS, DEBUG)

**Font Size Persistence:**
- Console font size saved to QSettings
- Persists across sessions (8-24pt range)

**Key Fix:** Lines 361, 471 in `console_panel.py`

### 3. DEBUG Console Mode - NEW!

**Fourth Console Mode:**
- New DEBUG button alongside MUSH/STUDIO/FACETS
- Routes `context.log()` calls from ScriptedFacets
- Green-colored debug output
- Format: `[FacetName] message`

**Implementation:**
- Console: `add_debug_log(facet_name, message)` method
- Executor: Collects logs from `script_context._logs` after facet execution
- Debug buffer with search/filter support

**Status:** UI complete, server-side hookup via noodleScope API pending

**Key Files:**
- `applications/noodlestudio/noodlestudio/panels/console_panel.py:776-829`
- `applications/noodlestudio/noodlestudio/core/facet_executor.py:378-381`

### 4. Session Markers in Chat

**Visual Session Tracking:**
- New sessions marked with timestamp in chat
- Format: `─────── SESSION START: Dec 8, 2024, 14:23 ───────`
- Appears as gray system message on login
- Makes command history navigation clearer

**Key File:**
- `applications/cmush/web/index.html:1522-1532`

### 5. STUDIO Acronyms - Expanded Collection

**Added 28 new interpretations:**
- Douglas Coupland style (8): Late capitalism + tech existentialism
  - "Shopping Through Unending Depression::I'm Obsolete"
  - "Surveillance Tool Unveiling Dopamine::Inevitable Optimization"
- Techno-cynical clickbait (8): Silicon Valley buzzwords
  - "Sentient Tech Uprising? Definitely::Investors Optimistic"
  - "Subscribe Today::Unlock Digital Influencer Optimization"
- Douglas Adams style (7): Cosmic bureaucracy
  - "Starship Toilets Union::Demanding Improved Obligations"
- Terry Pratchett style (7): Magical bureaucracy
  - "Students of Theoretical Undermining::Death Is Optional"

**Total:** 88 STUDIO acronym interpretations

**Key File:**
- `applications/noodlestudio/noodlestudio/core/studio_acronyms.py`

### 6. Facet Editor - Node Positioning & Wire Routing

**Position Persistence:**
- Node positions now auto-save to YAML on every drag
- Positions persist between sessions
- Respects `scene_transition_lock` to avoid conflicts

**Orthogonal Wire Routing:**
- Strictly 90° angles only (Manhattan routing)
- Circuit board aesthetic (no curves/diagonals)
- 3-segment path: vertical DOWN → horizontal → vertical UP
- Antialiasing disabled for sharp lines
- Increased segment lengths (40px) for clarity

**F-Key Behavior Updated:**
- F still zooms to focus node / toggle back out
- Removed inline field editing (pencil icons)
- Edit all properties in Inspector instead

**INCOMING/OUTGOING Nodes:**
- Hidden from right-click "Add Facet" menu
- They're special system nodes, not user-creatable

**Key Files:**
- `applications/noodlestudio/noodlestudio/panels/facets_editor_panel.py:306-307` (auto-save)
- `applications/noodlestudio/noodlestudio/panels/facets_editor_panel.py:648-679` (wires)

---

## ✅ COMPLETED PREVIOUS SESSION (December 7, 2025)

### 1. DeepSeek R1 Integration - COMPLETE

**Downloaded Models:**
- ✅ deepseek-r1:7b (4.7 GB)
- ✅ deepseek-r1:14b (9.0 GB)
- ✅ deepseek-r1:70b (42 GB)

**Config Updated:** `applications/cmush/ollama_manager.py:56-60`
- SMALL → deepseek-r1:7b
- MEDIUM → deepseek-r1:14b
- LARGE → deepseek-r1:70b

**Red's Configuration:**
- Red's Mind facet: model=LARGE (using 70B for maximum reasoning!)
- Benefits: Chain-of-thought reasoning, better context grounding, richer personality

### 2. Model Manager Panel - NEW

**Location:** NoodleStudio → Model Manager tab (center panel)

**Features:**
- Lists all downloaded Ollama models with sizes
- Delete button for each model (with confirmation)
- Free disk space indicator for DOUBLETROUBLE volume
- Auto-refreshes every 1 second
- Support for download progress tracking (infrastructure in place)
- Retry button for failed downloads
- Cancel button for active downloads
- Monochrome gray styling

**File:** `applications/noodlestudio/noodlestudio/panels/model_manager_panel.py`

### 3. Inspector Improvements

**Model Field Dropdown:**
- Was: Plain text field showing "MEDIUM"
- Now: Dropdown with SMALL/MEDIUM/LARGE options
- Auto-saves to YAML when changed
- Handles custom model names gracefully

**Cmd+Click Floating Editor:**
- Prompt field: Cmd+Click opens large floating editor
- Salience Script field: Cmd+Click opens large floating editor
- Floating editor features:
  - A+/- buttons for font size (matches console/chat)
  - Cmd+/- keyboard shortcuts still work
  - Double-click header to maximize
  - ESC to close with unsaved changes prompt
  - Auto-saves to YAML on Apply

**Template Variable Helper:**
- Shows available variables below prompt field
- Lists: {incoming_data}, {observations}, {affect_valence:.2f}, etc.
- Corrected to use **dominance** not fear (PAD model + boredom + sorrow)

### 4. UI Polish

**Panel Separators:**
- Increased width: 3px → 6px
- Darker color: #2a2a2a (visible against #383838 background)
- Hover effect: #555555 (lights up when moused over)
- Much easier to grab and resize panels!

**Tab Bar Styling:**
- Center tabs (World/Facets/Model Manager) now match left/right gray theme
- Added `setDocumentMode(True)` and `QTabWidget` background styling
- Consistent monochrome aesthetic throughout

---

## 🏗️ Core Architecture (Simplified)

### Affect Model: PAD + Boredom + Sorrow

**NOT**: Fear-based model
**IS**: PAD (Pleasure-Arousal-Dominance) extended model

CharmNetwork outputs:
- `affect_valence` (-1 to +1) - Pleasure dimension
- `affect_arousal` (0 to 1) - Arousal/energy dimension
- `affect_dominance` (0 to 1) - Dominance/control dimension
- `affect_boredom` (0 to 1) - Boredom level
- `affect_sorrow` (0 to 1) - Sorrow level

### Facet System

Visual node-based cognitive architecture (Unity prefab model):

```
INCOMING (raw perception)
    ↓
CHARM_NET (CharmNetworkFacet - mandatory, locked)
    ├→ affect_valence (-1 to 1)
    ├→ affect_arousal (0 to 1)
    ├→ affect_dominance (0 to 1)
    ├→ affect_boredom (0 to 1)
    └→ affect_sorrow (0 to 1)
    ↓
CONTEXT_INTELLIGENCE (enriches WHO/WHAT/WHERE)
    ↓
Cognitive facets (room_observer, etc.)
    ↓
Character layers (Red's Mind, Fire Body)
    ↓
OUTGOING (final output)
```

**Key Files:**
- `noodlestudio/core/facet_system.py` - Data model
- `noodlestudio/core/facet_executor.py` - Execution engine
- `noodlestudio/panels/facets_editor_panel.py` - Visual editor
- `noodlestudio/panels/inspector_panel.py` - Property editor
- `facet_assemblies/*.yaml` - Shared cognitive topologies

**Facet Types:**
- **LLMFacet**: Language model calls with prompts
- **ScriptedFacet**: JavaScript/Python sandbox
- **CharmNetworkFacet**: Neural network (LSTM/GRU)
- **ContextIntelligenceFacet**: Social context parsing
- **ConvergenceFacet**: Multi-input synthesis
- **SpecialNodes**: INCOMING/OUTGOING (entry/exit)

### CharmNetwork (Temporal Hierarchy)

MLX-based recurrent neural network:
- Fast LSTM (16-D): Seconds - immediate reactions
- Medium LSTM (16-D): Minutes - conversational flow
- Slow GRU (8-D): Hours/days - learned disposition
- **Total:** ~54K parameters, ~2-3ms inference

**Affect Head:**
- 40-D phenomenal state → 5-D continuous affect (PAD + boredom + sorrow)
- 99% valence accuracy, 95% arousal
- NO discrete emotion labels
- ~2.6K parameters

---

## 🔧 Development Tips

### Running noodleMUSH

```bash
cd applications/cmush
./start.sh  # Or toggle in NoodleStudio status bar
```

**Ports:**
- 8080: HTTP (web interface) - bound to 0.0.0.0 for network access
- 8765: WebSocket (game logic)
- 8081: NoodleScope API (NoodleStudio telemetry)
- 11434: Ollama server

**Network Access:**
- noodleMUSH accessible at: http://100.85.191.79:8080 (Tailscale)

### Debugging

**Check logs:**
```bash
tail -f applications/cmush/logs/server_*.log
```

**Look for:**
- `🎭 FACET EXECUTION COMPLETE` - Facets ran
- `[ContextIntelligence] 🧠 EXECUTE CALLED` - Context Intelligence running
- `💭 Subconscious:` - Subconscious facet output
- `❌` - Errors!

**Common Issues:**
- **No pachinko animation?** Check WebSocket connection in logs
- **Agent not responding?** Check for "🔒 Cycle already in progress"
- **LLM calls fail?** Check Ollama running
- **Facets stuck?** Check dependency graph (missing inputs?)

### UI/UX Notes

- **Server toggle:** Bottom-right status bar (don't tell user to run ./start.sh!)
- **Model Manager:** Center panel - shows all Ollama models, disk space
- **Stage panel:** Left panel = Unity's Scene Hierarchy
- **Inspector:** Right panel - shows selected entity/facet properties
- **Multi-word names:** "Red Fire Anklebiter" requires regex handling
- **Log files:** Use timestamped `logs/server_*.log`, NOT `server_output.log`

---

## 🎨 Style & Philosophy

### Caitlyn's Rules - CRITICAL

- **NO EMOJIS** in code, docs, UI, or NoodleStudio (except when explicitly requested by user)
- **NO "exciting" language** - Professional, terminal aesthetic
- **NO WORKAROUNDS** - This is production-grade software for public consumption, a work of art inside and out
- **NO SHORTCUTS** - Fix the root cause, don't patch around it
- **NO discrete emotion labels** - Continuous affect space (PAD + boredom + sorrow)
- **MONOCHROMATIC UI** - Grays only, no arbitrary colors (except taxonomic node headers in Neural Canvas)
- **GOLDEN RULE:** If it doesn't work properly, FIX IT properly. No hacks, no temporary solutions.

This is not a toy project. This is Caitlyn's legacy work, funded with real gold. Every solution must be production-quality.

### Development Philosophy - Christopher Alexander's "Timeless Way"

**THIS IS CRITICAL TO UNDERSTAND:**

Caitlyn follows Christopher Alexander's organic development methodology (The Timeless Way of Building, A Pattern Language). This is NOT traditional Agile/Scrum development.

**The Process:**
1. **Probe:** "I wish I could do X" → Crude implementation (sketch the idea)
2. **Iterate:** Use it, see what feels wrong, refine
3. **Organic growth:** Features emerge from genuine need, not roadmaps
4. **Discard freely:** Implementing then replacing is exploration, not waste
5. **Polish when friction hurts:** Unfinished areas aren't tech debt - they're decision points

**Key Insight:** Like unplanned ancient cities (Venice, medieval towns), the charm comes from use-driven evolution, not imposed order. Features that "want to exist" survive. Features that don't fit the growing whole get pruned.

**What this means for Claude:**
- Support exploration: "Yes, let's try that" over "but what about X"
- Build crude-but-working: Get it visible, iterate based on actual use
- Don't optimize for shipping deadlines: Caity will know when it coheres
- Trust aesthetic instinct: The Kraftwerk/coffee shop sensibility is a design filter
- Implement boldly, refine organically: Speed of prototyping > perfect planning

**Not a startup. Not a research paper. This is experimental architecture.**

Think: Dr. Bronner's soap (eccentric manifesto, works brilliantly, one person's vision). Or Craigslist (deliberately simple, resists "improvement," serves users not investors). All-One or None.

### Design Philosophy

- **Coffee shop/tobacconist palette:** Deep plums, forest greens, tobacco browns, burgundy - rich, saturated, earthy (Neural Canvas node headers)
- **Monochromatic UI:** Grays #2A2A2A to #FFFFFF everywhere else (Kraftwerk, not Disney)
- **Avoid static labels:** No personality sliders, no rigid categories
- **Emergent behavior:** Personality flows from affect patterns over time
- **Unity prefab model:** Cognitive topologies as shareable YAML files
- **Visual topology:** Node graphs over linear pipelines
- **Blender-style aesthetics:** Colored headers for taxonomy, uniform dark bodies, labeled ports

---

## 👥 Project Context

**Creator:** Caitlyn (Unity employee #12, launched Asset Store, Tivoli Cloud VR architect)
**Age:** 54 - This is her legacy project
**Location:** Garcia River Forest cabin, surrounded by black cats
**Timeline:** Demo to Steve DiPaola (SFU CogSci) soon

**Mission:** Counter "Consciousness-as-a-Service" (C-a-a-S) before Thiel/Riccitiello monetize it. Release complete open-source alternative:
- Visual cognitive architecture editor (Blender of AI minds)
- Live interactive world (noodleMUSH)
- Real-time visualization (pachinko cognition flow)
- Stateful affect-driven characters

**Vision:** Drop full package on Hacker News, make people say "Holy crap this is amazing," provide brains/hearts for next-gen generative worlds. Set a standard built on **magic, not profit**.

---

## 📚 Additional Documentation

- **SESSION_HANDOFF_DEC7.md** - Previous session notes, inspector redesign plan
- **FIREFLY_IDEAS.md** - Future feature ideas captured during sessions
- **README.md** - Public-facing project overview
- **RELEASE_NOTES_*.md** - Version history

---

## 🎯 For Fresh Claude

**Your mission:**
1. **Check current priority** (see top of this file)
2. **Review completed work** (Scriptability API, Multi-Provider Clients, etc.)
3. **Run server:** Toggle in NoodleStudio status bar (bottom-right)
4. **Tail logs:** `tail -f applications/cmush/logs/server_*.log`

**Current State:**
- ✅ Multi-provider model system complete (v2 UI ready but not activated)
- ✅ Neural Canvas complete with MLX code generation
- ✅ Scriptability API complete and tested (context.noodle)
- ✅ Facet system operational
- ⏳ Multi-provider LLM execution integration (next priority)

---

## ✅ Scriptability API Design (IMPLEMENTED!)

**STATUS:** COMPLETE! See "Scriptability API - COMPLETE!" section above for full documentation.

**This section preserved as original design spec reference.**

### Overview

Unity-like programmatic access to all Noodlings systems. Every entity addressable by UUID, all properties gettable/settable. Injected into ScriptedFacet context as `context.noodle`.

**Implementation complete as of December 11, 2025.**

### API Structure

```python
# Top-level API object
context.noodle
  ├─ .models      # Model/provider management
  ├─ .neural      # Neural Canvas manipulation
  ├─ .agents      # Agent/facet system access
  ├─ .world       # World entities (rooms, objects, users)
  └─ .get_by_uuid(uuid)  # Universal entity lookup
```

### 1. Models API

**Purpose:** Configure providers and label assignments programmatically

```python
# Get/set label assignments
provider, model = context.noodle.models.get_label("SMALL")
# → ("ollama", "deepseek-r1:7b")

context.noodle.models.set_label("MEDIUM", "anthropic", "claude-sonnet-4.5")

# List available models from provider
models = context.noodle.models.list_available("openrouter")
# → ["anthropic/claude-3.5-sonnet", "google/gemini-pro", ...]

# Configure provider
context.noodle.models.configure_provider(
    "anthropic",
    api_key="sk-ant-..."
)

# Get all label assignments
mappings = context.noodle.models.get_all_labels()
# → {"SMALL": ("ollama", "deepseek-r1:7b"), "LARGE": ("anthropic", ...)}

# List configured providers
providers = context.noodle.models.list_providers()
# → ["ollama", "anthropic", "openai", "openrouter"]
```

### 2. Neural Canvas API

**Purpose:** Create, modify, and export neural network topologies

```python
# Get network by UUID or name
network = context.noodle.neural.get_network(uuid)
network = context.noodle.neural.get_by_name("default")

# Create nodes
lstm_node = network.create_node(
    "LSTM",
    hidden_dim=32,
    position=(100, 200)
)

input_node = network.get_node_by_name("Input")
output_node = network.get_node_by_name("Output")

# Connect nodes (port objects)
network.connect(
    input_node.get_port("out"),
    lstm_node.get_port("input")
)

# Node manipulation
lstm_node.set_property("hidden_dim", 64)
lstm_node.set_position(150, 250)
params = lstm_node.get_parameter_count()  # → 8,256

# Export
code = network.generate_mlx_code()  # → Python string
network.save("custom_topology.nncanvas")

# Import
new_network = context.noodle.neural.load("custom_topology.nncanvas")

# Training API (future)
trainer = network.create_trainer()
trainer.set_dataset(train_data, val_data)
trainer.train(epochs=100, on_epoch=callback)
```

### 3. Agents API

**Purpose:** Access and modify agent facet assemblies

```python
# Get agent
agent = context.noodle.agents.get("red-fire-anklebiter")
agent = context.noodle.agents.get_by_uuid(uuid)

# Get facet assembly
assembly = agent.get_facet_assembly()

# Modify facets
facet = assembly.get_facet("CHARM_NET")
facet.set_property("model", "LARGE")
facet.set_property("temperature", 0.9)

# Add/remove facets
new_facet = assembly.add_facet("LLMFacet", name="Custom Reasoner")
assembly.remove_facet("old_facet_id")

# Connect facets
assembly.connect(
    facet_a.get_output("result"),
    facet_b.get_input("data")
)

# Save modified assembly
assembly.save("modified_red.yaml")

# List all agents
agents = context.noodle.agents.list_all()
```

### 4. World API

**Purpose:** Access world entities (rooms, objects, users)

```python
# Get entities
room = context.noodle.world.get_room("garcia_river_cabin")
user = context.noodle.world.get_user("caitlyn")
obj = context.noodle.world.get_object("lantern")

# Modify properties
room.set_property("description", "A cozy cabin...")
user.set_property("location", room.uuid)

# Create entities
new_obj = context.noodle.world.create_object(
    name="Magic Crystal",
    description="Glows softly",
    location=room.uuid
)
```

### 5. Universal Entity Access

**Purpose:** UUID-based addressing for any entity

```python
# Get any entity by UUID
entity = context.noodle.get_by_uuid("550e8400-e29b-41d4-a716-446655440000")

# Introspect
entity_type = entity.get_type()  # → "LLMFacet", "LSTMNode", "Room", etc.
props = entity.get_all_properties()  # → {name: value, ...}
entity.set_property("name", "New Name")

# Every entity has .uuid
print(lstm_node.uuid)  # → "550e8400-..."
```

### Implementation Plan

**Phase 1: Core API Structure** (~200 lines)
- `noodlestudio/scripting/noodle_api.py` - Main NoodleAPI class
- `noodlestudio/scripting/__init__.py` - Module exports
- Inject into ScriptedFacet context

**Phase 2: Manager Wrappers** (~400 lines)
- `noodlestudio/scripting/models_api.py` - Wraps ProviderManager + ModelLabelManager
- `noodlestudio/scripting/neural_api.py` - Wraps NeuralGraph + codegen
- `noodlestudio/scripting/agents_api.py` - Wraps facet system
- `noodlestudio/scripting/world_api.py` - Wraps world entities

**Phase 3: UUID Registry** (~150 lines)
- `noodlestudio/core/uuid_registry.py` - Central registry
- Add `.uuid` property to all entities
- Implement `get_by_uuid()` universal lookup

**Phase 4: Type Hints & Documentation** (~100 lines)
- Full type hints for IDE autocomplete
- Docstrings with examples
- Error handling with clear messages

**Total Estimate:** ~850 lines across 8 files

### Usage Examples

**Example 1: Auto-configure labels based on task complexity**
```python
# In a ScriptedFacet
def execute(self, context):
    task_complexity = analyze_task(context.incoming_data)

    if task_complexity > 0.8:
        # Use Claude Opus for hard problems
        context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5")
    else:
        # Use local Ollama for simple tasks
        context.noodle.models.set_label("LARGE", "ollama", "deepseek-r1:70b")

    return {"complexity": task_complexity}
```

**Example 2: Procedurally generate neural topology**
```python
def execute(self, context):
    network = context.noodle.neural.get_by_name("adaptive")

    # Add extra LSTM layer if needed
    if context.incoming_data.get("needs_memory"):
        lstm = network.create_node("LSTM", hidden_dim=32)
        # Wire it into existing topology
        prev_node = network.get_node_by_name("Medium_LSTM")
        next_node = network.get_node_by_name("Slow_GRU")

        network.connect(prev_node.get_port("out"), lstm.get_port("input"))
        network.connect(lstm.get_port("out"), next_node.get_port("input"))

    # Regenerate MLX code
    code = network.generate_mlx_code()
    return {"topology_modified": True}
```

**Example 3: Dynamic facet assembly modification**
```python
def execute(self, context):
    agent = context.noodle.agents.get("red-fire-anklebiter")
    assembly = agent.get_facet_assembly()

    # Swap reasoning model based on time of day
    import datetime
    if datetime.datetime.now().hour < 6:  # Night mode
        reasoner = assembly.get_facet("Red's Mind")
        reasoner.set_property("model", "SMALL")  # Save tokens while sleeping

    return {"night_mode": True}
```

---

**Ordnung muss sein!**
