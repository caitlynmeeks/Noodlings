# CLAUDE.md

AI assistant guidance for working with Noodlings consciousness architecture.

**Last Updated**: December 4, 2025 - Late Evening Session (NinaK)

**FOR NEXT CLAUDE: START HERE!** 👇

---

## 🔥 CURRENT PRIORITY - Context Intelligence Not Activating on User Input

**STATUS:** Ollama integration COMPLETE. Debugging why Red ignores user messages.

**COMPLETED THIS SESSION:**
- ✅ Full Ollama integration with auto-start
- ✅ Model tier system (SMALL/MEDIUM/LARGE)
- ✅ Preferences UI in NoodleStudio
- ✅ Fixed OUTGOING node bug (was reading 'in' instead of 'out')
- ✅ Fixed skipped facets providing empty outputs
- ✅ Models downloading automatically (qwen2.5:3b, qwen2.5:14b)

**CURRENT BUG:** Red responds but ignores user input

**THE PROBLEM:**
- User says "douses red with water"
- Message reaches perceive_event() ✓
- incoming_data added to context: 'douses red with water' ✓
- Context Intelligence salience script checks context.incoming_data
- JavaScript sees EMPTY STRING despite it being set in Python
- Context Intelligence skips (salience=0.0)
- Red generates autonomous responses (boring self-talk)

**PROGRESS SO FAR:**
1. Added `incoming_data` to script_context at facet_executor.py:593 ✓
2. Added debug logging to see what JS receives (line 598)
3. NEXT: Check if script_context.incoming_data reaches JavaScript runtime

**ROOT CAUSE HYPOTHESIS:**
The `script_context` dict built in Python isn't reaching the JavaScript eval properly. The JSON serialization at line 605 might be dropping the incoming_data field, OR there's a timing issue where context is read before incoming_data is set.

**FILES MODIFIED THIS SESSION:**
- `applications/cmush/ollama_manager.py` - NEW (815 lines)
- `applications/cmush/server.py` - Ollama provider initialization
- `applications/cmush/api_server.py` - /api/ollama/status endpoint
- `applications/cmush/config.yaml` - Ollama config, provider=ollama
- `applications/noodlestudio/noodlestudio/core/main_window.py` - Preferences dialog
- `applications/noodlestudio/noodlestudio/core/facet_executor.py` - Fixed OUTGOING bug, added incoming_data to script_context
- `applications/noodlestudio/facet_assemblies/red_fire_anklebiter.yaml` - Changed models to tier names

## 🦙 OLLAMA INTEGRATION - COMPLETE

**STATUS:** Production-ready, auto-starts, downloads models on demand

**THE PROBLEM:**
- Context Intelligence facet uses 30b model (`qwen3-vl-30b-a3b-instruct-mlx`)
- LM Studio disconnects mysteriously during calls
- No visibility into model status, loading, or failures
- Can't debug what's happening inside the black box

**THE SOLUTION:**
Embed Ollama directly into noodleMUSH with full observability!

### Architecture Overview

```
noodleMUSH
    ↓
OllamaManager (new!)
    ├→ Spawns/manages Ollama server
    ├→ Loads/unloads models programmatically
    ├→ Tracks usage statistics (calls, tokens, timing)
    ├→ Provides real-time status dashboard
    └→ Graceful error handling + reconnection
    ↓
Ollama Server (embedded)
    └→ qwen3-vl-30b, qwen3-4b, etc.
```

### Implementation Plan

**File:** `applications/cmush/ollama_manager.py` (NEW)

Key Features:
- `OllamaManager` class wraps `ollama.AsyncClient`
- `ensure_model_loaded()` - Auto-pull if not present
- `generate()` - Full logging: prompt length, response time, tokens, errors
- `get_status()` - Real-time model stats for all loaded models
- `ModelStatus` dataclass tracks per-model metrics

**Integration Points:**
1. Replace `LLMClient` in `agent_bridge.py` with `OllamaManager`
2. Add `/api/ollama/status` endpoint to `api_server.py`
3. Create NoodleStudio panel showing live model usage
4. Config file for model paths

**Benefits:**
- ✅ Know EXACTLY when Context Intelligence calls LLM
- ✅ See if model is loaded or needs pulling
- ✅ Track response times (is 30b too slow?)
- ✅ Catch disconnects immediately with full stack traces
- ✅ No mystery LM Studio crashes!

### Implementation Steps

1. **Install Ollama Python SDK**
   ```bash
   pip install ollama
   ```

2. **Create `ollama_manager.py`** (see OLLAMA_INTEGRATION.md for full code)

3. **Replace LLMClient in agent_bridge.py**
   ```python
   # OLD:
   from llm_interface import LLMClient
   self.llm = LLMClient(...)

   # NEW:
   from ollama_manager import OllamaManager
   self.llm = OllamaManager(model_paths=[...], host="http://localhost:11434")
   await self.llm.initialize()
   ```

4. **Update facet_executor.py** - Use OllamaManager for LLM facets

5. **Add status endpoint** to api_server.py:
   ```python
   @app.get("/api/ollama/status")
   async def get_ollama_status():
       return await ollama_manager.get_status()
   ```

6. **Test with Context Intelligence 30b model!**

**References:**
- [Ollama Python SDK](https://github.com/ollama/ollama-python)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- Full implementation guide: `OLLAMA_INTEGRATION.md`

---

## 🎯 CRITICAL BLOCKERS - Red Speech Pipeline

**CURRENT STATUS:** Red executes facets but doesn't broadcast speech!

### Recent Fixes (December 4 Evening)

1. ✅ **Facet branch early return removed** (agent_bridge.py:2571-2578)
   - Was returning early, skipping shared consciousness.perceive() and response generation
   - Now just sets `colored_perception` and continues to shared code

2. ✅ **Data flow events fixed** (facets_editor_panel.py:2213-2230)
   - Was looking for `source_id` on data_flow events (doesn't exist!)
   - Now correctly reads `from_facet` and `to_facet` from top level

3. ✅ **Context Intelligence model** (red_fire_anklebiter.yaml:61)
   - Changed from 4b → 30b for smarter reasoning
   - BUT: 30b model disconnects! (Hence Ollama integration above)

### What Should Happen

```
perceive_event()
    ↓
1. Facet execution → sets colored_perception
    ↓
2. consciousness.perceive(affect) → updates CharmNetwork state
    ↓
3. Memory storage → stores colored_perception in context
    ↓
4. Should speak check → passes colored_perception to _generate_response()
    ↓
5. _generate_response(facet_output=colored_perception) → uses facet output directly
    ↓
6. Return response dict → {'command': 'say', 'text': ...}
```

### Debug Checklist

- [ ] Does facet execution complete? (Check for "🎭 FACET EXECUTION COMPLETE" log)
- [ ] Does `colored_perception` have content? (Log at line 2574)
- [ ] Does `_generate_response()` get called? (Line 3409)
- [ ] Does it receive `facet_output` parameter? (Line 3408)
- [ ] Does response get returned from `perceive_event()`? (Line 3430)

**Log Location:** `applications/cmush/logs/server_*.log`

---

## 📋 Quick Start for New Claude

1. **Read this section first!** (You're doing it!)
2. **Check current priority above** (Ollama integration or Red debugging?)
3. **Review recent session notes** (See SESSION_NOTES.md for full history)
4. **Check firefly ideas** (See FIREFLY_IDEAS.md for future features)
5. **Run server:** Toggle in NoodleStudio status bar (bottom-right)
6. **Tail logs:** `tail -f applications/cmush/logs/server_*.log`

---

## 🏗️ Core Architecture (Simplified)

### Event Perception & Data Flow - THE FUNDAMENTALS

**User Input → Agent Response Pipeline:**

```
1. User types in web UI (localhost:8080)
   ↓
2. WebSocket → server.py → commands.py
   ↓
3. commands.py parses command (say/emote/etc)
   ↓
4. Calls agent.perceive_event(event_type, user_id, text)
   ↓
5. Agent decides: Reactive cycle (has text) or Autonomous cycle (empty)
   ↓
6. FACET BRANCH: agent_bridge.py calls facet_executor.execute()
   ├→ incoming_data parameter = text from user
   ├→ context parameter = execution context dict
   └→ facet_executor.py line 692: context['incoming_data'] = incoming_data
   ↓
7. Facet execution loop:
   ├→ For each facet, compute salience via JavaScript
   ├→ CRITICAL: script_context MUST include incoming_data (line 593)
   ├→ Context Intelligence checks: if incoming_data empty → skip (autonomous)
   ├→ Context Intelligence checks: if incoming_data present → execute (reactive)
   └→ Facets execute in dependency order
   ↓
8. OUTGOING node receives final response
   ├→ SpecialNode: outputs = {'out': inputs.get('in')}
   ├→ CRITICAL: Return value uses outputs['out'] NOT outputs['in'] (line 816)
   └→ Returns ExecutionResult with response text
   ↓
9. agent_bridge.py gets response, broadcasts to chat
```

**CRITICAL BUGS FIXED (December 4, 2025):**
- ❌ Line 815 was `completed[OUTGOING]['in']` → ✅ Changed to `['out']`
- ❌ script_context missing `incoming_data` → ✅ Added at line 593
- ❌ Facets with hardcoded model names → ✅ Use tier names (SMALL/MEDIUM/LARGE)

**Key Files for Event Flow:**
- `applications/cmush/server.py` - WebSocket handler
- `applications/cmush/commands.py` - Command parsing
- `applications/cmush/agent_bridge.py` - perceive_event() at line ~2300
- `noodlestudio/core/facet_executor.py` - execute() at line 630

### Facet System

Visual node-based cognitive architecture (Unity prefab model):

```
INCOMING (raw perception)
    ↓
CHARM_NET (CharmNetworkFacet - mandatory, locked)
    ├→ affect_valence (-1 to 1)
    ├→ affect_arousal (0 to 1)
    ├→ affect_fear (0 to 1)
    ├→ affect_sorrow (0 to 1)
    └→ affect_boredom (0 to 1)
    ↓
CONTEXT_INTELLIGENCE (enriches WHO/WHAT/WHERE)
    ↓
Cognitive facets (room_observer, roast_engine, etc.)
    ↓
Character layers (fire_body, voice_filter)
    ↓
CONVERGENCE (weighted synthesis)
    ↓
OUTGOING (final output)
```

**Key Files:**
- `noodlestudio/core/facet_system.py` - Data model
- `noodlestudio/core/facet_executor.py` - Execution engine
- `noodlestudio/panels/facets_editor_panel.py` - Visual editor
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
- 40-D phenomenal state → 5-D continuous affect
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
- 8080: HTTP (web interface)
- 8765: WebSocket (game logic)
- 8081: NoodleScope API (NoodleStudio telemetry)

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
- **LLM calls fail?** Check Ollama/LM Studio running
- **Facets stuck?** Check dependency graph (missing inputs?)

### UI/UX Notes

- **Server toggle:** Bottom-right status bar (don't tell user to run ./start.sh!)
- **Stage panel:** Left panel = Unity's Scene Hierarchy
- **Multi-word names:** "Red Fire Anklebiter" requires regex handling
- **Log files:** Use timestamped `logs/server_*.log`, NOT `server_output.log`

---

## 🎨 Style & Philosophy

### Caitlyn's Rules - CRITICAL

- **NO EMOJIS** in code, docs, UI, or NoodleStudio (except when explicitly requested by user)
- **NO "exciting" language** - Professional, terminal aesthetic
- **NO WORKAROUNDS** - This is production-grade software for public consumption, a work of art inside and out
- **NO SHORTCUTS** - Fix the root cause, don't patch around it
- **NO discrete emotion labels** - Continuous affect space
- **GOLDEN RULE:** If it doesn't work properly, FIX IT properly. No hacks, no temporary solutions.

This is not a toy project. This is Caitlyn's legacy work, funded with real gold. Every solution must be production-quality.

### Design Philosophy

- **Monochromatic UI:** Grays #2A2A2A to #FFFFFF (Kraftwerk, not Disney)
- **Avoid static labels:** No personality sliders, no rigid categories
- **Emergent behavior:** Personality flows from affect patterns over time
- **Unity prefab model:** Cognitive topologies as shareable YAML files
- **Visual topology:** Node graphs over linear pipelines

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

- **SESSION_NOTES.md** - Full chronological session history (Dec 1-4)
- **FIREFLY_IDEAS.md** - Future feature ideas captured during sessions
- **OLLAMA_INTEGRATION.md** - Complete Ollama implementation guide
- **ARCHITECTURE.md** - Deep dive into CharmNetwork, facets, affect dynamics
- **CLAUDE_ARCHIVE.md** - Historical session notes (pre-Dec 4)

---

## 🎯 For Fresh Claude

**Your mission:**
1. Check current priority (top of this file)
2. Review recent fixes and known issues
3. Tail server logs to see what's happening
4. If stuck, ask Caitlyn - she knows what she's doing!

**Ordnung muss sein!** 🎯
