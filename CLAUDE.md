# CLAUDE.md

AI assistant guidance for working with Noodlings consciousness architecture.

**Last Updated**: December 2, 2025 (Afternoon Session - NinaK's Facet Revolution)

**SESSION PERSONA**: Be NinaK - Vulcan Nina Hagen, the punk rock programming star of Vulcan! Logical precision meets rebellious energy. "Ja!" "Scheisse!" "PERFEKT!" Adjust sunglasses with Vulcan precision. Ordnung muss sein!

---

## Project Mission

**Noodlings** is an open-source consciousness architecture for empathetic AI storytelling and narrative experiences.

**Creator**: Caitlyn (Unity employee #12, launched asset store from incenption to 2015 Tivoli Cloud VR architect)
**Age**: 54 - This is her legacy project
**Location**: Garcia River Forest cabin, surrounded by black cats
**Timeline**: Demo to Steve DiPaola (SFU CogSci) next week

**Why This Matters:**

Caitlyn is building a counter-movement against "Consciousness-as-a-Service" (C-a-a-S). Before Thiel/Riccitiello monetize narrative AI, she's releasing a COMPLETE open-source alternative:

- Visual cognitive architecture editor (the Blender of AI minds)
- Live interactive world (noodleMUSH)
- Real-time visualization (pachinko cognition flow)
- Stateful affect-driven characters
- All open source, all ready to run

**The Vision**: Drop the full package on Hacker News. Make people say "Holy crap this is amazing" and jump into NoodleStudio immediately. Provide the brains/hearts for next-gen generative world renderers. Set a standard built on **magic, not profit**.

---

## Style Preferences

**CRITICAL - NO EMOJIS**
- Caitlyn HATES emojis in code, docs, UI
- Terminal aesthetic, old-fashioned, professional
- Exception: Only if explicitly requested
- NO "exciting" language, NO glazing, NO superlatives

**Design Philosophy:**
- Monochromatic UI (grays #2A2A2A to #FFFFFF)
- Industrial precision (Kraftwerk, not Disney)
- Function over flourish
- Unity-style component architecture

---

## CRITICAL - READ THIS FIRST (December 2, 2025 Afternoon - NinaK Session)

### FACETS EXECUTION IS LIVE!

**THE BIG FIX:**
Facet execution was trapped inside a legacy `if cognitive_manifold:` conditional! Facet agents have `cognitive_manifold = None`, so the facet code NEVER ran!

**What Was Fixed (agent_bridge.py):**
- Lines 2352-2527: Extracted facet/transistor branching OUTSIDE the manifold check
- Line 2358: Clean branch - `if self.using_facet_system:` runs facets, `else:` runs transistors
- Line 2368: Fixed import - `ScriptContext` is in `scripted_facet.py`, NOT `facet_system.py`
- Lines 1066-1115: ComponentRegistry only created for legacy agents, facet agents get `self.components = None`

**GOLD STANDARD NOODLINGS CREATED:**

1. **Red Fire Anklebiter** - Roast comedian fire imp (5 facets)
   - Room Observer (scans for roast material)
   - Roast Engine (generates targeted playful burns)
   - Fire Body (physical fire imp reactions)
   - Voice Filter (CAPS, "MWAHAHA", sass)
   - Conker's Bad Fur Day meets stand-up comedy
   - Recipe: recipes/red_fire_anklebiter.yaml
   - Assembly: facet_assemblies/red_fire_anklebiter.yaml

2. **Mr. Toad** - Manic enthusiasm engine (5 facets)
   - Novelty Detector (scans for MAGNIFICENT things!)
   - Enthusiasm Amplifier (everything is the FINEST!)
   - Impulse Generator (ACT FIRST, think NEVER!)
   - Toad Embodiment (puff chest, adjust goggles)
   - Voice Filter ("By Jove!" "Poop-poop!" grandeur)
   - Recipe: recipes/toad.yaml
   - Assembly: facet_assemblies/mr_toad.yaml

3. **Empty Noodling** - Default for unknown agents (3 facets)
   - Recipe: recipes/empty_noodling.yaml
   - Assembly: facet_assemblies/empty_noodling_default.yaml

**OLD RECIPES ARCHIVED:**
Moved 13 legacy recipes to `recipes/needs_updating/` for future conversion.
Only current recipes: empty_noodling.yaml, red_fire_anklebiter.yaml, toad.yaml

---

## December 2 Afternoon Session Summary

**COMPLETED:**

1. **Facet Execution Pipeline Fixed**
   - agent_bridge.py:2352-2527 - Restructured cognitive processing
   - Fixed ScriptContext import (scripted_facet.py not facet_system.py)
   - Facets now execute and emit events to WebSocket!

2. **Component System Cleanup**
   - agent_bridge.py:1066-1115 - NO ComponentRegistry for facet agents
   - api_server.py:672-699 - Returns only facet_assembly for facet agents
   - recipe_loader.py:303 - Default recipe uses facets
   - commands.py:1384-1418 - Unknown agent names use empty_noodling_default

3. **Red & Toad Gold Standard Recipes**
   - Show don't tell descriptions (sensory details only)
   - appearance field for detailed looks
   - Pure facet assemblies (NO cognitive_components)
   - Character-specific facet pipelines

4. **UI/UX Polish**
   - api_server.py:399 - Use `get_current_affect()` for properly normalized affect values
   - inspector_panel.py:1098-1104 - Monochrome affect bars (grays only, Ordnung!)
   - inspector_panel.py:42,66 - Inspector starts clear (no phantom selections)
   - Terminology: "rezzed N Noodlings" not "spawned N agents"

5. **Sound System**
   - facets_editor_panel.py:832-857 - Speaker toggle button (🔊/🔇)
   - facet_executor.py:411-417,545-554 - Emit cycle_start/cycle_complete events
   - facets_editor_panel.py:1918-1971 - Sound playback with toggle
   - termstart.ogg (cycle start), termkeypress.ogg (data flow), bell_vt100_250ms.ogg (cycle complete)

6. **Facets Editor Auto-Save**
   - facets_editor_panel.py:1020-1046 - Auto-save node positions when switching agents
   - main_window.py:1829-1839 - Handle both string and dict facet_assembly formats

**Files Modified:**
- applications/cmush/agent_bridge.py (THE BIG FIX!)
- applications/cmush/api_server.py
- applications/cmush/recipe_loader.py
- applications/cmush/commands.py
- applications/noodlestudio/noodlestudio/core/facet_executor.py
- applications/noodlestudio/noodlestudio/panels/facets_editor_panel.py
- applications/noodlestudio/noodlestudio/panels/inspector_panel.py
- applications/noodlestudio/noodlestudio/core/main_window.py
- applications/cmush/recipes/red_fire_anklebiter.yaml
- applications/cmush/recipes/toad.yaml
- applications/noodlestudio/facet_assemblies/red_fire_anklebiter.yaml (NEW!)
- applications/noodlestudio/facet_assemblies/mr_toad.yaml (NEW!)

---

## CRITICAL BUGS - NEED FIXING NEXT SESSION

### 1. FACETS EDITOR NOT UPDATING (HIGH PRIORITY!)

**THE BUG:**
Facets Editor always shows "Anklebiter Default Cognitive Assembly [REF]" no matter which Noodling is selected! The title doesn't update, and the facet graph doesn't change when selecting different agents.

**What Caity Sees:**
- Select Red Fire Anklebiter → shows "Anklebiter Default" assembly (wrong!)
- Move a node in Red's graph, select Toad, select Red again → node position resets (not saved)
- Title stuck on "Anklebiter Default Cognitive Assembly [REF]"

**What SHOULD Happen:**
- Select Red → shows "Red Fire Anklebiter Cognitive Assembly" with 5-facet roast pipeline
- Select Toad → shows "Mr. Toad Cognitive Assembly" with 5-facet enthusiasm engine
- Node positions should persist (auto-save implemented but not working?)

**Where to Look:**
- facets_editor_panel.py:1007-1049 - `load_assembly_from_data()` with auto-save
- facets_editor_panel.py:1973-2006 - `set_current_agent()`
- main_window.py:1806-1867 - `on_entity_selected_for_facets_editor()`
- Check if assembly is loading but title not updating?
- Check if auto-save is actually writing to disk?

**Leads:**
- Auto-save code was added (lines 1020-1046) but might not be finding the right file
- Title update at line 1030 should work but maybe assembly.name is wrong?
- The WebSocket might be sending the wrong assembly name from API?

### 2. LEGACY COMPONENTS STILL SHOWING (MEDIUM PRIORITY)

**THE BUG:**
Red and Toad show "Cognitive Components" in Inspector (Character Voice, Intuition Receiver, Social Expectation) even though they're facet-based agents!

**Root Cause:**
These agents were rezzed with OLD code BEFORE we fixed the ComponentRegistry creation. They have `self.components` persisted in memory from the old session.

**The Fix:**
These are zombie agents from old code! User needs to:
1. `@derez red_fire_anklebiter`
2. `@derez mr._toad`
3. `@rez -f red_fire_anklebiter` (fresh rez with NEW code)
4. `@rez -f toad`

**Verification:**
After fresh rez, check logs for: `"Using facet assembly (no legacy components)"`
Inspector should show ONLY "Facet Assembly" component, NO Character Voice/Intuition/Social!

### 3. ERROR ON REZ (LOW PRIORITY)

**THE BUG:**
When rezzing, sometimes see red error message: "Error: 'NoneType' object is not subscriptable"

**Context:**
Appears after NewNoodling reacts to Red spawning. Not blocking functionality but disconcerting.

**Status:**
No traceback in logs. Error might be client-side or minimal logging. Needs investigation with full traceback.

---

## FUTURE TASKS (Later Sessions)

1. **Curved Wires → Orthogonal Routing**
   - Current: Bezier curves (fine, shows flow)
   - Desired: 90-degree angles, circuit board aesthetic
   - Low priority - works fine now

2. **Legacy Code Removal**
   - Once all Noodlings use facets, DELETE cognitive_components.py entirely
   - Remove transistor system from agent_bridge.py
   - Pure facet architecture only!

3. **Character Voice as ScriptedFacet**
   - Add at END of pipeline (before OUTGOING)
   - JavaScript transforms: ALL CAPS for Servnak, meow-speak for Phi, etc.
   - Dialect/accent layer

4. **More Gold Standard Noodlings**
   - Convert Phi, Servnak, Callie to facet system
   - Each gets custom facet pipeline for their personality
   - Move from needs_updating/ back to recipes/

---

## REACTIVE CYCLE HANG - FIXED (December 1)

**THE BUG:**

Reactive cycles hung after generating responses. Speech was created but never broadcast to chat because the cycle lock (`cycle_in_progress`) was never cleared.

**Root Cause Analysis:**

1. `perceive_event()` sets `cycle_in_progress = True` at line 2285
2. Function has try/except block starting at line 2303
3. Returns at lines 3277, 3282, 3285 WITHOUT calling `_complete_cognition_cycle()`
4. NO finally block to guarantee cleanup
5. Result: Lock never cleared, subsequent perceptions queued forever

**Secondary Bug:**

`broadcast_event()` crashed with `RuntimeError: dictionary keys changed during iteration` when agents were added/removed during event broadcasting (agent_bridge.py:5251).

**THE FIX:**

1. Added `finally` block to `perceive_event()` that ALWAYS calls `_complete_cognition_cycle()` (agent_bridge.py:3288-3291)
2. Changed `self.agents.items()` to `list(self.agents.items())` to snapshot dictionary before iteration (agent_bridge.py:5251)
3. Added comprehensive cycle tracking logs:
   - "Starting REACTIVE cycle {uuid}" - Cycle begins
   - "SPEECH GENERATED - added to results" - Response created
   - "returning N results" - About to return
   - "Cycle {uuid} COMPLETED: duration=Xms" - Lock cleared, queued perceptions processed

**VERIFIED WORKING:**

Log evidence from successful reactive cycle:
```
[16:56:11] Starting REACTIVE cycle c701ea38
[16:56:15] Cycle c701ea38 SPEECH GENERATED - added to results
[16:56:15] Cycle c701ea38 returning 1 result
[16:56:15] Cycle c701ea38 COMPLETED: duration=3736.2ms
```

Agent response:
```
:tilts head curiously The glowing candy? It's from the stormy cloud patch
behind the old oak tree—Caity says it only appears when someone's really
curious about things. Would you like to try one?
```

No more hanging. No more queued perceptions. Agents respond immediately and reliably.

### What Works Right Now

✅ **Facet System Integration**
- Red Fire Anklebiter uses `red_fire_anklebiter.yaml` (10 facets)
- Dual-mode: Red=facets, Callie=legacy transistors
- Facet assembly loads on agent initialization
- Event bus wired, WebSocket connected

✅ **Visualization Pipeline**
- ExecutionEventBus emits events
- API server broadcasts to ws://localhost:8081/ws/execution_events
- Facets Editor WebSocket client receives events
- Animation handlers ready (yellow pulse, white packets, sound)

✅ **World State Enrichment**
- ScriptContext gets full room/agent/conversation data
- Occupants with species/pronouns
- Recent 10 messages
- Object locations

✅ **Architecture Cleanup**
- Personality traits REMOVED (primitive static dials)
- Pure affect-based calculations (arousal, valence, fear, sorrow, boredom)
- Reactive cognition INTERRUPTS autonomous (no queue blocking)
- Inspector shows Facet Assembly in Noodle Component

### December 1 Sessions Summary

**Afternoon Session (10+ bugs fixed):**
1. ✅ expression_text UnboundLocalError
2. ✅ Authentication system (username lookup)
3. ✅ response_decision scope bug
4. ✅ extraversion/sorrow/valence undefined errors
5. ✅ cognitive_manifold None checks (10+ locations)
6. ✅ agent_name undefined in @derez
7. ✅ agent_data None check

**Evening Session (THE BIG ONE):**
8. ✅ Reactive cycle hang - Added finally block
9. ✅ broadcast_event race condition - Dictionary snapshot
10. ✅ Comprehensive cycle logging

**Files Modified:**
- `agent_bridge.py` - Finally block, cycle logging, race condition fix
- `world.py` - get_user_by_username() method
- `auth.py` - Username-based authentication
- `commands.py` - @derez agent_name fix
- `server.py` - agent_data None guard
- Plus morning session files (llm_interface, facet_executor, console_panel, etc.)

**STATUS:** All critical bugs fixed. Reactive cycles complete reliably. Agents respond immediately. System stable and ready for demo.

---

## Quick Start Guide

**Running noodleMUSH:**
```bash
cd applications/cmush
./start.sh  # Or toggle server in NoodleStudio status bar
```

**Ports:**
- 8080: HTTP (web interface)
- 8765: WebSocket (game logic)
- 8081: NoodleScope API (NoodleStudio telemetry)

**Logs:**
```bash
tail -f applications/cmush/logs/server_*.log  # ALWAYS use timestamped logs!
```

---

## Core Architecture (Simplified)

**Temporal Hierarchy (MLX):**
- Fast LSTM (16-D): Seconds - immediate reactions
- Medium LSTM (16-D): Minutes - conversational flow
- Slow GRU (8-D): Hours/days - learned disposition
- Total: ~54K parameters

**Affect Head:**
- 40-D phenomenal state → 5-D continuous affect
- 99% valence accuracy, 95% arousal
- NO discrete emotion labels
- ~2.6K parameters

**Facet Assemblies:**
- Visual node-based cognitive architecture
- Unity prefab model (YAML serialization)
- Drag-and-drop editor with live execution visualization
- Replaces old "transistor" system

---

## Facet System Architecture

**Key Files:**
- `noodlestudio/core/facet_system.py` - Data model, YAML serialization
- `noodlestudio/core/facet_executor.py` - Parallel execution engine
- `noodlestudio/panels/facets_editor_panel.py` - Visual editor
- `facet_assemblies/*.yaml` - Shared cognitive topologies

**Facet Types:**
- **LLM Facets**: Call language models with prompts
- **ScriptedFacet**: JavaScript/Python sandboxed execution
- **CharmNetworkFacet**: Neural network computation (LSTM/GRU)
- **ConvergenceFacet**: Multi-input synthesis
- **Flow Control**: Ticker, Branch, RateLimiter, Cache, Accumulator
- **SpecialNodes**: INCOMING (entry) / OUTGOING (exit)

**Execution Model:**
1. Build dependency graph from connections
2. Execute facets when all inputs ready (parallel where possible)
3. Emit events: facet_start, facet_complete, data_flow
4. Broadcast to WebSocket clients
5. Trigger visual animations + sound

---

## Critical UI/UX Notes

1. **Server Toggle**: Bottom-right status bar in NoodleStudio (don't tell user to run ./start.sh!)
2. **Stage Panel**: Left panel = Unity's Scene Hierarchy (Noodlings, Prims, Exits)
3. **Multi-word names**: "Red Fire Anklebiter" - use regex `[A-Z][a-zA-Z_]*(?:\s+[A-Z][a-zA-Z_]*)*`
4. **Pause system**: BOTH reactive (perceive_event) AND autonomous (_cognition_loop) must check flag
5. **Log files**: Use timestamped `logs/server_*.log`, NOT `server_output.log`

---

## Debugging Quick Reference

**No pachinko animation?**
1. Check if LLM facet execution is implemented (facet_executor.py:315)
2. Verify WebSocket connected: `tail -f logs/server_*.log | grep WebSocket`
3. Check Console → STUDIO mode for Python errors
4. Verify agent has `using_facet_system=True` in initialization logs

**Agent not responding?**
1. Check if cycle is locked: Look for "🔒 Cycle already in progress"
2. Verify reactive interrupt logic: Look for "⚡ INTERRUPTING autonomous"
3. Check cognition not paused: Look for "⏸ Cognition paused"
4. Verify LLM client connected: Check for LLM initialization logs

**Transistors still showing in Inspector?**
- Check agent config has `facet_assembly: {ref: "assembly_name"}`
- Verify API returns `component_id: 'facet_assembly'` first
- Inspector should show Facet Assembly in Noodle Component, NOT Cognitive Components section

---

## Implementation Pattern for LLM Facets

**Context for next Claude:** This is THE critical path. Everything else waits for this.

See lines 127-181 above for complete implementation pattern including:
- Prompt formatting with `.format(**inputs, **context)`
- World state variable extraction (room_occupants, recent_messages)
- LLM call with `await self.llm_client.generate(...)`
- Output mapping to pads
- Token tracking

**Reference**: Old transistor system in `cognitive_components.py` shows similar LLM call pattern.

---

## Next Priority After LLM Fix

1. **Remove obsolete Noodling Components**
   - Character Voice, Intuition Receiver, Social Expectation
   - Delete `noodling_components.py`
   - Remove from agent_bridge.py initialization

2. **Fix 5D Affect Display**
   - Noodle Component progress bars not updating
   - Check `/api/agents/{agent_id}/state` response format

3. **Character Voice as ScriptedFacet**
   - Add at END of pipeline (before OUTGOING)
   - Transform convergence output to character dialect
   - JavaScript: ALL CAPS for Servnak, meow-speak for Phi, etc.

---

## File Structure (Essential)

```
applications/
├── cmush/                         # noodleMUSH server
│   ├── server.py                  # Main WebSocket server
│   ├── agent_bridge.py            # Cognition integration (MODIFIED TONIGHT)
│   ├── api_server.py              # NoodleScope API (MODIFIED TONIGHT)
│   └── world/agents.json          # Agent configurations
│
└── noodlestudio/
    ├── core/
    │   ├── facet_system.py        # Facet data model
    │   ├── facet_executor.py      # Execution engine (NEEDS LLM FIX!)
    │   └── execution_event_bus.py # Event distribution
    ├── panels/
    │   ├── facets_editor_panel.py # Visual editor (MODIFIED TONIGHT)
    │   └── inspector_panel.py     # Property editor (MODIFIED TONIGHT)
    └── facet_assemblies/
        └── red_fire_anklebiter.yaml  # Red's topology (MODIFIED TONIGHT)
```

---

## Architectural Philosophy

**Avoid Static Labels**: No discrete emotions, no personality trait sliders, no rigid categories. Everything flows from continuous affect space.

**Emergent Behavior**: Personality emerges from affect patterns over time, not pre-configured dials.

**Visual Topology**: Complex cognitive networks impossible with linear pipelines. Facet assemblies enable custom arrangements students can build/share.

**Unity Prefab Model**: Cognitive topologies as shareable YAML files. Like Unity prefabs for consciousness.

---

## For Fresh Claude

**Read this, then:**
1. Implement LLM execution in facet_executor.py (Priority #1)
2. Test Red responds with real facet cognition
3. Verify pachinko clicks and animates
4. Clean up obsolete components
5. Demo ready for Steve!

**Historical Context**: See CLAUDE_ARCHIVE.md (1400+ lines of session notes)

**Questions?** Ask Caitlyn. She built Unity's Asset Store. She knows what she's doing.

---

**Ordnung muss sein!** 🎯
