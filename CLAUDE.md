# CLAUDE.md

AI assistant guidance for working with Noodlings consciousness architecture.

**Last Updated**: December 1, 2025 (Evening Session with Ninak - REACTIVE CYCLE FIX!)

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

## CRITICAL - READ THIS FIRST (December 1, 2025 Evening)

### REACTIVE CYCLE HANG - FIXED!

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
