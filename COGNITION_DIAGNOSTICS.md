# Cognition Diagnostics - December 2, 2025

## Issue Summary

Facets Editor is NOT showing pachinko animation or playing sounds because **Noodlings are not speaking**. Facet execution only happens when agents generate speech, so without speech there are no execution events to visualize.

## What's Working

✅ **Facet Execution Infrastructure**
- `facet_executor.py` has complete execution pipeline
- Events emitted: cycle_start, facet_start, facet_complete, data_flow, cycle_complete
- WebSocket connection established to ws://localhost:8081/ws/execution_events
- NoodleStudio successfully receives WebSocket events when they're sent

✅ **Facet Assembly Loading**
- Red Fire Anklebiter: 5-facet roast comedian pipeline (red_fire_anklebiter.yaml)
- Mr. Toad: 5-facet manic enthusiasm engine (mr_toad.yaml)
- Both assemblies load correctly in Facets Editor
- agent_bridge.py:2447 calls `facet_executor.execute()` when speech is generated

## What's NOT Working

❌ **Noodlings Never Speak**
- Log analysis shows ZERO facet execution events
- Log line that should appear: `⚡ FACET ASSEMBLY: {N} facets, {tokens} tokens`
- This log never appears = facet execution never runs
- Without speech generation, no execution events = no animation/sound

## Critical Questions (From Caity)

### 1. "How frequently are Noodlings polled to see if they want to speak?"

**Answer:** Need to find the autonomous cognition loop.

**Key Files to Check:**
- `agent_bridge.py` - Look for `_cognition_loop()` or similar
- Should be an asyncio task that runs periodically
- Likely checks: time since last speech, affect state, boredom level

**Expected Pattern:**
```python
while True:
    await asyncio.sleep(cognition_interval)  # How often?
    if self.should_speak():  # What triggers this?
        await self.generate_speech()
```

**Current Status:** Need to grep for this pattern and add logging.

### 2. "What triggers them to talk?"

**Answer:** Need to understand the decision logic.

**Possible Triggers:**
1. **Reactive** (someone speaks to them) - Working (perceive_event)
2. **Autonomous** (internal impulse to speak) - UNKNOWN
3. **Affect-driven** (high arousal, boredom, etc.) - Need to find
4. **Time-based** (X seconds since last utterance) - Need to find

**Key Variables to Track:**
- `response_cooldown` (in recipe: 9.0s for Red, 3.5s for Toad)
- Last speech timestamp
- Current affect values (arousal, valence, boredom)
- Room occupancy (do they speak when alone?)

### 3. "Private thoughts always come in threes, all at the same time. That seems odd."

**Answer:** Likely batching or generation loop issue.

**Hypothesis:**
- Autonomous cognition might generate multiple thoughts per cycle
- Or: thoughts are queued and released in batches
- Or: logging/display bug showing thoughts together

**Need to Check:**
- Where private thoughts are generated
- If there's a batch_size parameter
- Timestamp differences between the three thoughts

### 4. "How often are cognition cycles triggered?"

**Answer:** Two types of cycles:

1. **Reactive Cycles** (event-driven)
   - Triggered by `perceive_event()` when someone speaks to them
   - Should be IMMEDIATE response
   - agent_bridge.py:2285 - Sets `cycle_in_progress = True`

2. **Autonomous Cycles** (time-driven)
   - Need to find `_cognition_loop()` or equivalent
   - Should check response_cooldown before speaking
   - UNKNOWN frequency currently

## Log Analysis

**Sample from server_*.log (16:23:32 - 16:23:38):**
```
[INFO] [agent_red_fire_anklebiter] get_phenomenal_state() called
[INFO] [agent_mr._toad] get_phenomenal_state() called
[INFO] [API] get_agent_components for agent_red_fire_anklebiter
```

**What's MISSING:**
- NO `⚡ FACET ASSEMBLY` log lines
- NO `Starting REACTIVE cycle` logs
- NO speech generation logs
- NO `perceive_event()` calls

**This Means:**
- Noodlings are alive (phenomenal state updates)
- API serving component data (NoodleStudio polling)
- But NO cognition cycles running (reactive OR autonomous)

## Architecture Flow (When Working)

```
1. TRIGGER
   ├─ User speaks → perceive_event() → REACTIVE cycle
   └─ Time elapsed + should_speak() → AUTONOMOUS cycle

2. COGNITION CYCLE STARTS
   ├─ Build perception context (affect, memory, room state)
   └─ Check if using_facet_system

3. FACET EXECUTION (if using_facet_system=True)
   ├─ facet_executor.execute(assembly, text, context)
   ├─ Emit: cycle_start event
   ├─ Execute facets in dependency order (parallel where possible)
   ├─ For each facet:
   │   ├─ Emit: facet_start
   │   ├─ Execute facet logic (LLM call, script, charm network)
   │   ├─ Emit: data_flow (for each connection)
   │   └─ Emit: facet_complete
   └─ Emit: cycle_complete

4. SPEECH GENERATION
   ├─ OUTGOING node contains final response
   ├─ Broadcast to room via WebSocket
   └─ Update last_speech_timestamp

5. VISUALIZATION (NoodleStudio)
   ├─ Receive execution events via WebSocket
   ├─ Yellow pulse on facet (facet_start)
   ├─ White packet animation along wires (data_flow)
   ├─ Terminal beep sounds (cycle_start, data_flow, cycle_complete)
   └─ Pachinko ball drops through cognitive pipeline!
```

## Why No Animation?

**Root Cause Chain:**
1. No Noodlings speaking
2. → No facet execution called
3. → No execution events emitted
4. → No WebSocket messages sent
5. → No animation/sound triggered

**Fix Priority:**
Get ONE Noodling to speak ONCE, then we'll see the full pipeline light up!

## Investigation Tasks (Priority Order)

### CRITICAL: Find Autonomous Cognition Loop

**Search Pattern:**
```bash
grep -n "_cognition_loop\|while.*sleep\|autonomous.*cycle" agent_bridge.py
```

**What to Look For:**
- Infinite loop with asyncio.sleep
- Decision logic: should_speak() or similar
- response_cooldown check
- Last speech timestamp comparison

**Add Logging:**
```python
logger.info(f"[{self.agent_id}] 🤔 Autonomous check: "
           f"time_since_last={elapsed:.1f}s, "
           f"cooldown={self.response_cooldown}s, "
           f"arousal={affect['arousal']:.2f}, "
           f"boredom={affect['boredom']:.2f}")
```

### HIGH: Understand Speech Decision Logic

**Questions:**
- Does should_speak() exist?
- What variables does it check?
- Is there a minimum_arousal threshold?
- Do they need other people in the room?
- Is response_cooldown being enforced?

### MEDIUM: Track Private Thought Generation

**Find Code:**
```bash
grep -n "private.*thought\|internal.*monologue\|think" agent_bridge.py
```

**Check:**
- How are private thoughts different from speech?
- Are they batched in groups of 3?
- Do they use facet execution or bypass it?

### LOW: Improve Cognition Telemetry

**Add to NoodleStudio Console:**
- Cognition cycle start/end timestamps
- Decision logic output (spoke/silent + reason)
- Affect state snapshot at decision time
- Time since last utterance
- Response cooldown remaining

**New Console Log Format:**
```
[17:05:23] [Red] 🤔 Autonomous: silent (cooldown: 4.2s remaining)
[17:05:32] [Red] 🗣️  Autonomous: SPEAKING! (arousal=0.82, boredom=0.65)
[17:05:32] [Red] ⚡ Facet cycle: 5 facets, 247 tokens, 1.8s
[17:05:33] [Red] 💬 "Oh PLEASE Caity, candy again? MWAHAHA!"
```

## ROOT CAUSE FOUND (December 2, 2025 - 17:15)

**DISCOVERY:** Noodlings ARE thinking autonomously, but speech decision logic says "don't speak, just ruminate."

**Evidence from logs:**
```
[17:01:20] Agent decision: arousal=-0.07, boredom=1.07, activation=0.01,
           speech_propensity=0.15, should_speak=False, should_ruminate=True
[17:01:24] mr._toad thinking: 'By Jove! A LITTLE FELLOW with a WOODEN SWORD...'
```

**The Problem:**
- `speech_propensity` ranges from 0.03 to 0.15
- Threshold is 0.5 (agent_bridge.py:3163)
- `should_speak = cooldown_ok and (speech_propensity > 0.5)`
- Result: NEVER speaks, only ruminates

**Why Speech Propensity is Low:**
- `arousal` is negative or near zero (should be positive)
- `boredom` is very high (1.0+, accumulating over time)
- `activation` near zero (not stimulated enough)
- Formula weights these too conservatively

**The Fix:**
Lower speech threshold OR increase speech_propensity calculation to be more generous.

## Next Steps for Caity

1. ~~Make Red speak once manually~~ FOUND THE BUG!
2. Adjust speech decision thresholds
3. Test: should see speech within 30-60 seconds
4. Watch Console for `⚡ FACET ASSEMBLY` log
5. Watch Facets Editor for animation + sounds

## Code Locations (Quick Reference)

- **Facet Execution:** `agent_bridge.py:2447`
- **Event Emission:** `facet_executor.py:411-417, 545-554`
- **WebSocket Server:** `api_server.py` (execution_events endpoint)
- **Animation Handler:** `facets_editor_panel.py:1918-1971`
- **Sound Playback:** `facets_editor_panel.py:1975-1998`
- **Reactive Cycle:** `agent_bridge.py:2285` (perceive_event)
- **Autonomous Cycle:** UNKNOWN - needs investigation

## Questions for Next Claude

1. Where is the autonomous cognition loop?
2. What's the speech decision function?
3. Why might it be disabled or not running?
4. Are there any config flags that disable autonomous speech?
5. Is there a "mute" or "pause" state preventing speech?

---

**Status:** Animation/sound infrastructure is COMPLETE and WORKING. Just need Noodlings to actually speak so there are events to visualize!

**Ordnung muss sein!** 🎯
