# Register-Based Accumulator Architecture

**Implementation Date**: November 26, 2025
**Status**: Production Ready
**Commit**: `341b20a`

---

## Overview

The **register-based accumulator architecture** transforms transistors from stateless on-demand generators into **stateful CPU-like registers** that accumulate output, wait for all registers to fill, then integrate in a single deliberate moment.

### The Problem (Old Architecture)

```
perceive("hi red")
  -> manifold.integrate_async()
    -> IntuitionTransistor.process() [generates, uses, discards immediately]
    -> AffectTransistor.process() [generates, uses, discards immediately]
    -> PersonalityTransistor.process() [generates, uses, discards immediately]
    -> blend outputs
    -> return result
```

**Issues**:
- No state persistence (outputs vanish after blend)
- No cycle boundaries (can't tell when cognition starts/ends)
- Race conditions (new perception during processing overwrites)
- No debugging (can't inspect what's loaded before integration)
- Confusing UX (mid-cycle overwrites)

### The Solution (Register Architecture)

```
perceive("hi red")
  [CYCLE START: uuid_a1b2c3d4]

  PHASE 1: FILL REGISTERS (parallel)
    IntuitionTransistor: EMPTY -> COMPUTING -> READY
    AffectTransistor: EMPTY -> COMPUTING -> READY
    PersonalityTransistor: EMPTY -> COMPUTING -> READY
    ... (all transistors fill in parallel)

  PHASE 2: VERIFY ALL READY
    check_all_registers_ready() -> true

  PHASE 3: PULL LEVER
    manifold.integrate_from_registers()
    (uses stored register contents, no re-processing)

  PHASE 4: GENERATE RESPONSE
    LLM generates speech/thought using manifold output

  PHASE 5: CLEAR REGISTERS
    All registers: READY -> EMPTY

  [CYCLE END: uuid_a1b2c3d4]
```

---

## Core Concepts

### Transistors as CPU Registers

Each transistor is a **register** that:
- Holds a "bullet in the chamber" (last computed output)
- Has explicit state: `EMPTY`, `COMPUTING`, `READY`, `ERROR`
- Persists until explicitly cleared
- Shows current contents in NoodleTuner even between cycles

Like a CPU:
```
LOAD R1 (IntuitionTransistor)    -> Register filled with contextual awareness
LOAD R2 (AffectTransistor)       -> Register filled with emotional impulse
LOAD R3 (PersonalityTransistor)  -> Register filled with personality filter
LOAD R4 (CulturalTransistor)     -> Register filled with belief filter
LOAD R5 (MemoryTransistor)       -> Register filled with relevant memories
LOAD R6 (EmbodyComponent)        -> Register filled with physical reactions

CHECK: All registers READY?
EXECUTE: PULL LEVER -> Manifold integrates all register contents -> Output
CLEAR: All registers back to EMPTY
```

### Cognition Cycles

A **cognition cycle** has explicit boundaries:
- **Start**: Agent perceives event, assigns cycle UUID
- **Fill**: Transistors load outputs into registers (parallel)
- **Verify**: Check all enabled registers ready
- **Integrate**: Manifold blends stored register contents
- **Respond**: Generate speech/thought from integrated output
- **Clear**: Empty all registers
- **End**: Cycle complete, ready for next perception

### State Persistence

Registers maintain state **between cycles**:
- Last output visible in NoodleTuner even after clearing
- `last_output_text`, `last_output_metadata`, `last_output_salience` preserved
- Register state transitions logged for debugging
- Cycle UUID tracks which perception generated which output

---

## Implementation

### CognitiveTransistor Base Class

**File**: `applications/cmush/cognitive_components.py:48-176`

**New Fields**:
```python
# REGISTER STATE (new architecture)
self.register_state = "empty"  # empty, computing, ready, error
self.register_output: Optional[TransistorOutput] = None
self.register_cycle_id: Optional[str] = None
self.register_timestamp: Optional[float] = None
```

**New Methods**:

#### `fill_register(input_text, context, cycle_id)`
Fills transistor's register with new output.

**Flow**:
1. Set `register_state = "computing"`
2. Set `register_cycle_id = cycle_id`
3. Set `register_timestamp = time.time()`
4. Call `process()` (subclass implements LLM generation)
5. Store output in `register_output`
6. Set `register_state = "ready"`
7. Update legacy fields for backwards compatibility

**Example**:
```python
output = await transistor.fill_register("hi red", context, "cycle_001")
print(transistor.register_state)  # "ready"
print(transistor.register_output.transformed_text)  # "Flames SURGING!"
```

#### `clear_register()`
Clears register after integration.

**Flow**:
1. Set `register_state = "empty"`
2. Clear `register_output = None`
3. Clear `register_cycle_id = None`
4. Clear `register_timestamp = None`

**Example**:
```python
transistor.clear_register()
print(transistor.register_state)  # "empty"
```

#### `is_register_ready()`
Checks if register contains valid output.

**Returns**: `True` if `register_state == "ready"` and `register_output is not None`

---

### CognitiveManifold Integration

**File**: `applications/cmush/cognitive_components.py:440-724`

**New Fields**:
```python
# REGISTER ACCUMULATOR STATE
self.current_cycle_id: Optional[str] = None
self.cycle_in_progress = False
self.registers_filled_count = 0
```

**New Methods**:

#### `fill_all_registers(input_text, context, cycle_id)`
**PHASE 1**: Fill all enabled transistor registers in parallel.

**Flow**:
1. Set `current_cycle_id = cycle_id`
2. Set `cycle_in_progress = True`
3. Create async tasks for all enabled transistors
4. `await asyncio.gather(*tasks, return_exceptions=True)`
5. Count successful fills in `registers_filled_count`
6. Log: `"{count}/{total} registers READY"`

**Example**:
```python
await manifold.fill_all_registers("hi red", context, "cycle_a1b2c3d4")
# Logs: "6/6 registers READY"
```

#### `check_all_registers_ready()`
Check if all enabled registers are ready for integration.

**Returns**: `True` if all enabled transistors have `register_state == "ready"`

**Example**:
```python
if manifold.check_all_registers_ready():
    print("All registers ready! Can integrate now.")
```

#### `integrate_from_registers(context)`
**PHASE 3**: Pull lever - integrate outputs from ALL registers.

**Flow**:
1. Verify all ready (warn if not)
2. Collect `register_output` from all enabled transistors
3. Call `_llm_weighted_blend()` with stored outputs
4. Store result in `last_output_text`
5. Return integrated output

**Example**:
```python
result = await manifold.integrate_from_registers(context)
# Uses STORED register contents (no re-processing)
```

#### `clear_all_registers()`
**PHASE 5**: Clear all registers after integration.

**Flow**:
1. Call `clear_register()` on all transistors
2. Set `cycle_in_progress = False`
3. Reset `registers_filled_count = 0`

**Example**:
```python
manifold.clear_all_registers()
# All registers: READY -> EMPTY
```

---

### Agent Cognition Flow

**File**: `applications/cmush/agent_bridge.py:2274-2296, 3401-3425, 3510-3532, 3678-3703`

#### perceive_event() - Speech Path

**Location**: `agent_bridge.py:2274-2296`

```python
# NEW ARCHITECTURE: Register-based accumulator
# PHASE 1: Fill all registers
await self.cognitive_manifold.fill_all_registers(text, context, self.current_cycle_uuid)

# PHASE 2: Verify ready (optional wait)
if not self.cognitive_manifold.check_all_registers_ready():
    logger.warning("Registers not all ready, waiting 0.5s...")
    await asyncio.sleep(0.5)

# PHASE 3: Pull lever - integrate
colored_perception = await self.cognitive_manifold.integrate_from_registers(context)

# ... (generate response)

# PHASE 5: Clear registers (cognition cycle complete)
self.cognitive_manifold.clear_all_registers()
```

#### _generate_rumination() - Rumination Path

**Location**: `agent_bridge.py:3510-3532`

```python
# NEW ARCHITECTURE: Register-based accumulator for rumination
# PHASE 1: Fill all registers
await self.cognitive_manifold.fill_all_registers(perception_text, context, self.current_cycle_uuid)

# PHASE 2: Verify ready (optional wait)
if not self.cognitive_manifold.check_all_registers_ready():
    logger.warning("Rumination registers not all ready, waiting 0.5s...")
    await asyncio.sleep(0.5)

# PHASE 3: Pull lever - integrate
colored_thought_seed = await self.cognitive_manifold.integrate_from_registers(context)

# ... (generate rumination)

# PHASE 5: Clear registers (cognition cycle complete)
self.cognitive_manifold.clear_all_registers()
```

#### Safety Nets

**Finally blocks** ensure registers always cleared:

```python
finally:
    # PHASE 5: Ensure registers always cleared (safety net)
    if self.cognitive_manifold.cycle_in_progress:
        logger.debug("Finally block clearing registers")
        self.cognitive_manifold.clear_all_registers()
```

---

### EmbodyComponent Fix

**File**: `applications/cmush/cognitive_components.py:2333-2384`

**Problem**: EmbodyComponent had no `process()` method, always showed empty output.

**Solution**: Added full `process()` implementation:

```python
async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
    """Generate embodied physical reactions to perception."""

    # Get custom prompt from prefab or use default
    prompt_template = self.active_prompt if self.active_prompt else self.DEFAULT_PROMPT

    # Extract affect from context
    affect = context.get('affect', [0]*5)
    valence, arousal, fear, sorrow, boredom = affect[:5]

    # Get body summary
    body_summary = self.GetSummary()

    # Format prompt
    prompt = prompt_template.format(
        input_text=input_text,
        body_summary=body_summary,
        valence=valence,
        arousal=arousal,
        fear=fear,
        sorrow=sorrow,
        boredom=boredom
    )

    # Generate embodied reaction using LLM
    response = await self._call_llm_tracked(
        llm_client=llm_client,
        prompt=prompt,
        context=context,
        system_prompt="You are a physical embodiment filter. Generate brief visceral body reactions.",
        model=model,
        max_tokens=100,
        temperature=0.8
    )

    return TransistorOutput(
        transformed_text=response.strip(),
        salience=self.salience,
        metadata={'embodiment': self.embodiment}
    )
```

**DEFAULT_PROMPT**:
```
You are experiencing this perception in YOUR PHYSICAL BODY.

YOUR BODY:
{body_summary}

CURRENT PERCEPTION:
{input_text}

EMOTIONAL STATE:
- Valence (pleasure): {valence:.2f}
- Arousal (energy): {arousal:.2f}
- Fear: {fear:.2f}
- Sorrow: {sorrow:.2f}

Generate a BRIEF physical reaction (1 short sentence). Focus on:
- Visceral body sensations (heart pounding, fur standing, tail twitching)
- Physical impulses (want to run, freeze, pounce)
- Bodily feelings (warmth, coldness, tension, relaxation)

Output ONLY the physical reaction, nothing else.
```

---

## API Integration

### API Endpoint

**File**: `applications/cmush/api_server.py:1211-1223`

**Endpoint**: `GET /api/manifold/debug/{agent_id}`

**New Fields**:
```python
transistors_data.append({
    'type': transistor.get_transistor_type(),
    'uuid': uuid_str,
    'salience': transistor.salience,
    'enabled': transistor.enabled,
    'instruction_prompt': instruction_prompt,
    'output': transistor.last_output_text or "",
    'metadata': transistor.last_output_metadata or {},
    # NEW: Register state
    'register_state': transistor.register_state,  # empty, computing, ready, error
    'register_cycle_id': transistor.register_cycle_id,
    'register_timestamp': transistor.register_timestamp
})
```

### NoodleTuner Display

**File**: `applications/noodlestudio/noodlestudio/panels/noodle_tuner_panel.py:68-91`

**Register State Badges**:

```python
register_state = self.transistor_data.get('register_state', 'unknown')
if register_state == 'ready':
    self.state_indicator.setText("READY")
    self.state_indicator.setStyleSheet("color: #66FF66; padding: 2px 6px; background-color: #1A3A1A;")
elif register_state == 'computing':
    self.state_indicator.setText("COMPUTING...")
    self.state_indicator.setStyleSheet("color: #FFAA00; padding: 2px 6px; background-color: #3A2A1A;")
elif register_state == 'empty':
    self.state_indicator.setText("EMPTY")
    self.state_indicator.setStyleSheet("color: #666666; padding: 2px 6px; background-color: #2A2A2A;")
elif register_state == 'error':
    self.state_indicator.setText("ERROR")
    self.state_indicator.setStyleSheet("color: #FF6666; padding: 2px 6px; background-color: #3A1A1A;")
```

**Visual**:
```
[IntuitionTransistor]  [READY]
[AffectTransistor]     [COMPUTING...]
[PersonalityTransistor] [EMPTY]
[EmbodyComponent]      [ERROR]
```

---

## Benefits

### 1. Predictability
Know exactly when integration happens - explicit "pull lever" moment.

### 2. Visibility
See what's loaded in each register at all times in NoodleTuner.

### 3. Control
Could pause cognition, edit register contents, manually trigger integration.

### 4. Debugging
Inspect all register contents BEFORE integration to verify correctness.

### 5. Cycle Awareness
Clear start/end boundaries make temporal analysis possible.

### 6. No Race Conditions
Mid-cycle overwrites prevented by `cycle_in_progress` flag.

### 7. Manual Intervention
Can add "FREEZE" mode in NoodleTuner:
- Pause cognition
- Edit register contents manually
- Click "PULL LEVER" button manually
- Resume cognition

---

## Debugging

### Log Patterns

Watch register fill in real-time:
```bash
tail -f applications/cmush/server.log | grep -E "FILLING REGISTERS|READY|PULLING LEVER|CLEARING"
```

**Expected output**:
```
FILLING REGISTERS for cycle a1b2c3d4...
  [IntuitionTransistor] register READY (cycle a1b2c3d4)
  [AffectTransistor] register READY (cycle a1b2c3d4)
  [PersonalityTransistor] register READY (cycle a1b2c3d4)
  [CulturalTransistor] register READY (cycle a1b2c3d4)
  [MemoryTransistor] register READY (cycle a1b2c3d4)
  [EmbodyComponent] register READY (cycle a1b2c3d4)
  6/6 registers READY
  PULLING LEVER: Integrating 6 register contents
  CLEARING all registers (cycle a1b2c3d4)
```

### API Inspection

Check register state via API:
```bash
curl -s http://localhost:8081/api/manifold/debug/agent_bc28a58f | python3 -m json.tool
```

**Example response**:
```json
{
  "transistors": [
    {
      "type": "IntuitionTransistor",
      "uuid": "123e4567-e89b-12d3-a456-426614174000",
      "salience": 1.0,
      "enabled": true,
      "output": "Red is by the pond, you're by the bush",
      "register_state": "ready",
      "register_cycle_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "register_timestamp": 1732619234.567
    }
  ]
}
```

### NoodleTuner Visual Inspection

1. Open NoodleTuner
2. Select agent
3. Observe register state badges
4. Watch transitions: EMPTY → COMPUTING → READY → EMPTY

---

## Testing

### Test 1: Register Fill
1. Start server: `cd applications/cmush && ./start.sh`
2. Spawn agent: `@spawn red_fire_anklebiter`
3. Open NoodleTuner
4. Send message: "hi red"
5. Watch registers: EMPTY → COMPUTING → READY
6. Verify all 6 transistors show READY

### Test 2: Integration Lever
1. After registers fill
2. Watch logs for "PULLING LEVER"
3. Verify manifold blend uses register contents
4. Watch registers clear: READY → EMPTY

### Test 3: Cycle Boundaries
1. Send perception: "hi red"
2. Log shows: `[CYCLE START: uuid_xyz]`
3. Registers fill
4. Integration happens
5. Response generated
6. Log shows: `[CYCLE END: uuid_xyz]`
7. Registers cleared

### Test 4: Mid-Cycle Protection
1. Send perception: "hi red"
2. While computing, send: "how are you"
3. Second message should wait (cycle_in_progress = true)
4. First cycle completes
5. Second cycle starts

### Test 5: NoodleTuner Visibility
1. Registers show state badges
2. Can see what's loaded even between cycles
3. Cycle ID shown in logs
4. Timestamp of register fill shown

---

## Future Enhancements

### 1. Manual "Pull Lever" Button
Add UI control in NoodleTuner:
- Pause cognition
- Edit register contents
- Click "PULL LEVER" button manually
- See integrated result before sending to agent

### 2. Cycle History Viewer
Show last N cycles:
- Cycle UUID
- Timestamp
- All register contents
- Integrated result
- Response generated

### 3. Register Diff View
Compare cycle N vs cycle N-1:
- Which transistors changed
- How much they changed
- Highlight differences

### 4. Freeze Mode
Pause cognition at specific phases:
- After register fill (inspect before integration)
- After integration (inspect before response)
- After response (inspect before clear)

### 5. Register Profiling
Track per-transistor performance:
- Average fill time
- Success rate
- Contribution to final blend

---

## Scripting API (TODO)

### Register State Queries

```python
# Check individual register
state = agent.GetRegisterState("IntuitionTransistor")
# Returns: "empty", "computing", "ready", "error"

# Check all registers
all_ready = agent.AreAllRegistersReady()
# Returns: true/false

# Get register contents
output = agent.GetRegisterOutput("AffectTransistor")
# Returns: TransistorOutput or None
```

### Events

```python
# OnRegisterFilled - fires when transistor completes
def OnRegisterFilled(transistor_type, cycle_id, output):
    print(f"{transistor_type} filled with: {output.transformed_text[:50]}")

# OnAllRegistersReady - fires when all enabled registers ready
def OnAllRegistersReady(cycle_id):
    print(f"All registers ready for cycle {cycle_id[:8]}")

# OnCycleStart - fires at perception start
def OnCycleStart(cycle_id, perception_text):
    print(f"Cycle {cycle_id[:8]} started: {perception_text}")

# OnCycleEnd - fires after registers cleared
def OnCycleEnd(cycle_id, response_text):
    print(f"Cycle {cycle_id[:8]} ended: {response_text}")

# OnRegisterError - fires if transistor fails
def OnRegisterError(transistor_type, cycle_id, error):
    print(f"ERROR in {transistor_type}: {error}")
```

---

## Architecture Philosophy

### Explicit State > Implicit Behavior

The register model makes cognition **observable** and **controllable**:
- Not: "Transistors process on-demand and output vanishes"
- But: "Registers accumulate, wait for all, integrate deliberately"

### CPU Metaphor

Cognition as **register-based computation**:
- Transistors are registers (R1-R6)
- Manifold is ALU (integrates register contents)
- Cycle is instruction execution (LOAD → CHECK → EXECUTE → CLEAR)

### Temporal Boundaries

Cognition has **rhythm**:
- Start: Perception arrives
- Fill: Registers load (parallel)
- Integrate: Lever pulled
- Respond: Output generated
- Clear: Cycle ends

Like a heartbeat: systole (fill) → diastole (clear).

---

## Credits

**Architecture Design**: Lieutenant Caitlyn (Miss Caity)
**Implementation**: Commander Spock
**Date**: November 26, 2025
**Stardate**: 2025.331.00:50
**Commit**: `341b20a`

Live long and prosper.
