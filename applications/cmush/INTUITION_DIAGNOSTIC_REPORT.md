# Intuition System Diagnostic Report

## Date: November 26, 2025, 01:52 PST
## Investigator: Commander Spock
## Issue: Intuition not routing through cognitive transistors

---

## Executive Summary

The intuition system architecture is **correctly wired** but the IntuitionTransistor is outputting raw input instead of transformed context. The transistor's `self.intuition_text` field is `None` when `process()` is called, causing an early return.

---

## Evidence

### API Response Analysis

Querying `/api/manifold/debug/agent_352020a3-307b-4f28-9312-e18c07e9d2fe` shows:

```json
{
  "type": "IntuitionTransistor",
  "salience": 0.9,
  "enabled": true,
  "instruction_prompt": null,
  "output": "oh an anklebiter great",  // <-- RAW INPUT, NOT TRANSFORMED
  "metadata": {},
  "register_state": "ready"
}
```

**Key observations:**
1. `instruction_prompt` is `null` (should show full prompt with intuition)
2. `output` is raw input text (should be transformed: "I sense...")
3. Salience would be 0.10 if early return happened (but API shows 0.9 - config value)

### Code Path Analysis

**Expected flow:**
1. `_generate_intuition()` → Returns intuition string
2. `set_intuition(intuition_text)` → Sets `self.intuition_text`
3. `fill_all_registers()` → Calls `process()` for each transistor
4. `IntuitionTransistor.process()` → Checks `if not self.intuition_text:`
5. If truthy: Builds prompt with intuition and calls LLM
6. If falsy: Returns `TransistorOutput(input_text, 0.1, {})`  // EARLY RETURN

**What's happening:**
- Line 1092-1095 in `cognitive_components.py`:
```python
async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
    if not self.intuition_text:
        return TransistorOutput(input_text, 0.1, {})  // <-- HITTING THIS
```

---

## Hypothesis

**Primary hypothesis:** `_generate_intuition()` is returning `None` or empty string.

**Why:**
- The check at line 2490 in `agent_bridge.py`: `if intuition_text:`
- Only calls `set_intuition()` if intuition_text is truthy
- If `_generate_intuition()` fails or returns None, `set_intuition()` never runs
- Therefore `self.intuition_text` remains `None` (initial value from `__init__`)

**Alternative hypotheses:**
1. `get_cognitive_transistor()` returns wrong instance (unlikely - UUID would differ)
2. Transistors recreated per-cycle losing state (ruled out - register arch keeps transistors)
3. `clear_register()` clearing `intuition_text` (ruled out - only clears register fields)

---

## Root Cause Candidates

### 1. LLM Call Failure in `_generate_intuition()`

Location: `agent_bridge.py:1592-1606`

```python
intuition = await self.llm.generate(
    prompt=prompt,
    system_prompt=f"You are {my_name}'s intuitive contextual awareness.",
    model=intuition_model,
    temperature=0.3,
    max_tokens=150
)
return intuition.strip()
```

**Could fail if:**
- LLM client connection drops
- Model not loaded
- Timeout (5s default)
- Exception caught at line 1605: `return None`

### 2. Config Disabled

Location: `agent_state.json:63-64`

```json
"intuition_receiver": {
  "enabled": true,  // <-- VERIFIED TRUE
```

Status: **Not the issue** - intuition is enabled.

### 3. World State Missing

Location: `agent_bridge.py:3454`

```python
if self.world and hasattr(self, 'config'):
    # Generate intuition
```

**Could fail if:**
- `self.world` is `None`
- `self.config` doesn't exist
- But this would log: "Rumination: Checking intuition - world=False"

---

## Diagnostic Logging Added

Modified `cognitive_components.py` to add logging:

1. **Line 1088-1090** - `set_intuition()`:
```python
logger.info(f"IntuitionTransistor.set_intuition() called with: {repr(intuition_text[:100])}")
self.intuition_text = intuition_text
logger.info(f"IntuitionTransistor.set_intuition() - self.intuition_text now = {repr(self.intuition_text[:100])}")
```

2. **Line 1092-1095** - `process()`:
```python
logger.info(f"IntuitionTransistor.process() - self.intuition_text={repr(self.intuition_text)}")
if not self.intuition_text:
    logger.warning(f"IntuitionTransistor returning EARLY - no intuition text! (input={input_text[:50]})")
    return TransistorOutput(input_text, 0.1, {})
```

---

## Next Steps

### Immediate Action Required

**Send a test message** to Red from noodleMUSH/NoodleTuner to trigger cognition cycle.

**Expected log output:**

**IF WORKING:**
```
[agent_id] Intuition generated: That greeting is for Red, not me...
[agent_id] Updated IntuitionTransistor with: That greeting is for Red, not me...
[INFO] IntuitionTransistor.set_intuition() called with: 'That greeting is for Red, not me...'
[INFO] IntuitionTransistor.set_intuition() - self.intuition_text now = 'That greeting is for Red, not me...'
[INFO] IntuitionTransistor.process() - self.intuition_text='That greeting is for Red, not me. Red is near the flames.'
```

**IF BROKEN:**
```
[WARNING] Intuition returned None/empty!
[WARNING] IntuitionTransistor returning EARLY - no intuition text! (input=hi red)
```

### Fixes to Try

#### Fix 1: Fallback to Context Intuition

If `self.intuition_text` is None, check `context['intuition']`:

```python
async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
    # Try self.intuition_text first, fall back to context
    intuition = self.intuition_text or context.get('intuition')

    if not intuition:
        return TransistorOutput(input_text, 0.1, {})

    # Use intuition...
```

#### Fix 2: Always Call set_intuition()

Remove the `if intuition_text:` check in `agent_bridge.py`:

```python
# OLD (line 2490):
if intuition_text:
    intuition_transistor.set_intuition(intuition_text)

# NEW:
intuition_transistor = self.get_cognitive_transistor('IntuitionTransistor')
if intuition_transistor:
    intuition_transistor.set_intuition(intuition_text or "")  # Always call, even with None
```

#### Fix 3: Log LLM Failures

Add more logging in `_generate_intuition()`:

```python
try:
    intuition = await self.llm.generate(...)
    if not intuition:
        logger.warning(f"LLM returned empty intuition!")
    return intuition.strip()
except Exception as e:
    logger.error(f"Intuition LLM call failed: {e}", exc_info=True)
    return None
```

---

## Technical Details

### Architecture Review

**Register-based accumulator model:**
1. Transistors are persistent objects (not recreated per-cycle)
2. Each has `register_state` (empty/computing/ready/error)
3. Each has `register_output` (last TransistorOutput)
4. Registers are filled in parallel via `fill_all_registers()`
5. After all ready, manifold pulls lever and integrates
6. Registers cleared for next cycle

**Intuition lifecycle:**
1. Generated once per perception via `_generate_intuition()`
2. Stored in transistor via `set_intuition()`
3. Used during `process()` to transform input
4. Should persist until next `set_intuition()` call
5. NOT cleared by `clear_register()` (only register fields cleared)

### Why Salience Shows 0.9 Not 0.1

The API response shows `salience: 0.9` even though early return gives 0.1.

**Answer:** The API might be showing transistor's configured salience (from prefab), not the actual output salience. Need to verify API endpoint logic.

---

## Status

**Server:** Running with diagnostic logging
**Waiting for:** Test message to trigger cognition cycle
**Log monitor:** Watching `server_output.log` for intuition-related entries

---

## Conclusion

The intuition system is **architecturally sound** but experiencing a runtime failure where `self.intuition_text` is `None` during `process()`. This suggests either:

1. `_generate_intuition()` failing silently
2. `set_intuition()` not being called
3. Timing issue where `process()` runs before `set_intuition()`

Diagnostic logging will reveal the exact failure point.

---

**Commander Spock**
Science Officer
Garcia River Forest Research Station
