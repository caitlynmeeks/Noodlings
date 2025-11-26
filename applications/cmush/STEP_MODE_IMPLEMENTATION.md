# Step Mode Implementation

## Overview

Step mode allows single-step debugging of the register-based cognition architecture. When enabled, cognition pauses after all registers fill, allowing inspection before integration.

## Architecture

### Flow with Step Mode DISABLED (normal):
```
Perception → Fill Registers (parallel) → All Ready → Integrate → Response → Clear Registers
```

### Flow with Step Mode ENABLED:
```
Perception → Fill Registers (parallel) → All Ready → [PAUSE & BEEP] → [Wait for Continue] → Integrate → Response → Clear Registers
```

## API Endpoints

### Enable/Disable Step Mode
```
POST /api/agents/{agent_id}/step_mode
Body: {"enabled": true/false}
```

### Continue from Pause (Pull Lever)
```
POST /api/agents/{agent_id}/step/continue
```

### Check Status
```
GET /api/manifold/debug/{agent_id}
```

Returns:
```json
{
  "step_mode_enabled": true/false,
  "step_mode_waiting": true/false,
  "step_mode_cycle_id": "uuid...",
  "transistors": [...]
}
```

## Implementation Details

### Agent Fields (agent_bridge.py:747-750)
```python
self.step_mode_enabled = False   # Toggle for step mode
self.step_mode_waiting = False   # True when paused at breakpoint
self.step_mode_cycle_id = None   # Current paused cycle UUID
```

### Pause Point (cognitive_components.py:675-693)
After `fill_all_registers()` completes:
```python
if agent.step_mode_enabled:
    agent.step_mode_waiting = True
    # Wait for continue signal (max 5 minutes)
    while agent.step_mode_waiting:
        await asyncio.sleep(0.1)
```

### Continue Signal
```python
agent.step_mode_waiting = False  # Release the wait loop
```

## NoodleTuner UI Integration

### Step Mode Toggle Button
```python
# When clicked:
POST /api/agents/{agent_id}/step_mode
{"enabled": step_mode_checkbox.isChecked()}
```

### Continue Button
```python
# Enabled only when step_mode_waiting == True
POST /api/agents/{agent_id}/step/continue

# Play beep when registers fill
if data['step_mode_waiting']:
    play_audio('file:///path/to/pc_beep_896hz250ms.ogg')
```

### UI States
1. **Step Mode OFF**: Normal operation, continue button disabled
2. **Step Mode ON, Not Waiting**: Waiting for next perception
3. **Step Mode ON, Waiting**: BEEP + Show "Registers Filled - Ready to Integrate" + Enable [CONTINUE] button
4. **After Continue**: Resume normal flow, wait for next perception

## Benefits

1. **Inspect Register Contents**: See all transistor outputs before blend
2. **No Concurrent Cycles**: Prevents race conditions during debugging
3. **Manual Control**: Pull lever by hand to proceed
4. **Audio Feedback**: Beep alerts when ready for inspection

## Example Session

```
1. Enable step mode in NoodleTuner
2. Send "hi red" from noodleMUSH
3. Registers fill in parallel...
4. [BEEP] - All registers READY
5. NoodleTuner shows:
   - IntuitionTransistor: "That greeting is for ME"
   - AffectTransistor: "Flames SURGING!"
   - PersonalityTransistor: "Oh please, you think..."
   - ...
6. Inspect each register's output
7. Click [CONTINUE] button
8. Manifold integrates
9. Response generated
10. Registers cleared
11. Ready for next perception (loop to step 2)
```

## Concurrency Control

When step mode is enabled:
- New perceptions arriving during pause are queued
- Only one cognition cycle active at a time
- Timeout after 5 minutes (prevents infinite hang)

## Timeout Handling

If no continue signal after 5 minutes:
```
[WARNING] STEP MODE: Timeout waiting for continue signal
```
Cycle proceeds anyway to prevent deadlock.

## Technical Notes

- Uses async/await for non-blocking pause
- Compatible with existing pause/resume system
- Works with both speech and rumination paths
- Register state persists during pause (not cleared until after integration)

---

**Status**: Implemented November 26, 2025
**Files Modified**:
- `applications/cmush/cognitive_components.py` (pause logic)
- `applications/cmush/agent_bridge.py` (agent fields)
- `applications/cmush/api_server.py` (API endpoints)
