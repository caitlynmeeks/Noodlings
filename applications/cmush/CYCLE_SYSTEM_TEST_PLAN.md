# Cognition Cycle UUID System - Test Plan

**Status**: In Progress
**Date**: November 25, 2025 (Evening Session)
**Tester**: Lieutenant Caitlyn
**System**: Cognition Cycle Management + NoodleTuner Improvements + Memory System

**Completed:**
- ✅ Cycle UUID tracking system
- ✅ LLM call instrumentation (all 14 calls tracked)
- ✅ Cycle status API endpoint
- ✅ Pause/resume with cycle completion wait
- ✅ NoodleTuner manifold instruction prompt display
- ✅ Text field scroll behavior (Escape to unfocus)
- ✅ MemoryTransistor added to Red Fire Anklebiter prefab
- ✅ Splash screen acronym fixed

**In Progress:**
- 🔧 Scrollbar UX (testing ScrollBarAsNeeded policy)
- 📋 Memory-driven salience (designed, not yet implemented)

---

## Test 1: Basic Cycle Tracking
**Difficulty**: Easy | **Time**: 2 minutes

```bash
# In terminal:
curl -s http://localhost:8081/api/agents/agent_red_fire_anklebiter/cycle/status | python3 -m json.tool
```

**Expected output**:
- `cycle_uuid`: Some UUID string
- `cycle_in_progress`: true or false
- `pending_llm_calls`: A number (probably 0 when idle)
- `cycle_complete`: true when not processing

**Pass criteria**: You see valid JSON with these fields

- [ ] Test 1 PASSED

---

## Test 2: Scroll Behavior Fix
**Difficulty**: Easy | **Time**: 1 minute

1. Open NoodleStudio: `cd applications/noodlestudio && python run_studio.py`
2. Open NoodleTuner panel
3. Select Red Fire Anklebiter
4. Resize panel to make content taller than viewport (drag sections larger)
5. Two-finger scroll on trackpad - panel should scroll
6. Click INTO a text field - now scroll should move text content
7. Press Escape - unfocuses, scroll returns to panel

**Pass criteria**:
- Panel scrolls when text fields NOT focused
- Text scrolls when focused
- Escape key unfocuses
- Scrollbar appears when content overflows

- [ ] Test 2 PASSED

---

## Test 3: Manifold Instruction Prompt Display
**Difficulty**: Easy | **Time**: 1 minute

1. In NoodleTuner, look at the "Manifold Blend Output" section
2. You should see TWO text fields now:
   - "Manifold Instruction Prompt" (blue background)
   - "Manifold Output" (gray background)

**Pass criteria**: Both fields are visible and contain text (or say "no instruction prompt available")

- [ ] Test 3 PASSED

---

## Test 4: Pause Waits for Cycle Completion
**Difficulty**: Medium | **Time**: 3 minutes

1. In noodleMUSH web interface (http://localhost:8080), type: `hey Red, tell me a story!`
2. IMMEDIATELY click "Pause Cognition" in NoodleTuner while Red is processing
3. Watch the button - it should say "Waiting..." or similar while LLMs finish
4. After a few seconds, button should change to "Resume Cognition"
5. Check that transistor outputs are all filled in (no empty fields)

**Pass criteria**:
- Pause button waits before enabling
- All transistor outputs populated
- No "(no output yet)" or empty instruction prompts

- [ ] Test 4 PASSED

---

## Test 5: Export .tuner File with Complete Data
**Difficulty**: Easy | **Time**: 2 minutes

1. With cognition paused, click "Export .tuner" button
2. Save file somewhere
3. Open the .tuner file in a text editor

**Look for these fields**:
```json
{
  "perception": { ... },
  "transistors": [
    {
      "instruction_prompt": "...",
      "output": "..."
    }
  ],
  "phenomenal_state": [...],
  "predicted_affect": {...}
}
```

**Pass criteria**:
- All instruction_prompts filled in (NOT empty strings)
- Phenomenal_state has numbers
- No major empty fields

- [ ] Test 5 PASSED

---

## Test 6: Edit and Resume
**Difficulty**: Medium | **Time**: 2 minutes

1. With cognition paused, edit any transistor's output text
2. Change salience slider
3. Click "Resume Cognition"
4. Say something to Red again
5. Check if behavior changed

**Pass criteria**:
- No errors when resuming
- Agent responds (even if behavior didn't obviously change)

- [ ] Test 6 PASSED

---

## BONUS Test 7: Cycle UUID Changes Between Cycles
**Difficulty**: Easy | **Time**: 1 minute

```bash
# Run this twice with a message to Red in between:
curl -s http://localhost:8081/api/agents/agent_red_fire_anklebiter/cycle/status | python3 -m json.tool | grep cycle_uuid
```

**Pass criteria**: The UUID is different each time

- [ ] Test 7 PASSED

---

## If Something Fails

**Don't panic.** Just note which test failed and what happened. I can debug from there.

Most likely failure points:
- Test 4: Timeout issues (could need adjustment)
- Test 5: Empty instruction prompts (means we missed a transistor)
- Test 2: Scroll still broken (PyQt6 version issue)

---

---

## Test 8: MemoryTransistor in Manifold
**Difficulty**: Medium | **Time**: 3 minutes

**Purpose**: Verify Red Fire Anklebiter now has memory-based thought coloring

1. In noodleMUSH: `say Hey Red, remember when we talked about candy?`
2. In NoodleTuner: Look at transistor list
3. Should see **4 transistors** now (not 3):
   - AffectTransistor
   - PersonalityTransistor
   - CulturalTransistor
   - MemoryTransistor ← NEW

4. Check MemoryTransistor output - should reference past interactions

**Pass criteria**:
- MemoryTransistor appears in list
- Has instruction prompt filled in
- Output references episodic memories

- [ ] Test 8 PASSED

---

## Summary

Total tests: 8
Required for success: 6/8 (Tests 1, 3, 4, 5, 6, 8)
Bonus: Tests 2, 7

**Known Issues:**
- Scrollbar UX still being refined (handle fills entire space when no overflow)
- Memory-driven salience designed but not yet implemented

**Next Session:**
- Implement memory-driven salience (LLM-evaluated, NO keywords)
- Continue scrollbar UX refinement

**When done, report results to Commander Spock.**

Live long and prosper.
