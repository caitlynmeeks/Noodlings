# QA Test Plan: November 20, 2025 Session

**Testing CollapsibleSection Fix, Inspector Migration, and Hierarchy Enhancements**

---

## Test Session Metadata

**Tester**: Caitlyn
**Date**: November 20, 2025
**Build**: Commits e316f37 → 565f3bc
**Environment**: NoodleSTUDIO on macOS
**Duration**: ~30-45 minutes

---

## Pre-Test Setup

### 1. Start noodleMUSH Server

```bash
cd /Users/thistlequell/git/noodlings_clean/applications/cmush
./start.sh
```

**Verify**:
- [ ] Server starts without errors
- [ ] Ports 8765, 8080, 8081 listening
- [ ] At least 2-3 agents loaded (Servnak, Phi, Callie recommended)

### 2. Start NoodleSTUDIO

```bash
cd /Users/thistlequell/git/noodlings_clean/applications/noodlestudio
source venv/bin/activate
python run_studio.py
```

**Verify**:
- [ ] Splash screen displays correctly (new layout)
- [ ] No Python errors in terminal
- [ ] Main window opens maximized
- [ ] All panels visible (Hierarchy, Inspector, Console, Chat, Timeline)

---

## Test Suite 1: Splash Screen Layout

**Feature**: Repositioned ASCII art, title, acronym, version

### Test 1.1: Visual Composition

**Procedure**:
1. Close NoodleSTUDIO if running
2. Run `python run_studio.py`
3. Observe splash screen for 7 seconds

**Expected Results**:
- [ ] ASCII art banner visible and positioned lower (not cramped at top)
- [ ] "NoodleSTUDIO" text appears ONE line above acronym
- [ ] "NoodleSTUDIO" text is size 20pt (larger, readable)
- [ ] Random STUDIO acronym displays (changes each launch)
- [ ] Version "v1.0.0-alpha" appears below acronym (around Y=470)
- [ ] Elements don't overlap
- [ ] Green terminal aesthetic maintained

**Failure modes to watch for**:
- [ ] Text overlapping or clipping
- [ ] Version too far from acronym
- [ ] ASCII art cut off at edges

---

## Test Suite 2: Scene Hierarchy Text-Click Expansion

**Feature**: Clicking tree item text (not just arrow) expands/collapses

### Test 2.1: Basic Expansion

**Procedure**:
1. In Stage Hierarchy panel, locate "Noodlings" folder
2. **Click on the TEXT "Noodlings"** (not the arrow)
3. Observe expansion

**Expected Results**:
- [ ] Folder expands to show children (Servnak, Phi, Callie, etc.)
- [ ] Arrow changes from ▶ to ▼
- [ ] Expansion is smooth (no flicker)

### Test 2.2: Collapse

**Procedure**:
1. With "Noodlings" expanded, **click on text "Noodlings"** again
2. Observe collapse

**Expected Results**:
- [ ] Folder collapses (children hidden)
- [ ] Arrow changes from ▼ to ▶
- [ ] Collapse is smooth

### Test 2.3: Rapid Clicking

**Procedure**:
1. Click "Noodlings" text 10 times rapidly
2. Observe behavior

**Expected Results**:
- [ ] Toggles expand/collapse each click
- [ ] No crashes
- [ ] No stuck states
- [ ] No visual glitches

### Test 2.4: Leaf Items Don't Toggle

**Procedure**:
1. Expand "Noodlings"
2. **Click on individual agent** (e.g., "Servnak [robot, they]")
3. Observe behavior

**Expected Results**:
- [ ] Agent is **selected** (shows in Inspector)
- [ ] NO expansion attempt (leaf items have no children)
- [ ] Selection highlight appears
- [ ] Inspector loads agent properties

### Test 2.5: All Folder Types

**Procedure**: Test text-click on each folder:
- [ ] "Stage: The Nexus" (root)
- [ ] "Connected Users"
- [ ] "Noodlings"
- [ ] "Prims"
- [ ] "Exits"

**Expected Results**:
- [ ] All folders expand/collapse on text click
- [ ] Behavior consistent across all folder types

---

## Test Suite 3: CollapsibleSection Bounce-Back Fix

**Feature**: Inspector sections no longer bounce closed after expanding

### Test 3.1: Cognitive Components (New CollapsibleSection)

**Procedure**:
1. Select a Noodling (Servnak recommended) in hierarchy
2. Inspector loads
3. Scroll to "Cognitive Components" section
4. Click header for **"Character Voice"**
5. Wait 5 seconds (let Inspector timer fire)
6. Observe state

**Expected Results**:
- [ ] Section expands on first click
- [ ] Section **stays expanded** for full 5 seconds
- [ ] No bounce-back
- [ ] No flicker
- [ ] Can click header again to collapse
- [ ] Collapse works smoothly

**Repeat for**:
- [ ] "Intuition Receiver"
- [ ] "Social Expectation Detector"

**Failure mode**:
- [ ] If section snaps closed after <5 seconds → State preservation failed

### Test 3.2: Legacy Sections (Migrated to CollapsibleSection)

**Procedure**: Test these sections with same procedure as 3.1:
- [ ] "Identity"
- [ ] "LLM Configuration"
- [ ] "Personality Traits"
- [ ] "Noodle Component"

**Expected Results**:
- [ ] Each section expands and **stays expanded**
- [ ] No bounce-back on any section
- [ ] Consistent behavior with Cognitive Components sections

### Test 3.3: State Persistence Across Inspector Refreshes

**Procedure**:
1. Expand **"Intuition Receiver"** section
2. **Wait 10 seconds** (Inspector timer fires at 1000ms intervals)
3. Observe section state throughout

**Expected Results**:
- [ ] Section remains expanded for full 10 seconds
- [ ] Console shows `[STATE]` messages if diagnostics enabled
- [ ] No visible flicker or bounce
- [ ] Can interact with widgets inside section during timer fires

### Test 3.4: Multiple Sections Expanded

**Procedure**:
1. Expand **ALL** sections in Inspector:
   - Identity
   - LLM Configuration
   - Personality Traits
   - Noodle Component
   - Character Voice
   - Intuition Receiver
   - Social Expectation Detector
2. Wait 10 seconds
3. Observe all sections

**Expected Results**:
- [ ] ALL sections stay expanded
- [ ] No sections collapse unexpectedly
- [ ] No performance issues (lag, stutter)

### Test 3.5: Rapid Toggle Stress Test

**Procedure**:
1. Click "Character Voice" header 20 times rapidly
2. Observe behavior

**Expected Results**:
- [ ] Toggles expand/collapse each click
- [ ] No crashes
- [ ] No stuck states (half-open/half-closed)
- [ ] No error messages in Console
- [ ] Final state matches click count parity (even=collapsed, odd=expanded)

### Test 3.6: Edit Field During Timer Refresh

**Procedure**:
1. Expand "Intuition Receiver" section
2. Click in **prompt template field** (starts editing)
3. **Keep focus in field** for 5+ seconds
4. Observe section state

**Expected Results**:
- [ ] Section stays expanded (focus protection works)
- [ ] Field retains focus (not stolen by refresh)
- [ ] No visual jump or flicker
- [ ] Can continue editing normally

**Failure mode**:
- [ ] If section collapses while editing → Focus protection failed
- [ ] If cursor jumps or field content resets → Widget recreation issue

---

## Test Suite 4: Inspector Section Content Integrity

**Feature**: Verify all migrated sections display correct content

### Test 4.1: Identity Section

**Procedure**:
1. Select Servnak in hierarchy
2. Expand "Identity" section

**Expected Results**:
- [ ] Name field shows "Servnak"
- [ ] Species field shows "robot"
- [ ] All fields editable
- [ ] Content matches agent data

### Test 4.2: Personality Traits Section

**Procedure**:
1. Expand "Personality Traits"
2. Check all sliders

**Expected Results**:
- [ ] 5 sliders visible (Extraversion, Agreeableness, Openness, Conscientiousness, Neuroticism)
- [ ] Slider values match agent configuration
- [ ] Can drag sliders
- [ ] Values update in real-time

### Test 4.3: Noodle Component Section

**Procedure**:
1. Expand "Noodle Component"
2. Observe live data

**Expected Results**:
- [ ] 5-D Affect Vector shows with progress bars
- [ ] Valence, Arousal, Fear, Sorrow, Boredom all visible
- [ ] Values update every ~1 second (live polling)
- [ ] 40-D Phenomenal State displays
- [ ] Surprise metric shows

### Test 4.4: Cognitive Components Section

**Procedure**:
1. Expand each cognitive component
2. Verify content

**Character Voice**:
- [ ] Description visible
- [ ] Enabled checkbox
- [ ] Prompt template (green-on-black, editable)
- [ ] Parameters (model, temperature, max_tokens)

**Intuition Receiver**:
- [ ] Description visible
- [ ] Enabled checkbox
- [ ] Prompt template
- [ ] Parameters (model, temperature, max_tokens, timeout)

**Social Expectation Detector**:
- [ ] Description visible
- [ ] Enabled checkbox
- [ ] Prompt template
- [ ] Parameters

---

## Test Suite 5: Cross-Entity Testing

**Feature**: Verify sections work for all entity types

### Test 5.1: Different Noodlings

**Procedure**: Select each agent and verify Inspector loads:
- [ ] Servnak (robot)
- [ ] Phi (kitten)
- [ ] Callie (noodling)

**Expected Results**:
- [ ] All sections load correctly for each agent
- [ ] No crashes when switching between agents
- [ ] Data specific to each agent displays
- [ ] State preservation works across agent switches

### Test 5.2: Non-Noodling Entities

**Procedure**: Select other entity types:
- [ ] User (caity)
- [ ] Prim/Object
- [ ] Exit
- [ ] Stage/Room

**Expected Results**:
- [ ] Appropriate sections load (not all entities have cognitive components)
- [ ] No errors about missing CollapsibleSection
- [ ] Sections that do appear work correctly

---

## Test Suite 6: Console Diagnostic Output

**Feature**: Verify diagnostic logging is working

### Test 6.1: State Preservation Logging

**Procedure**:
1. Open Console panel
2. Select Servnak
3. Expand "Intuition Receiver"
4. Wait 2 seconds
5. Check Console for `[STATE]` messages

**Expected Results**:
- [ ] `[STATE] User toggled 'Intuition Receiver': expanded=False` appears
- [ ] After refresh: `[STATE] Saved 'Intuition Receiver': expanded=False`
- [ ] After refresh: `[STATE] Restored 'Intuition Receiver': expanded=False`

**Optional (if verbose diagnostics still enabled)**:
- [ ] `[DIAGNOSTIC] set_expanded()` messages with stack traces
- [ ] `[DIAGNOSTIC] load_entity()` messages

### Test 6.2: No Error Messages

**Procedure**:
1. Perform all tests above
2. Monitor Console for errors

**Expected Results**:
- [ ] No Python tracebacks
- [ ] No "AttributeError" messages
- [ ] No "QWidget::setLayout" warnings
- [ ] No crashes

**Acceptable warnings**:
- [ ] WebSocket connection errors (if noodleMUSH not running) - OK
- [ ] Font warnings - OK
- [ ] Skia Graphite warnings - OK

---

## Test Suite 7: Performance and Responsiveness

### Test 7.1: Inspector Refresh Latency

**Procedure**:
1. Select Servnak
2. Expand "Noodle Component" (live updates)
3. Observe affect bar updates
4. Count seconds between updates

**Expected Results**:
- [ ] Affect bars update approximately every 1 second
- [ ] Updates are smooth (no stuttering)
- [ ] No UI freezing during updates

### Test 7.2: Rapid Entity Selection

**Procedure**:
1. Click Servnak → Phi → Callie → Servnak rapidly (4 clicks in 2 seconds)
2. Observe Inspector

**Expected Results**:
- [ ] Inspector updates to show correct entity
- [ ] No crashes
- [ ] No half-loaded states
- [ ] Final state matches final selection (Servnak)

### Test 7.3: Memory Usage (Optional)

**Procedure**:
1. Open Activity Monitor (macOS)
2. Find NoodleSTUDIO process
3. Note memory usage
4. Expand/collapse sections 50 times
5. Check memory again

**Expected Results**:
- [ ] Memory usage stable (no significant leak)
- [ ] CPU usage reasonable (<20% while idle)

---

## Test Suite 8: Edge Cases and Error Handling

### Test 8.1: Missing Server (Inspector Graceful Failure)

**Procedure**:
1. Stop noodleMUSH server (`pkill -f cmush`)
2. In NoodleSTUDIO, select a Noodling
3. Observe Inspector

**Expected Results**:
- [ ] Inspector shows "Could not load data" or similar
- [ ] No crash
- [ ] Can still navigate hierarchy
- [ ] Reconnects gracefully when server restarts

### Test 8.2: Expand While Focus in Text Field

**Procedure**:
1. Expand "Identity" section
2. Click in "Name" field (QLineEdit)
3. **While field has focus**, click another section header
4. Observe behavior

**Expected Results**:
- [ ] Other section expands/collapses as expected
- [ ] Name field keeps focus (or loses it gracefully)
- [ ] No crashes
- [ ] Can continue editing

### Test 8.3: Switching Entities While Section Expanded

**Procedure**:
1. Select Servnak
2. Expand "Intuition Receiver"
3. Immediately select Phi (before timer fires)
4. Observe both sections

**Expected Results**:
- [ ] Servnak's sections destroyed cleanly
- [ ] Phi's sections load
- [ ] Phi's "Intuition Receiver" state is independent (may be collapsed or expanded based on previous state)
- [ ] No crashes

---

## Test Suite 9: Regression Testing (Did We Break Anything?)

### Test 9.1: Save Functionality Still Works

**Procedure**:
1. Select Servnak
2. Expand "Identity"
3. Change name to "SERVNAK_TEST"
4. Tab out of field (trigger save)
5. Check console for "Saved changes" message
6. Deselect and reselect Servnak
7. Verify name persists

**Expected Results**:
- [ ] Save triggers on focus loss
- [ ] Data persists in backend
- [ ] Reload shows updated name
- [ ] No save handler spam (should be single save, not 6x)

### Test 9.2: Live Affect Updates Still Work

**Procedure**:
1. Select Servnak
2. Expand "Noodle Component"
3. In Chat panel, send message to Servnak
4. Watch affect bars

**Expected Results**:
- [ ] Affect bars update after Servnak responds
- [ ] Values change appropriately (e.g., valence increases after positive interaction)
- [ ] Updates continue every ~1 second

### Test 9.3: Component Parameter Editing

**Procedure**:
1. Expand "Intuition Receiver"
2. Find "timeout" parameter
3. Change value (e.g., 5 → 10)
4. Tab out
5. Check for save confirmation

**Expected Results**:
- [ ] Parameter editable
- [ ] Save triggers (if handlers enabled)
- [ ] Value persists

**Note**: Save handlers may still be disabled from debugging. If so:
- [ ] Mark as "DEFERRED - Save handlers disabled for testing"

---

## Test Suite 10: Visual Polish and Aesthetics

### Test 10.1: Consistent Styling

**Procedure**: Visually inspect all CollapsibleSection headers

**Expected Results**:
- [ ] All headers have consistent height (~28px)
- [ ] Arrow symbols consistent (▼ expanded, ▶ collapsed)
- [ ] Text color consistent (#FFFFFF)
- [ ] Background color consistent (#3C3C3C)
- [ ] Hover cursor shows pointer (indicates clickable)

### Test 10.2: Content Indentation

**Procedure**: Expand sections and observe content indentation

**Expected Results**:
- [ ] Content indented from left edge (~12px)
- [ ] Form layouts (Identity, Personality) aligned properly
- [ ] Labels and fields aligned in columns
- [ ] No overlapping text

### Test 10.3: Scrolling Behavior

**Procedure**:
1. Expand ALL sections
2. Scroll Inspector up and down
3. Observe behavior

**Expected Results**:
- [ ] Scrolling smooth
- [ ] Headers don't stick/jump
- [ ] Content doesn't clip unexpectedly
- [ ] Scrollbar appears when content exceeds viewport

---

## Test Suite 11: Hierarchy Features

### Test 11.1: Arrow Click Still Works

**Procedure**:
1. Collapse "Noodlings" folder (if expanded)
2. **Click on the ARROW** (not text) to expand
3. Observe

**Expected Results**:
- [ ] Clicking arrow expands folder
- [ ] Behavior unchanged from before (backwards compatible)

### Test 11.2: Selection Still Works

**Procedure**:
1. Click on "Servnak [robot, they]" text
2. Observe Inspector

**Expected Results**:
- [ ] Servnak selected (highlighted in tree)
- [ ] Inspector loads Servnak properties
- [ ] No expansion attempt (leaf item)

---

## Test Suite 12: Inspector Component Save Functionality

**Note**: Save handlers may be disabled in current build. If so, skip this suite and mark as "DEFERRED."

### Test 12.1: Prompt Template Editing

**Procedure**:
1. Expand "Intuition Receiver"
2. Click in prompt template field
3. Add text: "TEST EDIT"
4. Tab out of field
5. Check console

**Expected Results**:
- [ ] Save triggers on focus loss
- [ ] Console shows: "Component intuitionreceiver saved for agent_servnak"
- [ ] Single save message (not 6x spam)

### Test 12.2: Parameter Editing

**Procedure**:
1. In "Intuition Receiver", find "temperature" spinbox
2. Change value: 0.3 → 0.5
3. Click elsewhere (lose focus)
4. Check console

**Expected Results**:
- [ ] Save triggers
- [ ] Single save message
- [ ] No spam

### Test 12.3: Checkbox Toggle

**Procedure**:
1. Find "Enabled" checkbox in "Character Voice"
2. Toggle off
3. Toggle on
4. Observe saves

**Expected Results**:
- [ ] Each toggle triggers one save
- [ ] No spam (2 saves total, not 12)

---

## Test Suite 13: Multi-Agent Cognitive Components

**Feature**: Verify different agents show correct cognitive components

### Test 13.1: Agent-Specific Components

**Procedure**: Select each agent and check cognitive components:

**Servnak**:
- [ ] Character Voice shows (species: robot)
- [ ] Intuition Receiver shows
- [ ] Social Expectation Detector shows

**Phi**:
- [ ] Character Voice shows (species: kitten)
- [ ] Intuition Receiver shows
- [ ] Social Expectation Detector shows

**Callie**:
- [ ] Character Voice shows (or not, depending on species)
- [ ] Intuition Receiver shows
- [ ] Social Expectation Detector shows

**Expected Results**:
- [ ] Components match each agent's configuration
- [ ] No components missing
- [ ] No duplicate components

---

## Test Suite 14: Terminal Diagnostic Output

**Feature**: Verify diagnostic logging provides useful debugging info

### Test 14.1: Diagnostic Verbosity

**Procedure**:
1. In terminal, observe output while testing
2. Note level of diagnostic detail

**Expected Results**:
- [ ] `[STATE]` messages show save/restore operations
- [ ] `[DIAGNOSTIC]` messages show set_expanded calls (if enabled)
- [ ] Stack traces available for debugging (if needed)

**Decision point**:
- [ ] Keep verbose diagnostics? (Helpful for debugging)
- [ ] Remove before next release? (Clean up logs)
- [ ] Make configurable? (--debug flag)

### Test 14.2: No Error Spam

**Procedure**: Observe Console panel during all tests

**Expected Results**:
- [ ] No repeated error messages
- [ ] No infinite loops
- [ ] No stack overflow

---

## Test Suite 15: Backwards Compatibility

**Feature**: Verify old features still work

### Test 15.1: Scene Hierarchy Refresh

**Procedure**:
1. Click "Refresh Scene" button in hierarchy
2. Observe reload

**Expected Results**:
- [ ] Tree rebuilds
- [ ] Expanded state preserved (items stay expanded if they were before)
- [ ] Selection preserved
- [ ] No crashes

### Test 15.2: Context Menu

**Procedure**:
1. Right-click on Servnak in hierarchy
2. Observe context menu

**Expected Results**:
- [ ] Menu appears
- [ ] Options visible (Inspect, Toggle Enlightenment, Export, De-Rez, etc.)
- [ ] Selecting "Inspect Properties" loads Inspector

### Test 15.3: Chat Panel

**Procedure**:
1. Open Chat panel
2. Type message: "hello servnak"
3. Send

**Expected Results**:
- [ ] Message sends to noodleMUSH
- [ ] Response appears in chat
- [ ] No errors

---

## Test Suite 16: Platform Compatibility

### Test 16.1: Window Resizing

**Procedure**:
1. Resize NoodleSTUDIO window (make smaller, make larger)
2. Observe Inspector panel

**Expected Results**:
- [ ] Inspector resizes gracefully
- [ ] Scroll area adjusts
- [ ] CollapsibleSection headers don't break
- [ ] Content remains readable

### Test 16.2: Panel Rearrangement

**Procedure**:
1. Drag Inspector panel to different position
2. Drag Hierarchy panel
3. Observe

**Expected Results**:
- [ ] Panels can be rearranged
- [ ] No crashes
- [ ] CollapsibleSections still work after rearrangement

---

## Critical Bug Checklist (From BUGS.md)

### Verify Fixed Bugs Stay Fixed

**Bug #001: CollapsibleSection bounce-back** (Fixed in e316f37)
- [ ] Test Suite 3 passes → Bug remains fixed

**Bug #002: QGroupBox double-trigger** (Fixed in e316f37)
- [ ] No QGroupBox sections remain (all migrated to CollapsibleSection)
- [ ] Test Suite 3 passes → Bug eliminated by migration

**Bug #000: Agent broadcast failure** (Fixed in 7cbefde)
- [ ] Not directly testable in Inspector, but verify agents still respond in noodleMUSH

---

## Test Completion Checklist

After completing all test suites:

- [ ] **All critical tests passed** (Test Suites 1-3)
- [ ] **All functionality tests passed** (Test Suites 4-6)
- [ ] **No regressions detected** (Test Suite 9)
- [ ] **Performance acceptable** (Test Suite 7)
- [ ] **Edge cases handled** (Test Suite 8)

**If all tests pass**:
- ✅ Build is **STABLE** and ready for use
- ✅ Commits e316f37 → 565f3bc **VALIDATED**
- ✅ Can proceed with new features

**If any critical test fails**:
- ⚠️ File bug in BUGS.md
- ⚠️ Note which test failed
- ⚠️ Capture error output
- ⚠️ Create GitHub issue

---

## Quick Smoke Test (5 minutes)

**If time-limited, run this abbreviated test**:

1. [ ] Start NoodleSTUDIO (splash looks good)
2. [ ] Click "Noodlings" text in hierarchy (expands)
3. [ ] Click "Servnak" (Inspector loads)
4. [ ] Expand "Intuition Receiver" (opens)
5. [ ] Wait 5 seconds (stays open)
6. [ ] Click header again (closes smoothly)
7. [ ] Expand "Identity" (opens)
8. [ ] Expand "Personality Traits" (opens)
9. [ ] Click "Noodlings" text again (collapses)
10. [ ] Check Console (no errors)

**Expected**: All 10 steps complete without errors

**If smoke test passes**: High confidence build is stable

---

## Test Results Template

**Copy this to BUGS.md when testing complete**:

```markdown
## QA Session: November 20, 2025

**Build**: e316f37 → 565f3bc
**Tester**: Caitlyn
**Date**: [DATE]
**Duration**: [TIME]

### Results Summary

- Tests passed: X / Y
- Critical bugs found: N
- Non-critical issues: M
- Build status: ✅ STABLE / ⚠️ UNSTABLE / ❌ BROKEN

### Failed Tests

[If any tests failed, list them here with details]

### New Bugs Discovered

[If new bugs found, add to Active Bugs section above]

### Notes

[Any observations, performance notes, or recommendations]
```

---

## Post-Test Actions

**If all tests pass**:
1. Remove verbose diagnostic logging (optional)
2. Clean up test files (`test_collapsible.py` - can delete or keep)
3. Proceed with next feature development

**If tests fail**:
1. File bugs in BUGS.md
2. Create GitHub issues for critical bugs
3. Fix before proceeding

---

**Test plan complete. Execute at your convenience.** 🖖

**Estimated time**: 30-45 minutes for full suite, 5 minutes for smoke test

Good hunting.
