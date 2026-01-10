# Guide Implementation Brief

**Date:** 2025-01-08
**Goal:** Get Guide (Ajo Majo the axolotl) talking in Let's Consciousness!
**Priority:** Prove the full pipeline works end-to-end

---

## What We're Building

A minimal but complete instructor experience:
- User types a question
- Guide's facet assembly processes it
- Guide responds in speech bubble
- Guide's avatar renders via Radiance

This is the first Noodling character running in a UI. If this works, NoodleStudio is proven.

---

## Critical Architecture: Engine/Runtime Duality

**There is no separate Let's Consciousness! runtime.** Let's Consciousness! IS NoodleStudio with:
- A Let's Consciousness!-specific UI shell (`ui.yaml`)
- View Source permissions (can see how things work, can't modify)
- The same NoodleStudio core running underneath

From con-splo-spec.md: *"There is no 'runtime' separate from the 'editor.' There is only NoodleStudio, with different UI shells and permission levels."*

### Publisher Permission Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **Locked** | Pure player experience, no peeking | Commercial games, controlled experiences |
| **View Source** | Can see how things are built, cannot modify | Educational, "learn from this" |
| **Sandbox** | Can modify, changes don't persist / can reset | Safe experimentation |
| **Full Access** | Complete editor access | Open creation, community building |

**Let's Consciousness! ships at View Source** - users can inspect Guide's facets, see the assembly graph, understand how it works. They can't modify it in Let's Consciousness!, but that's the invitation to download NoodleStudio proper.

### What This Means for Implementation

- We're NOT creating a separate `applications/lets-consciousness/` app
- We're creating a Let's Consciousness! **project** that runs in NoodleStudio
- The project has a `ui.yaml` that defines the Let's Consciousness! experience
- Permission level is set in `project.yaml`: `permission: view_source`
- The same window that runs Let's Consciousness! can reveal the editor if user has Full Access

### The Meta-Demo

When NoodleStudio is running Let's Consciousness!:
1. Show Let's Consciousness! running with Guide
2. User clicks "View Source" (or similar affordance)
3. The editor UI appears - **same window, same app**
4. User sees Guide's assembly, facets, recipe
5. "This is what NoodleStudio is. You've been using it the whole time."

**Canonical source:** `/claudechat/projects/noodling-studio/con-splo-spec.md` (lines 820-1000)

---

## Files Already In Place

```
Noodlings/guide/
├── recipe.yaml           # Personality, affect model, voice
├── assembly.yaml         # Cognition: INCOMING → Perception → Response → OUTGOING
└── Radiances/
    └── AjoMajo.vrm       # Source avatar (needs conversion)
```

---

## Implementation Steps

### Step 1: Convert VRM to Radiance

Use the existing tool:

```bash
cd /Users/thistlequell/git/noodlings_clean
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance \
  Noodlings/guide/Radiances/AjoMajo.vrm \
  -o Noodlings/guide/Radiances/ajo_majo.radiance \
  -v
```

**Expected output:** `ajo_majo.radiance` with GAUS, SKEL, SKIN chunks.

**Test:** Load in a standalone RadianceViewport to verify rendering.

---

### Step 2: Create Let's Consciousness! Project Structure

Let's Consciousness! is a NoodleStudio PROJECT. Create the project folder:

```
Projects/lets-consciousness/
├── project.yaml          # Project config (permission: view_source)
├── ui.yaml               # Let's Consciousness! UI definition
├── Noodlings/
│   └── guide/            # Symlink or copy from top-level Noodlings/guide/
└── assets/               # Any Let's Consciousness!-specific assets
```

Or reference the top-level `Noodlings/guide/` directly in paths.

---

### Step 3: Create Let's Consciousness! UI

**UI Layout:**

```
┌─────────────────────────────────────────┐
│  ┌─────────────────┐  ┌──────────────┐  │
│  │                 │  │              │  │
│  │  RadianceView   │  │   Speech     │  │
│  │  (Guide avatar) │  │   Bubble     │  │
│  │                 │  │              │  │
│  └─────────────────┘  └──────────────┘  │
│                                         │
│  ┌─────────────────────────────────────┐│
│  │  TextInput: "Ask Guide..."         ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

**UI Canvas definition** (`Projects/lets-consciousness/ui.yaml`):

```yaml
# ui.yaml
root:
  type: Panel
  properties:
    background: "#1a1a1a"
  children:
    - type: Panel
      name: instructor_row
      layout: horizontal
      children:
        - type: RadianceViewport
          name: guide_viewport
          properties:
            radiance_path: "Noodlings/guide/Radiances/ajo_majo.radiance"
            width: 300
            height: 400

        - type: Panel
          name: speech_panel
          children:
            - type: Label
              name: guide_speech
              properties:
                text: "Hello! I'm Guide. Ask me anything about consciousness!"
                wrap: true

    - type: TextInput
      name: user_input
      properties:
        placeholder: "Ask Guide something..."
      events:
        onSubmit:
          action: run_assembly
          assembly: "Noodlings/guide/assembly.yaml"
          input_binding: "{user_input.text}"
          output_target: "guide_speech.text"
```

---

### Step 4: Wire Assembly Execution (Already Done!)

The `run_assembly` action in `event_dispatcher.py` handles all of this:

1. Capture `user_input.text` via `"{user_input.value}"` binding
2. Load assembly (cached after first load)
3. Execute assembly async
4. Bind outputs to UI components

**The wiring is in the `ui.yaml`** - the `onSubmit` event binding handles everything.

**What we need to verify:**
- `set_facet_executor()` is called on dispatcher during app init
- Assembly path resolves correctly
- OUTGOING facet outputs are named correctly for binding

**Reference:**
- `runtime/ui/event_dispatcher.py` lines 362-565
- `/docs/noodlestudio/facets.md`
- `/docs/formats/assembly.md`

---

### Step 5: Test the Loop

**Test case 1:** Basic question
- Input: "What is consciousness?"
- Expected: Warm, accessible response in Guide's voice

**Test case 2:** Follow-up
- Input: "Can you explain that differently?"
- Expected: Rephrased explanation (note: no memory yet, each query is independent)

**Test case 3:** Exhibit context (future)
- Input: "What am I looking at?"
- Expected: Guide describes the current exhibit (requires exhibit state in context)

---

## What We're NOT Doing Yet

- Affect dynamics (no CharmNetwork wired up)
- PAD → expression binding (no blend shapes animated)
- Computer use / ghost cursor for teaching
- Exhibit integration
- Memory / conversation history

These come later. First: **make the axolotl talk.**

---

## This Is The Battle Test

The Radiance system, facet execution, UI Canvas event wiring, and assembly loading have not been fully integration-tested together. **Expect bugs.** This implementation will surface:

- Edge cases in VRM → Radiance conversion
- Missing wiring between components
- Async timing issues
- Path resolution bugs
- UI component property binding gaps

**This is good.** Every bug fixed here hardens the platform.

---

## Testing Strategy: NoodleCode Computer-Use Tests

NoodleCode (Claude Code in-editor) has computer-use capabilities - screenshot, click, type, verify. We can write test scripts that NoodleCode executes to validate the full pipeline.

### Test Script Format

Create `tests/noodlecode/` with test scripts NoodleCode can run:

```yaml
# tests/noodlecode/guide_conversation_test.yaml
name: Guide Conversation Test
description: Verify Guide responds to user questions

steps:
  - action: open_project
    project: Projects/lets-consciousness

  - action: wait_for
    component: guide_viewport
    state: rendered
    timeout: 10s

  - action: screenshot
    name: initial_state

  - action: type
    target: user_input
    text: "What is consciousness?"

  - action: click
    target: user_input  # Submit via enter or find submit button

  - action: wait_for
    component: guide_speech
    state: not_empty
    timeout: 30s  # LLM calls take time

  - action: screenshot
    name: after_response

  - action: verify
    component: guide_speech.text
    contains: ["consciousness", "experience", "aware"]  # Reasonable response keywords

  - action: verify
    component: guide_speech.text
    not_contains: ["error", "failed", "undefined"]
```

### Test Categories

**1. Radiance Pipeline Tests**
```yaml
# Does VRM convert without errors?
# Does .radiance load in viewport?
# Is avatar visible and not corrupted?
- verify_vrm_conversion
- verify_radiance_load
- verify_avatar_render
```

**2. Assembly Execution Tests**
```yaml
# Does assembly load?
# Do LLM calls complete?
# Does output bind to UI?
- verify_assembly_load
- verify_perception_facet
- verify_response_facet
- verify_output_binding
```

**3. UI Integration Tests**
```yaml
# Does input capture text?
# Does submit trigger assembly?
# Does response display?
- verify_text_input
- verify_submit_event
- verify_response_display
- verify_loading_state
```

**4. Error Handling Tests**
```yaml
# What happens when LLM times out?
# What happens with malformed assembly?
# What happens with missing radiance file?
- verify_timeout_handling
- verify_assembly_error
- verify_missing_asset_error
```

### Running Tests

NoodleCode can execute these via a command:

```
/test guide_conversation
/test radiance_pipeline
/test all
```

Or run the full suite:

```bash
# Traditional pytest for unit tests
pytest -m "not slow" -v

# NoodleCode integration tests (requires running NoodleStudio)
python -m noodlestudio.test.noodlecode_runner tests/noodlecode/
```

### Why NoodleCode Tests?

1. **Visual verification** - Screenshots catch rendering bugs pytest can't
2. **Full integration** - Tests the actual app, not mocked components
3. **Async handling** - NoodleCode naturally handles wait states
4. **Debugging** - When tests fail, NoodleCode can investigate live
5. **Regression** - Save screenshots as baselines, diff on future runs

### Test Output

```
Guide Conversation Test
├── initial_state.png     ✓ Avatar rendered
├── after_response.png    ✓ Response displayed
├── verify: guide_speech  ✓ Contains expected keywords
└── PASSED

Radiance Pipeline Test
├── vrm_conversion        ✓ No errors
├── radiance_load         ✓ 847 gaussians loaded
├── avatar_render         ✗ FAILED: viewport shows black
└── FAILED
    → Investigating: checking RadianceViewport.set_component()...
```

---

## Expected Bug Categories & Triage

When things break (and they will), here's where to look:

| Symptom | Likely Cause | Where to Fix |
|---------|--------------|--------------|
| VRM conversion fails | Missing blend shapes, unsupported features | `vrm_to_radiance.py` |
| Viewport shows black | Radiance not loaded, camera wrong, shader issue | `RadianceViewport`, `GaussianRenderer` |
| Avatar looks wrong | Bone weights, scale, coordinate system | VRM conversion pipeline |
| Submit does nothing | Event not wired, dispatcher not set | `event_dispatcher.py`, `ui.yaml` |
| Assembly load fails | Path resolution, YAML parsing | `FacetAssembly.load_yaml()` |
| LLM call hangs | API key, network, model config | `LLMFacet`, model config |
| Output not appearing | Output binding wrong, component name mismatch | `_run_assembly_and_bind()` |
| Response in wrong component | Output binding target typo | `ui.yaml` outputs section |

### Debug Workflow

1. **Check logs** - `logger.debug` throughout event_dispatcher
2. **Screenshot** - NoodleCode can capture state at failure
3. **Isolate** - Test each piece independently:
   - Radiance alone in test viewport
   - Assembly alone via pytest
   - UI alone with mock responses
4. **Fix forward** - Don't revert, fix the bug and add a test

---

## Success Criteria

**Phase 1: Axolotl Appears**
1. VRM converts to .radiance without errors
2. Radiance renders in viewport (visible, not black)
3. Avatar is recognizable (not corrupted/inside-out)

**Phase 2: Axolotl Talks**
4. User can type a question and submit
5. Assembly executes (both LLM calls complete)
6. Response displays in speech label
7. Response sounds like Guide (warm, curious, accessible)

**Phase 3: Loop Works**
8. Can ask multiple questions in sequence
9. Loading state shows during LLM calls
10. Errors display gracefully (not silent failures)

**Stretch: First NoodleCode Test Passes**
11. `guide_conversation_test.yaml` runs and passes
12. Screenshots captured for regression baseline

---

## Architecture Notes

### Why Two LLM Calls?

Perception → Response separation allows:
- Different models (fast/cheap for perception, main for response)
- Cleaner prompts (perception extracts, response generates)
- Future: inject affect state between them

### Assembly Execution is Async

LLM calls are async. UI should show loading state while assembly runs.

### Recipe Informs Prompts

The `recipe.yaml` personality/voice data should inform the Response LLM prompt. Currently hardcoded in assembly - could be templated:

```yaml
system_prompt: |
  You are {recipe.display_name}, {recipe.backstory}
  Voice: {recipe.voice.style}
  ...
```

This is a future enhancement. For now, prompts are self-contained.

---

## File References

| What | Where |
|------|-------|
| Guide recipe | `Noodlings/guide/recipe.yaml` |
| Guide assembly | `Noodlings/guide/assembly.yaml` |
| Guide VRM | `Noodlings/guide/Radiances/AjoMajo.vrm` |
| VRM converter | `noodlestudio/tools/vrm_to_radiance.py` |
| Facet executor | `noodlestudio/core/facet_system/facet_executor.py` |
| Radiance format | `docs/formats/radiance.md` |
| Assembly format | `docs/formats/assembly.md` |
| UI Canvas | `docs/noodlestudio/ui-canvas.md` |

---

## Questions - RESOLVED

1. **Where does Let's Consciousness! UI live?**
   **RESOLVED:** Let's Consciousness! is a NoodleStudio PROJECT, not a separate app. Create it as a project folder with `ui.yaml`, assemblies, and noodlings. It runs inside NoodleStudio with View Source permissions.

2. **How does event system invoke assembly?**
   **RESOLVED:** `run_assembly` action is FULLY IMPLEMENTED in `event_dispatcher.py` (lines 362-565). Supports input/output bindings, caching, async execution. Ready to use.

3. **Loading state UX?**
   **NOT YET IMPLEMENTED.** Assembly execution is async but there's no visual feedback. Add this:
   - Before `asyncio.create_task`: Set loading indicator (disable input, show spinner)
   - In `_run_assembly_and_bind` completion: Clear loading state
   - On error: Clear loading state, show error message

---

## First Steps for Trench Claude

Start here. Do these in order:

### 1. Convert the VRM (5 min)
```bash
cd /Users/thistlequell/git/noodlings_clean
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance \
  Noodlings/guide/Radiances/AjoMajo.vrm \
  -o Noodlings/guide/Radiances/ajo_majo.radiance \
  -v
```

If this fails, that's your first bug to fix. Check the VRM structure, blend shapes, bone hierarchy.

### 2. Test Radiance Render Standalone
Before wiring up the full UI, verify the radiance loads:
- Open NoodleStudio
- Create a test viewport
- Load `Noodlings/guide/Radiances/ajo_majo.radiance`
- Confirm avatar renders

If black screen or corruption, fix before proceeding.

### 3. Create Let's Consciousness! Project Skeleton
```
mkdir -p Projects/lets-consciousness
```

Create `Projects/lets-consciousness/project.yaml`:
```yaml
name: "Let's Consciousness!"
version: 0.1.0
permission: view_source
ui: ui.yaml
```

### 4. Create Minimal ui.yaml
Start with just the viewport - no assembly wiring yet:
```yaml
version: 1
root:
  type: Panel
  name: root
  background: "#1a1a1a"
  children:
    - type: RadianceViewport
      name: guide_viewport
      radiance_path: "../../Noodlings/guide/Radiances/ajo_majo.radiance"
      width: 300
      height: 400
```

Open project, verify avatar appears.

### 5. Add Input + Label
Once avatar renders, add the UI components.

### 6. Wire Assembly
Once UI works, add the `run_assembly` event binding.

### 7. Test Full Loop
Type a question, get a response.

### 8. Add Loading State
Implement spinner/disable during LLM calls.

---

## Reminder: Run Regression Tests

Before and after major changes:
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest -m "not slow" -v
```

Don't break existing functionality while adding new.

---

*Brief prepared by: Architecture Claude (claudechat)*
*For: Implementation Claude (noodlings_clean)*
*Date: 2025-01-08*
*Status: Ready for implementation*
