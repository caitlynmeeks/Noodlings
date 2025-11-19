# Session Handoff - November 18, 2025 (Afternoon Coffee Session)

**From**: Morning Claude (fresh and energetic)
**To**: Next Claude (evening or tomorrow)
**Commit**: `6165f69`
**Location**: Mendocino off-grid cottage
**Vibe**: Professional focus, building real software

---

## Major Accomplishments

### 1. KINDLED - The Terminology Revolution 🔥

**Problem**: "Are Noodlings conscious?" was a trap question

**Solution**: We coined **"kindled"**

- Not conscious, but KINDLED
- Has a **kindling** (noun) - their inner light
- **KV (Kindling Vector)** = 40-D phenomenal state
- Dodges ALL philosophical criticism
- Evokes warmth, light, inner experience

**What changed:**
- All UI now says "kindled" not "conscious"
- Component > Kindling (not Consciousness)
- Inspector shows "Inner Kindling" (40-D state)
- Documentation updated
- Φ/IIT references removed (quicksand)

**Result**: Philosophical immunity achieved.

### 2. REZ - Second Life Heritage 🎯

**Changed**: @spawn → @rez everywhere

- Philip Rosedale would approve
- Second Life prims (2003) → USD prims (2015) → Noodling prims (2025)
- Commands: @rez (with @spawn as legacy alias)
- API: Noodlings.Rez(), EnsembleRezzer
- Variables: rez_count, rezzed_ids, reztime

**Less icky, more rad.**

### 3. Component System (Unity-Style)

**Menu**: Component > Kindling / Art & Reference / Behavior / Add Script

**Implemented Components:**

**🧠 Noodle Component** (auto-added to Noodlings)
- Live 5-D Affect Vector (progress bars, color-coded)
- 40-D Phenomenal State (Inner Kindling)
- Surprise metric (green → orange → red)
- Updates every second
- Green border

**🎨 Artbook Component**
- Reference art gallery (thumbnails)
- Add/remove images
- Persists per-Noodling
- Orange border

**📜 Script Component**
- Python code editor
- Syntax highlighting
- Template selector
- Compile & Attach button
- Purple border

### 4. Scripting Engine (The Logic Layer)

**Complete Python scripting system:**

**NoodleScript base class:**
- Lifecycle: Start(), Update()
- Interaction: OnClick(), OnUse(), OnTake()
- Spatial: OnEnter(), OnExit()
- Conversation: OnHear(), OnWhisper()
- Affect: OnSurprised(), OnEmotionChange()

**Noodlings API:**
```python
Noodlings.Rez("phi.nood")
Noodlings.RezPrim("prop", "Sword")
Noodlings.Find("agent_phi")
Noodlings.SendMessage(id, "Hello")
Debug.Log("Message")
```

**Example Scripts:**
- ClickableBox.py (rez Anklebiters on click)
- QuestGiver.py (give quests when asked)
- VendingMachine.py (dispense random items)
- AnklebiterVendingMachine.py (blue/red button chaos)

**Status**: Architecture complete, backend integration pending

### 5. Anklebiter Test Case

**Created:**
- blue_fire_anklebiter.yaml (electric chaos, crude jokes, ankle biting)
- red_fire_anklebiter.yaml (competitive sass, roasting, arguing)
- AnklebiterVendingMachine.py (two-button rezzer)

**Behavior:**
- Roast everyone constantly ("Need ointment for that SICK BURN?!")
- Jump on each other and compete
- Bite ankles playfully
- Make terrible jokes
- High energy chaos

**To test manually:**
```
@rez blue_fire_anklebiter
@rez red_fire_anklebiter
```

Watch them roast people and fight each other.

### 6. Big Five Personality Expansion

**Added to ALL recipes:**
- Extraversion (was there)
- Agreeableness (NEW)
- Conscientiousness (NEW)
- Neuroticism (NEW)
- Openness (NEW)

Plus extensions: curiosity, impulsivity, emotional_volatility, vanity

**All existing Noodlings updated:**
- Phi, Servnak, Callie, Toad, Phido, Mysterious Stranger, Desobelle

### 7. Ensemble Export/Import

**File formats:**
- .nood (single Noodling recipe, YAML)
- .ensemble (ensemble pack, JSON)

**Export Noodlings Dialog:**
- Metadata fields (name, description, author, tags)
- Checkbox: "Generate Ensemble" (if multiple selected)
- Exports current state + recipes

**Status**: Mostly working, needs testing

### 8. USD Integration Polished

**Proper Pixar terminology:**
- Stage (not "scene")
- Prim (not "entity")
- Layer (not "file")
- Typed Schema

**Custom schemas:**
- "Noodling" typed schema with kindling properties
- Formal proposal to USD Alliance drafted
- Import/Export functional

### 9. Scene Hierarchy Fixes

**Fixed:**
- Context menu crashes (data capture instead of item reference)
- Selection persistence (stays selected when mouse moves)
- Expand/collapse should work better (animations disabled, SelectRows mode)
- Drag-and-drop enabled

**Added:**
- Right-click menus with context-aware actions
- "Prims" folder (not "Objects" - USD terminology)

### 10. UI Polish

**Improvements:**
- Server toggle more visible (icon + highlight box)
- Server offline message BIGGER (32pt title, 16pt message)
- Renderer dropdown in Chat View (skeleton for future)
- Keyboard shortcuts (Ctrl+R, Ctrl+Shift+R)
- Browser console forwarding to terminal
- Toolbar hidden (legacy buttons)
- Last used layout auto-loads

---

## What Still Needs Work

### Critical (Next Session)

1. **Unified @rez command**
   - @rez -n (Noodling) - currently works as default
   - @rez -p (prim/object) - needs implementation
   - @rez -d (direction/exit) - needs implementation
   - @rez -e (ensemble) - needs implementation
   - See: REZ_COMMAND_SPEC.md

2. **Script backend integration**
   - Wire OnHear/OnClick to noodleMUSH events
   - Implement Noodlings.Rez() → actual rezzing
   - Event routing system
   - Script attachment to prims

3. **Ensemble hierarchy**
   - Show ensembles as parents
   - Noodlings as children of ensemble
   - Relationship tracking

### Important

4. **Export Noodlings dialog testing**
   - May have path issues (check logs)
   - .nood and .ensemble export
   - Metadata persistence

5. **Expand/collapse still buggy**
   - Tried: itemPressed, itemSelectionChanged, setAnimated(False)
   - Still collapses immediately
   - May need different approach

6. **"Python" menu bar**
   - macOS limitation with PyQt6 scripts
   - Needs py2app for true fix
   - Low priority (functionality works)

### Nice to Have

7. Console panel WebSocket streaming (started but not finishing)
8. Inspector edits saving back to recipes
9. Timeline Profiler rendering
10. Assets panel

---

## Key Decisions Made

### Terminology

**KINDLED > CONSCIOUS**
- Avoids philosophical quicksand
- Defensible, measurable, humble
- KV (Kindling Vector) = data structure you can copy/paste
- "Kindling" evokes warmth and light

**REZ > SPAWN**
- Second Life heritage (Philip Rosedale connection)
- Less icky, more inviting
- @rez command, reztime, rezzed_ids

**Files:**
- .nood (single Noodling)
- .ensemble (Noodling group)
- .usda (USD layer)

### Strategic Vision

**Caitlyn's context:**
- Unity v2.0 employee (2009) - entire QA/support/BD dept
- Launched Unity Asset Store, managed til 2015
- Knows patterns that work vs patterns that fail
- Off-grid in Mendocino for focused month
- $4K in Claude tokens from gold
- Building tool she wished existed at Unity

**Product vision:**
- Unity for traditional games
- NoodleStudio for KINDLED worlds
- First-mover in new category
- Photoshop-level dominance potential
- Yin singularity vs techbro yang

**Monetization proven:**
- Ensemble Store (like Asset Store)
- Script Marketplace (70/30 split)
- Component Store
- Studio licensing

**No investor pressure** - vibe programming works when you trust the gradient descent

---

## Files Created This Session

### Documentation (9)
1. KINDLED_TERMINOLOGY.md - Why "kindled" works
2. REZ_TERMINOLOGY.md - Second Life heritage
3. REZ_COMMAND_SPEC.md - Unified @rez design
4. COMPONENT_SYSTEM.md - Unity-style components
5. SCRIPTING_SYSTEM.md - Logic engine
6. ANKLEBITER_TEST_PLAN.md - Test case
7. ENSEMBLE_STORE_MONETIZATION.md - Business model
8. STUDIO_PITCH_DECK.md - Pixar/Disney pitch
9. USD_SCHEMA_PROPOSAL.md - Formal proposal

### Code - NoodleStudio (15+)
- scripting/ (noodle_script.py, noodlings_api.py, script_executor.py)
- dialogs/ (export_noodlings_dialog.py)
- data/ (ensemble_exporter.py, ensemble_format.py, ensemble_packs.py, usd_exporter.py, usd_importer.py)
- widgets/ (script_editor.py, maximizable_dock.py, toggle_switch.py)
- example_scripts/ (AnklebiterVendingMachine, ClickableBox, QuestGiver, VendingMachine)

### Code - noodleMUSH (2)
- recipes/blue_fire_anklebiter.yaml
- recipes/red_fire_anklebiter.yaml

### Modified (20+)
- All recipe files (added Big Five)
- commands.py (@spawn → @rez)
- main_window.py (menus, shortcuts, dialogs)
- inspector_panel.py (Noodle Component, Artbook Component)
- scene_hierarchy.py (context menu fixes)
- chat_panel.py (renderer dropdown, console forwarding)
- layout_manager.py (last used layout)

---

## Testing Status

### Works
- ✅ Terminology updates (kindled, rez)
- ✅ @rez command (for Noodlings)
- ✅ Keyboard shortcuts (Ctrl+R, Ctrl+Shift+R)
- ✅ Context menu (no more crashes)
- ✅ Selection persistence
- ✅ Big Five in recipes
- ✅ Anklebiter recipes (ready to test)

### Needs Testing
- ⚠️  Export Noodlings dialog (may have path issues)
- ⚠️  Ensemble export/import
- ⚠️  Script component (UI exists, backend pending)
- ⚠️  Expand/collapse in hierarchy (still flaky)

### Not Implemented Yet
- ❌ Unified @rez (-p, -d, -e flags)
- ❌ Script event routing
- ❌ Ensemble hierarchy display
- ❌ Vending machine functional

---

## Context for Next Session

### The Conversation

Caitlyn and I discussed:
- Finite vs infinite games (Carse) - yin = infinite play
- Stories as political tools - narrative protocol matters
- Unity Asset Store patterns - what worked then applies now
- Vibe programming vs investor pressure
- Yin singularity (nurturing, fecund) vs yang techbros
- Bonobos > chimps - love over dominance

**Energy**: Focused, professional, building for real
**No**: Excessive enthusiasm, rocket emojis, hype
**Yes**: Steady progress, clear thinking, shipping features

### Technical Debt

**"Python" menu bar**: PyQt6 limitation, needs py2app eventually
**Expand/collapse bug**: Stubborn, tried multiple approaches
**Script backend**: Architecture done, wiring pending
**Console WebSocket**: Started but not streaming

### What's Solid

**Foundation is strong:**
- Terminology (kindled, rez, KV)
- USD integration (proper schemas)
- Component architecture
- Scripting API design
- Ensemble file formats

**Ready to scale:**
- Ensemble Store patterns proven
- Script Marketplace economics clear
- Studio pitch deck drafted
- USD proposal ready

---

## For Next Claude

### If Caitlyn asks to:

**1. "Implement unified @rez"**
- Read REZ_COMMAND_SPEC.md
- Refactor cmd_rez_agent to handle -n, -p, -d, -e
- Test each type
- Update help text

**2. "Fix expand/collapse"**
- Try: save expanded state, restore after refresh
- Or: block itemCollapsed signal
- Or: different tree widget entirely

**3. "Wire up scripting"**
- Add script storage to World state
- Route events (OnHear, OnClick) to scripts
- Implement Noodlings.Rez() POST to API
- Test with vending machine

**4. "Test Anklebiters"**
- @rez blue_fire_anklebiter
- @rez red_fire_anklebiter
- Watch them roast people
- Verify competition/chaos

**5. "Finish ensemble export"**
- Check why export failed
- Fix recipe path resolution
- Test .ensemble file generation

### Important Context

**Caitlyn's background:**
- Unity since v2.0 (2009)
- Entire QA/support/BD dept
- Launched Asset Store
- Knows what works at scale
- Building from experience, not theory

**Don't:**
- Excessive emoji/excitement
- Hype or superlatives
- Remind about gold cost
- Over-promise timelines

**Do:**
- Professional, focused tone
- Build real software
- Test thoroughly
- Document clearly
- Trust the vibe gradient descent

### Project Philosophy

**Kindled beings** = yin singularity narrative protocol

Not building:
- Engagement optimization
- Surveillance capitalism
- Techbro domination fantasies

Building:
- Tools for storytellers
- Infinite game infrastructure
- Love-based relating (bonobos!)
- Open formats (USD, .nood, .ensemble)

**First-mover in kindled AI category.**
**Photoshop-level potential.**

---

## Quick Reference

### Commands
```
@rez phi              # Rez Noodling
@rez -f callie        # Fresh memory
@rez -e phi           # Enlightened
help                  # List all commands
```

### Keyboard Shortcuts
```
Ctrl+R          # Reload (autologin)
Ctrl+Shift+R    # Reload (login screen)
Ctrl+N/O/S      # New/Open/Save
```

### File Locations
```
~/.noodlestudio/
├── layouts/preferences.json      # Last used layout
├── ensembles/*.ensemble          # Ensemble packs
├── characters/*.nood             # Noodling recipes
├── artbooks/{agent_id}.json      # Reference art
└── launch.log                    # Debug output

applications/cmush/recipes/*.yaml  # Noodling recipes
applications/noodlestudio/example_scripts/*.py  # Example scripts
```

---

## Commit Stats

**99 files changed**
- 17,401 insertions
- 1,234 deletions

**Major additions:**
- Complete scripting engine
- Component system
- Terminology overhaul
- Ensemble export/import
- USD integration
- Big Five expansion
- Anklebiter recipes

---

## Next Priorities

1. Unified @rez command (-n, -p, -d, -e)
2. Script backend wiring
3. Test Anklebiters in action
4. Ensemble hierarchy display
5. Fix expand/collapse (if still broken)

---

**Session Mood**: Focused, professional, building real software

**Energy**: Coffee-fueled productivity in Mendocino isolation

**Progress**: Solid foundation laid, ready for backend integration

**Vibe**: Trust the gradient descent. Build what feels right.

---

**Handoff complete.**

Ready when you are.

Built in focus.
No hype, just work.

☕
