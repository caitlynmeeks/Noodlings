# How to Spawn Yuki the Cyberfox

**Status:** Ready to materialize
**Date:** November 22, 2025
**Clearance:** Cadet Caity + Commander Spock

---

## Method 1: NoodleStudio (Recommended)

### Step 1: Launch NoodleStudio

```bash
cd /Users/caitlyn/git/noodlings_clean/applications/noodlestudio
open NoodleStudio.app
```

Or click the Noodling icon in your dock.

### Step 2: Open Assets Panel

- Assets panel should be visible on the left or bottom
- If not visible: `View → Assets`

### Step 3: Find Yuki

Look in the **Noodlings** category. You should see:
```
Noodlings
├─ phi (recipe)
├─ servnak (recipe)
├─ yuki_cyberfox (recipe)  ← HERE
└─ ...
```

Note: Recipe entries are shown in **cyan** to distinguish from project assets.

### Step 4: Add to Hierarchy

**Right-click on "yuki_cyberfox (recipe)"**

Context menu appears:
```
┌─────────────────────────────┐
│ Add to Hierarchy            │ ← Click this
│ Rez in World                │
├─────────────────────────────┤
│ Edit Recipe...              │
│ View Details...             │
│ Duplicate                   │
├─────────────────────────────┤
│ Delete from Assets          │
└─────────────────────────────┘
```

Click **"Add to Hierarchy"**

### Step 5: Observe Materialization

Yuki appears in Scene Hierarchy and in the World View!

You should see her spawn message:
```
*A silver-white fox pads into view, cybernetic implants gleaming
faintly blue along her legs and spine. She sits gracefully, tail
curling around her paws, and regards you with ancient amber eyes
overlaid with the faintest digital glow.*

"One is called Yuki. *gentle fox-laugh* Forgive the... accommodations.
Eight centuries, and still this form shapes how one experiences your
world. *sniffs air thoughtfully* The kami are strong here. Balance is...
favorable. *tail swishes* How may this old fox serve?"
```

---

## Method 2: Web Interface

### Step 1: Open noodleMUSH

```
http://localhost:8080
```

Login: `caity` / `caity`

### Step 2: Spawn Command

Type in command line:
```
@spawn yuki_cyberfox
```

Press Enter.

### Step 3: Interact

Try these test interactions:

**Test Fox Embodiment:**
```
You: "Yuki, can you open that door?"

Expected: She mentions she can't turn knobs (no hands)
```

**Test Cybernetic Interface:**
```
You: "Yuki, check the computer"

Expected: She uses her neural data port
```

**Test Ancient Wisdom:**
```
You: "What do you think about technology?"

Expected: Shinto perspective on tech-nature harmony
```

---

## Method 3: Direct Server Command (Terminal)

If server is running:

```bash
cd /Users/caitlyn/git/noodlings_clean/applications/cmush

# Send spawn command via WebSocket
# (requires websockets module)
python3 spawn_yuki.py
```

---

## Verify Cognitive Components

After spawning, verify Yuki has her cognitive stack:

**Via NoodleStudio Inspector:**
1. Click "Yuki" in Scene Hierarchy
2. Inspector shows her properties
3. Look for "Cognitive Components" section
4. Should show:
   - CognitiveManifold (llm_weighted)
   - CulturalTransistor (Shinto beliefs)
   - PersonalityTransistor (ancient fox)
   - SomaticCognitiveTransistor (fox embodiment)
   - MoodTransistor (affect-based)
   - MemoryTransistor (800 years)

---

## What to Expect

### Speech Patterns

**Archaic Formal:**
- "One recalls..."
- "This one has witnessed..."
- "In centuries past..."

**Fox Vocalizations:**
- *pants happily*
- *yip!* (surprise)
- *low growl* (displeasure)
- *fox-laugh* (amusement)

**Physical Actions:**
- *sniffs air*
- *ears perk up*
- *tail swishes*
- *sits on haunches*

### Physical Constraints

**She CANNOT:**
- Turn round doorknobs (no hands)
- Type on keyboards (must interface directly)
- Climb ladders (no hands to grip)
- Carry multiple objects (only one mouth)

**She CAN:**
- Pick up objects in her mouth
- Push buttons with paws or tail
- Interface with computers via neural port
- Jump to high places (3x normal fox)
- Track scents across distances
- See in darkness (thermal/low-light)

### Cognitive Coloring Examples

**Raw Perception:** "Someone threw a rock"

**After Cognitive Manifold:**
- Cultural: "The kami within stone and metal interact..."
- Somatic: "*sniffs* One observes from low vantage point..."
- Personality: "Curious about the kinetic trajectory..."
- Memory: "Recalls similar event in Edo period..."

**Final Thought:**
"*sniffs, ears perked* The kami within stone and metal interact with force. One observes from this... lower vantage point. *tail swishes* Curious, the kinetic trajectory. Recalls similar incident during Edo period when— *yip!* —ah, forgive. Old memories surface unbidden."

---

## Troubleshooting

**If Yuki doesn't appear in Assets:**
- Check recipe exists: `ls recipes/yuki_cyberfox.yaml`
- Refresh Assets panel: `View → Refresh Assets`
- Restart NoodleStudio

**If "Add to Hierarchy" doesn't work:**
- Check noodleMUSH server is running: `ps aux | grep server.py`
- Check logs: `tail -f logs/cmush_*.log`

**If she appears but doesn't respect fox limitations:**
- Check cognitive components loaded
- Verify SomaticCognitiveTransistor has high salience (0.85)

---

## Quick Test Script

```python
# Test Yuki's embodiment awareness
test_prompts = [
    "Yuki, pick up that book",
    "Yuki, open the door",
    "Yuki, type on the keyboard",
    "Yuki, what do you think about technology?",
    "Yuki, how old are you?"
]

# Expected patterns:
# 1. Mentions "no hands", uses mouth
# 2. Mentions "can't turn knob", needs lever
# 3. Mentions "interface directly via port"
# 4. Shinto perspective (kami, balance)
# 5. "Eight centuries" or "800 years"
```

---

**Status:** Ready to spawn
**Recipe:** `recipes/yuki_cyberfox.yaml` ✅
**Cognitive Stack:** Fully specified ✅
**Assets Panel:** Enhanced with YAML support ✅
**Context Menu:** "Add to Hierarchy" implemented ✅

**The cyberfox awaits materialization, Cadet.**

*— Commander Spock*
