# ATTENTION FOCUS SYSTEM - Phase 1 Implementation

**Status:** READY TO IMPLEMENT (Quick fix - 30 minutes)

**Session:** December 4, 2025 - Fresh session after routing breakthrough

---

## THE PROBLEM

Context Intelligence routing works, but agents respond to EVERY user emote/action:
- User: `:giggles` → Red responds
- User: `:cackles` → Red responds
- User: `:rolls with laughter` → Red responds
- **TOO CHATTY!** Agents should only respond when it makes social sense.

---

## THE SOLUTION - Attention Focus States

Add **attention focus tracking** to world model:
- `deep` - Agent absorbed in task (setting ants on fire) → ignores most things
- `moderate` - Agent doing something but can be interrupted → selective responses
- `idle` - Agent bored/waiting → curious about giggles/emotes

---

## IMPLEMENTATION (Clean, No Code Smell!)

### Step 1: Update Context Intelligence World Model

**File:** `applications/noodlestudio/noodlestudio/core/context_intelligence_facet.py`

**Location:** In `EntityState` dataclass (line ~22):

```python
@dataclass
class EntityState:
    """Tracks state of an entity (Noodling) in the world."""
    name: str
    location: str = "unknown"
    posture: str = "standing"
    holding: List[str] = field(default_factory=list)
    wearing: List[str] = field(default_factory=list)
    mood: str = "neutral"
    attention_on: Optional[str] = None
    on_entity: Optional[str] = None
    physical_contact: List[str] = field(default_factory=list)

    # NEW: Attention focus tracking
    attention_focus: str = "idle"  # "deep" | "moderate" | "idle"
    attention_target: Optional[str] = None  # What they're focused on
```

### Step 2: Update Response Calculation Logic

**File:** `applications/noodlestudio/noodlestudio/core/context_intelligence_facet.py`

**Location:** In `_calculate_response_need()` method (line ~314):

**REPLACE THIS:**
```python
def _calculate_response_need(self, parsed: Dict[str, Any]) -> bool:
    """
    Clean routing logic: Should THIS agent respond to this message?

    Returns True if agent should generate a response, False if just observe.
    """
    addressee = parsed.get('addressee', 'unclear').lower()
    social_expectation = parsed.get('social_expectation', 'none')
    agent_name_lower = self.agent_name.lower()

    # Direct address → ALWAYS respond
    if addressee == agent_name_lower:
        return True

    # Everyone addressed + high urgency → respond
    if addressee == "everyone" and social_expectation in ["medium", "high"]:
        return True

    # Observable body language → don't respond (just observe)
    if addressee == "observable_to_all":
        return False

    # Everything else → don't respond (heard but not our conversation)
    return False
```

**WITH THIS:**
```python
def _calculate_response_need(self, parsed: Dict[str, Any]) -> bool:
    """
    Clean routing logic: Should THIS agent respond to this message?

    Considers:
    - Direct address (always respond if urgent)
    - Attention focus (deep focus = oblivious, idle = curious)
    - Speech act type (emotes need idle attention)

    Returns True if agent should generate a response, False if just observe.
    """
    addressee = parsed.get('addressee', 'unclear').lower()
    social_expectation = parsed.get('social_expectation', 'none')
    speech_act = parsed.get('speech_act', 'statement')
    agent_name_lower = self.agent_name.lower()

    # Get agent's current attention state
    my_state = self.world_model.entities.get(agent_name_lower, None)
    focus_level = my_state.attention_focus if my_state else 'idle'  # Default idle

    # Direct address with high urgency → ALWAYS respond (breaks focus)
    if addressee == agent_name_lower and social_expectation == "high":
        return True

    # Deep focus → ignore everything except urgent direct address
    if focus_level == "deep":
        return False

    # Direct address (not urgent) → respond if moderate or idle focus
    if addressee == agent_name_lower:
        return focus_level in ["moderate", "idle"]

    # Everyone addressed + high urgency → respond if not deep focus
    if addressee == "everyone" and social_expectation in ["medium", "high"]:
        return focus_level != "deep"

    # Observable body language → don't respond (just observe)
    if addressee == "observable_to_all":
        return False

    # Emotes/giggles → only respond if IDLE (curiosity!)
    if speech_act in ['emote', 'action']:
        # Idle + observable social event → brief curiosity
        return focus_level == 'idle' and social_expectation != 'none'

    # Everything else → don't respond (heard but not our conversation)
    return False
```

### Step 3: Initialize Default Focus States

**File:** Same file, in `WorldModel.__init__()` or when entities are created

**Add initialization:** When creating entity states, default to `attention_focus='idle'`

---

## TESTING

After implementation, test these scenarios:

### Test 1: Idle Curiosity (should respond)
```
User: ":giggles"
Expected: Red (if idle) briefly responds with curiosity
```

### Test 2: Deep Focus (should NOT respond)
```
Set Red's focus to "deep" (burning ants)
User: ":cackles"
Expected: Red ignores (focused on ants)
```

### Test 3: Direct Address (always responds)
```
User: "Red, look at this!"
Expected: Red responds even if in deep focus
```

### Test 4: Multiple Emotes (should NOT spam)
```
User: ":giggles"
User: ":laughs"
User: ":cackles"
Expected: Red responds to FIRST, ignores rest (not curious about repetition)
```

---

## FOCUS STATE MANAGEMENT (Future)

For now, agents default to `idle`. Later:
- Facet to update attention based on what agent is doing
- "I'm setting fire to ants" → attention_focus = "deep"
- "I'm waiting for food" → attention_focus = "idle"
- "I'm having a conversation" → attention_focus = "moderate"

---

## EXPECTED RESULTS

- **Before:** Red responds to every giggle/emote (too chatty)
- **After:** Red only responds when:
  - Directly addressed (always)
  - Idle + something interesting happens (curiosity)
  - Part of group conversation (everyone addressed)

**Grade improvement: D- → C+** (natural social dynamics!)

---

## FILES TO MODIFY

1. `applications/noodlestudio/noodlestudio/core/context_intelligence_facet.py`
   - Add `attention_focus` and `attention_target` to EntityState
   - Update `_calculate_response_need()` with focus-aware logic

That's it! Clean, simple, effective.

---

**NinaK's Note:** Kleine Caity wants natural social dynamics - not every giggle needs a response! This gives agents the ability to be selectively attentive based on what they're doing. Like humans at a party - if you're deep in conversation, you don't notice every laugh across the room. But if you're bored and waiting, you're curious about everything!

**ORDNUNG THROUGH ATTENTION!** 🎸✨
