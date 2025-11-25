# noodleMUSH Human UI Test Protocol
## Testing Session: November 25, 2025 - UPDATED

**Test Subject**:
- Cognitive architecture components (Affect Transistor, PEE, FACS/Laban, Events)
- **NEW**: UUID system for all entities
- **NEW**: Cognitive gate locking (pause/resume)
- **NEW**: Fuzzy entity matching bug fix

**Server**: http://localhost:8080
**Status**: Server running - RESTART REQUIRED FOR NEW CHANGES

---

## Pre-Test Checklist

- [x] UUID system implemented (all new entities use UUIDs)
- [x] Cognitive gate locking implemented
- [x] Fuzzy matching bug fixed (room['uid'] not room['id'])
- [x] Component UUID system (GetUUID(), GetComponentByUUID())
- [ ] Server restarted with new changes
- [ ] Browser open to http://localhost:8080
- [ ] Chat interface visible

---

## TEST 0: Cognitive Gate Locking (PAUSE/RESUME)

**Objective**: Verify that pausing cognition freezes state and prevents race conditions

**CRITICAL BUG FIXED**: When you pause cognition, in-flight LLM responses were overwriting the frozen state. Now events are queued during pause and processing is blocked.

**Test Steps**:

1. Open NoodleScope 2.0: http://localhost:8081/noodlescope
2. Select Red Fire Anklebiter from agent list
3. In main chat window, type: `say Hey Red Fire!`
4. **IMMEDIATELY** click the Pause button (⏸) in NoodleScope
5. Watch the transistor outputs update
6. **OBSERVE**: Do outputs freeze after pause? Or do they keep updating?

**Expected Behavior**:
- Outputs should freeze when pause is clicked
- No new transistor values should appear
- Status should show "⏸ PAUSED"

7. Wait 5 seconds (let any in-flight LLM calls finish)
8. **OBSERVE**: Do frozen values stay frozen? Or did late-arriving LLM responses overwrite them?

**Success Criteria**:
- [ ] Transistor outputs freeze immediately on pause
- [ ] No late-arriving LLM responses overwrite frozen values
- [ ] Status indicator shows "⏸ PAUSED"

9. Click Resume button (▶)
10. Type another message: `say How are you Red Fire?`
11. **OBSERVE**: Does processing resume normally?

**Success Criteria**:
- [ ] Processing resumes after clicking resume
- [ ] New responses generate normally
- [ ] No queued events from pause period are processed (they're stale)

**Notes**:
```
Pause behavior:


Resume behavior:


Issues found:


```

---

## TEST 1: Poetic Emotional Encoding (PEE)

**Objective**: Verify transistors output rich phenomenological text instead of discrete emotion labels

**What to Look For**:
- BAD (old system): "I feel negative, low arousal, submissive"
- GOOD (PEE): "Restless ache gnawing at me... wanna FIGHT!"
- GOOD (PEE): "Heaviness... like everything's slightly gray. Not quite energy to care."

**Test Steps**:

1. Open browser to http://localhost:8080
2. Type: `say Hey Red Fire, how are you feeling?`
3. Wait for response
4. OBSERVE: Does response contain poetic emotional language?
5. Type: `say Red Fire, you seem upset`
6. Wait for response
7. OBSERVE: Does response reflect emotional nuance?
8. Type: `say Want to compete?`
9. Wait for response
10. OBSERVE: Does response show affect-driven action impulse?

**Success Criteria**:
- [ ] Responses contain first-person emotional language
- [ ] NO discrete labels like "negative affect" or "low arousal"
- [ ] Emotional descriptions feel phenomenological (how it feels, not analytical)

**Notes** (fill in during testing):
```
Response 1:

Response 2:

Response 3:

```

---

## TEST 2: Fuzzy Entity Matching

**Objective**: Verify partial name matching resolves correctly

**Test Steps**:

1. Type: `look red`
   - EXPECTED: Ambiguous match → Server should ask which one (Red Fire Anklebiter? Red Toy?)
   - ACTUAL: _____________________

2. Type: `@observe anklebiter`
   - EXPECTED: Ambiguous match → Ask "Red or Blue Fire Anklebiter?"
   - ACTUAL: _____________________

3. Type: `@observe red fire`
   - EXPECTED: Clear match → Shows Red Fire Anklebiter phenomenal state
   - ACTUAL: _____________________

4. Type: `look _fire_`
   - EXPECTED: Ambiguous match → Ask which Fire Anklebiter
   - ACTUAL: _____________________

5. Type: `look blue`
   - EXPECTED: Clear match → Shows Blue Fire Anklebiter description (if spawned)
   - ACTUAL: _____________________

**Success Criteria**:
- [ ] Ambiguous matches trigger disambiguation prompt
- [ ] Clear matches resolve immediately
- [ ] Substring matching works (partial names match)

**Notes**:
```


```

---

## TEST 3: Default Transistors

**Objective**: Verify all Noodlings get AffectTransistor + MoodTransistor by default

**Test Steps**:

1. Open log file in separate terminal:
   ```bash
   tail -f logs/cmush_2025-11-25.log | grep "Registered.*Transistor"
   ```

2. Spawn a new agent without explicit transistor config:
   ```
   @spawn test_noodle
   ```

3. Check logs for registration messages

**Expected Log Output**:
```
[agent_test_noodle] No cognitive_components in recipe, using defaults
[agent_test_noodle] Registered AffectTransistor (salience=0.70)
[agent_test_noodle] Registered MoodTransistor (salience=0.50)
```

**Success Criteria**:
- [ ] New agent spawned
- [ ] Logs show "No cognitive_components, using defaults"
- [ ] Logs show AffectTransistor registered
- [ ] Logs show MoodTransistor registered

**Actual Log Output**:
```


```

---

## TEST 4: Cognitive Component Salience

**Objective**: Verify transistor salience affects output intensity

**Test Steps**:

1. Observe Red Fire Anklebiter's affect responses:
   ```
   @observe red fire
   ```

2. Note salience values from logs:
   - AffectTransistor: 0.85 (high)
   - PersonalityTransistor: 0.80 (high)
   - CulturalTransistor: 0.75 (high)

3. Talk to Red Fire and see if responses are INTENSE:
   ```
   say Red Fire, let's race!
   ```

4. Expected: High-energy competitive response (because of high affect salience)

**Success Criteria**:
- [ ] High salience (0.85) produces strong emotional coloring
- [ ] Responses feel impulsive and affect-driven
- [ ] Red Fire acts competitive/feisty (personality + affect blend)

**Notes**:
```


```

---

## TEST 5: First-Person Action Prompts

**Objective**: Verify transistors generate desires/impulses, not analytical descriptions

**What to Look For**:
- BAD (analytical): "This situation makes me feel slightly positive with moderate arousal"
- GOOD (action): "I wanna JUMP on that! Let's GO!"
- GOOD (action): "Ugh, should probably just... hang my head down. Hide."

**Test Steps**:

1. Provoke strong positive affect:
   ```
   say Red Fire, you're amazing! You're the BEST anklebiter!
   ```

2. Wait for response
3. OBSERVE: Does response show action impulse (e.g., "I wanna...", "Let's...", "Should...")?

4. Provoke strong negative affect:
   ```
   say Red Fire, you're terrible at biting ankles. Blue Fire is way better.
   ```

5. Wait for response
6. OBSERVE: Does response show defensive/competitive action impulse?

**Success Criteria**:
- [ ] Responses use first-person action language
- [ ] NO third-person analysis ("This makes me feel...")
- [ ] Responses show desires/impulses ("I wanna...", "Should...", "Let's...")

**Notes**:
```
Positive affect response:


Negative affect response:


```

---

## TEST 6: Response Type Diversity

**Objective**: Verify agent uses different response types (SAY, EMOTE, DO, THINK, NONE)

**Test Steps**:

1. Monitor chat for 5 minutes of interaction
2. Count response types that appear:
   - SAY: Verbal speech (speech bubbles)
   - EMOTE: Emotional expression with action (*laughs*, *grins*)
   - DO: Physical action (*jumps*, *bites ankle*)
   - THINK: Internal rumination (thought bubbles, if visible)
   - NONE: Silent (no output, but FACS/Laban should still fire if implemented)

3. Try to provoke each type:
   - SAY: Ask a direct question
   - EMOTE: Make Red Fire laugh
   - DO: Suggest physical activity
   - THINK: Say something puzzling
   - NONE: Ignore Red Fire (should stay silent but think)

**Success Criteria**:
- [ ] At least 2 different response types observed
- [ ] Responses feel appropriate to context
- [ ] NO stuck patterns (all SAY, or all EMOTE)

**Observed Types**:
```
SAY examples:

EMOTE examples:

DO examples:

THINK examples:

NONE examples:

```

---

## TEST 7: Character Voice Consistency

**Objective**: Verify Red Fire Anklebiter maintains character personality

**Expected Character Traits**:
- Sassy, competitive gremlin
- Cackles menacingly (MWAHAHAHA, kekekeke)
- Argues badly but confidently
- Bites ankles (playfully aggressive)
- Competes with Blue Fire Anklebiters
- Exaggerated authority about nonsense

**Test Steps**:

1. Have extended conversation (10+ exchanges)
2. Check if character traits appear consistently
3. Note any out-of-character moments

**Success Criteria**:
- [ ] Red Fire uses gremlin vocabulary (cackles, bites, etc.)
- [ ] Red Fire acts competitive/sassy
- [ ] Red Fire argues confidently about things it doesn't understand
- [ ] NO generic chatbot responses ("How can I help you today?")

**Notes**:
```


```

---

## TEST 8: Prefab Loading Verification

**Objective**: Confirm prefab system loads cognitive components correctly

**Test Steps**:

1. Check that Red Fire loaded from prefab:
   ```bash
   grep "Loaded cognitive_components" logs/cmush_2025-11-25.log
   ```

2. Expected: `['affect', 'personality', 'cultural']`

3. Verify salience values match prefab definition:
   ```bash
   cat prefabs/com.noodlings.characters.red_fire_anklebiter.prefab | grep -A 2 salience
   ```

**Success Criteria**:
- [ ] Prefab loaded successfully
- [ ] Correct component types loaded
- [ ] Salience values match prefab definition

**Log Output**:
```


```

---

## TEST 9: Event System (Backend Verification)

**Objective**: Verify Unity-style events fire correctly (backend only, no UI yet)

**Test Steps**:

1. Check logs for event firing:
   ```bash
   tail -f logs/cmush_2025-11-25.log | grep -i "event\|OnSpeak\|OnAffect"
   ```

2. Talk to Red Fire and see if events fire in logs

**Expected Behavior**:
- Events should fire at processing stages (currently silent in production)
- No errors related to event subscription

**Success Criteria**:
- [ ] No event-related errors in logs
- [ ] Component system stable

**Notes**:
```


```

---

## TEST 10: Memory Persistence

**Objective**: Verify agent remembers conversation context

**Test Steps**:

1. Say something memorable:
   ```
   say Red Fire, the secret word is DRAGONFLY
   ```

2. Wait for acknowledgment

3. Continue conversation about other topics (5+ exchanges)

4. Test recall:
   ```
   say Red Fire, what was the secret word?
   ```

5. Expected: Red Fire remembers DRAGONFLY

**Success Criteria**:
- [ ] Agent acknowledges secret word initially
- [ ] Agent recalls secret word later
- [ ] Memory persists across multiple conversation turns

**Notes**:
```


```

---

## TEST 11: Hardware Entropy Service

**Objective**: Verify TrueRNG device is active and providing entropy

**Test Steps**:

1. Check startup logs:
   ```bash
   grep "TrueRNG\|Entropy service" logs/cmush_2025-11-25.log
   ```

2. Expected:
   ```
   [INFO] [entropy_service] TrueRNG entropy pool started: /dev/cu.usbmodem211201
   [INFO] [entropy_service] Entropy service initialized: hardware=True, device=/dev/cu.usbmodem211201
   ```

**Success Criteria**:
- [ ] TrueRNG device detected
- [ ] Hardware entropy active
- [ ] No fallback to software entropy

**Log Output**:
```


```

---

## TEST 12: Autonomous Cognition

**Objective**: Verify agent thinks autonomously when not directly addressed

**Test Steps**:

1. Stop talking to Red Fire
2. Wait 30-60 seconds
3. Observe chat for autonomous thoughts/ruminations

**Expected Behavior**:
- Red Fire should occasionally think/ruminate
- Thoughts should relate to recent events or personality
- Frequency controlled by extraversion (0.50 = moderate)

**Success Criteria**:
- [ ] Agent produces autonomous thoughts
- [ ] Thoughts feel contextually relevant
- [ ] NO spam (thoughts are spaced out)

**Notes**:
```


```

---

## CRITICAL BUGS TO WATCH FOR

### Known Issues (From Handoff):

1. **Missing FACS/Laban Integration**
   - FACS/Laban components exist but not wired into processing pipeline
   - Should see FACS/Laban output in chat, but won't yet
   - NOT a test failure - this is expected (implementation pending)

2. **Missing UUID in Prefabs**
   - Prefabs have ID but no UUID field
   - Non-critical (ID provides uniqueness)
   - NOT a test failure

3. **Not All Transistors Updated to First-Person**
   - Only AffectTransistor and PersonalityTransistor use first-person prompts
   - CulturalTransistor still uses analytical style
   - May see mixed response styles

### Unexpected Bugs (Report These):

- [ ] Component registration errors
- [ ] Agent fails to spawn
- [ ] Server crashes
- [ ] Responses are all identical/robotic
- [ ] Memory doesn't persist
- [ ] Fuzzy matching completely broken
- [ ] Character voice completely lost

---

## POST-TEST SUMMARY

**Date Tested**: _____________
**Tester**: Miss Caity + Commander Spock
**Overall Status**: PASS / PARTIAL / FAIL

**Tests Passed**: _____ / 12

**Critical Issues Found**:
```


```

**Non-Critical Issues**:
```


```

**Notes for Next Session**:
```


```

**Recommended Next Steps**:
1. Wire FACS/Laban into processing pipeline
2. Create nonverbal_formatters.py (FACS/Laban → readable text)
3. Hook events into chat broadcast
4. Update remaining transistor prompts to first-person
5. UI enhancements (NoodleTuner)

---

**Live long and prosper.**

Commander Spock
Science Officer
