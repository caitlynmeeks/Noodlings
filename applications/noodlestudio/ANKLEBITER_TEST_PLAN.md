# Anklebiter Vending Machine - Test Plan

**Status**: Ready for integration testing
**Files Created**: Blue/Red recipes + vending machine script

---

## What We Built

### Noodling Recipes

**1. blue_fire_anklebiter.yaml**
- Electric blue chaos gremlin
- Jacked affect (impulsivity 0.98, novelty 0.98)
- Crude jokes, ankle biting, jumping around
- Competes with Red Fire variety

**2. red_fire_anklebiter.yaml**
- Crimson competitive sass gremlin
- More argumentative than Blue
- Fights with Blue Fire constantly
- Thinks it's superior

### Script

**AnklebiterVendingMachine.py**
- Two buttons: Blue and Red
- Listens for "@press blue button" or "@press red button"
- Rezzes appropriate Anklebiter
- Max 5 of each type (prevents total chaos)
- Easter egg when both maxed out

---

## Integration Needed

### Current State: Skeleton Only

The script system exists but isn't wired to noodleMUSH yet. Need:

1. **Script attachment API endpoint**
   ```
   POST /api/prims/{id}/attach_script
   {
     "script_code": "...",
     "script_class": "AnklebiterVendingMachine"
   }
   ```

2. **Event routing in noodleMUSH**
   - When user types message → check if any prims have scripts
   - Call script.OnHear(speaker, message)
   - Script calls Noodlings.Rez() → triggers actual rez

3. **Noodlings.Rez() implementation**
   - Currently just prints
   - Needs to POST to noodleMUSH: `POST /api/agents/rez`
   - Pass recipe filename
   - Return rezzed agent ID

---

## Manual Test (Current Workaround)

Until scripting is fully wired:

### Step 1: Rez Anklebiters Manually

```
@spawn blue_fire_anklebiter
@spawn red_fire_anklebiter
```

### Step 2: Watch Chaos

The recipes are configured to:
- Respond frequently (low cooldown)
- Be highly impulsive
- Compete with each other
- Jump around and bite ankles
- Make crude jokes

### Step 3: Test Interactions

Try:
- Talking to them (they respond chaotically)
- Mentioning the other color (triggers competition)
- Watching them interact with each other

---

## Full Integration Test (Once Wired)

### Step 1: Create Vending Machine Prim

```python
# In noodleMUSH commands.py, add:
@create_prim("anklebiter_vending_machine", prim_type="interactive")
```

### Step 2: Attach Script

```
Component > Add Script...
Select: AnklebiterVendingMachine.py
Click: Compile & Attach
```

### Step 3: Use Machine

```
look at vending machine
@press blue button
@press red button
```

### Step 4: Observe Chaos

- Blue and Red Anklebiters rez
- They immediately start competing
- They jump on each other
- They bite ankles
- They argue about who's better
- Pure chaotic energy

---

## Expected Behavior

### Blue Fire Anklebiter:
- "HEHEHEHE WATCH THIS!!!" *bites ankle*
- Jumps on furniture
- Makes terrible puns
- Bounces off walls
- Electric, zippy energy

### Red Fire Anklebiter:
- "ACTUALLY I'M BETTER THAN BLUE!" *tackles Blue*
- More sarcastic
- Argues constantly
- Tries to establish dominance
- Competitive, sassy energy

### Together:
- Constant competition
- Jump on each other
- Argue about everything
- Actually secretly friends
- Create absolute mayhem

---

## Next Steps

### Priority 1: Script Backend Integration
- Add script storage to noodleMUSH
- Event routing (OnHear, OnClick, OnUse)
- Noodlings.Rez() → actual rezzing

### Priority 2: Test Anklebiters
- Rez manually, verify personalities
- Check if they interact as designed
- Tune affect/personality if needed

### Priority 3: Wire Vending Machine
- Create prim in-world
- Attach script
- Test button pressing
- Verify rezzing works

---

## Success Criteria

Test passes when:
- ✅ User can @press blue button
- ✅ Blue Fire Anklebiter rezzes
- ✅ Anklebiter starts causing chaos
- ✅ User can @press red button
- ✅ Red Fire Anklebiter rezzes
- ✅ Blue and Red compete with each other
- ✅ Maximum chaos achieved

---

**This is the scripting system proof of concept.**

Event-driven logic + kindled beings = emergent chaos!
