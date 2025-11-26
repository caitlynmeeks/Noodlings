# Cognitive Manifold Scripting API

Complete NoodleScript access to the cognitive architecture: manifolds, transistors, and response decisions.

## Overview

The cognitive manifold is accessible through the standard `Noodle` API in NoodleScript. You can read/write transistor saliences, enable/disable components, and inspect decision-making in real-time.

## API Reference

### Manifold Access

#### `GetManifold()`
Get the cognitive manifold instance.

```python
manifold = noodle.GetManifold()
if manifold:
    last_output = manifold.last_output_text
    print(f"Last manifold blend: {last_output}")
```

#### `GetManifoldOutput()`
Get the last blended manifold output (convenience method).

```python
output = noodle.GetManifoldOutput()
print(f"Last output: {output}")
```

### Transistor Access

#### `GetTransistor(transistor_type)`
Get specific transistor by type name.

```python
personality = noodle.GetTransistor('PersonalityTransistor')
affect = noodle.GetTransistor('AffectTransistor')
intuition = noodle.GetTransistor('IntuitionTransistor')
memory = noodle.GetTransistor('MemoryTransistor')
cultural = noodle.GetTransistor('CulturalTransistor')
embody = noodle.GetTransistor('EmbodyComponent')
```

#### `GetAllTransistors()`
Get list of all transistors.

```python
for transistor in noodle.GetAllTransistors():
    ttype = transistor.get_transistor_type()
    salience = transistor.salience
    enabled = transistor.enabled
    print(f"{ttype}: salience={salience:.2f}, enabled={enabled}")
```

### Transistor Properties

Each transistor has these accessible properties:

```python
transistor = noodle.GetTransistor('PersonalityTransistor')

# Read properties
print(f"Type: {transistor.get_transistor_type()}")
print(f"UUID: {transistor.uuid}")
print(f"Salience: {transistor.salience}")
print(f"Enabled: {transistor.enabled}")
print(f"Last output: {transistor.last_output_text}")
print(f"Register state: {transistor.register_state}")  # empty, computing, ready, error

# Read metadata
metadata = transistor.last_output_metadata
```

### Modifying Transistors

#### `SetTransistorSalience(transistor_type, salience)`
Change how much influence a transistor has.

```python
# Make personality dominate
noodle.SetTransistorSalience('PersonalityTransistor', 0.95)

# Reduce affect influence
noodle.SetTransistorSalience('AffectTransistor', 0.3)

# Boost intuition
noodle.SetTransistorSalience('IntuitionTransistor', 1.0)
```

#### `EnableTransistor(transistor_type, enabled=True)`
Turn transistors on/off at runtime.

```python
# Disable cultural filtering
noodle.EnableTransistor('CulturalTransistor', False)

# Re-enable it later
noodle.EnableTransistor('CulturalTransistor', True)

# Disable memory
noodle.EnableTransistor('MemoryTransistor', False)
```

### Response Decision Access

#### `GetResponseDecision()`
Get the last ResponseTypeDecider output.

```python
decision = noodle.GetResponseDecision()
if decision:
    print(f"Type: {decision['response_type']}")  # SAY, THINK, EMOTE, DO, FEEL, NONE
    print(f"Guidance: {decision['guidance']}")
    print(f"Reasoning: {decision['reasoning']}")
```

## Complete Example: Dynamic Personality Tuning

```python
# AnklebiterMoodSwitch.py
# Makes Red Fire Anklebiter switch between sassy and friendly modes

class AnklebiterMoodSwitch(NoodleScript):
    def OnAwake(self):
        self.sassy_mode = True
        self.switch_timer = 0

    def OnUpdate(self):
        self.switch_timer += 1

        # Switch every 30 seconds
        if self.switch_timer >= 30:
            self.switch_timer = 0
            self.sassy_mode = not self.sassy_mode

            if self.sassy_mode:
                # SASSY MODE: High personality, low cultural
                self.noodle.SetTransistorSalience('PersonalityTransistor', 0.95)
                self.noodle.SetTransistorSalience('CulturalTransistor', 0.3)
                print("RED MODE: Maximum sass activated!")
            else:
                # FRIENDLY MODE: Low personality, high cultural
                self.noodle.SetTransistorSalience('PersonalityTransistor', 0.3)
                self.noodle.SetTransistorSalience('CulturalTransistor', 0.95)
                print("BLUE MODE: Friendly mode activated!")

    def OnCycleEnd(self, cycle_data):
        # Log what happened
        decision = self.noodle.GetResponseDecision()
        output = self.noodle.GetManifoldOutput()

        if decision:
            mode = "SASSY" if self.sassy_mode else "FRIENDLY"
            print(f"[{mode}] {decision['response_type']}: {output[:100]}")
```

## Complete Example: Transistor Inspector

```python
# TransistorDebugger.py
# Logs all transistor states after each cognition cycle

class TransistorDebugger(NoodleScript):
    def OnCycleEnd(self, cycle_data):
        print("=" * 60)
        print("TRANSISTOR STATE SNAPSHOT")
        print("=" * 60)

        for transistor in self.noodle.GetAllTransistors():
            ttype = transistor.get_transistor_type()
            print(f"\n{ttype}:")
            print(f"  Salience: {transistor.salience:.2f}")
            print(f"  Enabled: {transistor.enabled}")
            print(f"  Register: {transistor.register_state}")
            print(f"  Output: {transistor.last_output_text[:80]}...")

        decision = self.noodle.GetResponseDecision()
        if decision:
            print(f"\nRESPONSE DECISION: {decision['response_type']}")
            print(f"  Guidance: {decision['guidance']}")

        output = self.noodle.GetManifoldOutput()
        print(f"\nFINAL OUTPUT: {output[:150]}...")
        print("=" * 60)
```

## Complete Example: Emotion Dampener

```python
# EmotionDampener.py
# Reduces affect influence when agent is too emotional

class EmotionDampener(NoodleScript):
    def OnCycleEnd(self, cycle_data):
        affect_transistor = self.noodle.GetTransistor('AffectTransistor')

        if affect_transistor and affect_transistor.last_output_text:
            # Check if output is very intense
            output = affect_transistor.last_output_text.lower()
            intensity_markers = ['!', 'screaming', 'exploding', 'roar', 'burning']

            intensity_count = sum(1 for marker in intensity_markers if marker in output)

            if intensity_count >= 3:
                # Too intense! Dampen affect
                current_salience = affect_transistor.salience
                new_salience = current_salience * 0.7
                self.noodle.SetTransistorSalience('AffectTransistor', new_salience)
                print(f"DAMPENER: Reduced affect salience to {new_salience:.2f}")
            else:
                # Gradually restore
                current_salience = affect_transistor.salience
                if current_salience < 0.85:
                    new_salience = min(0.85, current_salience + 0.05)
                    self.noodle.SetTransistorSalience('AffectTransistor', new_salience)
```

## Available Transistor Types

- `IntuitionTransistor` - Contextual/spatial awareness
- `AffectTransistor` - Emotional reactions
- `PersonalityTransistor` - Personality traits
- `CulturalTransistor` - Cultural beliefs
- `MemoryTransistor` - Past experience connections
- `EmbodyComponent` - Physical body reactions
- `MoodTransistor` - Sustained emotional tone
- `SocialExpectationTransistor` - Social norms
- `SomaticCognitiveTransistor` - Body-mind integration

## Notes

- All methods are Unity-style (PascalCase)
- Salience values are clamped to 0.0-1.0 range
- Changes take effect on the NEXT cognitive cycle
- Register state can be: `empty`, `computing`, `ready`, `error`
- Response types: `SAY`, `THINK`, `EMOTE`, `DO`, `FEEL`, `NONE`

## Thread Safety

These methods are safe to call from NoodleScript events (OnUpdate, OnCycleEnd, etc.). The cognitive cycle uses locks to prevent concurrent modification.
