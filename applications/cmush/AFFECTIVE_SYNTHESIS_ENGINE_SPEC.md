# Affective Synthesis Engine
## Inverse Affective Design: Generating Stimuli from Target Emotions

**Authors:** Lieutenant Caitlyn Meeks + Commander Spock
**Date:** November 22, 2025
**Status:** Advanced specification
**Concept:** Given target affect vector, synthesize world elements to evoke that emotion

---

## Core Concept

**The Inverse Problem:**

**Forward (Affect Extraction):** Stimulus → Vector
**Inverse (Affective Synthesis):** Vector → Stimuli

**Use case:**
```
Director: "I want Toad to feel longing for his old automobile"

System:
1. Defines target affect: [+0.2, 0.3, 0.0, 0.5, 0.0] (nostalgia/longing)
2. Searches affect library for matching stimuli
3. Synthesizes scene: old song on radio + photograph + engine sound
4. Toad experiences target emotion
```

**This is emotional cinematography through generative affective design.**

---

## 1. Target Affect Specification

### Longing/Nostalgia for Old Automobile

**Phenomenology:**
- Bittersweet memories (pleasant past, current absence)
- Low energy (wistful, not excited)
- Sadness from loss
- Engagement with memories

**Affect vector:**
```python
longing_for_car = [
    +0.2,   # Valence: Slightly positive (fond memories)
    0.3,    # Arousal: Low (wistful, not energetic)
    0.0,    # Fear: None (safe memories)
    0.5,    # Sorrow: Moderate (missing something)
    0.0     # Boredom: Engaged (lost in memories)
]
```

---

## 2. Affective Stimulus Library

### Database of Stimuli with Known Affect Vectors

```python
AFFECTIVE_STIMULUS_LIBRARY = {
    # Songs
    'old_motor_song_1920s': {
        'type': 'sound',
        'affect': [+0.2, 0.3, 0.0, 0.5, 0.0],  # Nostalgia
        'tags': ['automobile', 'vintage', '1920s', 'nostalgia'],
        'description': '1920s motor song',
        'properties': {
            'sound_type': 'music',
            'decibels': 50,
            'era': '1920s'
        }
    },

    # Visual stimuli
    'photograph_old_car': {
        'type': 'prim',
        'affect': [+0.3, 0.2, 0.0, 0.6, 0.0],  # Strong nostalgia
        'tags': ['automobile', 'vintage', 'photograph', 'memory'],
        'description': 'Faded photograph of vintage automobile',
        'properties': {
            'prim_type': 'photograph',
            'image': 'vintage_car_sepia.jpg',
            'condition': 'worn, faded edges'
        }
    },

    # Scent stimuli
    'scent_motor_oil_leather': {
        'type': 'scent',
        'affect': [+0.1, 0.2, 0.0, 0.4, 0.0],  # Mild nostalgia
        'tags': ['automobile', 'garage', 'mechanical', 'leather'],
        'description': 'Scent of motor oil and old leather',
        'properties': {
            'scent_type': 'motor_oil_leather',
            'intensity': 0.4,
            'pleasantness': 0.6
        }
    },

    # Ambient sounds
    'distant_engine_rumble': {
        'type': 'sound',
        'affect': [+0.1, 0.3, 0.0, 0.5, 0.0],  # Longing
        'tags': ['automobile', 'engine', 'distant', 'nostalgia'],
        'description': 'Distant rumble of vintage automobile engine',
        'properties': {
            'sound_type': 'engine',
            'decibels': 40,
            'frequency': 'low',
            'distance': 'far away'
        }
    },

    # Environmental
    'sunset_golden_hour': {
        'type': 'lighting',
        'affect': [+0.3, 0.2, 0.0, 0.3, 0.0],  # Wistful
        'tags': ['nostalgic', 'golden', 'sunset', 'melancholic'],
        'description': 'Golden hour sunset lighting',
        'properties': {
            'light_color': '#FFB347',
            'brightness': 500,
            'angle': 'low on horizon'
        }
    }
}
```

---

## 3. Synthesis Algorithm

### Multi-Modal Affective Synthesis

```python
def synthesize_affect_scene(
    target_affect: List[float],
    agent_id: str,
    room_id: str,
    context: Dict = None
) -> List[str]:
    """
    Generate scene elements to evoke target affect.

    Args:
        target_affect: Target 5-D affect vector
        agent_id: Agent to affect (for personalization)
        room_id: Room to modify
        context: Additional constraints (available prims, themes, etc.)

    Returns:
        List of generated prim IDs
    """
    # 1. Search library for matching stimuli
    candidates = search_affect_library(
        target_affect,
        tolerance=0.3,  # Vector distance threshold
        tags=context.get('tags', [])
    )

    # 2. Rank by affective distance
    ranked = sorted(candidates, key=lambda c: affect_distance(c['affect'], target_affect))

    # 3. Select diverse modalities (audio + visual + scent)
    selected = select_diverse_stimuli(ranked, max_count=5)

    # 4. Instantiate prims in room
    generated_prims = []
    for stimulus in selected:
        prim_id = instantiate_stimulus(stimulus, room_id, agent_id)
        generated_prims.append(prim_id)

    # 5. Verify combined affect
    combined_affect = calculate_combined_affect(selected)
    distance = affect_distance(combined_affect, target_affect)

    print(f"Target affect: {target_affect}")
    print(f"Synthesized affect: {combined_affect}")
    print(f"Distance: {distance:.3f}")

    return generated_prims


def search_affect_library(
    target: List[float],
    tolerance: float,
    tags: List[str]
) -> List[Dict]:
    """
    Search library for stimuli matching target affect.

    Args:
        target: Target affect vector
        tolerance: Maximum affective distance
        tags: Required tags (e.g., ['automobile', 'vintage'])

    Returns:
        List of matching stimuli
    """
    matches = []

    for stimulus_id, stimulus in AFFECTIVE_STIMULUS_LIBRARY.items():
        # Check tags
        if tags and not any(tag in stimulus['tags'] for tag in tags):
            continue

        # Check affective distance
        distance = affect_distance(stimulus['affect'], target)
        if distance <= tolerance:
            matches.append({
                **stimulus,
                'id': stimulus_id,
                'distance': distance
            })

    return matches


def affect_distance(a1: List[float], a2: List[float]) -> float:
    """Calculate Euclidean distance between affect vectors."""
    import math
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a1, a2)))


def select_diverse_stimuli(
    candidates: List[Dict],
    max_count: int = 5
) -> List[Dict]:
    """
    Select diverse stimuli across modalities.

    Prefer one of each: sound, visual, scent, lighting, tactile.

    Args:
        candidates: Ranked candidates
        max_count: Maximum stimuli to select

    Returns:
        Diverse stimulus list
    """
    selected = []
    modalities_used = set()

    for candidate in candidates:
        modality = candidate['type']

        # Prefer diversity
        if modality not in modalities_used or len(selected) < max_count:
            selected.append(candidate)
            modalities_used.add(modality)

        if len(selected) >= max_count:
            break

    return selected


def calculate_combined_affect(stimuli: List[Dict]) -> List[float]:
    """
    Calculate combined affect from multiple stimuli.

    Weighted average by stimulus intensity/salience.

    Args:
        stimuli: List of stimuli

    Returns:
        Combined affect vector
    """
    if not stimuli:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # Simple average (could be weighted)
    combined = [0.0] * 5

    for stimulus in stimuli:
        for i in range(5):
            combined[i] += stimulus['affect'][i]

    # Average
    for i in range(5):
        combined[i] /= len(stimuli)

    return combined
```

---

## 4. Example: Making Toad Feel Longing

### Command Interface

```
Director: @evoke longing automobile for agent_toad

System executes:
1. Define target: [+0.2, 0.3, 0.0, 0.5, 0.0]
2. Search library: tags=['automobile', 'nostalgia']
3. Select: song + photograph + scent + sound + lighting
4. Instantiate in Toad's room
5. Toad perceives → feels longing
```

### Generated Scene

```python
# System synthesizes:

# 1. Radio playing old motor song
radio = world.create_object(
    name="Vintage Radio",
    location=toad_room,
    sound_emitter=SoundEmitter(
        sound_type="music",
        decibels=50,
        audio_description="1920s motor song about open roads"
    )
)

# 2. Photograph on mantle
photo = world.create_object(
    name="Faded Photograph",
    description="Sepia-toned photo of Toad's first motor car",
    location=toad_room,
    pod=PhysicsObjectDescriptor(
        material="paper",
        semantic_properties=["nostalgic", "faded", "precious"]
    )
)

# 3. Scent of motor oil
scent_source = world.create_object(
    name="Old Garage Smell",
    location=toad_room,
    scent_emitter=ScentEmitter(
        scent_type="motor_oil_leather",
        intensity=0.4,
        pleasantness=0.6  # Pleasant to Toad
    )
)

# 4. Distant engine sound
ambient_sound = world.create_object(
    name="Distant Engine",
    location=toad_room,
    sound_emitter=SoundEmitter(
        sound_type="engine",
        decibels=35,  # Quiet, distant
        frequency="low"
    )
)

# 5. Golden hour lighting (sunset)
lighting = world.get_room(toad_room)
lighting['environment']['lighting'] = 'golden_hour'
lighting['environment']['light_color'] = '#FFB347'
```

### Toad's Response

**Somatic Cognitive Transistor receives:**
- Sound: Old motor song (0.3 salience)
- Visual: Photograph (0.4 salience)
- Scent: Motor oil + leather (0.3 salience)
- Sound: Distant engine (0.2 salience)
- Light: Golden sunset (0.2 salience)

**Manifold integrates:**
```
Cultural (0.5): "Automobiles represent freedom"
Personality (0.6): "I love motor cars!"
Memory (0.8): "Remembers first car vividly" ← HIGH
Somatic (0.4): "Song + photo + scent trigger memories"

Combined affect achieved: [+0.2, 0.3, 0.0, 0.5, 0.0] ✓ (matches target!)

Toad's response:
"*looks at photograph wistfully* Ah, my old motor car... *listens to song*
 Those were the days. *breathes in scent* I can almost smell the garage...
 *stares at sunset* I do miss her terribly. *single tear*"
```

**Result:** Target emotion successfully evoked!

---

## 5. Personalized Affective Synthesis

### Agent Memory Integration

**Problem:** Same stimulus affects different agents differently.

**Solution:** Personalize synthesis using agent's memory and personality.

```python
def synthesize_personalized_affect(
    target_affect: List[float],
    agent_id: str,
    room_id: str
) -> List[str]:
    """
    Generate personalized affective scene.

    Queries agent's memories to find what historically
    produced target affect for THIS specific agent.
    """
    agent = agent_manager.get(agent_id)

    # Search agent's episodic memory for similar affect
    matching_memories = agent.search_memories_by_affect(
        target_affect,
        tolerance=0.3
    )

    # Extract common themes/elements
    themes = extract_themes_from_memories(matching_memories)
    # For Toad: ['automobile', 'vintage', 'motor', 'speed', 'freedom']

    # Search library with agent-specific tags
    candidates = search_affect_library(
        target=target_affect,
        tolerance=0.3,
        tags=themes  # Personalized!
    )

    # Generate scene
    return instantiate_stimuli(candidates, room_id)
```

**Example:**

**Toad's memories with affect [+0.2, 0.3, 0.0, 0.5, 0.0]:**
- "First drive in motor car - wind in face"
- "Racing past Mole's house - shouted hello"
- "Crash that ended it all - sent to prison"

**Extracted themes:** automobile, speed, wind, freedom, loss

**Synthesized scene:**
- Song about open roads
- Photograph of his car
- Wind sound (breeze through window)
- Scent of gasoline
- Sound of engine

**Result:** Highly personalized nostalgia trigger.

---

## 6. Multi-Modal Synthesis

### Combining Stimuli for Target Affect

**Challenge:** Single stimulus rarely hits exact target. Combine multiple modalities.

**Strategy:**
```python
def find_optimal_stimulus_combination(
    target_affect: List[float],
    available_stimuli: List[Dict],
    max_stimuli: int = 5
) -> List[Dict]:
    """
    Find combination of stimuli that produces target affect.

    This is a search problem:
    - State space: All possible stimulus combinations
    - Objective: Minimize ||combined_affect - target||
    - Constraint: ≤ max_stimuli

    Uses greedy search with backtracking.
    """
    best_combo = []
    best_distance = float('inf')

    # Greedy search
    current_combo = []
    current_affect = [0.0] * 5

    while len(current_combo) < max_stimuli:
        # Find stimulus that moves us closer to target
        best_next = None
        best_next_distance = float('inf')

        for stimulus in available_stimuli:
            if stimulus in current_combo:
                continue

            # Calculate combined affect if we add this
            test_combo = current_combo + [stimulus]
            test_affect = calculate_combined_affect(test_combo)
            test_distance = affect_distance(test_affect, target_affect)

            if test_distance < best_next_distance:
                best_next = stimulus
                best_next_distance = test_distance

        if not best_next or best_next_distance >= best_distance:
            break  # No improvement

        # Add best next stimulus
        current_combo.append(best_next)
        current_affect = calculate_combined_affect(current_combo)
        best_distance = best_next_distance

    return current_combo
```

---

## 7. Generative Affective Design

### LLM-Powered Stimulus Creation

**If library doesn't have matching stimulus, GENERATE it.**

```python
def generate_stimulus_for_affect(
    target_affect: List[float],
    modality: str,  # "sound", "visual", "scent", "prim"
    context: Dict
) -> Dict:
    """
    Use LLM to generate new stimulus matching target affect.

    Args:
        target_affect: Target emotion
        modality: What kind of stimulus to generate
        context: Contextual constraints (agent memories, themes, etc.)

    Returns:
        Generated stimulus specification
    """
    # Describe target affect to LLM
    affect_desc = describe_affect_vector(target_affect)
    # → "Bittersweet longing - slightly positive but sad, low energy, wistful"

    # Build prompt
    prompt = f"""
    Design a {modality} stimulus that evokes this emotion:
    {affect_desc}

    Context:
    - Agent: {context.get('agent_name')} ({context.get('species')})
    - Themes: {', '.join(context.get('themes', []))}
    - Setting: {context.get('setting')}

    What {modality} would evoke this specific emotional state?
    Describe in detail.
    """

    # LLM generates
    response = call_llm(prompt, model="deepseek/deepseek-chat", max_tokens=200)

    # Parse and create stimulus
    return parse_stimulus_description(response, modality)
```

**Example:**

```
Target: [+0.2, 0.3, 0.0, 0.5, 0.0] (longing for automobile)
Modality: "sound"
Context: {agent: "Toad", themes: ["automobile", "vintage"], setting: "parlor"}

LLM generates:
"A scratchy gramophone recording of 'The Open Road Waltz' from 1922,
 with the sound of distant motor engines in the background. The music
 has that tinny, nostalgic quality of early recordings, punctuated by
 the occasional backfire and engine rumble."

System creates:
- Gramophone prim
- SoundEmitter with generated description
- Affect validated: [+0.2, 0.3, 0.0, 0.5, 0.0] ✓
```

---

## 8. Command Interface

### @evoke Command

```bash
# Evoke specific emotion in agent
@evoke <emotion> for <agent> [using <themes>]

# Examples:
@evoke longing automobile for agent_toad
@evoke joy for agent_mole using picnic, sunshine
@evoke fear for agent_rat using darkness, sounds
@evoke nostalgia childhood for agent_badger

# System:
1. Maps emotion word → affect vector
2. Searches/generates stimuli
3. Instantiates in agent's room
4. Agent experiences target emotion
```

### Emotion → Affect Vector Mapping

```python
EMOTION_AFFECT_MAPPINGS = {
    'longing': [+0.2, 0.3, 0.0, 0.5, 0.0],
    'nostalgia': [+0.3, 0.2, 0.0, 0.4, 0.0],
    'joy': [+0.8, 0.7, 0.0, 0.0, 0.0],
    'sadness': [-0.4, 0.2, 0.0, 0.8, 0.0],
    'fear': [-0.5, 0.8, 0.9, 0.2, 0.0],
    'anger': [-0.6, 0.9, 0.3, 0.1, 0.0],
    'contentment': [+0.4, 0.2, 0.0, 0.0, 0.0],
    'boredom': [-0.2, 0.1, 0.0, 0.0, 0.9],
    'excitement': [+0.6, 0.9, 0.1, 0.0, 0.0],
    'anxiety': [-0.3, 0.7, 0.7, 0.2, 0.0],
    'bittersweet': [+0.3, 0.5, 0.0, 0.6, 0.0],  # "Hey Ya"
    'melancholy': [+0.1, 0.2, 0.0, 0.7, 0.0],
    'awe': [+0.4, 0.6, 0.2, 0.0, 0.0],
    'disgust': [-0.7, 0.5, 0.1, 0.0, 0.0]
}
```

---

## 9. Cross-Modal Invariance Application

**Your insight: Same affect, different modalities**

```python
# Target: Make Toad feel longing
target = [+0.2, 0.3, 0.0, 0.5, 0.0]

# Option A: Song
synthesize_affect_scene(target, modality='sound')
→ "1920s motor song"
→ Toad feels longing ✓

# Option B: Visual
synthesize_affect_scene(target, modality='visual')
→ "Faded photograph of old car"
→ Toad feels longing ✓

# Option C: Scent
synthesize_affect_scene(target, modality='scent')
→ "Motor oil and leather smell"
→ Toad feels longing ✓

# All three produce same emotional state!
# Cross-modal invariance validated.
```

---

## 10. Advanced: Emotional Narrative Arcs

### Scripted Emotional Journeys

```python
# Create emotional arc for scene
emotional_arc = [
    (0, [+0.5, 0.6, 0.0, 0.0, 0.0]),    # t=0s: Happy (party starting)
    (30, [+0.6, 0.8, 0.0, 0.0, 0.0]),   # t=30s: Excited (dancing)
    (60, [+0.3, 0.7, 0.1, 0.4, 0.0]),   # t=60s: Bittersweet ("Hey Ya" plays)
    (90, [+0.1, 0.4, 0.0, 0.6, 0.0]),   # t=90s: Wistful (song ends)
    (120, [+0.3, 0.3, 0.0, 0.2, 0.0])   # t=120s: Gentle contentment
]

# System generates stimuli for each timestamp
for timestamp, target_affect in emotional_arc:
    schedule_affect_change(
        time=timestamp,
        target=target_affect,
        agent_id="agent_toad",
        room_id="room_parlor"
    )

# Result: Toad experiences emotional journey
# Happy → Excited → Bittersweet → Wistful → Content
# (Orchestrated by generated stimuli)
```

---

## 11. Validation: Did It Work?

### Measuring Success

```python
# After synthesizing scene, check if agent reached target affect

# 1. Generate scene
synthesize_affect_scene(
    target_affect=[+0.2, 0.3, 0.0, 0.5, 0.0],
    agent_id="agent_toad",
    room_id="room_parlor"
)

# 2. Wait for agent to perceive and process (5-10 seconds)
await asyncio.sleep(10)

# 3. Read agent's actual affect
actual_affect = agent_toad.get_current_affect()

# 4. Calculate distance
distance = affect_distance(target_affect, actual_affect)

# 5. Validate
if distance < 0.3:
    print("✓ Success - Target affect achieved")
    print(f"  Target: {target_affect}")
    print(f"  Actual: {actual_affect}")
    print(f"  Distance: {distance:.3f}")
else:
    print("✗ Failed - Affect not achieved")
    # Refine and try again
```

---

## 12. Applications

### Interactive Fiction / Narrative Design

```python
# Emotional beats for story
story_beats = {
    'act1_intro': [+0.4, 0.5, 0.0, 0.0, 0.0],      # Pleasant introduction
    'act1_conflict': [-0.3, 0.7, 0.5, 0.2, 0.0],   # Rising tension
    'act2_climax': [-0.6, 0.9, 0.8, 0.4, 0.0],     # Peak danger
    'act3_resolution': [+0.7, 0.4, 0.0, 0.0, 0.0], # Happy ending
    'denouement': [+0.3, 0.2, 0.0, 0.1, 0.0]       # Gentle closure
}

# System generates appropriate stimuli for each beat
# Agents experience emotional journey matching narrative arc
```

### Therapeutic Applications

```python
# Move agent from anxious to calm
initial_affect = [-0.4, 0.8, 0.7, 0.3, 0.0]  # Anxious
target_affect = [+0.3, 0.2, 0.0, 0.0, 0.0]   # Calm, content

# Generate calming stimuli
synthesize_affect_scene(
    target_affect,
    agent_id="agent_anxious",
    tags=['calming', 'peaceful', 'safe']
)

# Generates:
# - Soft music (low arousal)
# - Warm lighting
# - Comfortable furniture
# - Pleasant scent (lavender)
# - Gentle sounds (stream, wind chimes)

# Agent gradually calms over 2-3 minutes
```

### Game Design

```python
# Boss battle should feel terrifying
boss_room_affect = [-0.6, 0.9, 0.9, 0.1, 0.0]

synthesize_affect_scene(boss_room_affect, tags=['danger', 'threatening'])

# Generates:
# - Ominous music (low frequency, loud)
# - Dark lighting with red accents
# - Rumbling vibration
# - Distant roars
# - Hot, oppressive temperature

# Player Noodling: "This place is terrifying... *sweating* *heart pounding*"
```

---

## 13. Theoretical Implications

### Affective Control Theory

**You've discovered affective servomechanism:**

```
Target Affect (setpoint)
    ↓
Synthesis Engine (controller)
    ↓
Generated Stimuli (actuator)
    ↓
Agent Perception (sensor)
    ↓
Actual Affect (measured)
    ↓
Error = ||target - actual||
    ↓
(feedback to synthesis)
```

**This is closed-loop affective control!**

Like a thermostat, but for emotions.

---

## 14. Summary

**Affective Synthesis Engine:**

✅ **Inverse affect problem** (vector → stimuli)
✅ **Affect library** (database of known stimuli)
✅ **Multi-modal synthesis** (sound + visual + scent)
✅ **Personalization** (agent memory integration)
✅ **Generative design** (LLM creates new stimuli)
✅ **Cross-modal invariance** (song = photo = scent if affect matches)
✅ **Validation** (measure actual vs target affect)

**Command interface:**
```bash
@evoke longing automobile for agent_toad
```

**Result:**
- Radio plays vintage motor song
- Photograph appears on mantle
- Scent of motor oil fills room
- Distant engine sound
- Golden sunset lighting

**Toad experiences:** [+0.2, 0.3, 0.0, 0.5, 0.0] - longing for his old car

**Cross-modal invariance in action.**

---

**This is emotional cinematography as inverse affective design.**

**Highly logical, Lieutenant—er, Principal Researcher Meeks.**

*— Commander Spock*

*The gumball center represents optimal confection compression. Logical.*
