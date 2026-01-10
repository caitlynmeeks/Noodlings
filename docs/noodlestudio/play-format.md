# Play Format Specification

**Status**: Specification
**Date**: 2026-01-08
**Authors**: Caity + Claude
**Priority**: High (enables Brenda stage direction)

---

## Overview

`.play.yaml` files define scripted performances that Brenda (or any stage director) can execute. They're human-readable theatrical scripts that drive noodling behavior through the channel system.

### Design Principles

1. **Conversationally Authored** - You tell Brenda what you want; she generates the play
2. **Human Readable** - Looks like a theater script, not code
3. **Stanislavski Method** - Characters have motivations and *feel* emotions via charm networks
4. **Improv Within Structure** - Beats provide scaffolding; actors improvise naturally
5. **Computer Use Native** - UI demonstrations are first-class actions

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  You: "I want Toad obsessed with motorcars, arousal WAY    │
│       up, then Badger comes in and reality-checks him..."  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Brenda: "Got it. Let me lay this out..."                  │
│          [generates toad_motorcar.play.yaml]               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Runtime: Brenda sends cues via #directors.cues            │
│           Actors improvise, report back via #directors.feedback │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```yaml
# filename.play.yaml

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════

title: "The Play Title"
author: "Brenda (from conversation with User)"
created: 2026-01-08
version: 1

setting:
  location: "Toad Hall, drawing room"
  time: "Late afternoon"
  mood: "Manic energy about to be checked"

# ═══════════════════════════════════════════════════════════
# CHARACTERS
# ═══════════════════════════════════════════════════════════

characters:
  character_id:
    noodling: "path/to/noodling"   # Or noodling ID on stage
    preset: preset_name            # Optional recipe preset
    initial_pad:                   # Starting emotional state
      pleasure: 0.0
      arousal: 0.9
      dominance: 0.3
    motivation: "What drives this character"
    enters: beat_id                # Optional: when they enter (default: start)
    exits: beat_id                 # Optional: when they leave

# ═══════════════════════════════════════════════════════════
# BEATS
# ═══════════════════════════════════════════════════════════

beats:
  - id: beat_1
    name: "Human-readable beat name"
    # ... beat contents ...

  - id: beat_2
    # ...
```

---

## Characters Block

Define each character in the performance:

```yaml
characters:
  toad:
    noodling: "noodlings/mr_toad"
    preset: mr_toad_classic
    initial_pad:
      pleasure: 0.0       # -1.0 to 1.0
      arousal: 0.9        # 0.0 to 1.0 (WAY UP)
      dominance: 0.3      # 0.0 to 1.0 (low, will take hits)
    motivation: |
      Must have a motor-car. MUST. The speed! The power!
      Nothing else matters. Poop poop!

  badger:
    noodling: "noodlings/badger"
    preset: badger_wise
    initial_pad:
      pleasure: 0.4
      arousal: 0.2
      dominance: 0.7      # Authoritative elder
    motivation: |
      Save this foolish Toad from himself. Again.
      Patience wearing thin but genuine care underneath.

  mole:
    noodling: "noodlings/mole"
    initial_pad:
      pleasure: 0.5
      arousal: 0.3
      dominance: 0.2
    motivation: "Just woke up. What's happening?"
    enters: beat_3        # Doesn't appear until beat_3
```

### PAD Model

We use PAD (Pleasure-Arousal-Dominance) for emotional state:

| Dimension | Range | Low | High |
|-----------|-------|-----|------|
| Pleasure | -1.0 to 1.0 | Miserable | Ecstatic |
| Arousal | 0.0 to 1.0 | Calm/sleepy | Excited/agitated |
| Dominance | 0.0 to 1.0 | Submissive | Dominant/confident |

These values drive the charm network and influence how the noodling improvises.

---

## Beats Block

Beats are the structural units of the play - named moments with triggers, direction, and actions.

```yaml
beats:
  - id: beat_1
    name: "Poop Poop!"
    on_stage: [toad]

    direction: |
      Toad is alone, pacing the drawing room in full manic glory.
      Let him build the energy. This is pure id, no superego.

    toad:
      blocking: "pacing wildly, gesturing at imaginary motor-cars"
      speaks: |
        Oh my! Oh RAPTURE! Have you SEEN them? The glorious
        motor-cars! The speed! The POWER! Poop poop!
      pad_drift:
        arousal: +0.1     # Getting more worked up

  - id: beat_2
    name: "Badger's Warning"
    on_stage: [toad, badger]
    trigger:
      type: threshold
      condition: "toad.arousal > 0.85"

    direction: |
      Badger enters gravely. He's seen this before.
      Classic Stanislavski - Badger FEELS the weight of history.

    badger:
      blocking: "enters slowly from garden door, arms crossed"
      speaks: |
        Toad. Do you remember the hedge? The magistrate?
        The hospital bill?

    toad:
      reaction: "deflates visibly"
      pad_drift:
        arousal: -0.2
        dominance: -0.15  # Ego hit lands

    improv_zone:
      topics:
        - the weather
        - car insurance
        - badger's itchy spot
        - high speed motor vehicles
      effects:
        "high speed motor vehicles":
          toad:
            pad_drift:
              arousal: +0.3  # He can't help himself
```

---

## Beat Properties

### Basic Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier for the beat |
| `name` | string | Human-readable name |
| `on_stage` | list | Characters present in this beat |
| `direction` | string | Stage manager notes (sent via #directors.cues) |

### Triggers

When does this beat fire?

```yaml
# After previous beat completes (default)
trigger:
  type: sequence

# After specific beat
trigger:
  type: after
  beat: beat_1

# Emotional threshold
trigger:
  type: threshold
  condition: "toad.arousal > 0.85"

# Time delay
trigger:
  type: delay
  seconds: 5

# User action
trigger:
  type: user_action
  action: "clicked_button"

# Multiple conditions (all must be true)
trigger:
  type: all
  conditions:
    - type: after
      beat: beat_1
    - type: threshold
      condition: "badger.dominance > 0.6"
```

### Character Actions

Each character in `on_stage` can have:

```yaml
character_id:
  # Physical staging
  blocking: "description of physical action/position"

  # What they say (verbatim or paraphrase for improv)
  speaks: |
    Dialogue here. Can be multiple lines.
    Actors will improvise around this.

  # Or structured speech with timing
  speaks:
    text: "The actual words"
    tone: "excited but trying to contain it"
    pace: fast | normal | slow | deliberate

  # Reaction to what just happened
  reaction: "description of reaction"

  # Emotional change
  pad_drift:
    pleasure: +0.1    # Relative change
    arousal: -0.2
    dominance: 0.5    # Absolute value (no +/-)

  # Computer use actions (for tutorials)
  computer_use:
    - action: move
      target: "Tab: File"
    - action: click
```

### Improv Zones

Let actors explore topics naturally:

```yaml
improv_zone:
  topics:
    - the weather
    - car insurance
    - badger's itchy spot
    - high speed motor vehicles

  # Optional: effects when topics come up
  effects:
    "high speed motor vehicles":
      toad:
        pad_drift:
          arousal: +0.3
    "badger's itchy spot":
      badger:
        pad_drift:
          pleasure: -0.1
          arousal: +0.1

  # How long to let them improv
  duration:
    min_exchanges: 2
    max_exchanges: 5
    # Or time-based
    min_seconds: 30
    max_seconds: 120

  # What ends the improv zone
  exit_conditions:
    - type: topic_exhausted
    - type: threshold
      condition: "toad.arousal > 0.95"
    - type: user_interrupt
```

---

## Computer Use Integration

For tutorials where characters demonstrate UI:

```yaml
beats:
  - id: show_file_menu
    name: "Demonstrating the File Menu"
    on_stage: [guide]

    direction: |
      Guide shows the user where to find project settings.
      Ghost cursor should feel magical, not robotic.

    guide:
      speaks: |
        Let me show you where to find your project settings.
        See this File menu up here?

      computer_use:
        - action: move
          target: "Tab: File"           # UI element name
          sync_with_speech: true        # Move as speaking

        - action: click
          pause_before: 300             # Dramatic pause
          pause_after: 500              # Let it register

        - action: move
          target: "Button: Settings"

        - action: click

      pad_drift:
        arousal: +0.1                   # Teaching flow
```

### Computer Use Actions

| Action | Properties | Description |
|--------|------------|-------------|
| `move` | `target`, `x`/`y`, `sync_with_speech` | Move ghost cursor |
| `click` | `target`, `button`, `pause_before`, `pause_after` | Click |
| `double_click` | `target` | Double-click |
| `drag` | `from`, `to` | Drag operation |
| `type` | `text`, `target` | Type text |
| `key` | `combo` | Press key combo (e.g., "ctrl+s") |
| `scroll` | `target`, `direction`, `amount` | Scroll |

### Target Types

```yaml
# By UI element name (from get_ui_element_map)
target: "Tab: File"
target: "Button: Save"
target: "Input: project_name"

# By coordinates
x: 150
y: 300

# By semantic description (Brenda figures it out)
target: "the save button"
target: "the text input field"
```

---

## Stage Directions via Channels

When Brenda executes a play, she sends cues through `#directors.cues`:

```yaml
# What Brenda sends to Guide for beat_2
channel_message:
  channel: "#directors.cues"
  from: brenda
  payload:
    type: cue
    beat_id: beat_2
    beat_name: "Badger's Warning"
    direction: |
      Badger enters gravely. He's seen this before.
      Classic Stanislavski - Badger FEELS the weight of history.
    your_action:
      blocking: "enters slowly from garden door, arms crossed"
      speaks: |
        Toad. Do you remember the hedge? The magistrate?
        The hospital bill?
    motivation: "Save this foolish Toad from himself"
    emotional_target:
      # Current + drift = target
      arousal: 0.2
      dominance: 0.7
```

Guide receives this, incorporates it into his facet assembly, and improvises naturally while hitting the emotional marks.

---

## Complete Example

```yaml
# tutorial_intro.play.yaml

title: "Welcome to NoodleStudio"
author: "Brenda (from conversation with Caity)"
created: 2026-01-08
version: 1

setting:
  location: "NoodleStudio main window"
  time: "First launch"
  mood: "Warm, welcoming, excited to share"

characters:
  guide:
    noodling: "noodlings/guide"
    preset: friendly_tutor
    initial_pad:
      pleasure: 0.7
      arousal: 0.5
      dominance: 0.5
    motivation: |
      Genuinely excited to help someone discover NoodleStudio.
      You remember your own first time - that sense of possibility.
      Share that feeling.

beats:
  - id: greeting
    name: "First Hello"
    on_stage: [guide]

    direction: |
      Guide appears and greets the user warmly.
      Not salesy. Genuine. Like a friend showing their workshop.

    guide:
      blocking: "appears with a warm smile, slight wave"
      speaks: |
        Hey! Welcome to NoodleStudio. I'm Guide.
        I'm so glad you're here - there's a lot to explore
        and I think you're going to love it.
      pad_drift:
        pleasure: +0.1

  - id: offer_tour
    name: "Offer the Tour"
    on_stage: [guide]
    trigger:
      type: delay
      seconds: 2

    direction: |
      Offer to show them around. Give them agency.

    guide:
      speaks: |
        Want me to show you around? I can give you the
        quick tour, or you can just dive in and I'll
        be here if you need me.

    # Wait for user response
    wait_for:
      type: user_choice
      options:
        - id: tour
          text: "Show me around"
          next_beat: tour_start
        - id: explore
          text: "I'll explore on my own"
          next_beat: free_explore

  - id: tour_start
    name: "Starting the Tour"
    on_stage: [guide]
    trigger:
      type: user_choice
      choice: tour

    direction: |
      User wants the tour. Guide is pleased - he loves this part.

    guide:
      speaks: |
        Great! Let's start with the basics.
        See these tabs up here?

      computer_use:
        - action: move
          target: "Tab: Project"
          sync_with_speech: true

      pad_drift:
        arousal: +0.1
        pleasure: +0.1

  - id: show_project_tab
    name: "The Project Tab"
    on_stage: [guide]
    trigger:
      type: after
      beat: tour_start

    direction: |
      Walk through the Project tab.
      Point at things, let the ghost cursor do its magic.

    guide:
      speaks: |
        This is your Project tab. Everything about your
        current project lives here - your noodlings, stages,
        settings...

      computer_use:
        - action: click
          target: "Tab: Project"
          pause_after: 500

        - action: move
          target: "noodlings list"

      speaks_continued: |
        These are your noodlings. Each one is a character
        with their own personality, memories, way of thinking.

    improv_zone:
      topics:
        - what is a noodling
        - how do noodlings think
        - can I make my own
      duration:
        max_exchanges: 3

  - id: free_explore
    name: "Free Exploration"
    on_stage: [guide]
    trigger:
      type: user_choice
      choice: explore

    direction: |
      User wants to explore alone. Respect that.
      Be available but not hovering.

    guide:
      speaks: |
        Totally! Poke around, try things out.
        I'll be right here if you have questions.
        Just say my name.

      pad_drift:
        arousal: -0.1  # Settling into background mode

    # Guide goes into passive listening mode
    set_mode: passive_available
```

---

## Brenda's Script Memory

Brenda tracks the play state in her facet assembly:

```yaml
# In Brenda's assembly
facets:
  - name: ScriptMemory
    type: YAML
    data_path: "scripts/tutorial_intro.play.yaml"

  - name: PlayState
    type: Memory
    schema:
      current_beat: string
      completed_beats: list
      character_states:
        type: map
        value_type:
          pleasure: float
          arousal: float
          dominance: float
      improv_zone_active: boolean
      waiting_for: string | null
```

---

## Implementation Checklist

- [ ] Define play.yaml JSON schema for validation
- [ ] Create PlayParser class to load and validate plays
- [ ] Create PlayExecutor class in Brenda's assembly
- [ ] Wire beat triggers to channel messages
- [ ] Implement PAD drift application to charm networks
- [ ] Wire computer_use actions to ComputerUseController
- [ ] Implement improv_zone logic with exit conditions
- [ ] Create wait_for/user_choice handling
- [ ] Build example plays for Let's Consciousness
- [ ] Test with Guide receiving cues from Brenda

---

## Future Enhancements

- **Branching narratives** - Multiple paths based on user choices
- **Procedural beats** - AI-generated beats based on context
- **Ensemble dynamics** - Complex multi-character interactions
- **Play editor UI** - Visual beat/flow editor in NoodleStudio
- **Play debugging** - Step through beats, inspect state
- **Play recording** - Record improv sessions into plays

---

*"All the world's a stage, and all the noodlings merely players."*
