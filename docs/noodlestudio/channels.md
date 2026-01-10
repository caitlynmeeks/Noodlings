# Channel Architecture

**Status**: Specification
**Date**: 2026-01-08
**Authors**: Caity + Claude
**Priority**: High (enables stage direction, ensemble dynamics)

---

## Overview

Channels are named message buses that noodlings can publish to and subscribe to. They enable decoupled communication patterns beyond direct noodling-to-noodling speech.

### Why Channels?

| Without Channels | With Channels |
|------------------|---------------|
| Direct speech only | Broadcast, scoped, private messaging |
| Tight coupling | Loose coupling |
| Hard to add observers | Easy pub/sub patterns |
| Stage direction is magic | Stage direction is visible |

### Inspiration

- ROS topics (robotics)
- Game engine event buses
- Actor model message passing
- Theater: director's headset channel

---

## Channel Types

### Public Channels

World-visible broadcasts. Any noodling can subscribe.

```
#world.weather      → "temperature dropped to 40°F"
#world.time         → "the sun is setting"
#world.events       → "a door slammed in the next room"
#world.ambiance     → "distant thunder rumbles"
```

Use for: Environmental context that anyone can perceive.

### Scoped Channels

Only subscribed members receive messages. Membership defined in assembly.

```
#directors.cues     → Stage direction from Brenda to actors
#bridge.comms       → Bridge crew internal channel
#away_team.radio    → Away team walkie-talkie
#villains.scheme    → Bad guys plotting (heroes can't hear)
```

Use for: Group communication, role-based channels.

### Private Channels

Point-to-point between specific noodlings.

```
#dm.brenda→guide    → Private notes from director to actor
#whisper.a→b        → Secret conversation
```

Use for: Private direction, secrets, asides.

---

## Channel Naming Convention

```
#<scope>.<purpose>

Examples:
#world.weather          Public environmental
#directors.cues         Scoped to director/actor relationship
#bridge.comms           Scoped to location/group
#dm.brenda→guide        Private directional
```

Reserved prefixes:
- `#world.*` - Public environmental channels
- `#directors.*` - Stage management channels
- `#dm.*` - Direct message (private)
- `#system.*` - Engine-level (timing, errors)

---

## Assembly Configuration

### Subscribing to Channels

```yaml
# In assembly.yaml
name: Guide Assembly

channels:
  subscribe:
    - "#directors.cues"      # Hear stage direction
    - "#world.context"       # Hear environmental updates
  publish:
    - "#directors.feedback"  # Report back to director
```

### INCOMING Node with Channel Pads

In the facets editor, subscribed channels appear as additional pads on INCOMING:

```
┌──────────────────────────────────────────────────────────────┐
│  INCOMING                                                     │
│  ┌────────────────┐                                          │
│  │ ○ user_input   │──→  What user says to me                 │
│  │ ○ perception   │──→  What I perceive in world             │
│  │ ○ memory       │──→  Retrieved memories                   │
│  ├────────────────┤                                          │
│  │ ○ #directors   │──→  Cues from stage director   [channel] │
│  │ ○ #world       │──→  Environmental context      [channel] │
│  └────────────────┘                                          │
└──────────────────────────────────────────────────────────────┘
```

### OUTGOING Node with Channel Pads

Published channels appear as additional pads on OUTGOING:

```
┌──────────────────────────────────────────────────────────────┐
│  OUTGOING                                                     │
│  ┌────────────────┐                                          │
│  │ ○ speech       │──→  What I say aloud                     │
│  │ ○ action       │──→  Physical actions I take              │
│  │ ○ muscles      │──→  Animation muscle values              │
│  ├────────────────┤                                          │
│  │ ○ #directors   │──→  Feedback to stage director [channel] │
│  └────────────────┘                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Message Format

Channel messages are structured:

```yaml
channel_message:
  channel: "#directors.cues"
  from: brenda              # Sender noodling (or "system")
  timestamp: 1704825600
  payload:
    type: cue               # Message type (app-defined)
    direction: "Walk them through the File menu"
    motivation: "you're excited to share this"
    target: guide           # Optional: specific recipient
```

### Payload Types (Examples)

**Stage Direction:**
```yaml
payload:
  type: cue
  direction: "React to the explosion"
  motivation: "you're terrified but trying to stay professional"
```

**Environmental:**
```yaml
payload:
  type: context
  event: "temperature_change"
  value: 40
  unit: "fahrenheit"
  description: "A cold front has arrived"
```

**Feedback:**
```yaml
payload:
  type: performance_report
  beat_id: "show_file_menu"
  status: completed
  notes: "User asked a clarifying question, handled it"
```

---

## Facet Integration

### Reading from Channels

Facets can read channel input like any other incoming data:

```yaml
# In facet definition
facets:
  - name: Response
    type: LLM
    incoming:
      - user_input
      - perception
      - channel:#directors.cues   # Channel as input
    prompt: |
      You are Guide.

      User said: {{user_input}}
      You perceive: {{perception}}

      {% if directors_cues %}
      Director's note: {{directors_cues.direction}}
      Your motivation: {{directors_cues.motivation}}
      {% endif %}

      Respond naturally, incorporating the direction.
```

### Publishing to Channels

Script facets or LLM facets can output to channels:

```yaml
- name: CueEmitter
  type: Script
  language: javascript
  outputs:
    - channel:#directors.cues
  code: |
    // Read current beat from script memory
    const beat = context.script_memory.current_beat;

    if (beat.action === 'cue') {
      emit('#directors.cues', {
        type: 'cue',
        direction: beat.direction,
        motivation: beat.motivation,
        target: beat.actor
      });
    }
```

---

## Stage Director Pattern

Brenda (or any stage director noodling) uses channels to orchestrate:

```yaml
# brenda/assembly.yaml
name: Brenda Stage Director
invisible: true              # Not rendered, but present in stage

channels:
  subscribe:
    - "#directors.feedback"  # Hear actor reports
    - "#user.input"          # Hear what user says (observer)
  publish:
    - "#directors.cues"      # Send cues to actors

facets:
  - name: ScriptMemory
    type: YAML
    data_path: "scripts/tutorial_intro.play.yaml"

  - name: DirectorLogic
    type: LLM
    model: anthropic/claude-3-haiku
    incoming:
      - channel:#directors.feedback
      - channel:#user.input
      - script_state         # From ScriptMemory
    system: |
      You are Brenda, a warm professional stage director.

      Current script state: {{script_state}}
      Actor feedback: {{directors_feedback}}
      User just said: {{user_input}}

      Decide:
      - If scene is progressing well: {"wait": true}
      - If actor needs direction: {"cue": "...", "motivation": "..."}
      - If user interrupted: {"handle_interrupt": true, "then_resume": "beat_id"}
      - If beat complete: {"advance": "next_beat_id"}

  - name: CueEmitter
    type: Script
    incoming:
      - director_decision    # From DirectorLogic
    outputs:
      - channel:#directors.cues
      - script_state_update  # Back to ScriptMemory
```

---

## Actor Pattern

Guide (or any actor noodling) listens to director and performs:

```yaml
# guide/assembly.yaml
name: Guide
visible: true
vrm_path: "Radiances/AjoMajo.vrm"

channels:
  subscribe:
    - "#directors.cues"
  publish:
    - "#directors.feedback"

facets:
  - name: Perception
    type: LLM
    incoming:
      - user_input
      - world_perception
      - channel:#directors.cues
    purpose: |
      Synthesize what I'm experiencing:
      - What the user said/did
      - What's happening in the world
      - Any direction from the stage manager

  - name: Response
    type: LLM
    model: anthropic/claude-sonnet-4
    incoming:
      - perception_synthesis
    system: |
      You are Guide, a friendly and warm tutor.
      You know you're performing in a tutorial, but you're genuine.

      {% if perception_synthesis.director_cue %}
      [Director's note: {{perception_synthesis.director_cue.direction}}]
      [Your motivation: {{perception_synthesis.director_cue.motivation}}]
      {% endif %}

      Improvise naturally. Don't be wooden. You're an actor, not a puppet.

  - name: PerformanceReport
    type: Script
    outputs:
      - channel:#directors.feedback
    code: |
      // After responding, report back to director
      emit('#directors.feedback', {
        type: 'performance_report',
        beat_id: context.current_beat,
        status: 'completed',
        user_engaged: context.user_responded
      });
```

---

## Implementation Checklist

- [ ] Add `channels` field to assembly schema
- [ ] Implement ChannelBus in runtime (pub/sub)
- [ ] Add channel pads to INCOMING/OUTGOING nodes in facets editor
- [ ] Add channel message routing in FacetExecutor
- [ ] Create `#world.*` system channels
- [ ] Create `#directors.*` stage management channels
- [ ] Build Brenda as example stage director noodling
- [ ] Wire Guide to receive cues from Brenda
- [ ] Test in Let's Consciousness

---

## Example: Let's Consciousness

The demo app is built entirely with these tools:

```
Let's Consciousness/
├── project.yaml
├── Noodlings/
│   ├── guide/
│   │   ├── assembly.yaml      # Actor facets
│   │   └── Radiances/
│   │       └── AjoMajo.vrm
│   └── brenda/
│       ├── assembly.yaml      # Director facets
│       └── scripts/
│           └── tutorial_intro.play.yaml
├── Stages/
│   └── tutorial_stage/
│       └── stage.yaml
└── ui.yaml
```

User downloads app → Uses it → Clicks "View Project" → Sees exactly how it works → Modifies it → Learns NoodleStudio.

The app IS the tutorial.

---

## Future Enhancements

- **Channel permissions**: Fine-grained access control
- **Channel history**: Replay/scrub through channel messages
- **Channel visualization**: See message flow in real-time
- **Cross-stage channels**: Communication between stages
- **Channel webhooks**: External system integration

---

*"The medium is the message. The demo is the documentation."*
