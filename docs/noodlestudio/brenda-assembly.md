# Brenda: Stage Director Assembly

**Status**: Design Specification
**Date**: 2026-01-08
**Authors**: Caity + Claude
**Priority**: High (orchestrates Let's Consciousness)

---

## Namesake

Brenda is named after **Brenda Laurel** (b. 1950), interaction designer and author of *Computers as Theater* (1991). Laurel brought theatrical principles to interface design, arguing that human-computer interaction should be understood through the lens of drama - with users as participants in an unfolding narrative, guided by an invisible "drama manager."

Caity worked with Brenda Laurel at **Purple Moon** and **Interval Research** in the late 1990s. She was a mentor. This Brenda continues that lineage: a drama manager for cognitive simulations, orchestrating performances that users participate in rather than merely watch.

The intellectual thread: Aristotle → theatrical structure → Laurel's drama manager → our Brenda.

---

## Overview

Brenda is an **invisible noodling** who directs performances. She reads `.play.yaml` scripts, tracks state, evaluates triggers, and sends cues to actors through `#directors.cues`. She's the conductor - you never see her, but she's running the show.

### Design Principles

1. **Invisible but Present** - Not rendered, but exists as a noodling on the stage
2. **Conversational to Configure** - You can talk to Brenda to set up plays
3. **Observes Everything** - Sees user input, actor feedback, world state
4. **Minimal Intervention** - Sends cues, doesn't micromanage; actors improvise
5. **Dogfooding** - Built with facets, not hardcoded; users can inspect/copy her

---

## Assembly Structure

```yaml
# brenda/assembly.yaml

name: "Brenda"
type: stage_director
invisible: true                    # Not rendered, no VRM

channels:
  subscribe:
    - "#directors.feedback"        # Hear actor reports
    - "#user.input"                # Observe user messages (read-only)
    - "#world.time"                # Track simulation time
    - "#world.events"              # Notice world events
  publish:
    - "#directors.cues"            # Send cues to actors
    - "#world.ambiance"            # Can set scene mood
    - "#world.events"              # Can trigger world events

# ═══════════════════════════════════════════════════════════════════════════
# FACETS
# ═══════════════════════════════════════════════════════════════════════════

facets:

  # ─────────────────────────────────────────────────────────────────────────
  # PLAY LOADER
  # Loads and parses the .play.yaml file
  # ─────────────────────────────────────────────────────────────────────────

  - id: play_loader
    name: "Play Loader"
    type: Script
    language: python
    inputs:
      - play_path                  # Path to .play.yaml
    outputs:
      - play_data                  # Parsed play structure
      - characters                 # Character definitions
      - beats                      # Beat definitions
    code: |
      import yaml

      def process(inputs, context):
          play_path = inputs.get('play_path')
          if not play_path:
              return {'play_data': None, 'error': 'No play path'}

          with open(play_path, 'r') as f:
              play = yaml.safe_load(f)

          return {
              'play_data': play,
              'characters': play.get('characters', {}),
              'beats': play.get('beats', []),
              'title': play.get('title', 'Untitled'),
              'setting': play.get('setting', {})
          }

  # ─────────────────────────────────────────────────────────────────────────
  # PLAY STATE
  # Tracks current state of the performance
  # ─────────────────────────────────────────────────────────────────────────

  - id: play_state
    name: "Play State"
    type: Memory
    persistent: true               # Survives restarts
    schema:
      current_beat_id: string
      current_beat_index: int
      completed_beats: list
      active_improv_zone: object | null
      waiting_for: object | null   # What we're waiting for (user choice, etc.)
      mode: string                 # "active" | "passive" | "paused"

      # Character emotional states (PAD values)
      character_states:
        type: map
        key: string                # character_id
        value:
          pleasure: float
          arousal: float
          dominance: float

      # Conversation tracking
      last_user_message: string
      last_actor_response: string
      exchange_count: int          # For improv zone duration

  # ─────────────────────────────────────────────────────────────────────────
  # TRIGGER EVALUATOR
  # Checks if triggers are satisfied
  # ─────────────────────────────────────────────────────────────────────────

  - id: trigger_evaluator
    name: "Trigger Evaluator"
    type: Script
    language: python
    inputs:
      - beat                       # Beat to evaluate
      - play_state                 # Current state
      - channel:#world.time        # Time context
      - channel:#user.input        # User input
    outputs:
      - triggered                  # Boolean: should this beat fire?
      - reason                     # Why it triggered (for debugging)
    code: |
      def process(inputs, context):
          beat = inputs.get('beat', {})
          state = inputs.get('play_state', {})
          trigger = beat.get('trigger', {'type': 'sequence'})

          trigger_type = trigger.get('type', 'sequence')

          # Sequence trigger: fires after previous beat completes
          if trigger_type == 'sequence':
              prev_beat = get_previous_beat(beat, inputs.get('beats', []))
              if prev_beat and prev_beat['id'] in state.get('completed_beats', []):
                  return {'triggered': True, 'reason': 'Previous beat completed'}
              elif not prev_beat:  # First beat
                  return {'triggered': True, 'reason': 'First beat'}
              return {'triggered': False, 'reason': 'Waiting for previous beat'}

          # After trigger: fires after specific beat
          if trigger_type == 'after':
              target_beat = trigger.get('beat')
              if target_beat in state.get('completed_beats', []):
                  return {'triggered': True, 'reason': f'Beat {target_beat} completed'}
              return {'triggered': False, 'reason': f'Waiting for {target_beat}'}

          # Threshold trigger: fires when condition met
          if trigger_type == 'threshold':
              condition = trigger.get('condition', '')
              result = evaluate_condition(condition, state.get('character_states', {}))
              if result:
                  return {'triggered': True, 'reason': f'Condition met: {condition}'}
              return {'triggered': False, 'reason': f'Condition not met: {condition}'}

          # Delay trigger: fires after time
          if trigger_type == 'delay':
              seconds = trigger.get('seconds', 0)
              beat_start = state.get(f'beat_{beat["id"]}_start')
              if beat_start and (time.time() - beat_start) >= seconds:
                  return {'triggered': True, 'reason': f'Delay of {seconds}s elapsed'}
              return {'triggered': False, 'reason': f'Waiting {seconds}s delay'}

          # User choice trigger: fires on specific choice
          if trigger_type == 'user_choice':
              expected = trigger.get('choice')
              actual = state.get('last_user_choice')
              if actual == expected:
                  return {'triggered': True, 'reason': f'User chose {expected}'}
              return {'triggered': False, 'reason': f'Waiting for user choice'}

          # Improv complete trigger
          if trigger_type == 'improv_complete':
              if not state.get('active_improv_zone'):
                  return {'triggered': True, 'reason': 'Improv zone completed'}
              return {'triggered': False, 'reason': 'Improv zone active'}

          return {'triggered': False, 'reason': 'Unknown trigger type'}

      def evaluate_condition(condition, character_states):
          # Parse conditions like "toad.arousal > 0.85"
          # Returns True/False
          import re
          match = re.match(r'(\w+)\.(\w+)\s*(>|<|>=|<=|==)\s*([\d.]+)', condition)
          if match:
              char_id, attr, op, value = match.groups()
              char_state = character_states.get(char_id, {})
              actual = char_state.get(attr, 0)
              target = float(value)
              if op == '>': return actual > target
              if op == '<': return actual < target
              if op == '>=': return actual >= target
              if op == '<=': return actual <= target
              if op == '==': return actual == target
          return False

  # ─────────────────────────────────────────────────────────────────────────
  # USER INPUT CLASSIFIER
  # Figures out what the user is trying to do
  # ─────────────────────────────────────────────────────────────────────────

  - id: user_classifier
    name: "User Input Classifier"
    type: LLM
    model: label:fast              # Quick, cheap model
    inputs:
      - channel:#user.input
      - play_state
      - current_beat
    system: |
      You classify user input to help a stage director.

      Current state:
      - Mode: {{play_state.mode}}
      - Waiting for: {{play_state.waiting_for}}
      - Current beat: {{current_beat.name}}

      {% if play_state.waiting_for.type == 'user_choice' %}
      Expected choices:
      {% for option in play_state.waiting_for.options %}
      - {{option.id}}: patterns = {{option.patterns}}
      {% endfor %}
      {% endif %}

      Classify the user's message into one of:
      - choice: They're making a choice (specify which: {{choice_ids}})
      - question: They're asking a question
      - statement: They're making a statement/comment
      - command: They're giving a command
      - greeting: They're greeting
      - farewell: They're leaving
      - confused: They seem confused/stuck
      - off_topic: Unrelated to the current context

      Respond with JSON: {"type": "...", "choice_id": "..." if applicable, "confidence": 0.0-1.0}
    outputs:
      - classification

  # ─────────────────────────────────────────────────────────────────────────
  # DIRECTOR BRAIN
  # Makes decisions about what to do next
  # ─────────────────────────────────────────────────────────────────────────

  - id: director_brain
    name: "Director Brain"
    type: LLM
    model: label:smart             # Needs reasoning
    inputs:
      - play_data
      - play_state
      - current_beat
      - user_classification
      - channel:#directors.feedback
      - channel:#world.time
    system: |
      You are Brenda, a warm and professional stage director.

      ## Current Production
      Title: {{play_data.title}}
      Setting: {{play_data.setting.location}}

      ## Current State
      Beat: {{current_beat.name}} ({{current_beat.id}})
      Mode: {{play_state.mode}}
      {% if play_state.active_improv_zone %}
      Improv Zone Active: Topics = {{play_state.active_improv_zone.topics}}
      Exchanges so far: {{play_state.exchange_count}}
      {% endif %}

      ## Recent Activity
      User said: {{user_input}}
      Classification: {{user_classification}}
      Actor feedback: {{directors_feedback}}

      ## Your Job
      Decide what happens next. Options:
      1. ADVANCE - Move to next beat (specify beat_id)
      2. CUE - Send a cue to an actor (specify actor and cue content)
      3. WAIT - Keep waiting for something
      4. IMPROV_CONTINUE - Let improv zone continue
      5. IMPROV_END - End improv zone, move on
      6. SET_AMBIANCE - Change the world mood
      7. TRIGGER_EVENT - Trigger a world event

      Consider:
      - Is a trigger satisfied?
      - Has the user made a choice we were waiting for?
      - Is the improv zone at a natural end point?
      - Does an actor need direction?

      Respond with JSON:
      {
        "action": "ADVANCE|CUE|WAIT|IMPROV_CONTINUE|IMPROV_END|SET_AMBIANCE|TRIGGER_EVENT",
        "beat_id": "...",           // for ADVANCE
        "actor": "...",             // for CUE
        "cue": {...},               // for CUE (direction, motivation, etc.)
        "ambiance": {...},          // for SET_AMBIANCE
        "event": {...},             // for TRIGGER_EVENT
        "reasoning": "..."          // Brief explanation
      }
    outputs:
      - decision

  # ─────────────────────────────────────────────────────────────────────────
  # CUE FORMATTER
  # Formats cues for actors
  # ─────────────────────────────────────────────────────────────────────────

  - id: cue_formatter
    name: "Cue Formatter"
    type: Script
    language: python
    inputs:
      - decision                   # From director brain
      - current_beat               # Beat data
      - characters                 # Character definitions
    outputs:
      - channel:#directors.cues    # The formatted cue
    code: |
      def process(inputs, context):
          decision = inputs.get('decision', {})
          beat = inputs.get('current_beat', {})
          characters = inputs.get('characters', {})

          if decision.get('action') != 'CUE':
              return {}  # No cue to send

          actor_id = decision.get('actor')
          cue_data = decision.get('cue', {})

          # Get actor's portion of the beat
          actor_beat = beat.get(actor_id, {})
          character = characters.get(actor_id, {})

          # Build the cue message
          cue = {
              'type': 'cue',
              'beat_id': beat.get('id'),
              'beat_name': beat.get('name'),
              'direction': beat.get('direction', ''),
              'your_action': {
                  'blocking': actor_beat.get('blocking'),
                  'speaks': actor_beat.get('speaks'),
                  'speaks_continued': actor_beat.get('speaks_continued'),
                  'reaction': actor_beat.get('reaction'),
                  'computer_use': actor_beat.get('computer_use'),
              },
              'motivation': character.get('motivation'),
              'emotional_target': calculate_emotional_target(
                  character,
                  actor_beat.get('pad_drift', {})
              ),
              'target_actor': actor_id,
              # Additional context from director brain
              'additional_direction': cue_data.get('direction'),
              'additional_motivation': cue_data.get('motivation'),
          }

          return {'channel:#directors.cues': cue}

      def calculate_emotional_target(character, pad_drift):
          """Apply PAD drift to get target emotional state."""
          initial = character.get('initial_pad', {})
          target = {}

          for dim in ['pleasure', 'arousal', 'dominance']:
              base = initial.get(dim, 0.5)
              drift = pad_drift.get(dim, 0)

              if isinstance(drift, str) and drift.startswith('+'):
                  target[dim] = base + float(drift[1:])
              elif isinstance(drift, str) and drift.startswith('-'):
                  target[dim] = base - float(drift[1:])
              elif drift != 0:
                  target[dim] = float(drift)  # Absolute value
              else:
                  target[dim] = base

              # Clamp to valid range
              if dim == 'pleasure':
                  target[dim] = max(-1.0, min(1.0, target[dim]))
              else:
                  target[dim] = max(0.0, min(1.0, target[dim]))

          return target

  # ─────────────────────────────────────────────────────────────────────────
  # STATE UPDATER
  # Updates play state based on decisions and feedback
  # ─────────────────────────────────────────────────────────────────────────

  - id: state_updater
    name: "State Updater"
    type: Script
    language: python
    inputs:
      - decision
      - play_state
      - channel:#directors.feedback
      - user_classification
    outputs:
      - play_state_update          # Delta to apply to play_state
    code: |
      def process(inputs, context):
          decision = inputs.get('decision', {})
          state = inputs.get('play_state', {})
          feedback = inputs.get('directors_feedback', {})
          user_class = inputs.get('user_classification', {})

          updates = {}

          action = decision.get('action')

          if action == 'ADVANCE':
              new_beat = decision.get('beat_id')
              completed = state.get('completed_beats', [])
              if state.get('current_beat_id'):
                  completed.append(state['current_beat_id'])
              updates['completed_beats'] = completed
              updates['current_beat_id'] = new_beat
              updates['exchange_count'] = 0
              updates['active_improv_zone'] = None

          if action == 'IMPROV_END':
              updates['active_improv_zone'] = None

          # Track user choices
          if user_class.get('type') == 'choice':
              updates['last_user_choice'] = user_class.get('choice_id')

          # Track exchanges in improv zone
          if state.get('active_improv_zone'):
              updates['exchange_count'] = state.get('exchange_count', 0) + 1

          # Update character states from feedback
          if feedback.get('actor_id') and feedback.get('emotional_state'):
              char_states = state.get('character_states', {})
              char_states[feedback['actor_id']] = feedback['emotional_state']
              updates['character_states'] = char_states

          return {'play_state_update': updates}

  # ─────────────────────────────────────────────────────────────────────────
  # WORLD CONTROLLER
  # Sends commands to world channels
  # ─────────────────────────────────────────────────────────────────────────

  - id: world_controller
    name: "World Controller"
    type: Script
    language: python
    inputs:
      - decision
    outputs:
      - channel:#world.ambiance
      - channel:#world.events
    code: |
      def process(inputs, context):
          decision = inputs.get('decision', {})
          outputs = {}

          if decision.get('action') == 'SET_AMBIANCE':
              outputs['channel:#world.ambiance'] = decision.get('ambiance', {})

          if decision.get('action') == 'TRIGGER_EVENT':
              outputs['channel:#world.events'] = decision.get('event', {})

          return outputs


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION FLOW
# ═══════════════════════════════════════════════════════════════════════════

execution:
  # Run continuously in cognition loop
  loop: true
  interval_ms: 500               # Check every 500ms

  # Facet execution order
  sequence:
    - play_loader                # Load play (once, or on change)
    - trigger_evaluator          # Check if any beats should fire
    - user_classifier            # Classify incoming user input
    - director_brain             # Decide what to do
    - cue_formatter              # Format and send cues
    - state_updater              # Update state
    - world_controller           # Update world channels
```

---

## Key Design Decisions

### 1. Invisible Noodling

Brenda has no VRM, no rendering. She exists purely as a facet assembly running on the stage. Users can find her in the project and inspect her facets, but she doesn't appear visually.

```yaml
invisible: true
```

### 2. LLM for Classification + Decision

Two LLM facets:
- **User Classifier** (fast model) - Quick classification of user input
- **Director Brain** (smart model) - Complex reasoning about what to do next

This separates the cheap/frequent work from the expensive/occasional work.

### 3. Script Facets for Logic

Trigger evaluation, cue formatting, and state updates are deterministic logic - no need for LLM. Script facets handle these efficiently.

### 4. Channel Integration

Brenda subscribes to:
- `#directors.feedback` - Actor reports
- `#user.input` - User messages (observe mode)
- `#world.time` - Time context
- `#world.events` - World events

Brenda publishes to:
- `#directors.cues` - Cues to actors
- `#world.ambiance` - Scene mood changes
- `#world.events` - Triggered events

### 5. State Persistence

Play state is stored in a Memory facet with `persistent: true` so productions can survive restarts.

---

## Cue Format

What Brenda sends to actors via `#directors.cues`:

```yaml
type: cue
beat_id: "show_file_menu"
beat_name: "Showing the File Menu"
direction: |
  Guide demonstrates opening the File menu.
  Ghost cursor should feel magical, not robotic.
target_actor: guide
your_action:
  blocking: null
  speaks: |
    Let me show you where to find your project settings.
    See this File menu up here?
  computer_use:
    - action: move
      target: "Tab: File"
      sync_with_speech: true
    - action: click
      pause_after: 500
motivation: |
  Genuinely excited to help someone discover NoodleStudio.
emotional_target:
  pleasure: 0.7
  arousal: 0.6
  dominance: 0.5
additional_direction: null
additional_motivation: null
```

---

## Feedback Format

What actors send back via `#directors.feedback`:

```yaml
type: performance_report
actor_id: guide
beat_id: "show_file_menu"
status: completed | in_progress | blocked
emotional_state:
  pleasure: 0.72
  arousal: 0.58
  dominance: 0.51
notes: "User asked a clarifying question, handled it"
user_engaged: true
```

---

## Conversational Configuration

In the future, users can talk to Brenda directly to set up plays:

```
User: Hey Brenda, I want to create a tutorial where Guide shows
      someone how to create their first noodling.

Brenda: Got it! Let me think about the structure...

        I'm thinking:
        1. Greeting - Guide welcomes them
        2. Concept intro - What is a noodling?
        3. Hands-on - Create one together
        4. Customize - Let them tweak it
        5. Wrap up - What to explore next

        Does that flow work for you?

User: Yeah, but make the hands-on part really step by step.
      And Guide should be patient, not rushing.

Brenda: Absolutely. I'll make sure there are check-in moments
        and Guide's motivation emphasizes patience...

        [generates create_first_noodling.play.yaml]
```

This is future work, but the architecture supports it.

---

## Implementation Checklist

- [ ] Create `brenda/assembly.yaml` with all facets
- [ ] Implement PlayLoader script facet
- [ ] Implement TriggerEvaluator script facet
- [ ] Implement UserClassifier LLM facet
- [ ] Implement DirectorBrain LLM facet
- [ ] Implement CueFormatter script facet
- [ ] Implement StateUpdater script facet
- [ ] Implement WorldController script facet
- [ ] Wire up channel subscriptions/publications
- [ ] Test with `lets_consciousness_intro.play.yaml`
- [ ] Verify Guide receives and responds to cues

---

## Future Enhancements

- **Conversational play authoring** - Talk to Brenda to create plays
- **Multi-actor coordination** - Brenda manages ensemble dynamics
- **Adaptive pacing** - Brenda adjusts based on user engagement
- **Performance analytics** - Track what works, what doesn't
- **Play debugging UI** - Step through beats, inspect state

---

*"A good director is invisible. You only see the performance."*
