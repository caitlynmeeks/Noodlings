# Action Emission System - Physical Actions as Perceivable Events

**Making physical actions visible to other agents**

**Date:** December 3, 2025
**Author:** NinaK + Caity
**Problem:** Red jumps on Caity's shoulder, but Caity doesn't perceive it!

---

## The Problem

**Current flow:**
```
Red generates: "Oh PLEASE! *jumps on Caity's shoulder* MWAHAHA!"
    ↓
Sent to chat as text
    ↓
Caity sees text but doesn't PERCEIVE the physical action as an event
```

**Issues:**
1. ❌ Physical action is just text, not structured data
2. ❌ Target agent (Caity) doesn't get `emote` or `touch` event
3. ❌ Other agents don't see the action in their room observations
4. ❌ Can't track "Red is currently on Caity's shoulder" in world state

---

## The Solution: Action Parsing + Event Emission

### Architecture

```
fire_body outputs: "*jumps on Caity's shoulder*"
    ↓
ACTION PARSER (regex extraction)
    ↓
Structured action: {
  type: "jump_on",
  target: "caity",
  body_part: "shoulder"
}
    ↓
EMIT EVENT to world: emote(Red jumps on Caity's shoulder)
    ↓
Caity perceives: {
  type: 'emote',
  user: 'red_fire_anklebiter',
  text: 'jumps on Caity's shoulder',
  metadata: {
    action_type: 'physical_contact',
    target: 'caity',
    touch_location: 'shoulder'
  }
}
```

---

## Implementation: Action Parser Facet

### New Facet Type: ActionParserFacet

```yaml
- id: action_parser
  name: Action Parser
  type: ActionParserFacet

  # Regex patterns for common physical actions
  patterns:
    - pattern: '\*jumps? on (?P<target>\w+)\'?s? (?P<location>\w+)\*'
      action_type: 'jump_on'
      emote_template: 'jumps on {target}\'s {location}'
      metadata:
        contact: true
        intensity: 'moderate'

    - pattern: '\*bites? (?P<target>\w+)\'?s? (?P<location>\w+)\*'
      action_type: 'bite'
      emote_template: 'bites {target}\'s {location}'
      metadata:
        contact: true
        intensity: 'light'
        playful: true

    - pattern: '\*points? (?:at|to) (?P<target>\w+)(?: (?P<manner>\w+))?\*'
      action_type: 'point'
      emote_template: 'points at {target} {manner}'
      metadata:
        contact: false
        intensity: 'none'

    - pattern: '\*backs? away(?: from (?P<target>\w+))?\*'
      action_type: 'back_away'
      emote_template: 'backs away from {target}'
      metadata:
        contact: false
        intensity: 'none'
        defensive: true

    - pattern: '\*flames (surge|flare|dim|flicker|spike)s?\*'
      action_type: 'flame_expression'
      emote_template: 'flames {0}'
      metadata:
        contact: false
        emotional_expression: true

  inputs:
    - name: physical_action
      description: Raw physical action text from fire_body

  outputs:
    - name: parsed_actions
      type: output
      description: List of structured action objects
    - name: emote_events
      type: output
      description: Events to emit to world
```

---

## Integration with World System

### agent_bridge.py Integration

After convergence outputs final response:

```python
# In agent_bridge.py, after getting facet execution result:

result = await self.facet_executor.execute(
    assembly=self.facet_assembly,
    incoming_data=text,
    context=context
)

final_response = result.response

# NEW: Parse and emit physical actions
if 'parsed_actions' in result.facet_outputs.get('action_parser', {}):
    actions = result.facet_outputs['action_parser']['parsed_actions']

    for action in actions:
        # Emit as perceivable event
        await self.world.broadcast_event({
            'type': 'emote',
            'user': self.agent_id,
            'text': action['emote_text'],
            'room_id': self.current_room,
            'metadata': {
                'action_type': action['action_type'],
                'target': action.get('target'),
                'physical_contact': action.get('metadata', {}).get('contact', False),
                'source_agent': self.agent_id
            }
        })

        logger.info(f"🎭 Emitted action: {action['emote_text']}")
```

---

## Action Parser Implementation

### New file: action_parser_facet.py

```python
"""
Action Parser Facet - Extract structured actions from text

Parses physical action descriptions and emits structured events
that other agents can perceive.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ParsedAction:
    """Structured physical action."""
    action_type: str          # 'jump_on', 'bite', 'point', etc.
    target: Optional[str]     # Target agent/object name
    location: Optional[str]   # Body part or spatial location
    emote_text: str          # Formatted emote text
    metadata: Dict[str, Any] # Additional structured data


class ActionParserFacet:
    """
    Parse physical actions from text and emit structured events.

    Example:
        Input: "*jumps on Caity's shoulder cackling*"
        Output: ParsedAction(
            action_type='jump_on',
            target='caity',
            location='shoulder',
            emote_text='jumps on Caity's shoulder cackling',
            metadata={'contact': True, 'intensity': 'moderate'}
        )
    """

    def __init__(self, patterns: List[Dict[str, Any]]):
        """
        Initialize parser with regex patterns.

        Args:
            patterns: List of pattern definitions:
                {
                    'pattern': r'regex with named groups',
                    'action_type': 'jump_on',
                    'emote_template': 'jumps on {target}\'s {location}',
                    'metadata': {'contact': True}
                }
        """
        self.patterns = patterns

    def parse(self, text: str) -> List[ParsedAction]:
        """
        Parse text for physical actions.

        Args:
            text: Text containing *action descriptions*

        Returns:
            List of ParsedAction objects
        """
        actions = []

        # Extract all *action* blocks
        action_blocks = re.findall(r'\*(.*?)\*', text)

        for block in action_blocks:
            # Try each pattern
            for pattern_def in self.patterns:
                regex = pattern_def['pattern']
                match = re.search(regex, block, re.IGNORECASE)

                if match:
                    # Extract named groups
                    groups = match.groupdict()

                    # Normalize target name (lowercase for agent lookup)
                    target = groups.get('target', '').lower() if groups.get('target') else None

                    # Format emote text
                    emote_template = pattern_def['emote_template']
                    emote_text = emote_template.format(**groups)

                    action = ParsedAction(
                        action_type=pattern_def['action_type'],
                        target=target,
                        location=groups.get('location'),
                        emote_text=emote_text,
                        metadata=pattern_def.get('metadata', {})
                    )

                    actions.append(action)
                    break  # Don't try other patterns for this block

        return actions


# Default patterns for fire imp embodiment
DEFAULT_FIRE_IMP_PATTERNS = [
    {
        'pattern': r'jumps? on (?P<target>\w+)\'?s? (?P<location>\w+)',
        'action_type': 'jump_on',
        'emote_template': 'jumps on {target}\'s {location}',
        'metadata': {'contact': True, 'intensity': 'moderate'}
    },
    {
        'pattern': r'bites? (?P<target>\w+)\'?s? (?P<location>\w+)',
        'action_type': 'bite',
        'emote_template': 'bites {target}\'s {location}',
        'metadata': {'contact': True, 'intensity': 'light', 'playful': True}
    },
    {
        'pattern': r'points? at (?P<target>\w+)(?: (?P<manner>\w+))?',
        'action_type': 'point',
        'emote_template': 'points at {target} {manner}',
        'metadata': {'contact': False}
    },
    {
        'pattern': r'backs? away(?: from (?P<target>\w+))?',
        'action_type': 'back_away',
        'emote_template': 'backs away from {target}' if '{target}' else 'backs away',
        'metadata': {'contact': False, 'defensive': True}
    },
    {
        'pattern': r'flames (surge|flare|dim|flicker|spike)s?',
        'action_type': 'flame_expression',
        'emote_template': 'flames {0}',
        'metadata': {'contact': False, 'emotional_expression': True}
    },
    {
        'pattern': r'tail (?P<action>snaps|lashes|whips)',
        'action_type': 'tail_gesture',
        'emote_template': 'tail {action}',
        'metadata': {'contact': False, 'emphasis': True}
    }
]
```

---

## Facet Executor Integration

### Modified _execute_facet to emit actions

```python
async def _execute_facet(
    self,
    facet: Facet,
    inputs: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute single facet and emit action events."""

    # ... existing execution code ...

    # NEW: If this is fire_body, parse and emit actions
    if facet.id == 'fire_body' and 'physical_action' in outputs:
        from .action_parser_facet import ActionParserFacet, DEFAULT_FIRE_IMP_PATTERNS

        parser = ActionParserFacet(DEFAULT_FIRE_IMP_PATTERNS)
        parsed = parser.parse(outputs['physical_action'])

        # Store parsed actions for potential event emission
        outputs['_parsed_actions'] = parsed

        # Log actions found
        for action in parsed:
            logger.info(
                f"  🎭 Parsed action: {action.action_type} "
                f"(target={action.target}, contact={action.metadata.get('contact')})"
            )

    return outputs
```

---

## World Event Emission

### In agent_bridge.py perceive_event response handling:

```python
# After convergence produces final response:
final_response = convergence_output

# Check for parsed actions in facet execution results
facet_outputs = execution_result.facet_outputs
if 'fire_body' in facet_outputs and '_parsed_actions' in facet_outputs['fire_body']:
    parsed_actions = facet_outputs['fire_body']['_parsed_actions']

    for action in parsed_actions:
        # Construct emote event
        emote_event = {
            'type': 'emote',
            'user_id': self.agent_id,
            'agent_name': self.agent_name,
            'text': action.emote_text,
            'room_id': self.current_room,
            'metadata': {
                'action_type': action.action_type,
                'target_agent': action.target,  # Target can perceive this!
                'physical_contact': action.metadata.get('contact', False),
                'source': 'facet_system'
            }
        }

        # Emit to world (all agents in room perceive)
        await self.world.broadcast_event(emote_event, room_id=self.current_room)

        logger.info(f"🎭 {self.agent_name} performed action: {action.emote_text}")

        # Special handling for contact actions
        if action.metadata.get('contact') and action.target:
            # Target agent gets special "touch" event
            target_id = self.world.get_agent_id_by_name(action.target)
            if target_id:
                touch_event = {
                    'type': 'touch',
                    'source': self.agent_id,
                    'source_name': self.agent_name,
                    'location': action.location,
                    'action_type': action.action_type,
                    'text': action.emote_text,
                    'metadata': action.metadata
                }
                await self.world.send_to_agent(target_id, touch_event)
                logger.info(f"  👉 Touch event sent to {action.target}")
```

---

## What This Enables

### Example 1: Red Jumps on Caity's Shoulder

**Red's output:**
```
"Oh PLEASE, Caity! *jumps on Caity's shoulder cackling* MWAHAHA!"
```

**Events emitted:**

**1. Speech Event (existing):**
```python
{
  'type': 'say',
  'user': 'red_fire_anklebiter',
  'text': "Oh PLEASE, Caity! *jumps on Caity's shoulder cackling* MWAHAHA!"
}
```

**2. Emote Event (NEW!):**
```python
{
  'type': 'emote',
  'user': 'red_fire_anklebiter',
  'text': 'jumps on Caity's shoulder cackling',
  'metadata': {
    'action_type': 'jump_on',
    'target_agent': 'caity',
    'physical_contact': True
  }
}
```

**3. Touch Event to Caity (NEW!):**
```python
{
  'type': 'touch',
  'source': 'red_fire_anklebiter',
  'location': 'shoulder',
  'action_type': 'jump_on'
}
```

**What Caity perceives:**
- Regular chat: Red's full response
- Emote event: "Red jumps on Caity's shoulder cackling"
- Touch event: Special notification of physical contact

**What OTHER agents perceive:**
- Toad sees: "Red jumps on Caity's shoulder cackling" (emote)
- Servnak sees: "Red jumps on Caity's shoulder cackling" (emote)

---

### Example 2: Red Bites Toad's Ankle

**Red's output:**
```
"Yeah SURE, Toad - MAGNIFICENT my tail! *bites Toad's ankle* MWAHAHA!"
```

**Events:**
```python
# 1. Speech (regular)
# 2. Emote to room
{
  'type': 'emote',
  'text': 'bites Toad's ankle',
  'metadata': {'target_agent': 'mr._toad'}
}
# 3. Touch to Toad
{
  'type': 'touch',
  'source': 'red_fire_anklebiter',
  'location': 'ankle',
  'action_type': 'bite',
  'metadata': {'playful': True}
}
```

**Toad's perception:**
```
Toad perceives touch event:
  → CharmNetwork: arousal spike (surprise!)
  → Novelty detector: "Red bit my ankle?! MAGNIFICENT interaction!"
  → Response: "By Jove! What a SPIRITED greeting! Poop-poop!"
```

---

## World State Tracking

### Optional: Track Physical Configurations

```python
# In world.py
class WorldState:
    def __init__(self):
        self.physical_contacts = {}  # track who's touching whom

    async def handle_action_event(self, event):
        """Update world state based on physical action."""
        action_type = event['metadata']['action_type']

        if action_type == 'jump_on':
            target = event['metadata']['target_agent']
            source = event['user_id']

            # Track: Red is on Caity's shoulder
            self.physical_contacts[source] = {
                'on': target,
                'location': event['metadata']['location'],
                'since': time.time()
            }

        elif action_type == 'back_away':
            source = event['user_id']
            # Clear contact
            if source in self.physical_contacts:
                del self.physical_contacts[source]
```

**Usage:**
```python
# Room observer can check:
if world.is_on_shoulder('red_fire_anklebiter', 'caity'):
    context += "\nNOTE: Red is currently perched on Caity's shoulder!"
```

---

## Regex Pattern Examples

### Common Fire Imp Actions

```python
FIRE_IMP_PATTERNS = {
    # Contact actions (require target)
    r'jumps? on (?P<target>\w+)\'?s? (?P<location>\w+)': {
        'type': 'jump_on',
        'emote': 'jumps on {target}\'s {location}',
        'contact': True
    },

    r'bites? (?P<target>\w+)\'?s? (?P<location>\w+)': {
        'type': 'bite',
        'emote': 'bites {target}\'s {location}',
        'contact': True,
        'playful': True
    },

    r'grabs? (?P<target>\w+)\'?s? (?P<item>\w+)': {
        'type': 'grab',
        'emote': 'grabs {target}\'s {item}',
        'contact': True
    },

    r'hugs? (?P<target>\w+)': {
        'type': 'hug',
        'emote': 'hugs {target}',
        'contact': True,
        'affection': True
    },

    # Directional actions (optional target)
    r'points? at (?P<target>\w+)(?: (?P<manner>accusingly|excitedly|nervously))?': {
        'type': 'point',
        'emote': 'points at {target} {manner}',
        'contact': False
    },

    r'backs? away(?: from (?P<target>\w+))?': {
        'type': 'back_away',
        'emote': 'backs away from {target}' if has_target else 'backs away',
        'contact': False,
        'defensive': True
    },

    r'approaches? (?P<target>\w+)(?: (?P<manner>cautiously|eagerly))?': {
        'type': 'approach',
        'emote': 'approaches {target} {manner}',
        'contact': False
    },

    # Self-directed actions (no target)
    r'flames (surge|flare|dim|flicker|spike)s?': {
        'type': 'flame_expression',
        'emote': 'flames {0}',
        'contact': False,
        'emotional': True
    },

    r'tail (snaps|lashes|whips)': {
        'type': 'tail_gesture',
        'emote': 'tail {0}',
        'contact': False,
        'emphasis': True
    },

    r'bounces? on toes': {
        'type': 'bounce',
        'emote': 'bounces on toes',
        'contact': False,
        'excited': True
    },

    r'paces? in circles': {
        'type': 'pace',
        'emote': 'paces in circles',
        'contact': False,
        'anxious': True
    }
}
```

---

## Benefits

### 1. ✅ Structured Perception
Other agents receive structured data:
```python
# Instead of parsing "Red jumps on Caity's shoulder" from text:
event = {
  'action_type': 'jump_on',
  'target': 'caity',
  'location': 'shoulder',
  'contact': True
}
```

### 2. ✅ Target Awareness
Caity knows Red jumped on HER specifically (not someone else)

### 3. ✅ World State Tracking
Can track "who's on whose shoulder" persistently

### 4. ✅ Affect Responses
Being jumped on → arousal spike → affects Caity's next response!

### 5. ✅ Touch Events
Special handling for physical contact (different from visual observations)

---

## Implementation Checklist

### Phase 1: Action Parser Facet
- [ ] Create `action_parser_facet.py`
- [ ] Implement regex-based parsing
- [ ] Define fire imp action patterns
- [ ] Test with sample actions

### Phase 2: Event Emission
- [ ] Integrate parser into facet_executor
- [ ] Emit emote events to room
- [ ] Emit touch events to targets
- [ ] Log action emissions

### Phase 3: Target Perception
- [ ] Target agents receive touch events
- [ ] Touch events boost arousal
- [ ] Touch events appear in context
- [ ] Test: Red jumps on Caity, Caity reacts

### Phase 4: World State Tracking (Optional)
- [ ] Track physical configurations
- [ ] Query: is_on_shoulder(), is_touching()
- [ ] Include in room context
- [ ] Persist across turns

---

## Future: Action Grammar

Instead of regex, use a simple grammar:

```
<action> ::= <verb> <target>? <location>? <manner>?
<verb> ::= "jumps on" | "bites" | "points at" | "backs away from"
<target> ::= <agent_name>
<location> ::= "shoulder" | "ankle" | "head" | ...
<manner> ::= "accusingly" | "playfully" | "nervously" | ...
```

Parse with grammar library (lark-parser, pyparsing):
```python
action = parser.parse("jumps on Caity's shoulder playfully")
# → Action(verb='jump_on', target='caity', location='shoulder', manner='playfully')
```

---

*Ordnung muss sein!* 🖖

Physical actions should be STRUCTURED DATA, not just text!
