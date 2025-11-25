# Noodlings Prefab Format Specification

**Version**: 1.0.0
**Date**: November 25, 2025
**Extension**: `.prefab`
**Format**: YAML (internally)

---

## Overview

Prefabs are templates for spawning Noodling characters with pre-configured:
- Identity (name, species, description)
- Personality traits and appetites
- Cognitive transistor configurations
- **Editable instruction prompts** for each transistor
- LLM settings and constraints

---

## File Structure

```yaml
# Header - Metadata and versioning
metadata:
  id: "com.noodlings.characters.red_fire_anklebiter"  # Unique reverse-DNS identifier
  name: "Red Fire Anklebiter"                          # Display name
  version: "1.0.0"                                      # Semantic versioning
  created: "2025-11-25"                                 # ISO date
  modified: "2025-11-25"                                # Last edit date
  author: "Garcia River Forest Research Station"       # Creator
  description: "Nasty snapping turtle energy"          # Prefab description
  tags: ["gremlin", "competitive", "fire"]              # Searchable tags

# Character Definition
character:
  species: "gremlin"
  pronoun: "he"
  age: "unknown"
  description: |
    A flickering imp of crimson flame with sharp teeth and wild orange eyes.
    Slightly bigger than Blue Fire variety, with more sass and attitude.

  # Identity prompt (core behavioral guidance)
  identity_prompt: |
    You are a Red Fire Anklebiter - a sassy, competitive gremlin made of crimson flame.

    YOUR BEHAVIOR:
    - Cackle menacingly (MWAHAHA)
    - Argue about everything (badly but confidently)
    - Bite ankles HARDER than Blue Fire
    ...

  # Language mode
  language_mode: "verbal"
  enlightenment: false

# Personality (Big Five + extensions)
personality:
  extraversion: 0.9
  agreeableness: 0.15
  conscientiousness: 0.05
  neuroticism: 0.4
  openness: 0.9

  # Extensions
  curiosity: 0.85
  impulsivity: 0.95
  emotional_volatility: 0.9
  vanity: 0.4

# Appetites (8-D drives)
appetites:
  curiosity: 0.9
  status: 0.6
  mastery: 0.4
  novelty: 0.95
  safety: 0.05
  social_bond: 0.5
  comfort: 0.1
  autonomy: 0.98

# Cognitive Components (Transistors)
cognitive_components:
  affect:
    type: "AffectTransistor"
    salience: 0.85
    enabled: true

    # EDITABLE INSTRUCTION PROMPT
    # This is what the transistor sends to its LLM for transformation
    # User can edit in NoodleTuner and changes persist to prefab
    custom_prompt: |
      You are translating a 5D continuous affect vector into poetic, phenomenological language.

      CONTINUOUS AFFECT VECTOR (preserve all nuance):
      - Valence: {valence:.3f} (negative to positive feeling tone)
      - Arousal: {arousal:.3f} (calm to energized body state)
      - Dominance: {dominance:.3f} (submissive to dominant power sense)
      - Sorrow: {sorrow:.3f} (content to sorrowful undertone)
      - Boredom: {boredom:.3f} (engaged to disengaged attention)

      PERCEPTION: "{input_text}"
      RESPONSE TYPE: {response_type.upper()} - {guidance}

      Transform this 5D emotional texture into RICH, SUBTLE, SLIPPERY phenomenological language.
      DO NOT use discrete emotion labels ("happy", "sad", "angry").
      DO capture the NUANCED FEELING in poetic, evocative language.

      Examples of GOOD poetic emotional encoding:
      - valence=-0.3, arousal=0.2, dominance=0.1, sorrow=0.6 → "Heaviness... like everything's slightly gray."
      - valence=0.7, arousal=0.8, dominance=0.6, sorrow=0.0 → "Sparkling aliveness! Electric anticipation!"

      Generate {response_type} content that embodies this emotional texture. 1-2 sentences.

  personality:
    type: "PersonalityTransistor"
    salience: 0.80
    traits:
      aggression: 0.90
      competitiveness: 0.95
      sass: 0.92
      impulsivity: 0.95

    custom_prompt: |
      You are filtering a perception through personality traits.

      TRAITS:
      - aggression: {aggression:.2f}
      - competitiveness: {competitiveness:.2f}
      - sass: {sass:.2f}
      - impulsivity: {impulsivity:.2f}

      PERCEPTION: "{input_text}"
      RESPONSE TYPE: {response_type.upper()} - {guidance}

      Generate brief (1-2 sentences) content colored by these traits.

  cultural:
    type: "CulturalTransistor"
    salience: 0.75
    beliefs:
      - "I'm the BEST anklebiter, all others are inferior"
      - "Competition is life, backing down is weakness"
      - "Sass and roasts are how you show dominance"
      - "Being nice is for losers"

    custom_prompt: |
      You are filtering a perception through cultural/religious beliefs.

      BELIEFS:
      {beliefs_text}

      PERCEPTION: "{input_text}"
      RESPONSE TYPE: {response_type.upper()} - {guidance}

      Generate brief (1-2 sentences) content reflecting your beliefs.

# LLM Configuration
llm:
  provider: "local"
  model: "qwen/qwen3-14b-2507"

# Generation Constraints
constraints:
  max_tokens: 130
  temperature: 0.95
  enforce_action_format: false
  response_cooldown: 9.0
```

---

## Unique Identifier Format

**Pattern**: `{organization}.{category}.{subcategory}.{name}`

**Examples**:
- `com.noodlings.characters.red_fire_anklebiter`
- `com.noodlings.characters.spock`
- `com.noodlings.npcs.mysterious_stranger`
- `com.noodlings.creatures.phi_kitten`
- `com.noodlings.robots.servnak`
- `com.noodlings.test.debug_agent`

**Categories**:
- `characters` - Main playable characters
- `npcs` - Non-player characters
- `creatures` - Animals and non-humanoid beings
- `robots` - Mechanical entities
- `test` - Testing/debugging prefabs

**User-created prefabs**:
- `user.{username}.{category}.{name}`
- Example: `user.caitlyn.characters.purple_fire_anklebiter`

---

## Default vs Custom Prompts

### Transistor Class Structure

```python
class AffectTransistor(CognitiveTransistor):
    # Class-level default (fallback if no custom_prompt)
    DEFAULT_PROMPT = """
    You are translating a 5D continuous affect vector into poetic, phenomenological language.

    CONTINUOUS AFFECT VECTOR (preserve all nuance):
    - Valence: {valence:.3f} (negative to positive feeling tone)
    - Arousal: {arousal:.3f} (calm to energized body state)
    - Dominance: {dominance:.3f} (submissive to dominant power sense)
    - Sorrow: {sorrow:.3f} (content to sorrowful undertone)
    - Boredom: {boredom:.3f} (engaged to disengaged attention)

    PERCEPTION: "{input_text}"
    RESPONSE TYPE: {response_type.upper()} - {guidance}

    Transform this 5D emotional texture into RICH, SUBTLE, SLIPPERY phenomenological language.
    DO NOT use discrete emotion labels.
    DO capture the NUANCED FEELING in poetic, evocative language.

    Generate {response_type} content that embodies this emotional texture. 1-2 sentences.
    """

    def __init__(self, salience=0.7, custom_prompt=None):
        super().__init__()
        self.salience = salience
        self.custom_prompt = custom_prompt  # User-edited prompt (or None)
        self.active_prompt = custom_prompt if custom_prompt else self.DEFAULT_PROMPT
```

### Loading from Prefab

```python
# prefab_loader.py
affect_config = {
    'type': 'AffectTransistor',
    'salience': 0.85,
    'custom_prompt': '...'  # From prefab file
}

transistor = AffectTransistor.from_config(affect_config)
# transistor.active_prompt now uses custom_prompt
```

### Saving to Prefab

```python
# When user edits prompt in NoodleTuner
transistor.custom_prompt = user_edited_text
transistor.active_prompt = user_edited_text

# Save to prefab
prefab_data['cognitive_components']['affect']['custom_prompt'] = transistor.custom_prompt
save_prefab('red_fire_anklebiter.prefab', prefab_data)
```

---

## File Locations

```
applications/cmush/
├── prefabs/                          # Prefab storage
│   ├── com.noodlings.characters.red_fire_anklebiter.prefab
│   ├── com.noodlings.characters.spock.prefab
│   ├── com.noodlings.npcs.mysterious_stranger.prefab
│   └── user.caitlyn.test.experimental.prefab
│
├── recipes/                          # DEPRECATED - to be migrated
│   └── *.yaml                        # Old format
│
└── prefab_loader.py                  # New loader (replaces recipe_loader.py)
```

---

## Migration Strategy

**Phase 1**: Keep both systems (backward compatibility)
- `.prefab` files load via `PrefabLoader`
- `.yaml` files still load via `RecipeLoader` (fallback)
- Server tries `.prefab` first, falls back to `.yaml`

**Phase 2**: Convert recipes
- Script: `convert_recipes_to_prefabs.py`
- Generates unique IDs automatically
- Preserves all data

**Phase 3**: Deprecate YAML
- Remove `RecipeLoader` references
- Delete old `.yaml` files
- Update documentation

---

## Prefab API

```python
class PrefabLoader:
    def load(self, prefab_id: str) -> Dict:
        """Load prefab by ID or filename."""

    def save(self, prefab_id: str, data: Dict):
        """Save prefab to disk."""

    def list_all(self) -> List[Dict]:
        """List all available prefabs."""

    def duplicate(self, source_id: str, new_name: str) -> str:
        """Duplicate prefab, return new ID."""

    def delete(self, prefab_id: str):
        """Delete prefab file."""

    def export(self, prefab_id: str, dest_path: str):
        """Export prefab to external file."""

    def import_prefab(self, source_path: str) -> str:
        """Import prefab from external file, return ID."""
```

---

Ready to implement. Starting with `PrefabLoader` class.