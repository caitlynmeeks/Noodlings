# Recipe Format

YAML format for Noodling character definitions.

---

## Overview

`recipe.yaml` defines a Noodling's identity: personality, appearance, voice,
baseline affect, and references to their cognitive architecture.

## Location

```
Noodlings/
└── character_name/
    ├── recipe.yaml      # This file
    ├── assembly.yaml    # Cognitive architecture
    └── Radiances/       # Avatar models
```

## Full Schema

```yaml
# Required
name: red
display_name: "Red the Fire Imp"

# Personality (Big Five, 0-1)
personality:
  openness: 0.8           # Curious vs conventional
  conscientiousness: 0.3  # Organized vs spontaneous
  extraversion: 0.9       # Outgoing vs reserved
  agreeableness: 0.6      # Cooperative vs challenging
  neuroticism: 0.4        # Sensitive vs stable

# Baseline affect (where they drift to at rest)
affect_baseline:
  valence: 0.3            # -1 to +1
  arousal: 0.6            # 0 to 1
  dominance: 0.5          # 0 to 1
  boredom: 0.2            # 0 to 1
  sorrow: 0.1             # 0 to 1

# Physical description (for LLM context)
appearance: |
  A small fire imp with flickering orange skin.
  Ember eyes that glow brighter when excited.
  Leaves tiny scorch marks on surfaces.

# Voice and speaking style
voice:
  style: "playful, curious, occasionally chaotic"
  quirks:
    - "Uses fire metaphors liberally"
    - "Gets distracted by shiny things"
    - "Laughs at own jokes"

# Backstory (for LLM context)
backstory: |
  Red emerged from the first spark of creativity.
  Has been exploring the world ever since.

# Cognitive architecture reference
assembly: assembly.yaml

# Avatar (optional)
radiance: Radiances/fire_imp.radiance

# Tags for categorization
tags:
  - elemental
  - playful
  - tutorial
```

## Minimal Recipe

```yaml
name: simple
display_name: Simple Character

personality:
  openness: 0.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5

affect_baseline:
  valence: 0.0
  arousal: 0.5
  dominance: 0.5
  boredom: 0.0
  sorrow: 0.0

appearance: A generic character.

assembly: assembly.yaml
```

## Loading in Code

```python
import yaml

with open("Noodlings/red/recipe.yaml") as f:
    recipe = yaml.safe_load(f)

print(recipe["display_name"])  # "Red the Fire Imp"
```
