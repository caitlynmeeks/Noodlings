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

# Affect model - flexible, dimension-agnostic
# NoodleStudio does not hardcode any particular affect model.
# Declare your dimensions; the animation system adapts.
affect:
  model: pad              # Label for reference (pad, vkp, occ, ekman, custom...)
  dimensions:
    - name: valence
      range: [-1, 1]
      baseline: 0.3
      description: "Pleasure/displeasure"
    - name: arousal
      range: [0, 1]
      baseline: 0.6
      description: "Energy/activation"
    - name: dominance
      range: [0, 1]
      baseline: 0.5
      description: "Sense of control"

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

affect:
  model: pad
  dimensions:
    - name: valence
      range: [-1, 1]
      baseline: 0.0
    - name: arousal
      range: [0, 1]
      baseline: 0.5
    - name: dominance
      range: [0, 1]
      baseline: 0.5

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
