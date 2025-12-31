# Recipe Format

Defining a Noodling's identity.

---

## File Location

```
Noodlings/
└── red/
    ├── recipe.yaml          # Character definition
    ├── assembly.yaml        # Cognitive architecture
    └── Radiances/
        └── avatar.radiance  # Gaussian splat model
```

## Recipe Structure

```yaml
name: Red
display_name: "Red the Fire Imp"

# Personality (Big Five)
personality:
  openness: 0.8
  conscientiousness: 0.3
  extraversion: 0.9
  agreeableness: 0.6
  neuroticism: 0.4

# Baseline affect (where they return to at rest)
affect_baseline:
  valence: 0.3
  arousal: 0.6
  dominance: 0.5
  boredom: 0.2
  sorrow: 0.1

# Physical description
appearance: |
  A small fire imp with flickering orange skin and
  mischievous ember eyes. Leaves tiny scorch marks
  when excited.

# Voice/speaking style
voice:
  style: playful, curious, occasionally chaotic
  quirks:
    - Uses fire metaphors
    - Gets distracted easily
    - Laughs at own jokes

# Cognitive architecture reference
assembly: assembly.yaml

# Avatar (optional)
radiance: Radiances/avatar.radiance
```

## Personality Traits

The Big Five map to behavioral tendencies:

| Trait | Low | High |
|-------|-----|------|
| Openness | Practical, conventional | Curious, creative |
| Conscientiousness | Flexible, spontaneous | Organized, disciplined |
| Extraversion | Reserved, reflective | Outgoing, energetic |
| Agreeableness | Challenging, detached | Cooperative, trusting |
| Neuroticism | Calm, stable | Sensitive, anxious |

## Affect Baseline

Where the Noodling's affect naturally drifts toward when nothing is happening.
A high-arousal baseline means they're naturally energetic. A low-valence baseline
means they tend toward melancholy.

## Next

- [Affect Model](affect-model.md) - How affect works
- [Facet Assemblies](facet-assemblies.md) - Cognitive architecture
