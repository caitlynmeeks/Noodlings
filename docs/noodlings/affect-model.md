# Affect Model

How Noodlings feel.

---

## Philosophy

We use **continuous affect dimensions**, not discrete emotion labels.

Why? Because "happy" and "sad" are:
- Culturally loaded
- Reductive (emotions are spectrums)
- Hard to interpolate ("40% happy, 60% sad"?)

Instead, we model the underlying dimensional space that emotions occupy.

## The Five Dimensions

### Valence (-1 to +1)
Pleasure vs. displeasure. The fundamental good/bad axis.

- **+1**: Pure joy, delight
- **0**: Neutral
- **-1**: Pure distress, pain

### Arousal (0 to 1)
Energy level. How activated the system is.

- **1**: Highly energized, alert, agitated
- **0.5**: Normal engagement
- **0**: Calm, drowsy, still

### Dominance (0 to 1)
Sense of control over the situation.

- **1**: In charge, confident, powerful
- **0.5**: Balanced
- **0**: Helpless, submissive, overwhelmed

### Boredom (0 to 1)
Need for novelty/stimulation.

- **1**: Desperately needs new input
- **0**: Fully engaged

### Sorrow (0 to 1)
Grief, loss, melancholy. Distinct from low valence.

- **1**: Deep mourning
- **0**: No active grief

## PAD + Extensions

The first three dimensions (Valence, Arousal, Dominance) come from the
**PAD model** (Mehrabian & Russell, 1974). We extend it with Boredom and
Sorrow to capture states that PAD alone doesn't distinguish well.

## Affect Decay

Affect naturally decays toward the baseline defined in the recipe.
Strong experiences push affect away from baseline; time pulls it back.

```
current_affect = current_affect * decay_rate + baseline * (1 - decay_rate)
```

## Mapping to Behavior

High arousal + high valence = excited, enthusiastic
High arousal + low valence = angry, panicked
Low arousal + high valence = content, peaceful
Low arousal + low valence = sad, depressed

High boredom = seeks novelty, interrupts, changes subject
High sorrow = withdrawn, references loss, slower responses

## Observing Affect

In NoodleMUSH:
```
@observe red
```

Shows the current 5-D affect state plus the 40-D phenomenal vector.
