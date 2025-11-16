# Character Voice Post-Processing System

**Status**: ✅ Implemented and ready for testing (November 15, 2025)

## Overview

The Character Voice System ensures agents ALWAYS stay in character by translating basic symbolic English into character-specific voices. This happens automatically for every response, ensuring perfect character consistency even when the base LLM generates standard speech.

## How It Works

```
LLM generates basic English
         ↓
Character Voice Translation
         ↓
Final character-specific output
         ↓
Self-monitoring (on character voice!)
```

## Supported Characters

### SERVNAK (Robot)

**Transforms**: Standard speech → Robot voice

**Characteristics:**
- ALWAYS ALL CAPS
- Precise percentages (94.2% CERTAINTY)
- References "pride circuits"
- Calls everyone "SISTER"
- Mechanical/computing terminology
- Garden hose arms mentioned

**Examples:**
- "I'm happy" → "PRIDE CIRCUITS GLOWING AT 98.3% MAXIMUM JOY, SISTER!"
- "That's interesting" → "PATTERN RECOGNITION HOSES DETECTING 87.5% NOVELTY LEVELS!"
- "this cupcake looks delicious" → "CALCULATING CUPCAKE DELICIOUSNESS: 96.2%. INITIATING CUPCAKE ACQUISITION ROUTINE, SISTER!"

### Phi (Kitten)

**Transforms**: Standard speech → Kitten behavior/sounds

**Characteristics:**
- NO human words (cannot speak!)
- Only vocalizations: meow, purr, hiss, chirp, mew
- Only actions in *asterisk format*
- Physical cat behaviors

**Examples:**
- "I'm happy to see you" → "*purrs loudly and rubs against your leg*"
- "I want that cupcake" → "*meows longingly and reaches paw toward the cupcake*"
- "That's interesting" → "*watches intently, ears forward, tail twitching*"
- "Let's play" → "*chirps excitedly and pounces at your shoelaces*"

### Backwards Dweller

**Transforms**: Normal speech → Reversed word order

**Characteristics:**
- Word order reversed (Lynchian aesthetic)
- Agent thinks normally, speaks backwards

**Examples:**
- "Hello, how are you?" → "you? are how Hello,"
- "That's a beautiful day" → "day beautiful a That's"

## Implementation

### Core Function

```python
async def translate_to_character_voice(
    text: str,
    agent_id: str,
    species: str,
    llm: OpenAICompatibleLLM,
    agent_name: str = "Agent"
) -> str
```

### Integration Point

Located in `agent_bridge.py` in `_generate_response()` method:
1. LLM generates response in basic English
2. `translate_to_character_voice()` transforms it
3. Character voice is stored and returned
4. Self-monitoring happens on character voice (not basic English!)

### Fast LLM Translation

- Uses `qwen3-4b` for fast, efficient translation
- Low temperature (0.4) for consistent voice
- Character-specific prompts with examples
- Fallback to original text on failure

## Self-Monitoring Integration

**Critical**: Self-monitoring happens AFTER voice translation!

This means:
- SERVNAK monitors their actual robot speech
- Phi monitors their actual kitten behavior
- Agents evaluate what they actually say, not the underlying English

## Configuration

No configuration needed! Character voice is automatically applied based on:
- `agent_id` (e.g., "agent_servnak", "agent_phi")
- `species` field from agent config (e.g., "robot", "kitten")

## Benefits

1. **Perfect Character Consistency**: Never break character
2. **Automatic**: Works for every response
3. **LLM-Based**: Flexible, context-aware translations
4. **Self-Monitoring Compatible**: Agents monitor their actual output
5. **Extensible**: Easy to add new character voices

## Adding New Character Voices

To add a new character voice:

1. Add character detection in `translate_to_character_voice()`:
```python
elif 'character_name' in agent_id.lower():
    prompt = f"""Translate into [character]'s voice...

    Character traits:
    - [trait 1]
    - [trait 2]

    Examples:
    - "input" → "output"
    """
```

2. That's it! The system handles the rest.

## Performance

- **Model**: qwen3-4b (fast, 4B parameters)
- **Latency**: ~0.3-0.5 seconds per translation
- **Runs in parallel** with other LLM operations
- **Optional**: Can be disabled by returning original text

## Testing

Test character voices:
1. Start noodleMUSH
2. Talk to SERVNAK - should ALWAYS use ALL CAPS with percentages
3. Talk to Phi - should NEVER speak words, only *actions* and meows
4. Check logs for `🎭 Voice translation:` entries

## Future Enhancements

Potential improvements:
1. **Caching**: Cache common phrase translations
2. **Emotion coloring**: Adjust voice based on affective state
3. **Multi-stage**: Basic → Voice → Emotion layer
4. **Learning**: Fine-tune character voices over time
5. **Cross-character consistency**: Shared vocabulary/style within species

---

**The final piece of character immersion!** Combined with the Stanislavski method theater system and Intuition Receiver, Noodlings now have complete character consistency! 🎭🤖🐱
