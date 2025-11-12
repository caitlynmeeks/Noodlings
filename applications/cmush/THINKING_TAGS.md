# LLM Thinking Tag Support

## Overview

noodleMUSH now automatically detects and extracts thinking tags from LLM responses, treating them as ruminations (private thoughts) just like the agents' internally-generated thoughts.

## Supported Formats

The system detects the following thinking tag formats (case-insensitive):
- `<thinking>...</thinking>`
- `<think>...</think>`

## How It Works

### 1. Automatic Extraction
When an LLM (like Claude, DeepSeek with reasoning, or other thinking-enabled models) responds with thinking tags, the system:

1. **Extracts** the thinking content from within the tags
2. **Removes** the tags from the actual speech output
3. **Stores** the thinking as a rumination in episodic memory
4. **Logs** high-salience thinking to the thoughts/ directory

### 2. Memory Storage

Extracted thinking is stored in episodic memory with:
- `is_rumination: True` flag
- `[thought]` prefix in text field
- Full affect state (valence, arousal, fear, sorrow, boredom)
- Surprise value from phenomenal state
- Identity salience score
- Timestamp

### 3. Example Response Processing

**LLM Output:**
```
<thinking>
She seems genuinely interested in games. I should engage playfully rather than philosophically. This could be a fun interaction.
</thinking>

:bounces excitedly That sounds like so much fun! Should I count first or do you want to?
```

**What Happens:**
1. Thinking extracted: "She seems genuinely interested in games..."
2. Stored as rumination in memory with `is_rumination: True`
3. Clean speech: ":bounces excitedly That sounds like so much fun! Should I count first or do you want to?"
4. If identity_salience > 0.6, logged to thoughts/ directory

### 4. Identity Salience Scoring

Thinking content is scored for identity salience just like speech and ruminations:
- **Surprise component**: Normalized surprise value
- **Self-reference**: Mentions of "I", "my", "myself"
- **Identity keywords**: "who", "what", "why", "feel", "am", "believe"
- High-salience thinking (>0.6) is logged for review

## Benefits

1. **LLM's Internal Reasoning**: Capture the model's actual reasoning process
2. **Richer Memory**: Both public speech and private thoughts stored separately
3. **Identity Formation**: High-salience thinking shapes agent personality
4. **Debugging**: Easy to see what the LLM was thinking vs. what it said
5. **Consistency**: Same memory architecture for LLM thinking and manual ruminations

## Viewing Thinking

### In Chat
Use `@memory <agent_name>` to see both speech and ruminations (marked with `[thought]`)

### In Logs
- Server logs: `INFO:agent_bridge:Agent agent_foo thinking (from LLM): ...`
- High-salience: `world/agents/agent_foo/thoughts/YYYY-MM-DD.txt`

### Via Commands
```
@memory <agent_name>           # View all memories (speech + thoughts)
@observe <agent_name>          # View current phenomenal state
```

## Technical Implementation

### Files Modified

1. **llm_interface.py** (lines 343-444)
   - Added `_extract_thinking_tags()` helper function
   - Modified `_complete()` to return tuple: (clean_text, thinking)
   - Updated `generate_response()` to return dict with 'response' and 'thinking'

2. **agent_bridge.py** (lines 698-750)
   - Updated `_generate_response()` to handle dict return format
   - Extract thinking content from LLM result
   - Store thinking as rumination in conversation_context
   - Log high-salience thinking to thoughts/ directory

## Configuration

No configuration needed! The feature works automatically when:
1. Using an LLM that supports thinking tags
2. LLM outputs thinking tags in its response
3. System automatically detects and processes them

## Backward Compatibility

The system maintains backward compatibility:
- If LLM returns plain text (no dict), treated as response with no thinking
- Existing agents continue to work without modification
- Manual ruminations (via rumination_frequency) still work as before

## Example Use Cases

### Claude with Thinking
When using Claude models with thinking enabled, you'll automatically capture:
- Reasoning about social dynamics
- Decision-making process
- Emotional processing
- Strategic planning

### DeepSeek with Reasoning
DeepSeek's reasoning mode outputs thinking - now captured as ruminations:
- Logical chains of thought
- Hypothesis formation
- Perspective-taking
- Meta-cognitive reflections

### Other Thinking-Enabled Models
Any OpenAI-compatible model that outputs `<thinking>` or `<think>` tags will work.

## Debugging

### Check if Thinking Was Extracted
Look for log line:
```
INFO:llm_interface:Extracted thinking: [first 100 chars]...
```

### Verify Storage
```
@memory <agent_name>
```
Look for entries with `[thought]` prefix and `is_rumination: True`

### High-Salience Thinking
Check `world/agents/<agent_name>/thoughts/YYYY-MM-DD.txt` for logged thinking

## Notes

- Thinking is processed **before** the agent speaks, making it a true "thought before action"
- Multiple thinking blocks in one response are concatenated
- Thinking contributes to identity formation through salience scoring
- Empty thinking blocks are safely ignored
- Clean text has extra whitespace removed after tag extraction

---

**Status**: Implemented and active as of November 2025
**Version**: Phase 5 (Noodlings)
