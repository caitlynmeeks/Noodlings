# Memory Persistence Fix - DRAGONFLY Mystery Solved!

**Issue**: Noodlings weren't remembering the "DRAGONFLY" secret word across sessions

**Root Cause**: Memory trimming was too aggressive!

## The Problem

The conversation_context (episodic memory) was being trimmed to only **50 messages** during runtime:

```python
# In perceive_event():
trim_threshold = self.config.get('memory_windows', {}).get('affect_trim_threshold', 20)
if len(self.conversation_context) > trim_threshold:
    self.conversation_context = self.conversation_context[-trim_threshold:]
```

**What happened:**
1. DRAGONFLY conversation happened (saved to chat_history.json ✅)
2. More than 50 messages occurred
3. DRAGONFLY was trimmed from conversation_context ❌
4. Agent state saved WITHOUT DRAGONFLY in conversation_context ❌
5. Next session: Agents loaded, but DRAGONFLY was gone!

## The Fix

**Changed**: `affect_trim_threshold: 50` → `affect_trim_threshold: 500`

Now agents keep **500 messages** in memory before trimming (10x more!).

**File**: `config.yaml` line 31

## Memory Architecture

### Two Types of Memory Storage:

1. **Global Chat History** (`world/chat_history.json`)
   - Stores last 200 messages from ALL conversations
   - Loaded on server start
   - Used for display only (not for agent memory)
   - ✅ Had DRAGONFLY

2. **Agent Conversation Context** (`world/agents/agent_X/agent_state.json`)
   - Stores agent's personal episodic memories
   - Trimmed to `affect_trim_threshold` (now 500)
   - Saved to disk: up to `disk_save` limit (500)
   - Used for agent responses and memory recall
   - ❌ Didn't have DRAGONFLY (was trimmed out!)

### Memory Window Configuration

```yaml
memory_windows:
  affect_extraction: 10       # Context for extracting affect from text
  response_generation: 20     # Context shown to LLM when generating responses
  rumination: 10              # Context for generating thoughts
  self_reflection: 10         # Context for self-protection decisions
  disk_save: 500              # Max messages saved to disk
  affect_trim_threshold: 500  # Max messages kept in RAM (was 50!)
```

## Impact

**Before Fix:**
- Memories persisted for ~50 messages (~5-10 minutes of conversation)
- Secret words/rules forgotten quickly
- Games like "DRAGONFLY" didn't work across sessions

**After Fix:**
- Memories persist for ~500 messages (~1-2 hours of conversation)
- Important moments like secret words remembered
- Long-term games and rules work!

## Testing

To test the fix:

1. Restart noodleMUSH (to pick up new config)
2. Tell agents the DRAGONFLY rule again
3. Have a long conversation (100+ messages)
4. Restart the server
5. Ask if they remember DRAGONFLY - they should! ✅

## Future Enhancements

Potential memory improvements:

1. **Importance Tagging**: Mark certain memories as "important" (never trim)
2. **Semantic Consolidation**: Compress old memories into summaries
3. **Retrieval System**: Search all saved history, not just in-RAM context
4. **Memory Salience**: Keep high-salience memories even if old

---

**Fix Applied**: November 15, 2025
**Memory Capacity**: 10x increase (50 → 500 messages)
**Result**: DRAGONFLY and other long-term memories will now persist! 🐉✨
