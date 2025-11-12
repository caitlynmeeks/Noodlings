# cMUSH Memory Integration Testing Checklist

**Date:** October 27, 2025
**Purpose:** Verify hierarchical memory and training data collection work

---

## Test Sequence

### ✅ Pre-flight Checks (DONE)

- [x] Created `training/data/cmush_real/` directory
- [x] Verified TrainingDataCollector imports correctly
- [x] Verified server imports work

### ⬜ Phase 1: Server Start (Next)

**Action:** Start cMUSH server

```bash
./start.sh
```

**Expected:**
- Server starts without errors
- Log shows: "cMUSH server ready!"
- No import errors related to memory systems

**Success criteria:**
- Server running
- Can access http://localhost:8080
- No Python tracebacks in console

### ⬜ Phase 2: Agent Spawn

**Action:** In browser

```
@spawn test_agent
```

**Expected:**
- Agent spawns successfully
- Log shows: "Agent created: agent_test_agent"
- Training collector initialized (check logs)

**Check in logs for:**
```
Training collector: <TrainingDataCollector...>
```

### ⬜ Phase 3: Basic Interaction

**Action:** Have a simple conversation

```
say Hello test_agent!
say How are you today?
say I'm feeling good
```

**Expected:**
- Agent may or may not respond (surprise-based)
- No errors in server logs
- Interactions logged

### ⬜ Phase 4: Memory System Check

**Action:** Check training data

```bash
# In another terminal:
ls ../../training/data/cmush_real/
cat ../../training/data/cmush_real/session_*.jsonl | head -20
```

**Expected:**
- Session file exists: `session_YYYYMMDD_HHMMSS.jsonl`
- File contains JSON entries with:
  - timestamp
  - user_text
  - affect (5-D vector)
  - phenomenal_state (fast/medium/slow)
  - surprise value

**Example entry:**
```json
{
  "timestamp": "2025-10-27T...",
  "agent_id": "agent_test_agent",
  "user_id": "your_username",
  "user_text": "Hello test_agent!",
  "affect": {...},
  "phenomenal_state": {...},
  "surprise": 0.234
}
```

### ⬜ Phase 5: Longer Conversation (Memory Test)

**Action:** Have 20+ turn conversation

```
say I've been feeling stressed lately
say Work has been really difficult
say My boss is being demanding
say I'm worried about my performance
... (continue for 20+ turns)
```

**Expected:**
- Surprise values vary
- Agent responds occasionally (when surprise > threshold)
- Training data file grows

**Check:** After 20 turns, look for consolidation

### ⬜ Phase 6: Verify Memory Persistence

**Action:** Restart server

```bash
# Stop server (Ctrl+C)
./start.sh
# Agent should reload from saved state
```

**Expected:**
- Agent state persists
- Conversation context preserved
- New session file created for new session

---

## What to Watch For

### Good Signs ✓

1. **Server starts cleanly** - No import errors
2. **Agent spawns** - Creates training collector
3. **Data files created** - Check `training/data/cmush_real/`
4. **JSON is valid** - Can parse session files
5. **Surprise varies** - Not always 0 or always 1
6. **Occasional responses** - Agent speaks when surprised

### Warning Signs ⚠️

1. **No training data files** - Collector not initialized
2. **Empty session files** - Logging not working
3. **Flat surprise (always same value)** - Model not processing
4. **Server crashes** - Integration issue

### Critical Issues ✗

1. **Import errors** - Path problems
2. **Can't spawn agent** - Core integration broken
3. **No memory persistence** - Save/load broken

---

## Expected File Structure After Testing

```
training/data/cmush_real/
├── session_20251027_143022.jsonl    # First session
├── session_20251027_145513.jsonl    # After restart
└── ...

applications/cmush/world/agents/
├── agent_test_agent/
│   ├── agent_state.json              # Agent state
│   └── checkpoint.npz                # Model weights (if exists)
```

---

## Quick Verification Commands

### Check if training data exists:
```bash
ls -lh ../../training/data/cmush_real/
```

### Count interactions logged:
```bash
wc -l ../../training/data/cmush_real/session_*.jsonl
```

### View last 5 interactions:
```bash
tail -5 ../../training/data/cmush_real/session_*.jsonl | python3 -m json.tool
```

### Check agent state:
```bash
cat world/agents/agent_test_agent/agent_state.json | python3 -m json.tool
```

---

## If Something Goes Wrong

### Agent won't spawn
- Check logs for errors
- Verify checkpoint path in config.yaml
- Try with simpler config

### No training data
- Check if collector initialized: grep "Training collector" logs/*.log
- Verify directory exists: ls training/data/cmush_real/
- Check config: `collect_training_data` should be true (or not set)

### Server won't start
- Check Python version: `python3 --version` (need 3.8+)
- Check dependencies: `pip3 list | grep -E "(mlx|websockets|yaml)"`
- Review logs/cmush_*.log

### Memory errors
- Check memory capacity settings
- Monitor RAM usage: `top` or `htop`
- Reduce capacities in config if needed

---

## Success Criteria

Integration is successful if:

1. ✅ Server starts without errors
2. ✅ Agent spawns successfully
3. ✅ Training data files are created
4. ✅ Session files contain valid JSON
5. ✅ Agent state persists across restarts
6. ✅ Surprise values vary (not constant)
7. ✅ Agent responds occasionally

---

Ready to test! Start with Phase 1: `./start.sh`
