# Quick Test: Training Data Persistence Fix

**What was fixed:** Added `agent.shutdown()` call in `save_all_agents()` so training data gets written to disk on server shutdown.

---

## Test Steps

### 1. Stop Current Server
```bash
# In the server terminal, press Ctrl+C
```

### 2. Check If Data Was Written
```bash
ls -lh ../../training/data/cmush_real/
```

**Expected:** You should now see a session file like `session_20251027_HHMMSS.jsonl`

### 3. Restart Server
```bash
./start.sh
```

### 4. Have a Brief Conversation

In browser (http://localhost:8080):
```
say Hello desobelle!
say How are you today?
say I'm feeling great
say Tell me about yourself
say Thanks for chatting
```

### 5. Stop Server Again
```bash
# Press Ctrl+C in server terminal
```

### 6. Verify New Session File
```bash
ls -lh ../../training/data/cmush_real/
```

**Expected:** Should see TWO session files now:
- `session_20251027_HHMMSS.jsonl` (from before)
- `session_20251027_HHMMSS.jsonl` (new one, slightly later timestamp)

### 7. View the Data
```bash
# View last 10 lines of latest session
tail -10 ../../training/data/cmush_real/session_*.jsonl | python3 -m json.tool
```

**Expected:** Should see JSON entries with:
- `timestamp`
- `agent_id`
- `user_id`
- `user_text`
- `affect` (5-D vector: valence, arousal, fear, sorrow, boredom)
- `phenomenal_state` (fast/medium/slow layer states)
- `surprise` value

---

## Success Criteria

✅ **Fixed** if:
1. Session files appear in `training/data/cmush_real/` after server shutdown
2. Files contain valid JSON
3. Each line has complete interaction data

❌ **Still broken** if:
1. Directory still empty after shutdown
2. Files exist but are empty
3. JSON is malformed

---

## What This Data Is For

This training data captures:
- User inputs and agent responses
- Full phenomenal state at each moment (40-D: fast+medium+slow layers)
- Affective context (5-D emotional vector)
- Surprise/prediction error values

It will be used to train the Phase 4 model with real conversation data using the script:
```bash
cd ../../training/scripts
python3 05_train_on_cmush_data.py --data-dir ../data/cmush_real
```

---

## If Something Goes Wrong

**Empty directory after shutdown:**
- Check server logs for errors during shutdown
- Verify `agent.shutdown()` is being called (should see log message)

**JSON parse errors:**
- Check if MLX arrays are being converted properly
- Verify all data types are JSON-serializable

**Missing fields in JSON:**
- Check if `log_interaction()` is being called in agent_bridge.py
- Verify all required fields are passed to collector
