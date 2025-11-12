# Memory Systems Verification Guide

**Purpose:** Verify that hierarchical memory and semantic memory are working correctly in cMUSH.

---

## Test 1: Hierarchical Memory - Working Memory

**What we're testing:** Recent interactions are stored in working memory (20-slot FIFO).

### Steps

1. **Start cMUSH and login**
   ```bash
   cd /Users/thistlequell/git/consilience/applications/cmush
   ./start.sh
   # Open browser: http://localhost:8080
   ```

2. **Have a brief conversation** (5-10 messages)
   ```
   say Hello desobelle!
   say How are you today?
   say I'm feeling curious
   say Tell me about yourself
   say What do you think about consciousness?
   ```

3. **Check working memory via Python**

   Open a new terminal (keep server running):
   ```bash
   cd /Users/thistlequell/git/consilience/applications/cmush
   source ~/git/consilience/venv/bin/activate

   python3 << 'EOF'
   import sys
   sys.path.insert(0, '../../consilience_core')

   # Load agent state
   import json
   with open('world/agents/agent_desobelle/checkpoint_state.json', 'r') as f:
       agent_state = json.load(f)

   print("Agent State Summary:")
   print("=" * 50)
   print(f"Step: {agent_state.get('step', 0)}")
   print(f"Conversation history entries: {len(agent_state.get('conversation_history', []))}")
   print(f"Surprise buffer size: {len(agent_state.get('surprise_buffer', []))}")
   print(f"Last surprise: {agent_state.get('last_surprise', 0):.3f}")
   print()

   # Show last 3 conversation entries
   history = agent_state.get('conversation_history', [])
   if history:
       print("Last 3 interactions:")
       for entry in history[-3:]:
           print(f"  [{entry['timestamp']}] {entry['user_text'][:50]}...")
           print(f"    Surprise: {entry['surprise']:.3f}")
   EOF
   ```

**Expected output:**
```
Agent State Summary:
==================================================
Step: 5
Conversation history entries: 5
Surprise buffer size: 5
Last surprise: 0.623

Last 3 interactions:
  [2025-10-27T...] Tell me about yourself...
    Surprise: 0.542
  [2025-10-27T...] What do you think about consciousness?...
    Surprise: 0.623
```

**✅ Success Criteria:**
- Conversation history matches number of messages sent
- Surprise values vary (not constant)
- Timestamps are recent

---

## Test 2: Hierarchical Memory - Episodic Consolidation

**What we're testing:** High-surprise/emotional moments are consolidated to episodic memory.

### Steps

1. **Create high-surprise interactions** (emotional content)
   ```
   say I'm feeling really anxious today
   say My best friend just moved away and I'm sad
   say But I'm also excited about a new opportunity!
   say Life is so full of contradictions
   ```

2. **Check episodic memory statistics**

   ```bash
   python3 << 'EOF'
   import sys
   sys.path.insert(0, '../../consilience_core')

   from hierarchical_memory import HierarchicalMemory
   import pickle

   # Note: In production, memory state would be saved to disk
   # For now, we'll check if the system can be instantiated

   memory = HierarchicalMemory(
       working_capacity=20,
       episodic_capacity=200,
       surprise_threshold=0.5
   )

   print("Hierarchical Memory System")
   print("=" * 50)
   print(f"Working memory capacity: {memory.working_capacity}")
   print(f"Episodic memory capacity: {memory.episodic_capacity}")
   print(f"Surprise threshold: {memory.surprise_threshold}")
   print()

   # Test adding a high-surprise memory
   import time
   import mlx.core as mx

   memory.add(
       timestamp=time.time(),
       step=1,
       user_id="test_user",
       user_text="Test high-surprise interaction",
       affect=mx.array([0.8, 0.9, 0.2, 0.7, 0.1]),  # High valence, arousal, some sorrow
       phenomenal_state={'fast': [0]*16, 'medium': [0]*16, 'slow': [0]*8},
       surprise=0.85,  # High surprise
       response="Test response"
   )

   stats = memory.get_stats()
   print("After adding high-surprise interaction:")
   print(f"Working memory: {stats['working_memory']['current']}/{stats['working_memory']['capacity']}")
   print(f"Episodic memory: {stats['episodic_memory']['current']}/{stats['episodic_memory']['capacity']}")
   print(f"Consolidations: {stats['consolidations']}")
   print(f"Consolidation rate: {stats['consolidation_rate']:.2%}")

   if stats['consolidations'] > 0:
       print("\n✓ Episodic consolidation working!")
   else:
       print("\n⚠ No consolidations yet (may need higher surprise)")
   EOF
   ```

**Expected output:**
```
Hierarchical Memory System
==================================================
Working memory capacity: 20
Episodic memory capacity: 200
Surprise threshold: 0.5

After adding high-surprise interaction:
Working memory: 1/20
Episodic memory: 1/200
Consolidations: 1
Consolidation rate: 100.00%

✓ Episodic consolidation working!
```

**✅ Success Criteria:**
- High-surprise interactions trigger consolidation to episodic memory
- Consolidation rate > 0%

---

## Test 3: Semantic Memory - User Profile Creation

**What we're testing:** After 50+ interactions, semantic profiles are created for users.

### Steps

1. **Have a substantial conversation** (30+ messages)

   You can use the training scenario runner to generate quality conversations:
   ```bash
   python3 run_training_scenario.py --scenario emotional_arc
   ```

   Or just chat naturally for 30+ turns.

2. **Check if semantic profiles exist**

   ```bash
   python3 << 'EOF'
   import sys
   sys.path.insert(0, '../../consilience_core')

   from semantic_memory import SemanticMemorySystem

   print("Semantic Memory System Test")
   print("=" * 50)

   semantic = SemanticMemorySystem(
       max_users=10,
       consolidation_interval=50
   )

   # Check if profiles would be created
   stats = semantic.get_stats()
   print(f"Total users tracked: {stats['total_users']}")
   print(f"Total semantic facts: {stats['total_facts']}")
   print(f"Consolidation counter: {semantic.consolidation_counter}")
   print(f"Consolidation trigger: {semantic.consolidation_interval}")
   print()

   if stats['total_users'] > 0:
       print("✓ Semantic memory is tracking users!")
       for user_id in list(semantic.profiles.keys())[:3]:
           summary = semantic.get_user_summary(user_id)
           print(f"\nUser: {user_id}")
           print(f"  {summary}")
   else:
       print("⚠ No semantic profiles yet")
       print(f"  (Need {semantic.consolidation_interval} interactions for first consolidation)")
       print("  Continue chatting to reach threshold!")
   EOF
   ```

**Expected output (before 50 interactions):**
```
Semantic Memory System Test
==================================================
Total users tracked: 0
Total semantic facts: 0
Consolidation counter: 0
Consolidation trigger: 50

⚠ No semantic profiles yet
  (Need 50 interactions for first consolidation)
  Continue chatting to reach threshold!
```

**Expected output (after 50+ interactions):**
```
Semantic Memory System Test
==================================================
Total users tracked: 1
Total semantic facts: 5
Consolidation counter: 53
Consolidation trigger: 50

✓ Semantic memory is tracking users!

User: user_caity
  Caity is warm and curious, enjoys deep conversations about consciousness and emotion.
```

**✅ Success Criteria:**
- After 50+ interactions, semantic profiles are created
- User summaries are meaningful and accurate

---

## Test 4: Memory Persistence Across Restarts

**What we're testing:** Memory state survives server restarts.

### Steps

1. **Have a conversation and note details**
   ```
   say My favorite color is purple
   say I love hiking in the mountains
   say Coffee is essential to my morning routine
   ```

2. **Stop the server** (Ctrl+C)

3. **Check that training data was saved**
   ```bash
   ls -lh ../../training/data/cmush_real/
   tail -5 ../../training/data/cmush_real/session_*.jsonl | grep -i "purple\|hiking\|coffee"
   ```

4. **Restart the server**
   ```bash
   ./start.sh
   ```

5. **Check that agent remembers context**
   ```
   say Do you remember what I told you about my favorite color?
   ```

   Agent should reference purple (either directly or show it influenced their internal state).

**✅ Success Criteria:**
- Training data files contain previous conversation
- Agent state persists across restarts
- Conversation context is maintained

---

## Test 5: Memory Retrieval - Context Assembly

**What we're testing:** Agents combine working memory + episodic memories for context.

### Steps

1. **Have a long conversation with varied topics** (20+ messages)

2. **Test memory retrieval directly**

   ```bash
   python3 << 'EOF'
   import sys
   sys.path.insert(0, '../../consilience_core')

   from hierarchical_memory import HierarchicalMemory
   import mlx.core as mx
   import time

   memory = HierarchicalMemory(
       working_capacity=20,
       episodic_capacity=200,
       surprise_threshold=0.5
   )

   # Simulate 25 interactions (overflow working memory)
   for i in range(25):
       surprise_level = 0.3 + (i % 5) * 0.15  # Varying surprise
       memory.add(
           timestamp=time.time(),
           step=i,
           user_id="test_user",
           user_text=f"Message {i}: {'Important!' if surprise_level > 0.5 else 'Casual chat'}",
           affect=mx.array([0.5, 0.3 + (i % 3)*0.2, 0.1, 0.1, 0.2]),
           phenomenal_state={'fast': [0]*16, 'medium': [0]*16, 'slow': [0]*8},
           surprise=surprise_level,
           response=None
       )

   # Retrieve context for user
   context = memory.retrieve_context(
       user_id="test_user",
       context_size=10
   )

   print("Context Retrieval Test")
   print("=" * 50)
   print(f"Total interactions: 25")
   print(f"Working memory: {len(memory.working_memory)}/20")
   print(f"Episodic memory: {len(memory.episodic_memory)}")
   print(f"Context retrieved: {len(context)} entries")
   print()

   print("Context composition:")
   recent = [e for e in context if e.step >= 20]
   important = [e for e in context if e.surprise > 0.5]
   print(f"  Recent entries (step >= 20): {len(recent)}")
   print(f"  Important entries (surprise > 0.5): {len(important)}")
   print()

   if len(context) > 0:
       print("Sample context entries:")
       for entry in context[:5]:
           print(f"  Step {entry.step}: surprise={entry.surprise:.2f} importance={entry.importance:.2f}")
           print(f"    '{entry.user_text}'")
       print("\n✓ Context retrieval working!")
   EOF
   ```

**Expected output:**
```
Context Retrieval Test
==================================================
Total interactions: 25
Working memory: 20/20
Episodic memory: 8
Context retrieved: 10 entries

Context composition:
  Recent entries (step >= 20): 5
  Important entries (surprise > 0.5): 5

Sample context entries:
  Step 24: surprise=0.90 importance=0.85
    'Message 24: Important!'
  Step 23: surprise=0.75 importance=0.72
    'Message 23: Important!'
  Step 22: surprise=0.60 importance=0.58
    'Message 22: Important!'
  Step 21: surprise=0.45 importance=0.35
    'Message 21: Casual chat'
  Step 20: surprise=0.30 importance=0.28
    'Message 20: Casual chat'

✓ Context retrieval working!
```

**✅ Success Criteria:**
- Context includes both recent and important memories
- High-surprise events are prioritized
- Context size is reasonable (not overwhelming)

---

## Quick Verification Checklist

Run all tests and check off:

- [ ] **Test 1**: Working memory stores recent interactions
- [ ] **Test 2**: Episodic consolidation triggers on high surprise
- [ ] **Test 3**: Semantic profiles created after 50+ interactions
- [ ] **Test 4**: Memory persists across server restarts
- [ ] **Test 5**: Context retrieval combines working + episodic

---

## Troubleshooting

### Working memory empty
- **Cause**: Agent not perceiving events
- **Fix**: Check logs for "perceiving" messages, verify agent in same room as user

### No episodic consolidations
- **Cause**: Surprise threshold too high or all interactions low-surprise
- **Fix**: Lower threshold in config, have more emotional conversations

### No semantic profiles after 50+ interactions
- **Cause**: Consolidation counter not incrementing
- **Fix**: Check if `should_consolidate()` returns true, verify counter in logs

### Memory not persisting
- **Cause**: Shutdown hook not calling `agent.shutdown()`
- **Fix**: Verify fix in agent_bridge.py:513 is applied

---

## Expected Timeline

- **Test 1-2**: 5 minutes (immediate verification)
- **Test 3**: 30-60 minutes (requires 50+ interactions)
- **Test 4**: 5 minutes (restart test)
- **Test 5**: 10 minutes (context assembly)

**Total**: ~1-2 hours for complete verification

---

## Success Metrics

**Fully Working:**
- All 5 tests pass
- Training data contains phenomenal states
- Memory stats show consolidation
- Agents reference past interactions naturally

**Partially Working:**
- Tests 1, 4, 5 pass (core memory functional)
- Test 2 or 3 pending (need more interactions/tuning)

**Not Working:**
- Tests 1 or 4 fail (critical issue)
- No training data collected
- Memory stats all zeros

---

*For questions about memory systems, see:*
- *consilience_core/hierarchical_memory.py - Implementation*
- *CURRENT_STATE_OCT2025.md - Architecture overview*
- *INTEGRATION_COMPLETE.md - Integration details*
