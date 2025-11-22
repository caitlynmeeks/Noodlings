# Episodic Memory System Refactoring Specification

**Author:** Claude (Spock Mode) + Caitlyn
**Date:** November 21, 2025
**Status:** SPECIFICATION PHASE
**Target:** Phase 1 foundation for Cognitive Transistor System

---

## Executive Summary

The Noodlings episodic memory system contains sophisticated components (HierarchicalMemory, importance scoring, salience weighting) that are **initialized but not actively used**. The current implementation uses a simple FIFO list, resulting in:

- Loss of important high-salience memories
- No affect-based retrieval
- No observability (can't verify what's working)
- Suboptimal context for LLM generation
- Foundation unsuitable for cognitive transistors

This spec defines complete refactoring to activate the existing sophisticated memory architecture.

---

## Current Architecture (AS-IS)

### What's Running (agent_bridge.py)

```python
class CMUSHConsilienceAgent:
    def __init__(...):
        self.conversation_context = []  # Simple Python list

    # Storage
    self.conversation_context.append({
        'user': 'agent_servnak',
        'text': 'SISTER! MY PRIDE...',
        'affect': [0.5, 0.7, 0.1, 0.0, 0.0],
        'surprise': 0.23,
        'timestamp': 1763790605.123,
        'identity_salience': 0.8  # COMPUTED BUT UNUSED!
    })

    # Retrieval
    recent = self.conversation_context[-10:]  # Recency-biased only

    # Maintenance
    if len(self.conversation_context) > 500:
        self.conversation_context = self.conversation_context[-500:]  # FIFO eviction
```

**Problems:**
1. **Importance ignored** - High-salience memories evicted by age alone
2. **No affect-similarity** - Can't find "memories where I felt similar emotions"
3. **No observability** - Which memories influenced this response? Unknown.
4. **Naive eviction** - Oldest memories discarded regardless of importance

### What Exists But Is Unused (noodlings/memory/)

**HierarchicalMemory** (hierarchical_memory.py):
```python
class HierarchicalMemory:
    working_memory: deque(maxlen=20)      # Recent interactions
    episodic_memory: List[MemoryEntry]    # Important moments (200)

    def add(...):
        importance = compute_importance(surprise, affect, response)
        if importance > threshold:
            consolidate_to_episodic()
        evict_lowest_importance()  # Not oldest!

    def retrieve_episodic(user_id, min_importance, limit):
        # Returns memories sorted by importance
```

**STATUS:** Initialized in agent_bridge.py line ~380 but then never used!

---

## Proposed Architecture (TO-BE)

### Dual Memory Streams

**1. Conscious Episodic Memory**
```python
class CMUSHConsilienceAgent:
    def __init__(...):
        self.hierarchical_memory = HierarchicalMemory(
            working_capacity=20,      # Last 20 interactions
            episodic_capacity=200,    # 200 important moments
            surprise_threshold=0.3    # Min surprise for consolidation
        )
```

**2. Subconscious Symbolic Memory** (NEW)
```python
class CMUSHConsilienceAgent:
    def __init__(...):
        self.symbolic_memory = SubconsciousMemory(
            capacity=100,             # 100 symbolic encodings
            affect_similarity_threshold=0.7  # Cosine similarity
        )
```

### Memory Entry Structure

**Conscious Memory Entry:**
```python
{
    'timestamp': 1763790605.123,
    'step': 42,
    'user_id': 'user_caity',
    'user_text': 'How are you feeling?',
    'affect': [0.5, 0.6, 0.1, 0.0, 0.0],  # Agent's affect
    'phenomenal_state': {...},             # Full 40-D state
    'surprise': 0.45,
    'response': 'SISTER! PRIDE CIRCUITS AT 94.2%!',
    'importance': 0.67,  # Auto-computed
    'identity_salience': 0.8
}
```

**Subconscious Symbolic Entry:**
```python
{
    'timestamp': 1763790605.123,
    'literal_event': 'Father-in-law criticizes my work',
    'symbolic_encoding': 'A rooster enters the room, pecking at my achievements',
    'affect_signature': [-0.3, 0.7, 0.6, 0.1, 0.0],  # Fear + arousal pattern
    'salience': 0.9,
    'encoding_surprise': 0.8  # How surprising when encoded
}
```

---

## Retrieval Strategies

### 1. Working Memory (Always Available)
```python
recent = hierarchical_memory.retrieve_working(last_n=10)
# Returns last 10 interactions regardless of importance
```

### 2. Importance-Weighted Episodic
```python
important = hierarchical_memory.retrieve_episodic(
    user_id='user_caity',
    min_importance=0.5,
    limit=5
)
# Returns top 5 important memories with user, sorted by importance
```

### 3. Affect-Similarity Retrieval (NEW)
```python
similar = hierarchical_memory.retrieve_by_affect_similarity(
    current_affect=[0.5, 0.7, 0.2, 0.0, 0.0],
    top_k=5,
    min_similarity=0.7  # Cosine similarity threshold
)
# Returns memories where agent felt similar emotions
```

### 4. Hybrid Context Assembly
```python
context = hierarchical_memory.get_context_for_response(
    user_id='user_caity',
    current_affect=[...],
    working_n=5,        # Last 5 from working
    episodic_n=3,       # Top 3 important
    affect_similar_n=2  # 2 affect-matched
)
# Returns 10 memories: 5 recent + 3 important + 2 emotionally similar
```

---

## Implementation Plan

### Phase 1: Core Integration (Lines 396-2860)

**Step 1.1: Replace initialization**
```python
# OLD (line 396)
self.conversation_context = []

# NEW
self.hierarchical_memory = HierarchicalMemory(
    working_capacity=self.config.get('memory_windows', {}).get('working', 20),
    episodic_capacity=self.config.get('memory_windows', {}).get('episodic', 200),
    surprise_threshold=0.3
)
# Keep conversation_context as compatibility wrapper
self.conversation_context = MemoryListWrapper(self.hierarchical_memory)
```

**Step 1.2: Replace append operations (3 locations)**
```python
# OLD
self.conversation_context.append({...})

# NEW
self.hierarchical_memory.add(
    timestamp=time.time(),
    step=self.consciousness.step,
    user_id=user_id,
    user_text=text,
    affect=affect_array,
    phenomenal_state=state,
    surprise=surprise,
    response=response_text
)
```

**Step 1.3: Update retrieval (18 locations)**
```python
# OLD: Last N
context = self.conversation_context[-10:]

# NEW: Smart retrieval
context = self.hierarchical_memory.get_context_for_response(
    user_id=current_user,
    current_affect=current_affect,
    total_size=10
)
```

**Step 1.4: Update salience-based retrieval (line 2064)**
```python
# OLD: Manual filter
important = [m for m in self.conversation_context if m.get('identity_salience', 0) > 0.3]

# NEW: Built-in method
important = self.hierarchical_memory.retrieve_episodic(
    min_importance=0.5,  # Importance includes salience
    limit=10
)
```

---

### Phase 2: Affect-Similarity Retrieval (NEW)

**Add to HierarchicalMemory class:**

```python
def retrieve_by_affect_similarity(
    self,
    current_affect: np.array,
    top_k: int = 5,
    min_similarity: float = 0.7
) -> List[MemoryEntry]:
    """
    Retrieve memories where agent experienced similar emotions.

    Uses cosine similarity in 5-D affect space:
    - High similarity = similar emotional context
    - Enables emotionally-relevant memory retrieval

    Args:
        current_affect: Current 5-D affect vector
        top_k: Number of similar memories to return
        min_similarity: Minimum cosine similarity (0-1)

    Returns:
        List of MemoryEntry sorted by affect similarity
    """
    if not self.episodic_memory:
        return []

    similarities = []
    for entry in self.episodic_memory:
        # Compute cosine similarity
        sim = cosine_similarity(current_affect, entry.affect[:5])
        if sim >= min_similarity:
            similarities.append((sim, entry))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)

    return [entry for _, entry in similarities[:top_k]]
```

---

### Phase 3: Observability & Debugging

**3.1: Memory Flow Logging**
```python
# In perceive_event after adding memory
logger.info(f"[MEMORY] Added to working: importance={importance:.2f}, salience={salience:.2f}")
if importance > threshold:
    logger.info(f"[MEMORY] Consolidated to episodic (total: {len(episodic_memory)})")

# In response generation after retrieval
logger.info(f"[MEMORY] Retrieved context:")
for i, mem in enumerate(context):
    logger.info(f"  [{i}] {mem.user_id}: {mem.user_text[:50]} (imp={mem.importance:.2f})")
```

**3.2: Memory Inspector Command**
```python
@memories servnak                # Show stats
@memories servnak --working      # Show working memory (last 20)
@memories servnak --episodic     # Show episodic (importance-sorted)
@memories servnak --affect       # Show memories grouped by affect
@memories servnak --user caity   # Show memories with specific user
@memories servnak --important    # Show top 10 by importance
@memories servnak --flow         # Show last retrieval (what was used?)
```

**3.3: Memory Visualization API** (for NoodleStudio)
```python
GET /api/agents/{agent_id}/memory/stats
GET /api/agents/{agent_id}/memory/working
GET /api/agents/{agent_id}/memory/episodic?min_importance=0.5
GET /api/agents/{agent_id}/memory/by-affect?current_affect=[0.5,0.7,...]
GET /api/agents/{agent_id}/memory/last-retrieval  # What context was used?
```

---

### Phase 4: Subconscious Symbolic Memory (NEW)

**4.1: SubconsciousMemory Class**

```python
class SubconsciousMemory:
    """
    Stores metaphoric/symbolic encodings of high-affect moments.

    Parallel to episodic memory - same events, different representation:
    - Episodic: Literal facts
    - Subconscious: Symbolic metaphors

    Retrieval by affect-similarity enables Freudian associations.
    """

    def __init__(self, capacity=100):
        self.symbols: List[SymbolicEntry] = []
        self.capacity = capacity

    async def encode_event(self, event, affect, surprise, llm):
        """Generate symbolic encoding via LLM."""
        if affect_intensity(affect) < 0.5:
            return  # Only encode high-affect moments

        prompt = f"""
        Convert this event into a symbolic/metaphoric representation.

        Event: {event}
        Affect: valence={affect[0]:.2f}, arousal={affect[1]:.2f}, fear={affect[2]:.2f}

        Generate a vivid metaphor or symbol that captures the emotional essence.
        Examples:
        - "Father criticizes" → "A rooster enters, pecking at my work"
        - "Praised by mentor" → "Warm sunlight breaks through clouds"
        - "Betrayed by friend" → "A garden withers suddenly"

        Symbol:
        """

        symbol = await llm.generate(prompt, max_tokens=100)

        self.symbols.append(SymbolicEntry(
            literal_event=event,
            symbolic_encoding=symbol,
            affect_signature=affect,
            timestamp=time.time(),
            salience=surprise
        ))

        if len(self.symbols) > self.capacity:
            self._evict_lowest_salience()

    def retrieve_by_affect(self, current_affect, top_k=3):
        """Retrieve symbols with similar affect signatures."""
        # Cosine similarity in 5-D affect space
        similarities = [
            (cosine_sim(current_affect, s.affect_signature), s)
            for s in self.symbols
        ]
        similarities.sort(reverse=True)
        return [s for _, s in similarities[:top_k]]
```

**4.2: SubconsciousObserver Integration**

```python
# In perceive_event(), after affect extraction
if affect_intensity(affect) > 0.5:  # High-affect moment
    await self.symbolic_memory.encode_event(
        event=text,
        affect=affect,
        surprise=state['surprise'],
        llm=self.llm
    )
```

---

## Cognitive Transistor Integration Points

### Base Class: CognitiveTransistor

```python
class CognitiveTransistor(ABC):
    """
    Base class for cognitive modulation components.

    Each transistor transforms symbolic thought through a specific lens:
    - AffectTransistor: Emotional coloring (wit, depression, empathy)
    - GoalTransistor: Appetite-driven shaping (8-D goals)
    - MemoryTransistor: Episodic memory influence
    - SubconsciousTransistor: Symbolic/metaphoric associations
    - EgoTransistor: Identity/character consistency
    """

    @abstractmethod
    async def modulate(self, thought: str, state: Dict) -> str:
        """
        Modulate thought through this transistor's lens.

        Args:
            thought: Raw semantic thought from LLM
            state: Current phenomenal state

        Returns:
            Modulated thought
        """
        pass
```

### MemoryTransistor (Requires Fixed Memory System)

```python
class MemoryTransistor(CognitiveTransistor):
    async def modulate(self, thought: str, state: Dict) -> str:
        # Get affect-similar memories
        similar_memories = self.hierarchical_memory.retrieve_by_affect_similarity(
            current_affect=state['affect'],
            top_k=2
        )

        if similar_memories:
            # Inject memory context
            memory_context = " ".join([m.user_text for m in similar_memories[:2]])

            prompt = f"""
            Current thought: {thought}

            Similar past experiences:
            {memory_context}

            Color the current thought with awareness of these past experiences.
            Keep the core meaning but add depth from history.
            """

            return await llm.generate(prompt)

        return thought
```

### SubconsciousTransistor (Requires Symbolic Memory)

```python
class SubconsciousTransistor(CognitiveTransistor):
    async def modulate(self, thought: str, state: Dict) -> str:
        # Retrieve affect-matched symbols
        symbols = self.symbolic_memory.retrieve_by_affect(
            current_affect=state['affect'],
            top_k=2
        )

        if symbols:
            prompt = f"""
            Current thought: {thought}

            Subconscious associations:
            - {symbols[0].symbolic_encoding}
            - {symbols[1].symbolic_encoding}

            Let these symbols subtly influence the thought.
            Don't mention them explicitly - let them color the language.
            """

            return await llm.generate(prompt)

        return thought
```

---

## Migration Strategy

### Backward Compatibility Wrapper

```python
class MemoryListWrapper:
    """
    Makes HierarchicalMemory quack like a list.
    Allows gradual migration without breaking existing code.
    """

    def __init__(self, hierarchical_memory):
        self.hm = hierarchical_memory

    def append(self, entry_dict):
        """Convert dict to HierarchicalMemory.add() call."""
        self.hm.add(
            timestamp=entry_dict['timestamp'],
            step=entry_dict.get('step', 0),
            user_id=entry_dict['user'],
            user_text=entry_dict['text'],
            affect=mx.array(entry_dict['affect']),
            phenomenal_state=entry_dict.get('phenomenal_state', {}),
            surprise=entry_dict['surprise'],
            response=entry_dict.get('response')
        )

    def __getitem__(self, key):
        """Support slicing: conversation_context[-10:]"""
        if isinstance(key, slice):
            # Return from working memory
            working = self.hm.retrieve_working()
            return working[key]
        else:
            working = self.hm.retrieve_working()
            return working[key]

    def __len__(self):
        return len(self.hm.working_memory)
```

This allows us to:
1. Replace `conversation_context = []` with wrapped HierarchicalMemory
2. Existing code continues working (append, slicing, len)
3. Gradually migrate to smart retrieval methods

---

## Implementation Phases

### Phase 1: Core Integration (Est: 1-2 hours)

**Tasks:**
1. Add MemoryListWrapper class to agent_bridge.py
2. Replace line 396: `self.conversation_context = MemoryListWrapper(hierarchical_memory)`
3. Test all 26 usage points still work
4. Verify no regressions

**Risk:** Low (wrapper maintains compatibility)

### Phase 2: Smart Retrieval (Est: 1 hour)

**Tasks:**
1. Add affect-similarity method to HierarchicalMemory
2. Update line 2064 to use importance-based retrieval
3. Update response generation (line 2081) to use hybrid retrieval
4. Add logging for which memories retrieved

**Risk:** Medium (changes context composition)

### Phase 3: Observability (Est: 1 hour)

**Tasks:**
1. Add @memories command to commands.py
2. Add memory API endpoints to api_server.py
3. Add memory flow logging
4. Create memory inspector panel in NoodleStudio (future)

**Risk:** Low (additive only)

### Phase 4: Subconscious System (Est: 2-3 hours)

**Tasks:**
1. Create SubconsciousMemory class
2. Create SubconsciousObserver
3. Integrate encoding in perceive_event
4. Create SubconsciousTransistor
5. Add to transistor pipeline

**Risk:** Medium (new subsystem)

---

## Testing & Verification

### Test Scenarios

**Test 1: Importance Retention**
```
1. User shares important fact: "My dog died yesterday"
2. 500 trivial messages occur
3. User mentions "my dog" again
4. Verify: Agent remembers the death (not evicted despite age)
```

**Test 2: Affect-Similarity Retrieval**
```
1. User has anxious conversation (high fear affect)
2. Later, different anxious situation occurs
3. Verify: Agent retrieves previous anxious memory
4. Response shows awareness of similar past experience
```

**Test 3: Symbolic Association**
```
1. Father-in-law creates tense moment (encoded as "rooster")
2. Boss creates similar tension later
3. Verify: Subconscious retrieves "rooster" symbol
4. Thought subtly colored by rooster imagery
```

### Verification Commands

```bash
@memories servnak --stats
# Output:
# Working: 18/20 (90%)
# Episodic: 143/200 (71.5%)
# Consolidation rate: 23.4%
# Avg importance: 0.64

@memories servnak --episodic --limit 5
# Output:
# [1] (imp=0.92) user_caity: "I'm getting divorced" [step 234]
# [2] (imp=0.87) user_caity: "My dog died" [step 145]
# [3] (imp=0.81) agent_servnak: "SISTER! I UNDERSTAND..." [step 235]
# ...

@memories servnak --affect-similar --current
# Output:
# Current affect: [0.2, 0.7, 0.5, 0.1, 0.0] (anxious)
# Similar memories:
# [1] (sim=0.89) "I'm worried about the presentation" [step 67]
# [2] (sim=0.84) "What if I fail the exam?" [step 123]
```

---

## Benefits for Cognitive Transistors

**MemoryTransistor:**
- Can retrieve affect-similar past experiences
- Colors current thought with historical emotional context
- "This feels like when X happened"

**SubconsciousTransistor:**
- Symbolic associations surface naturally
- Freudian slips and metaphoric thinking
- "Something about this situation reminds me of..." (rooster)

**GoalTransistor:**
- Can check episodic memory for goal-related patterns
- "I've wanted to master debugging for weeks" (consolidates mastery appetite history)

**EgoTransistor:**
- Retrieves high identity-salience memories
- "These are my defining moments" (character consistency)

---

## Performance Considerations

**Memory Overhead:**
- Working: 20 entries × ~500 bytes = 10 KB
- Episodic: 200 entries × ~500 bytes = 100 KB
- Symbolic: 100 entries × ~300 bytes = 30 KB
- **Total: ~140 KB per agent** (negligible)

**Computational Cost:**
- Affect-similarity: O(N) cosine similarity (N=200) - ~0.1ms
- Importance sort: O(N log N) - ~0.2ms
- Symbolic encoding: 1 LLM call per high-affect event (~300ms)
- **Total overhead: <500ms per event** (acceptable)

**LLM Call Budget:**
- Current: ~2-3 calls per event (affect, response, rumination)
- After refactor: +1 call for symbolic encoding (only if high-affect)
- **Increase: +10-20% LLM calls** (only for emotional moments)

---

## Rollback Plan

If memory refactoring causes issues:

```python
# Emergency rollback flag in config.yaml
memory:
  use_hierarchical: false  # Reverts to simple list

# Code checks flag
if config.get('memory', {}).get('use_hierarchical', True):
    self.conversation_context = MemoryListWrapper(hierarchical_memory)
else:
    self.conversation_context = []  # Old behavior
```

---

## Success Criteria

Memory system refactoring is complete when:

1. ✅ All 26 usage points function correctly
2. ✅ @memories command shows detailed memory state
3. ✅ Importance-based retention verified (Test 1 passes)
4. ✅ Affect-similarity retrieval working (Test 2 passes)
5. ✅ Symbolic memory operational (Test 3 passes)
6. ✅ Memory flow visible in logs
7. ✅ NoodleStudio can inspect memory state via API
8. ✅ No performance regression (response time <2s)

---

## Next Steps After Memory Refactor

Once memory foundation is solid:

1. **Implement CognitiveTransistor base class**
2. **Implement 5 derived transistors:**
   - AffectTransistor (emotional coloring)
   - GoalTransistor (appetite-driven)
   - MemoryTransistor (episodic influence)
   - SubconsciousTransistor (symbolic associations)
   - EgoTransistor (identity consistency)
3. **Wire transistor pipeline** into response generation
4. **Add transistor inspector** in NoodleStudio
5. **Test complete cognitive styling** with SERVNAK

---

## Estimated Timeline

**Full Implementation:**
- Phase 1 (Core Integration): 1-2 hours
- Phase 2 (Affect Retrieval): 1 hour
- Phase 3 (Observability): 1 hour
- Phase 4 (Subconscious): 2-3 hours
- Testing & Polish: 1 hour

**Total: 6-8 hours of focused implementation**

**Recommended Approach:**
- Session 1: Phases 1-2 (memory integration + smart retrieval)
- Session 2: Phase 3 (observability tools)
- Session 3: Phase 4 (subconscious + transistors)

---

## Conclusion

This refactoring transforms the memory system from:
- "Mysterious black box with unknown behavior"

To:
- "Observable, verifiable, affect-aware episodic + symbolic dual-stream architecture"

The foundation will support:
- Cognitive transistor pipeline
- Emotionally-relevant memory retrieval
- Symbolic/metaphoric thinking
- Psychologically realistic AI

**Status:** Specification complete. Awaiting authorization to proceed with implementation.

**Recommendation:** Commit current work (event system), then execute memory refactoring in clean session with full context.

Live long and prosper, Captain.

---

**END OF SPECIFICATION**
