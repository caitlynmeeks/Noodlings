# Performance Tracking & Event-Driven Cognition - Implementation Summary

## What We Built

### 1. Performance Tracking System (`performance_tracker.py`) ✓

A comprehensive operation tracking system that:
- Tracks all LLM calls, neural operations, and events with microsecond timestamps
- Maintains rolling buffers (100 ops per agent)
- Provides context manager for easy integration: `with tracker.track_operation(...)`
- Logs instant events (surprise spikes, stimuli, etc.)
- Calculates performance statistics by operation type

### 2. Next Steps - Integration Points

The tracking system is ready. Here's where to integrate it:

#### A. LLM Interface (`llm_interface.py`)

Add at top after imports:
```python
from performance_tracker import get_tracker
```

In `__init__` method, add:
```python
self.agent_id = agent_id  # Pass from agent_bridge
self.tracker = get_tracker()
```

Wrap key methods:
- `text_to_affect()` - Track as "llm_text_to_affect"
- `generate_response()` - Track as "llm_generate_response"
- `_complete_impl()` - Track as "llm_api_call"

#### B. Agent Bridge (`agent_bridge.py`)

Add tracking to:
- `perceive_event()` - Track full perception pipeline
- MLX forward pass - Track as "mlx_forward"
- Memory operations - Track as "memory_store", "memory_retrieve"
- Instant events - Log "stimulus_received", "surprise_spike", "name_mentioned"

#### C. Autonomous Cognition (`autonomous_cognition.py`)

**MAJOR CHANGE**: Replace fixed 45s timer with event-driven triggers

Add to `__init__`:
```python
# Event-driven thresholds (REPLACES wake_interval)
self.surprise_accumulation_threshold = config.get('surprise_threshold', 2.0)
self.accumulated_surprise = 0.0
self.min_think_interval = 10  # seconds
self.max_think_interval = 120  # seconds
```

Replace `_cognition_loop()`:
```python
async def _cognition_loop(self):
    while self.running:
        # Event-driven wait (NOT fixed 45s!)
        wait_time = self._calculate_next_think_interval()
        await asyncio.sleep(wait_time)

        # Only think if conditions warrant it
        if self._should_think():
            await self._do_cognition_cycle()
```

Add methods:
- `_calculate_next_think_interval()` - Exponential distribution based on personality
- `_should_think()` - Check event-driven conditions
- `on_surprise(surprise)` - Accumulate surprise for threshold triggering

#### D. Session Profiler (`session_profiler.py`)

Add new endpoint:
```python
@app.route('/api/profiler/operations/<agent_id>')
async def get_operations(agent_id):
    """Get recent operations for an agent."""
    tracker = get_tracker()
    last_n = request.args.get('last_n', 50, type=int)
    operations = tracker.get_recent_operations(agent_id, last_n)
    return jsonify(operations)
```

#### E. NoodleScope UI (noodlescope2.html)

Add timeline panel with:
- Scrollable console showing operations in real-time
- Color-coded by duration (green < 100ms, yellow < 500ms, orange < 1000ms, red > 1000ms)
- Filter by operation type (All/LLM/Neural/Events)
- Auto-scroll with pause/clear buttons

## Benefits

1. **Visibility** - See exactly where time is spent (LLM vs neural)
2. **Natural Behavior** - No more mechanical 45s ticks
3. **Personality-Driven** - Spontaneous agents think erratically, introverts think less
4. **Debugging** - Instantly identify bottlenecks
5. **Scientific** - Quantify performance characteristics

## Current Status

✓ Performance tracker core system complete
✓ Implementation plan documented
⏳ Integration into existing files (multi-file changes)
⏳ Event-driven cognition implementation
⏳ Profiler API endpoint
⏳ NoodleScope timeline UI

## Recommendation

Given the scope of changes across multiple files, the safest approach is:

1. **Test the tracker in isolation first** - Verify it works standalone
2. **Integrate into one file at a time** - Start with llm_interface.py
3. **Test incrementally** - Make sure each integration works before moving to next
4. **Deploy event-driven cognition last** - This is the most impactful change

Would you like me to:
- A) Continue with full integration across all files (careful, systematic)
- B) Start with just llm_interface.py integration as proof-of-concept
- C) Provide detailed patches for you to review/apply manually
- D) Create a branch/backup first, then proceed with full implementation

The foundation is solid. The question is how aggressively to integrate it.
