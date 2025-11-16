# Performance Tracking & Event-Driven Cognition Implementation Plan

## Overview
Adding real-time operation tracking + replacing fixed 45s timer with event-driven cognition.

## Part 1: Performance Tracking Integration

### Files to Modify:

1. **llm_interface.py**
   - Wrap `_complete_impl()` with tracking
   - Track `text_to_affect` calls
   - Track `generate_response` calls
   - Track rumination calls from autonomous cognition

2. **agent_bridge.py**
   - Track `perceive_event` full pipeline
   - Track MLX forward passes
   - Track memory operations
   - Track relationship updates
   - Log instant events (surprise spikes, stimulus received, etc.)

3. **autonomous_cognition.py**
   - Track rumination cycles
   - Track speech generation
   - Log cognitive pressure changes
   - Log event-driven triggers

4. **session_profiler.py**
   - Add endpoint for operation logs: `/api/profiler/operations/<agent_id>`
   - Include operations in realtime data

### Operation Types to Track:

**LLM Operations** (milliseconds):
- `llm_text_to_affect` - Affect extraction
- `llm_generate_response` - Response generation
- `llm_rumination` - Internal thought generation

**Neural Operations** (microseconds):
- `mlx_forward` - Forward pass through hierarchical model
- `mlx_surprise_calc` - Surprise calculation

**Memory Operations** (milliseconds):
- `memory_store_episodic` - Store memory
- `memory_retrieve` - Retrieve memories
- `memory_trigger_by_name` - Name-based memory trigger

**Instant Events** (no duration):
- `stimulus_received` - External event perceived
- `surprise_spike` - Surprise threshold crossed
- `name_mentioned` - Agent heard their name
- `cogn_pressure_update` - Cognitive pressure changed
- `speech_triggered` - Decided to speak

## Part 2: Event-Driven Cognition

### Replace Fixed Timer With:

**Event-Driven Triggers:**
1. **Surprise Accumulation** - Sum of recent surprise crosses threshold
2. **Cognitive Pressure Threshold** - Personality-based pressure crosses threshold
3. **Social Acknowledgment** - Being directly addressed
4. **Long Silence** - No interaction for personality-dependent duration
5. **Emotional Intensity** - Strong affect (high arousal or extreme valence)

**Stochastic Variation:**
- Base intervals drawn from exponential distribution (mean = personality-dependent)
- State-modulated (high arousal = shorter intervals)
- Never perfectly periodic

### Implementation in autonomous_cognition.py:

```python
class AutonomousCognitionEngine:
    def __init__(self, agent, config):
        # REMOVE: self.wake_interval = 45

        # ADD: Event-driven thresholds
        self.surprise_accumulation_threshold = config.get('surprise_threshold', 2.0)
        self.min_think_interval = config.get('min_think_interval', 10)  # seconds
        self.max_think_interval = config.get('max_think_interval', 120)  # seconds

        # State tracking
        self.accumulated_surprise = 0.0
        self.last_think_time = time.time()

    async def _cognition_loop(self):
        while self.running:
            # REPLACE fixed sleep with event-driven wait
            wait_time = self._calculate_next_think_interval()
            await asyncio.sleep(wait_time)

            # Check if we should actually think (event-driven conditions)
            if self._should_think():
                await self._do_cognition_cycle()

    def _calculate_next_think_interval(self) -> float:
        """Calculate next think interval based on state."""
        # Base interval from personality (exponential distribution)
        extraversion = self.personality['extraversion']
        spontaneity = self.personality['spontaneity']

        # More extraverted/spontaneous = shorter average interval
        mean_interval = 60 * (1.5 - extraversion) * (1.5 - spontaneity)

        # Draw from exponential distribution
        import random
        interval = random.expovariate(1.0 / mean_interval)

        # Clamp to reasonable bounds
        return max(self.min_think_interval, min(self.max_think_interval, interval))

    def _should_think(self) -> bool:
        """Check if event-driven conditions warrant thinking."""
        # Always think if directly addressed
        if self.directly_addressed:
            return True

        # Think if surprise accumulated
        if self.accumulated_surprise > self.surprise_accumulation_threshold:
            return True

        # Think if cognitive pressure high
        if self.cognitive_pressure > self.speech_urgency_threshold * 0.8:
            return True

        # Think if been too long since last thought
        time_since_think = time.time() - self.last_think_time
        if time_since_think > self.max_think_interval:
            return True

        # Otherwise, random chance based on spontaneity
        spontaneity = self.personality['spontaneity']
        return random.random() < (spontaneity * 0.1)

    def on_surprise(self, surprise: float):
        """Called when agent experiences surprise."""
        self.accumulated_surprise += surprise

        # Decay over time
        time_since_think = time.time() - self.last_think_time
        decay_factor = math.exp(-time_since_think / 60)  # Half-life of 1 minute
        self.accumulated_surprise *= decay_factor
```

## Part 3: NoodleScope Timeline UI

Add scrollable operation log console to profiler page.

### New UI Section:
```html
<div id="operation-timeline" class="timeline-panel">
    <h3>Operation Timeline</h3>
    <div class="timeline-controls">
        <button onclick="pauseTimeline()">Pause</button>
        <button onclick="clearTimeline()">Clear</button>
        <select id="operation-filter">
            <option value="all">All Operations</option>
            <option value="llm">LLM Only</option>
            <option value="neural">Neural Only</option>
            <option value="events">Events Only</option>
        </select>
    </div>
    <div class="timeline-log" id="timeline-log">
        <!-- Auto-scrolling log entries -->
    </div>
</div>
```

### Timeline Entry Format:
```
[12:34:56.123] agent_callie | llm_text_to_affect    | 82ms    | success
[12:34:56.205] agent_callie | mlx_forward           | 3ms     | success
[12:34:56.208] agent_callie | llm_generate_response | 1247ms  | success ← BOTTLENECK
[12:34:57.455] agent_callie | speech_triggered      | -       | event
```

Color coding:
- Green: < 100ms
- Yellow: 100-500ms
- Orange: 500-1000ms
- Red: > 1000ms
- Blue: events (no duration)

## Implementation Order:

1. ✓ Create performance_tracker.py
2. Integrate tracking into llm_interface.py
3. Integrate tracking into agent_bridge.py
4. Integrate tracking into autonomous_cognition.py
5. Add profiler API endpoint for operations
6. Implement event-driven cognition (replace timer)
7. Add timeline UI to NoodleScope
8. Test and tune thresholds

## Benefits:

1. **Visibility**: See exactly where time is spent
2. **Natural Behavior**: No more clockwork 45s ticks
3. **Debugging**: Identify bottlenecks instantly
4. **Scientific**: Quantify performance characteristics
5. **Personality-Driven**: Spontaneous agents think more erratically
