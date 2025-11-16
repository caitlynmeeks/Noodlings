# Intuition Receiver (Context Gremlin)

**Status**: ✅ Implemented and ready for testing (November 15, 2025)

## Overview

The Intuition Receiver provides each Noodling with integrated contextual awareness - like a radio tuned to spatial and routing signals. Instead of external scaffolding, this creates natural intuitive understanding of:

- **Message routing**: Who is being addressed
- **Spatial awareness**: Who is where
- **Prop tracking**: Who has what objects
- **Recent actions**: What just happened

## Architecture

### Components

1. **Context Analyzer** (`_generate_intuition()` in `agent_bridge.py`)
   - Uses fast LLM (qwen3-4b) to analyze each message
   - Accesses world state (rooms, objects, agents, inventories)
   - Generates natural first-person contextual awareness

2. **Integration Point** (`perceive_event()` in `agent_bridge.py`)
   - Intuition generated before response/rumination
   - Stored in phenomenal state as `state['intuition']`
   - Available to both speech and thought generation

3. **Prompt Injection** (in `llm_interface.py`)
   - Injected into both `generate_response()` and `generate_rumination()`
   - Appears as "📻 YOUR INTUITIVE AWARENESS"
   - Feels like natural awareness, not external information

## Configuration

**File**: `config.yaml`

```yaml
agent:
  intuition_receiver:
    enabled: true
    model: qwen/qwen3-4b-2507
    timeout: 5
```

## Implementation Details

### Files Modified

1. **config.yaml**
   - Added `intuition_receiver` configuration

2. **agent_bridge.py**
   - Added `world` parameter to `CMUSHConsilienceAgent.__init__()`
   - Added `_generate_intuition()` method
   - Integrated intuition generation in `perceive_event()`
   - Updated `AgentManager.create_agent()` to pass world reference
   - Added 'intuition_receiver' to config merge list

3. **llm_interface.py**
   - Modified `generate_response()` to inject intuition into prompt
   - Modified `generate_rumination()` to inject intuition into thoughts

### Data Flow

```
Event arrives
    ↓
perceive_event() called
    ↓
_generate_intuition() analyzes context
    ├─ Examines current message
    ├─ Reviews recent conversation
    ├─ Checks room occupants
    ├─ Checks object locations
    └─ Checks agent inventories
    ↓
Intuition stored in state['intuition']
    ↓
    ┌─────────────┬──────────────┐
    ↓             ↓              ↓
Rumination  Speech Gen    (future uses)
    ↓             ↓
Prompt with intuition section
    ↓             ↓
Agent responds with contextual awareness
```

## Example Output

### Intuition Examples

```
"That greeting is for Toad, not me. They're by the glowing pond while I'm near the hedge."

"Callie is asking everyone a question. I notice she's still holding the mysterious stone from earlier."

"Someone just entered the room - Servnak. The conversation was about the garden project."
```

### How It Appears in Prompts

```
╔═══════════════════════════════════════════════════════════╗
║  📻 YOUR INTUITIVE AWARENESS (Situational Context)       ║
╚═══════════════════════════════════════════════════════════╝

That greeting is for Toad, not me. They're by the glowing pond
while I'm near the hedge.

This contextual awareness naturally informs your understanding of the situation.
```

## Benefits

1. **Natural Understanding**: Agents intuitively know who messages are for
2. **Spatial Awareness**: Agents understand relative positions
3. **Object Tracking**: Agents know who has what props
4. **Reduced Confusion**: No more agents responding to others' names
5. **Theater Ready**: Essential for play performances with blocking

## Performance

- **Model**: qwen3-4b (fast, efficient)
- **Timeout**: 5 seconds (configurable)
- **Parallel**: Uses existing LLM pool infrastructure
- **Optional**: Can be disabled via config

## Testing

To test the Intuition Receiver:

1. Start noodleMUSH: `cd applications/cmush && ./start.sh`
2. Open browser to http://localhost:8080
3. Spawn multiple agents: `@spawn toad`, `@spawn callie`
4. Test scenarios:
   - Address one agent specifically: "how are you toad?"
   - Have one agent hold an object
   - Multiple agents in different locations
   - Check logs for intuition generation: `📻 Intuition:`

## Future Enhancements

Potential improvements:

1. **Cache recent intuitions** to avoid redundant generation
2. **Emotional coloring** of intuition based on relationships
3. **Prediction** of likely next actions based on context
4. **Multi-agent coordination** awareness (who is doing what together)
5. **Memory integration** with episodic memory for context

## Technical Notes

- Intuition uses same LLM pool as main generation (no bottleneck)
- World state snapshot taken at perception time (consistent view)
- Intuition generated before response decision (informs all choices)
- First-person perspective maintains character immersion
- Low temperature (0.3) ensures consistent, reliable analysis

## Philosophy

The Intuition Receiver implements **integrated contextual awareness** rather than external scaffolding. Instead of parsing metadata or using rigid rules, agents have a natural "radio sense" of their environment. This makes contextual understanding part of their phenomenology, not an add-on.

Like peripheral vision or proprioception, the Intuition Receiver provides background awareness that feels innate rather than computed.

---

**Next Steps**: Test with existing agents in noodleMUSH and monitor intuition quality/accuracy.
