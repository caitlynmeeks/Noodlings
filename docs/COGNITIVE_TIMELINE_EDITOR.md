# Cognitive Timeline Editor

**What Premiere did for video, what Maya did for 3D - we do for cognition.**

## Status: Phase 1 Complete (December 19, 2025)

### Completed
- TimelineRecorder bridges ExecutionEventBus to ProfilerPanel
- FacetTrack widget renders facet executions as colored swimlanes
- ProfilerPanel now has Facets + Affect + Inspector sections
- Recording controls (REC/PAUSE/CLEAR)
- Click-to-inspect facet details

### Pending
- Playback controls (play/pause/speed)
- A/B prompt testing (edit prompt, re-execute, compare)
- Session save/load

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     COGNITIVE TIMELINE EDITOR                    │
├─────────────────────────────────────────────────────────────────┤
│  COGNITIVE TIMELINE                    [LIVE] [REC] [Clear]     │
├─────────────────────────────────────────────────────────────────┤
│  FACET EXECUTION                          3 cycles | 18 facets  │
│    CYCLES   ████████░░░░░░░████████░░░░░░░████████░░░░░░░░░░░  │
│    INCOMING        █░░░░░░░░░█░░░░░░░░█░░░░░░░░░░░░░░░░░░░░░░  │
│    CharmNetwork    ░█░░░░░░░░░█░░░░░░░░█░░░░░░░░░░░░░░░░░░░░░  │
│    ContextIntel    ░░████░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░  │
│    RoastEngine     ░░░░░░████░░░░░░░░░░░░░░░████░░░░░░░░░░░░░  │
│    Convergence     ░░░░░░░░██░░░░░░░░░░░░░░░░░░██░░░░░░░░░░░░  │
│    OUTGOING        ░░░░░░░░░█░░░░░░░░░░░░░░░░░░░█░░░░░░░░░░░░  │
├─────────────────────────────────────────────────────────────────┤
│  AFFECT WAVEFORMS                                               │
│    Valence   ──────▁▂▃▄▅▆▇█▇▆▅▄▃▂▁──────────────────────────  │
│    Arousal   ████▇▆▅▄▃▂▁──────▁▂▃▄▅▆▇████░░░░░░░░░░░░░░░░░░  │
│    Boredom   ▁▂▃▄▅▆▇█████████▇▆▅▄▃▂▁░░░░░░░░░░░░░░░░░░░░░░░  │
├─────────────────────────────────────────────────────────────────┤
│  INSPECTOR (click facet or event to view)                       │
│    ==================================================           │
│     FACET: RoastEngine                                          │
│    ==================================================           │
│    Type:     LLMFacet                                           │
│    Cycle:    3                                                  │
│    Duration: 1234.5ms                                           │
│    Tokens:   847                                                │
│    INPUTS:                                                      │
│      affect_valence: 0.3                                        │
│      context_summary: "Caity said hi to Red"                    │
│    OUTPUTS:                                                     │
│      roast_text: "Oh PLEASE, 'hi red'? That's the BEST..."     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
ExecutionEventBus
    │
    ├─ 'cycle_start' ──────→ TimelineRecorder.on_cycle_start()
    │                              └─ Create new CycleRecord
    │
    ├─ 'facet_start' ──────→ TimelineRecorder.on_facet_start()
    │                              └─ Create FacetRecord, mark start_time
    │
    ├─ 'facet_complete' ───→ TimelineRecorder.on_facet_complete()
    │                              └─ Update FacetRecord with outputs, duration
    │
    ├─ 'cycle_complete' ───→ TimelineRecorder.on_cycle_complete()
    │                              └─ Finalize CycleRecord
    │                              └─ Emit cycleRecorded signal → ProfilerPanel
    │
    └─ (future: affect_update) → AffectSample for waveform
```

---

## Data Structures

### FacetRecord
```python
@dataclass
class FacetRecord:
    facet_id: str
    facet_name: str
    facet_type: str
    start_time: float
    end_time: float
    duration_ms: float
    token_count: int
    salience: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    prompt: Optional[str]  # For LLM facets
    execution_id: str
    cycle: int
```

### CycleRecord
```python
@dataclass
class CycleRecord:
    cycle_id: str
    cycle_number: int
    cycle_type: str  # 'reactive' | 'autonomous'
    start_time: float
    end_time: float
    duration_ms: float
    incoming_text: str
    outgoing_text: str
    assembly_name: str
    total_tokens: int
    facets: List[FacetRecord]
```

### TimelineSession
```python
@dataclass
class TimelineSession:
    session_id: str
    start_time: float
    agents: Dict[str, AgentTimeline]  # agent_id -> timeline
```

---

## Files

### New Files (Phase 1)
| File | Lines | Purpose |
|------|-------|---------|
| `core/timeline_recorder.py` | ~500 | EventBus listener, builds timeline data |
| `widgets/facet_track.py` | ~350 | FacetTrack, CycleTrack, FacetBlockItem widgets |

### Modified Files
| File | Changes |
|------|---------|
| `panels/profiler_panel.py` | Added FacetTimelineView, recording controls, facet inspector |

---

## Facet Color Coding

Following monochromatic palette guidelines:

| Facet Type | Color | Hex |
|------------|-------|-----|
| LLMFacet | Purple | #9C27B0 |
| CharmNetworkFacet | Green | #4CAF50 |
| ScriptedFacet | Blue | #2196F3 |
| ContextIntelligenceFacet | Teal | #009688 |
| ConvergenceFacet | Red | #F44336 |
| TickerGateFacet | Orange | #FF9800 |
| SpecialNode (INCOMING/OUTGOING) | Blue-gray | #607D8B |

---

## Implementation Phases

### Phase 1: Core Infrastructure (DONE)
- [x] TimelineRecorder bridging EventBus to ProfilerPanel
- [x] FacetRecord/CycleRecord data structures
- [x] FacetTrack widget for swimlane visualization
- [x] Click-to-inspect facet details
- [x] Recording controls (REC/PAUSE/CLEAR)

### Phase 2: Playback Controls
- [ ] Transport bar (Play/Pause/Stop)
- [ ] Speed selector (0.5x, 1x, 2x, 4x)
- [ ] Playhead animation (QTimer-driven)
- [ ] Sync affect waveforms to playhead

### Phase 3: Facet Inspector Enhancement
- [ ] Collapsible sections for inputs/outputs
- [ ] Full prompt display for LLM facets
- [ ] Copy output to clipboard
- [ ] Jump to facet in Facets Editor

### Phase 4: A/B Prompt Testing (Transformational!)
- [ ] Edit prompt in inspector
- [ ] Re-execute facet with modified prompt
- [ ] Side-by-side output comparison
- [ ] Save prompt variants as presets

### Phase 5: Session Persistence
- [ ] Save timeline session to file
- [ ] Load previous sessions for review
- [ ] Export timeline data for analysis

---

## Existing Infrastructure Leveraged

### ExecutionEventBus (already emits)
- `cycle_start` - assembly_name, execution_id, timestamp
- `facet_start` - facet_id, facet_name, facet_type, inputs
- `facet_complete` - outputs, execution_time, token_count
- `cycle_complete` - duration, total_tokens, facets_executed

### MultiTrackTimeline (already has)
- Per-agent collapsible tracks
- Affect waveform rendering
- Event markers (speech, thought, movement)
- Playhead slider
- Zoom controls

### ProfilerPanel (already had)
- Vertical splitter (timeline + inspector)
- Timecode display
- Basic event inspector

---

## Notes

1. **EventBus connection is deferred** - ProfilerPanel retries connection every 2s until server starts
2. **Affect waveforms remain separate** - Original SessionLoader API still feeds affect data
3. **Facet timeline is live** - Updates automatically as facets execute
4. **Click facet block** - Shows inputs/outputs/timing in inspector

---

## The Vision

> "What Premiere did for video, what Maya did for 3D - we do for COGNITIVE CONSCIOUSNESS"

This isn't just a debugger. It's a cognitive workbench where you can:
- **See** exactly what each facet received and produced
- **Understand** the timing relationships between facets
- **Edit** prompts and re-execute to compare outputs
- **Tune** the cognitive architecture by adjusting facet parameters
- **Share** recorded sessions for collaboration and analysis

The Cognitive Timeline Editor transforms the invisible process of AI cognition into something you can see, touch, and manipulate.
