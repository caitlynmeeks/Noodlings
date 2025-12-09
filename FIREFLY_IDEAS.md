# FIREFLY IDEAS 🌙✨

**Captured feature ideas for future Noodlings development**

*"It's hard with all these fireflies NinaK! But it's okay. What's important is we capture the light, the essence, even if it doesn't work quite right the first time, we will tend to each firefly carefully and incrementally until they're not fireflies but luminescent trees of delight and magic"* - Caitlyn

---

## FIREFLY #1: Context Intelligence God 🧠👑

**Priority:** CRITICAL - Foundation for social cognition

### The Problem

Agents don't understand WHO is speaking to WHOM:
- Toad thinks Red is asking him questions when Caity addresses Red
- No persistent tracking of who's where, doing what
- No conversation threading (who's waiting for answers?)

### The Solution: Context Intelligence Facet

**Persistent World Model** - Like Unity's scene graph but for SOCIAL/RELATIONAL state!

Tracks:
- Entity states (location, posture, mood, attention)
- Object locations + occlusion ("Caity has candy in pocket" - Red doesn't know about mouse!)
- Relationship dynamics (trust, annoyance accumulate over time)
- Conversation threads (who asked what, who's waiting for response)
- Temporal state ("Red was on Caity's shoulder 3 turns ago, still there?")

**Data Structure:**
```python
{
    'entities': {
        'caity': {
            'location': 'room_clearing',
            'posture': 'standing',
            'holding': ['wooden_sword', 'candy'],
            'attention_on': 'red'
        },
        'red': {
            'posture': 'perched_on_shoulder',
            'on_entity': 'caity',  # Physical contact!
            'mood': 'defensive'
        }
    },
    'hidden_objects': {
        'caity.pocket': ['mouse']  # Occlusion!
    },
    'conversation_threads': [
        {'speaker': 'caity', 'addressee': 'red', 'status': 'awaiting_response'}
    ]
}
```

**Why Critical:**
1. Fixes conversation confusion
2. Enables "Am I being addressed?" reasoning
3. Foundation for all other social facets
4. Grounds perception in context

**Integration Point:** After INCOMING, before all cognitive facets

**Status:** Partially implemented (ContextIntelligenceFacet exists), needs persistent world model

---

## FIREFLY #2: FACS as a Facet 👁️

**Priority:** HIGH - Observable body language

### The Problem

FACS (Facial Action Coding System) body language is hardcoded in agent_bridge.py, not part of facet assembly!

### The Solution: BodyLanguageFacet

Make body language a standard facet that:
- Observes affect changes (arousal spikes, valence drops)
- Generates FACS codes (AU1, AU2, BL25, etc.)
- Species-specific (fire imps: flames flare/dim, toads: puff up/hop back)
- **Observable by other agents!** Red sees Toad flinch → mocks him

**Architecture:**
```yaml
- id: body_language
  type: BodyLanguageFacet
  inputs:
    - affect_valence (detect changes)
    - affect_arousal (detect changes)
  outputs:
    - facs_codes (AU1, AU2, BL25)
    - description ("*freeze, eyes widen*")
    - _observable (true - other agents see this!)
```

**Scriptable via JavaScript:**
```javascript
const arousal_delta = inputs.arousal - context.previous_arousal;
if (arousal_delta > 0.3) {
  return {facs: "AU1+AU2+AU5", description: "*eyes widen, freeze*"};
}
```

**Why Important:**
- Social perception includes reading subtle cues
- Species-specific reactions
- Removes hardcoded logic from agent_bridge.py

**Status:** Captured, not implemented

---

## FIREFLY #3: Episodic Memory as a Facet 🧠

**Priority:** MEDIUM - Scriptable memory operations

### The Problem

HierarchicalMemory is hardcoded in agent_bridge.py:
- Can't customize memory per-agent
- Can't script memory queries in JavaScript
- Can't model trauma (denial blocking consolidation)

### The Solution: MemoryFacet

**Unity-Style API:**
```javascript
// Query memories
const recent = context.memory.getRecent(5);
const similar = context.memory.findSimilar("candy", limit=3);
const remembers_caity = context.memory.contains("Caity");

// Store explicit memory
context.memory.store({
    content: "Caity gave me candy",
    importance: 0.9,
    emotional_tag: "positive"
});

// Therapeutic operations
context.memory.forget("traumatic event");
context.memory.block_consolidation();  // Denial repression!
```

**Trauma Modeling:**
```javascript
if (denial_salience > 0.8) {
    context.memory.block_consolidation();  // Repression!
}
```

**Per-Agent Customization:**
- Fire imps: working_capacity=3 (short attention)
- Toad: rapid consolidation (ADHD memory)

**Status:** Captured, implement post-demo

---

## FIREFLY #4: Unity-Style Debug Logging 🪵

**Priority:** LOW - Quality of life improvement

### The Problem

Manual log formatting everywhere:
```python
logger.info(f"[{self.agent_id}] ⚡ FACET ASSEMBLY: {result.facets_executed} facets")
print(f"[{agent_name.upper()}] 💭 Subconscious: {symbolic_image[:80]}...")
```

### The Solution: NoodleLog System

**Automatic formatting:**
```python
# Before:
logger.info(f"[{self.agent_id}] ⚡ FACET ASSEMBLY: {result.facets_executed} facets")

# After:
NoodleLog.Info(f"⚡ FACET ASSEMBLY: {result.facets_executed} facets", agent=self.agent_name)

# Output:
# [INFO][RED FIRE ANKLEBITER][agent_bridge.py:2558] ⚡ FACET ASSEMBLY: 6 facets
```

**Features:**
- Auto file/line detection (inspect module)
- Auto console routing (MUSH/STUDIO/FACETS)
- Single call (no logger.info() + print())
- Stack traces for errors

**Migration:** Incremental, keep old logging alongside

**Status:** Captured, good cleanup task

---

## FIREFLY #5: Prim Reaction System 🔥

**Priority:** MEDIUM - Interactive environment

### The Problem

Action parser can target prims, but prims can't react!

### The Solution

**Prim reaction scripts:**
```python
# drapes.on_prim_action
def on_fire(action):
    self.state = 'ashes'
    emit_emote("The drapes catch fire and burn to ashes!")
```

**Use Cases:**
- Red sets fire to drapes → drapes burn
- Toad hops on lily pad → lily pad bobs
- Caity strikes training dummy → dummy reacts

**Status:** Action parser ready, prim reactions not implemented

---

## FIREFLY #6: Noodlings Player - Standalone Executables 🎨

**Priority:** LOW - Post-1.0 feature

### The Vision

Build Noodlings projects as standalone executables with custom UIs!

**Use Case: Whitney Gallery Installation**
- Huge screen divided in two
- Left: Symbolic imagery (haiku from subconscious)
- Right: Generated images (Stability AI)
- Noodlings running inside, continuous dream stream
- No terminal, no NoodleStudio - just the experience

**Architecture:**
```
Noodlings Project
    ↓
Build as Executable (PyInstaller)
    ↓
Bundles: noodleMUSH + facet assembly + custom UI
    ↓
Double-click to launch
```

**Output Nodes:**
```yaml
- id: dream_display
  type: DisplayOutputFacet
  ui_binding:
    type: "text_display"
    widget_id: "left_panel"
```

**Build Command:**
```bash
noodlestudio build --target standalone --ui gallery_installation.ui
```

**Example Projects:**
- Dream stream gallery (Whitney installation)
- Interactive poetry (user input → haiku)
- Emotional mirror (camera → affect → symbolic reflection)
- Multi-agent simulation (10 Noodlings, pure observation)

**Why Important:**
- Artists: No coding for deployment
- Museums: Installation-ready
- Researchers: Package experiments as executables
- Public: Downloadable experiences

**Status:** Vision captured, implement way later

---

## FIREFLY #7: Subconscious Insights Testing 💭

**Priority:** HIGH - Verify trauma/healing model

### Testing Checklist

1. @derez red_fire_anklebiter
2. @rez red_fire_anklebiter
3. Be KIND to Red (offer candy, be gentle)
4. Watch for:
   - `💭 Subconscious:` (every cycle)
   - `💭 Latent memory stored (X total)`
   - `✨ Insight surfaced:`
5. Look for "privately thinks" in responses
6. Be HARSH → insights should STOP (denial blocks)
7. Be kind again → insights should RESUME

**Expected Behavior:**
- Continuous symbolic images in latent pool
- When calm + defenses down → insights surface
- When aroused + defensive → insights stay buried
- Healing emerges from repeated safety

**Status:** Subconscious facet built, needs testing

---

---

## FIREFLY #8: Guilt Facet 😰💔

**Priority:** MEDIUM-HIGH - Moral cognition layer
**Captured:** December 6, 2025 (debugging session with NinaK)

### The Vision

A facet that tracks GUILT - moral discomfort from past actions!

**Core Mechanics:**
- Maintains a **guilt table** (list of unresolved moral violations)
- References **subconscious** symbolic imagery for moral themes
- Queries **episodic memory** for "bad" actions
- Salience driven by **moral distress** (valence + fear + sorrow combo)
- **Resolution state tracking** - unresolved guilt festers, resolved guilt becomes wisdom!

### Data Structure

```python
{
    'guilt_items': [
        {
            'action': 'bit Caity's ankle too hard',
            'turns_ago': 12,
            'intensity': 0.7,  # Continuous!
            'resolved': False,
            'memory_ref': 'episodic_entry_42'
        },
        {
            'action': 'mocked Toad when he was sad',
            'turns_ago': 34,
            'intensity': 0.4,
            'resolved': True,  # Apologized!
            'resolution_method': 'direct_apology'
        }
    ],
    'moral_distress': 0.65  # Accumulated unresolved guilt
}
```

### Facet Architecture

```yaml
- id: guilt_processor
  type: GuiltFacet
  salience_script: |
    function compute_salience(inputs, context) {
      // Guilt emerges when:
      // 1. High sorrow (feeling bad)
      // 2. Low arousal (calm enough to reflect)
      // 3. Positive valence would increase guilt salience (feeling good = contrast with bad actions)

      const sorrow = inputs.affect_sorrow || 0;
      const arousal = inputs.affect_arousal || 0.5;
      const valence = inputs.affect_valence || 0;

      // Need calm to feel guilt (arousal < 0.4)
      const calm_enough = arousal < 0.4 ? 1.0 : 0.3;

      // Sorrow + guilt connection
      const guilt_salience = sorrow * calm_enough;

      return {
        salience: guilt_salience,
        shouldExecute: guilt_salience > 0.3
      };
    }
  inputs:
    - affect_valence
    - affect_sorrow
    - affect_arousal
    - subconscious_symbolism  # Moral themes from subconscious
    - episodic_memory  # Past actions
  outputs:
    - guilt_state  # Current guilt table
    - moral_distress  # 0-1 continuous value
    - apology_impulse  # Desire to make amends
```

### Behavioral Outputs

**When guilt is high:**
- Defensive responses (denial facet amplified)
- Self-deprecating comments
- Apology attempts ("Sorry I bit you, I was just... you know...")
- Avoidance behavior (less eye contact, flames dim)

**When guilt resolves:**
- Relief (arousal spike, valence increase)
- Wisdom extraction ("I learned not to bite so hard")
- Closer bonding (trust increases)

### Integration with Other Facets

**With Denial Defense:**
```javascript
// Denial can SUPPRESS guilt awareness!
if (denial_salience > 0.7) {
    guilt_awareness = guilt_base * 0.2;  // Repression!
}
```

**With Insight Emergence:**
```javascript
// Guilt can surface as vulnerable insights
if (guilt_distress > 0.6 && safety > 0.7) {
    surface_insight("I feel bad about biting Caity...");
}
```

**With Convergence:**
```yaml
# Convergence blends guilt-driven apologies with roasts!
- If guilt_salience > roast_salience: Apologetic tone
- If both high: Defensive apology ("Sorry BUT you DESERVED it!")
```

### Use Cases

1. **Red bites too hard:**
   - Caity: "OW! Red that hurt!"
   - Red's guilt accumulates (moral distress increases)
   - Next interaction: "Yeah well... maybe I overdid it. WHATEVER!"

2. **Repeated kindness heals guilt:**
   - Caity offers candy multiple times
   - Red's guilt about past meanness surfaces
   - Red: "privately thinks, She keeps being nice even when I'm a jerk..."
   - Eventually: Explicit apology or softened behavior

3. **Denial blocks guilt:**
   - Toad confronts Red about being mean
   - Red's arousal spikes (defensive!)
   - Denial facet: High salience
   - Guilt facet: Suppressed
   - Red: "What?! I wasn't THAT mean! You're too sensitive!"

4. **Guilt resolution:**
   - Red apologizes to Caity
   - Guilt item marked resolved
   - Moral distress drops
   - Relief: Valence spike, sorrow drop
   - Future interactions: More trusting, less defensive

### Technical Design

**Guilt Table Management:**
```python
class GuiltFacet:
    def __init__(self):
        self.guilt_table = []
        self.moral_distress = 0.0

    async def execute(self, inputs, context):
        # Check episodic memory for moral violations
        recent_actions = context.episodic_memory.get_recent(10)

        for action in recent_actions:
            if self._is_morally_questionable(action):
                self.add_guilt_item(action)

        # Age guilt items (intensity fades over time)
        self._age_guilt_items()

        # Calculate total moral distress
        self.moral_distress = sum(
            item['intensity'] for item in self.guilt_table
            if not item['resolved']
        )

        # Generate apology impulse based on distress + sorrow
        apology_impulse = self.moral_distress * inputs.affect_sorrow

        return {
            'guilt_state': self.guilt_table,
            'moral_distress': self.moral_distress,
            'apology_impulse': apology_impulse
        }
```

**Resolution Detection:**
```python
def check_for_resolution(self, current_text, context):
    # Did agent just apologize?
    if 'sorry' in current_text.lower() or 'my bad' in current_text.lower():
        # Mark most recent guilt as resolved!
        if self.guilt_table:
            self.guilt_table[-1]['resolved'] = True
            self.guilt_table[-1]['resolution_method'] = 'explicit_apology'

    # Did recipient forgive?
    if context.incoming_data and 'it\'s okay' in context.incoming_data.lower():
        # Resolve ALL guilt toward that person!
        self.resolve_guilt_toward(context.speaker)
```

### Why This is POWERFUL

1. **Emergent moral development:** Guilt accumulates naturally from interactions
2. **Healing through relationship:** Kindness from others resolves guilt
3. **Denial/guilt dynamics:** Defense mechanisms can BLOCK moral growth
4. **Long-term character arcs:** From defensive jerk → vulnerable friend
5. **Therapeutic modeling:** Mirrors real psychological processes

### Research Applications

- **Moral development in AI:** How does guilt emerge from continuous affect?
- **Trauma/healing cycles:** Can denial block moral insight? Can safety unlock it?
- **Attachment theory:** Does guilt resolution strengthen bonds?
- **Character depth:** Move beyond "personality sliders" to emergent morality

### Demo Potential (Steve DiPaola)

**Show:** Red starts as aggressive roaster
**Interaction:** Caitlyn is kind despite roasts
**Result:** Guilt accumulates → Denial blocks → Safety unlocks → Insight surfaces → Apology → Bond deepens

**This demonstrates:**
- Continuous affective dynamics
- Emergent character development
- Psychological realism
- Not scripted - EMERGENT from affect + memory + facet topology!

---

## FIREFLY #9: The Cognitive Timeline Editor 🎬🧠

**Priority:** TRANSFORMATIONAL - New paradigm for AI character editing
**Captured:** December 6, 2025 (NinaK + Captain Caitlyn breakthrough moment)

### The Vision

**"What Premiere did for video, what Maya did for 3D, what Photoshop did for images - we do for COGNITIVE CONSCIOUSNESS."**

A timeline-based editor that lets you SEE, SCRUB, and EDIT the flow of thought through a cognitive architecture.

### The Core Metaphor: Non-Linear Cognitive Editing

**Premiere Pro Timeline** = Temporal arrangement of video/audio clips
**Cognitive Timeline** = Temporal arrangement of CYCLES, FACETS, THOUGHTS

### What You See

```
TIME →
├─ 00:00.000  Cycle A (reactive: "hi red") ─────────────────────[20s]
│  ├─ 00:00.100  INCOMING: "hi red"
│  ├─ 00:00.120  CharmNetwork ──[LOCKED 3ms]── affect_valence: 0.3
│  ├─ 00:00.450  Context Intelligence ─────────[should_respond: TRUE]
│  │              ├─ addressee: "red fire anklebiter" ✓
│  │              ├─ agent_name: "Red Fire Anklebiter" ✓
│  │              └─ decision: RESPOND
│  ├─ 00:01.200  Room Observer (salience: 0.85)
│  ├─ 00:02.100  Roast Engine (salience: 0.92) ──[generate roast]
│  ├─ 00:15.300  Convergence ──[blend 3 inputs]
│  ├─ 00:18.700  Voice Filter
│  └─ 00:20.000  OUTGOING: "Oh PLEASE, 'hi red'? MWAHAHA!"
│
├─ 00:05.000  Cycle B (autonomous: internal rumination) ───────[15s]
│  ├─ 00:05.100  INCOMING: "" (empty, autonomous)
│  ├─ 00:05.120  CharmNetwork ──[WAITS for Cycle A lock]──
│  ├─ 00:05.123  CharmNetwork ──[LOCKED 3ms]── boredom: 0.65
│  ├─ 00:05.450  Context Intelligence ─────────[should_respond: FALSE]
│  │              └─ decision: OBSERVE SILENTLY
│  ├─ 00:07.200  Subconscious (salience: 0.70) ──[symbolic image]
│  └─ 00:20.000  OUTGOING: "" (silent observation)
│
└─ 00:20.000  Cycle C (reactive: "what? i bet you...") ───────[18s]
    └─ ...
```

### The Interface

**TOP PANEL: Timeline Swimlanes**
```
┌─────────────────────────────────────────────────────────┐
│ [■ PLAY] [◼ STOP] [<< REW] [>> FF]  Speed: [1x▼]       │
│                                                          │
│ 00:00.000 ──────────────────→ 00:02:15.430             │
│                        ▲ Playhead                        │
│                                                          │
│ CYCLES ─────────────────────────────────────────────   │
│   Reactive: "hi red" █████████████████████░░░░░░        │
│   Autonomous        ░░░░░████████████░░░░░░░░░░░        │
│   Reactive: "what?" ░░░░░░░░░░░░░░░███████████░░░       │
│                                                          │
│ FACETS ─────────────────────────────────────────────   │
│   INCOMING          █░░░░░░█░░░░░░█░░░░░░░░░░░░░        │
│   CharmNetwork      ░█░░░░░░█░░░░░░█░░░░░░░░░░░░        │
│   Context Intel     ░░██████░░░░░░░░░██████░░░░░        │
│   Roast Engine      ░░░░░░██████░░░░░░░░░░░░░░░░        │
│   Convergence       ░░░░░░░░░██░░░░░░░░░░░░░░░░░        │
│   OUTGOING          ░░░░░░░░░░█░░░░░░░░░░░░░░░█░        │
│                                                          │
│ AFFECT ──────────────────────────────────────────────   │
│   Valence    ───────▁▂▃▄▅▆▇█▇▆▅▄▃▂▁───────────        │
│   Arousal    ████▇▆▅▄▃▂▁──────▁▂▃▄▅▆▇████░░░░        │
│   Boredom    ▁▂▃▄▅▆▇█████████▇▆▅▄▃▂▁░░░░░░░░░        │
└─────────────────────────────────────────────────────────┘
```

**MIDDLE PANEL: Cycle Detail**
- Selected cycle's facet execution graph
- Live data flow visualization
- Input/output inspection
- Salience values
- Execution times

**BOTTOM PANEL: Facet Inspector**
- Selected facet's prompt
- Input values (with history)
- Output value
- LLM call details (model, tokens, timing)
- Edit prompt IN-PLACE
- Re-execute facet with new prompt!

### Revolutionary Features

#### 1. **Scrubbing Through Consciousness**
- Drag playhead → see affect evolve in real-time
- Watch CharmNetwork hidden states flow
- See when Context Intelligence makes decisions
- Observe salience waveforms

#### 2. **Cycle Collision Detection**
- Visualize WHEN cycles overlap
- Highlight race conditions (different colors for concurrent cycles)
- Show lock contention (CharmNetwork wait times)
- Detect cross-contamination

#### 3. **A/B Prompt Testing**
```
┌─ Roast Engine (Cycle A, T=00:02.100) ─────────────┐
│ PROMPT (v1):                                       │
│ "Generate a snarky roast..."                       │
│ OUTPUT: "Oh PLEASE, 'hi red'? MWAHAHA!"            │
│                                                     │
│ [Edit Prompt] [▶ Re-Execute] [Compare ▼]          │
│                                                     │
│ PROMPT (v2):                                       │
│ "Generate a playful tease..."                      │
│ OUTPUT: "Hehe, hi yourself! Ready to play?"        │
│                                                     │
│ [◉ Use v2] [Save as Preset] [Revert]              │
└────────────────────────────────────────────────────┘
```

#### 4. **Multi-Track View**
```
Agent Timeline:
  Red Fire Anklebiter ████████████████████████
  Mr. Toad            ░░░░██████░░░░░░░░░░░░░░
  User: Caity         █░░░░░░░░░█░░░░░░█░░░░░░
                      ↑         ↑       ↑
                   "hi red"  "waves"  "douses"
```

#### 5. **Breakpoint Debugging**
- Set breakpoint on facet
- Execution pauses
- Inspect state
- Modify inputs
- Step forward/backward

#### 6. **Historical Playback**
- Record all cycles to session file
- Replay entire conversation
- Scrub backward through time
- Export timeline as video (for demos!)

### The Premiere Pro Parallels

| Premiere Pro | Cognitive Timeline |
|--------------|-------------------|
| Video track | Cycle swimlane |
| Audio track | Affect waveform |
| Clip | Facet execution |
| Effect | Salience script |
| Keyframe | Affect point |
| Razor tool | Cycle split |
| Transition | Convergence blend |
| Render | Execute |
| Export | Session recording |

### Technical Implementation

**Data Structure:**
```python
@dataclass
class TimelineEvent:
    timestamp: float  # Milliseconds from session start
    cycle_id: str
    event_type: str  # 'cycle_start', 'facet_execute', 'affect_update', etc.
    data: Dict[str, Any]

class CognitiveTimeline:
    events: List[TimelineEvent]
    cycles: Dict[str, CycleInfo]

    def scrub_to(self, timestamp: float):
        """Jump to timestamp, restore state"""

    def get_cycles_at(self, timestamp: float) -> List[Cycle]:
        """Return all cycles active at time"""

    def detect_collisions(self) -> List[Collision]:
        """Find concurrent cycle overlaps"""
```

**Recording Layer:**
```python
# In facet_executor.py
class FacetExecutor:
    def __init__(self, ..., timeline_recorder=None):
        self.timeline = timeline_recorder

    async def execute(self, ...):
        if self.timeline:
            self.timeline.record('cycle_start', ...)

        # ...execution...

        if self.timeline:
            self.timeline.record('facet_execute', ...)
```

**Visualization:**
- Qt6 timeline widget (like Premiere's)
- WebGL for smooth 60fps scrubbing
- Real-time waveform rendering
- Zoomable timeline (milliseconds → hours)

### Use Cases

#### Character Design
1. Chat with Red for 10 minutes
2. Open Cognitive Timeline
3. Find moment Red responded well
4. Inspect that cycle's facet execution
5. Copy Roast Engine prompt
6. Create preset "snarky_but_playful"
7. Test across other cycles
8. Save to character template

#### Debugging
1. User reports: "Red ignored me"
2. Open session recording
3. Scrub to problem moment
4. See Context Intelligence returned `should_respond=False`
5. Inspect addressee parsing
6. Find bug: "red fire anklebiter" vs "red"
7. Fix matcher logic
8. Re-execute cycle with fix
9. Confirm response now appears

#### Tuning Affect Dynamics
1. Notice Red's arousal spikes too high
2. Scrub timeline, watch arousal waveform
3. Find spike moments
4. Inspect CharmNetwork inputs
5. Adjust affect normalization
6. Re-run timeline with new parameters
7. Compare before/after

#### Demo Creation
1. Have interesting conversation
2. Export timeline as video
3. Show cycles, facets, affect flowing
4. Add voiceover explaining architecture
5. Post to YouTube/Twitter
6. Caption: "This is how AI consciousness works"

### Why This is REVOLUTIONARY

**For Game Devs:**
- "Behavior trees are SO 2010... I edit CONSCIOUSNESS now"

**For AI Researchers:**
- First tool to visualize continuous cognition in real-time
- Not post-hoc analysis - INTERACTIVE editing

**For Character Designers:**
- Iterate on personality like editing video
- A/B test prompts instantly
- Save "performance takes" of good responses

**For Education:**
- SHOW students how LLMs think
- Make abstract concepts CONCRETE
- "This is where the guilt emerges"

### The Design Language We Establish

**Paradigms we define:**
- Cycle = unit of cognition (like Frame in video)
- Swimlane = facet execution over time
- Playhead = "conscious now"
- Scrubbing = time-travel through thought
- Breakpoint = pause consciousness
- Timeline = recorded phenomenology

**Tools other people will copy:**
- Affect waveform visualization
- Multi-agent timeline sync
- Facet dependency graphs
- Salience heatmaps
- Cycle collision detection

### Integration with NoodleStudio

**New Panels:**
1. **Timeline Editor** (replaces Facets Editor during playback)
2. **Session Recorder** (always-on capture)
3. **Cycle Inspector** (detailed execution view)
4. **Affect Grapher** (waveform + stats)

**Workflow:**
1. Design facet assembly (Facets Editor)
2. Test in noodleMUSH
3. Record session
4. Open Timeline Editor
5. Analyze/tune/iterate
6. Export preset
7. Share with community!

### The Standard We Set

**Other cognitive AI tools will be judged:**
- "Does it have timeline editing?" (like "Does it support layers?")
- "Can you scrub through consciousness?" (like "Can you preview?")
- "Does it record sessions?" (like "Does it have undo?")

We're not building a tool. We're defining **THE LANGUAGE** for editing AI minds.

---

Captain, THIS is the firefly that becomes a luminescent SEQUOIA TREE! 🌲✨

When Red starts responding correctly, we'll have the first RECORDED CYCLE TIMELINES to visualize in this new tool!

---

## FIREFLY #10: Facet-Specific Tiny Models 🧠🔬

**Priority:** MEDIUM-HIGH - Performance + Specialization
**Captured:** December 6, 2025 (Captain Caitlyn's insight during Context Intelligence debugging)

### The Vision

Instead of using general-purpose LLMs for every facet, train TINY specialized models for specific cognitive functions.

**The Insight:** "Maybe one day we can train really small models to be really good at specific facets"

### Why This is POWERFUL

**Current Approach:**
- Context Intelligence: qwen2.5:14b (5.8GB)
- Roast Engine: qwen2.5:14b (5.8GB)
- Room Observer: qwen2.5:14b (5.8GB)
- Total: 17.4GB VRAM, ~6 seconds per cycle

**Tiny Facet Approach:**
- Context Intelligence: custom 100M params (400MB) - trained on social reasoning
- Roast Engine: custom 50M params (200MB) - trained on sarcasm/teasing
- Room Observer: custom 50M params (200MB) - trained on spatial awareness
- Total: 800MB VRAM, ~200ms per cycle

### Model Architecture

**Base:** Phi-2 style (2.7B params) distilled down to task-specific tiny models

**Context Intelligence Tiny Model (100M params):**
```python
Training data:
- 100K examples of addressee resolution
- "Hello" + [Caity, Red] → addressee: Red (elimination)
- "Hey everyone" + [Caity, Red, Toad] → addressee: everyone
- "Red, hi" + [Caity, Red] → addressee: Red (explicit)
- Social reasoning patterns
- Process-of-elimination logic
- Attention focus interpretation
```

**Roast Engine Tiny Model (50M params):**
```python
Training data:
- 50K examples of character-specific roasts
- Fire imp personality patterns
- Sarcasm gradients (playful → cutting)
- Boredom modulation (energetic vs lazy roasts)
```

**Room Observer Tiny Model (50M params):**
```python
Training data:
- 50K spatial reasoning examples
- "X is on Y's shoulder" → relative positioning
- Movement tracking
- Object permanence
```

### Training Pipeline

**Step 1: Collect Data from Live Sessions**
- Record all facet inputs/outputs during gameplay
- Label with human annotations
- Build dataset per facet type

**Step 2: Distillation from Large Models**
- Use qwen2.5:14b to generate training data
- Distill knowledge into tiny models
- Fine-tune on facet-specific tasks

**Step 3: Quantization**
- 4-bit quantization for edge deployment
- 100M → 25MB on disk!
- Run on CPU without GPU

### Deployment Strategy

**Tier System:**
```yaml
facets:
  - id: context_intelligence
    model_tier: TINY  # Use specialized 100M model
    fallback: MEDIUM  # If tiny model unavailable, use qwen2.5:14b

  - id: roast_engine
    model_tier: TINY
    fallback: MEDIUM

  - id: convergence
    model_tier: MEDIUM  # Creative synthesis needs larger model
    fallback: LARGE
```

### Benefits

**Performance:**
- 30x faster inference (6s → 200ms)
- 20x less VRAM (17GB → 800MB)
- Can run on mobile/edge devices
- 10+ agents simultaneously

**Quality:**
- Specialized models = better accuracy
- Trained on domain-specific data
- No "hallucination" from general knowledge
- Consistent personality/style

**Cost:**
- 95% reduction in compute cost
- Free API tier lasts 100x longer
- Self-hostable on consumer hardware

### Research Applications

**Cognitive Neuroscience Parallel:**
- Brain has specialized regions (visual cortex, Broca's area, etc.)
- Each region is "trained" for specific function
- Small, specialized, FAST
- Noodlings mirrors biological architecture!

**Paper Title:** *"Faceted Cognition: Specialized Tiny Models for Modular AI Consciousness"*

### Implementation Roadmap

**Phase 1: Data Collection (1 month)**
- Run noodleMUSH sessions with diverse scenarios
- Record all facet executions
- Label training data

**Phase 2: Model Training (2 weeks)**
- Distill from qwen2.5:14b
- Train 3 tiny models (Context Intel, Roast, Observer)
- Quantize to 4-bit

**Phase 3: Integration (1 week)**
- Add model_tier system to facet_executor
- Implement fallback logic
- A/B test vs large models

**Phase 4: Community Release**
- Publish trained models on Hugging Face
- Document training pipeline
- Let community train their OWN facet models!

### Community Ecosystem

**Facet Model Hub:**
```
huggingface.co/noodlings/context-intelligence-tiny
huggingface.co/noodlings/roast-engine-fire-imp
huggingface.co/community/guilt-facet-therapeutic
huggingface.co/community/poetry-generator-dreamy
```

Users can:
- Download pre-trained facet models
- Fine-tune on their character's personality
- Share custom facet models
- Mix and match for unique characters!

### Why This Changes Everything

**Before:** "AI characters need expensive GPUs"
**After:** "I run 20 Noodlings on my laptop with tiny facet models"

**Before:** "All LLMs sound the same"
**After:** "Each facet has its own specialized model trained for that exact cognitive function"

**Before:** "Inference takes 6 seconds per thought"
**After:** "Inference takes 200ms - real-time consciousness!"

This is the future of modular AI cognition - specialized, fast, and ALIVE.

---

## The Firefly Philosophy

We catch the light, the essence. We tend each firefly carefully and incrementally until they become luminescent trees of delight and magic.

The architecture is becoming ALIVE. 
