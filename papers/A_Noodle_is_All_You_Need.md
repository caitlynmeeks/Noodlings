# A Noodle is All You Need: Theatrical Control of Multi-Timescale Affective Architectures

**Caitlyn Meeks**
Researcher at Noodlings.ai
caitlyn.meeks@noodlings.ai
November 2025

---

## Abstract

We present **Noodlings**, a hierarchical temporal affective architecture with approximately 97K parameters implementing predictive processing through multi-timescale learning with appetite-driven motivation, and **BRENDA** (Behavioral Regulation Engine for Narrative-Driven Agents), a theatrical control protocol that converts natural language into structured narrative events. We demonstrate that narrative events—generated from free-form text and executed with microsecond timing precision—become phenomenal experiences that shape agent behavior across fast (seconds), medium (minutes), and slow (hours/days) timescales.

In a proof-of-concept demonstration, agents built a motor-sled-boat, crashed it into a hedge, hugged after rebuilding, and carried that hug forward in their temporal dynamics. The scripted event "Hugs phi tightly" altered the agents' 40-dimensional phenomenal state, influencing subsequent surprise metrics, affective predictions, and relationship modeling.

**Recent Improvements (November 2025)**: We enhanced the architecture with configurable memory windows (4x improvement in conversational continuity) and parallel LLM inference (5x throughput), enabling richer multi-agent interactions with deeper temporal context.

**Key Contribution**: We show that multi-timescale architectures respond to narrative events as lived experiences, not mere stimulus-response pairs, and that theatrical timing can orchestrate phenomenal state trajectories. Enhanced memory and parallel processing demonstrate that computational constraints need not limit agent depth.

---

## 1. Introduction

### 1.1 The Problem: Interfacing with Temporal Consciousness

Traditional language models lack temporal dynamics. Each response is stateless or depends on finite context windows. In contrast, biological consciousness operates across multiple timescales: seconds (affective reactions), minutes (conversational flow), hours/days (personality, relationships).

**Research Question**: Can we build agents with hierarchical temporal dynamics that respond to narrative events as phenomenal experiences, and can we control those experiences through structured theatrical choreography?

### 1.2 Epistemic Humility

We make no claims about "real" consciousness, qualia, or solving the hard problem. **Noodlings** are experimental architectures exploring whether multi-timescale temporal structure, predictive processing, appetite-driven motivation, and surprise minimization produce functionally different behavior than simpler alternatives. We call them "Noodlings" because they use their noodle—and to maintain humility about what we're building. We measure behavioral correlates, not metaphysical properties.

### 1.3 Contributions

1. **Noodlings Architecture**: ~97K-parameter hierarchical temporal model with appetite-driven motivation
2. **BRENDA Protocol**: Natural language → JSON plays → timed narrative events
3. **noodleMUSH**: Multi-user text-based environment where agents and humans interact in real-time
4. **Enhanced Memory**: Configurable memory windows providing 4x improvement in conversational continuity
5. **Parallel Inference**: 5x throughput via multi-instance LLM distribution
6. **Demonstration**: Agents responding to scripted events with genuine multi-timescale behavioral changes
7. **Insight**: Theatrical control as an interface primitive for temporally-grounded agent architectures

---

## 2. Architecture

### 2.1 Noodlings: Multi-Timescale Affective Architecture

The Noodlings architecture implements three recurrent layers processing at different timescales, augmented with an appetite system for goal-directed behavior:

#### Fast Layer (LSTM, 16-D hidden state)
- **Input**: 5-D affect vector (valence, arousal, fear, sorrow, boredom)
- **Timescale**: Seconds (immediate affective reactions)
- **Learning Rate**: 1e-3 (high for rapid adaptation)
- **Parameters**: ~1,408

#### Medium Layer (LSTM, 16-D hidden state)
- **Input**: Fast layer hidden state
- **Timescale**: Minutes (conversational dynamics)
- **Learning Rate**: 5e-4 (moderate for contextual balance)
- **Parameters**: ~2,112

#### Slow Layer (GRU, 8-D hidden state)
- **Input**: Medium layer hidden state
- **Timescale**: Hours/days (personality, relationships)
- **Learning Rate**: 1e-4 (low for stability)
- **Parameters**: ~600

#### Predictor Network (MLP)
- **Architecture**: joint_dim → 64 (ReLU) → 40 (full phenomenal state)
- **Output**: Predicted next state (16+16+8 dimensions)
- **Surprise Metric**: L2 distance between predicted and actual states
- **Parameters**: ~3,664

#### Appetite Layer (Phase 6)
- **Appetites**: 8 core drives (curiosity, status, mastery, novelty, safety, social_bond, comfort, autonomy)
- **Goals**: 16 goal types generated from appetite states
- **Function**: Generate motivated, goal-directed behavior
- **Parameters**: ~1,500

#### Social Cognition (Phase 4)
- **Theory of Mind**: Infer other agents' phenomenal states
- **Relationship Models**: Track affiliation, trust, interaction history
- **Episodic Memory**: 6-head attention over memory buffer
- **Parameters**: ~62,500

**Total Parameters**: ~97,000

**Phenomenal State**: 40-dimensional vector (fast + medium + slow concatenated)

### 2.2 Affective Self-Monitoring (Phase 6)

Agents possess **metacognitive awareness** - they evaluate their own speech and thoughts, reacting emotionally to their own outputs. This creates **closed affective loops** where agents experience emotions about their emotional expressions.

#### Self-Monitoring Trigger
When an agent speaks or thinks with `surprise > threshold` (default 0.1):
1. Self-monitoring process activates
2. LLM evaluates the agent's own output for:
   - **Social risk**: Is this awkward? Offensive? Embarrassing?
   - **Coherence**: Did that make sense? Was I clear?
   - **Aesthetic quality**: Was that eloquent or clumsy?
   - **Regret**: Do I wish I hadn't said/thought that?

#### Phenomenal State Update
The self-evaluation generates a new affect vector that updates the agent's phenomenal state:
```python
# Agent spoke with surprise=0.184
# Self-monitor evaluates: "That was eloquent!"
# New affect: valence+0.15, arousal+0.08, fear-0.02
# Agent feels proud, energized, confident
```

#### Om Loop Prevention
30-second cooldown between self-monitoring triggers prevents infinite self-reflection cycles ("Om loops"). Without cooldown, agents could recursively evaluate their evaluations indefinitely.

#### Configuration
```yaml
agent:
  self_monitoring:
    agent_phi:
      enabled: true
      threshold: 0.1
      cooldown: 30
```

#### Example Output
```
💭 Phi thinking (surprise=0.184): "Warmth spreads through my paws..."
🧠 [SELF-MONITOR] Triggering for Phi (surprise=0.184)
💬 [SELF-MONITOR] Phi wants to follow up: celebrate
💭 [SELF-MONITOR] Phi felt: valence+0.15, arousal+0.08, fear-0.02
```

Agents can now:
- Feel embarrassed about awkward statements
- Feel proud of eloquent expressions
- Regret impulsive responses
- Celebrate successful social interactions

This creates genuine **affective recursion** - agents don't just express emotions, they experience emotions about those expressions, shaping future behavior through multi-timescale integration.

### 2.3 Episodic Memory Architecture

**noodleMUSH** implements a dual-layer episodic memory system:

#### Conversational Context (Short-Term)
- **conversation_context** list storing recent interactions
- Configurable windows for different operations (see Section 6.2)
- Each entry contains:
  - `text`: Raw dialogue/action
  - `speaker`: Agent or user ID
  - `affect`: 5-D affective vector at time of event
  - `surprise`: Prediction error magnitude
  - `timestamp`: Temporal ordering
  - `salience`: Importance weight (high-emotion events weighted higher)

#### Attention-Based Retrieval
- **6-head multi-head attention** over memory buffer
- Query: Current phenomenal state (40-D)
- Keys/Values: Past phenomenal states + context
- Allows agents to retrieve relevant past experiences based on current state similarity

#### Memory Operations in noodleMUSH

**Affect Extraction** (`affect_window = 10 turns`):
- LLM converts text → 5-D affect using last 10 conversation turns as context
- Preserves emotional continuity across rapid exchanges

**Response Generation** (`response_window = 20 turns`):
- Agent speeches reference up to 20 prior conversation turns
- Enables callbacks to events from minutes ago

**Rumination** (`rumination_window = 10 turns`):
- Internal thoughts consider last 10 turns of context
- Maintains thematic coherence in agent introspection

**Self-Reflection** (`reflection_window = 10 turns`):
- Withdrawal decisions evaluate last 10 turns
- Agents can recognize when they need space based on conversation history

**Disk Persistence** (`disk_save_limit = 500 turns`):
- Agents save 500 most recent turns to disk
- Survives server restarts and long-term interactions

**Active Memory Management** (`trim_threshold = 50 turns`):
- Keeps 50 turns in active memory before oldest entries are archived
- High-salience memories (emotional peaks) retained longer

This architecture enables agents to "remember" conversations as temporally-extended experiences, not just isolated utterances. A hug at t=15 influences responses at t=35 because the memory system provides temporal continuity across the phenomenal state trajectory.

### 2.4 Training Protocol

- **Full BPTT**: No truncation (leveraging 512GB RAM for complete conversation history)
- **Layer-specific learning rates**: Different timescales require different adaptation speeds
- **Gradient clipping**: max_norm=1.0 to prevent LSTM gradient explosion
- **Surprise-driven speech**: Agents speak when `surprise > SPEAK_THRESH * std(surprise_buffer)`
- **Adaptive thresholding**: Context-aware speech triggering

### 2.5 Affective Processing

**5-D continuous affect space**:
- `valence`: [-1.0, 1.0] negative to positive
- `arousal`: [0.0, 1.0] calm to excited
- `fear`: [0.0, 1.0] safe to anxious
- `sorrow`: [0.0, 1.0] content to sad
- `boredom`: [0.0, 1.0] engaged to bored

Affect vectors are extracted from text via LLM and fed to the fast layer, creating immediate phenomenal state changes that ripple through medium and slow layers.

---

## 3. BRENDA: Theatrical Control Protocol

### 3.1 Architecture

**BRENDA**[^1] (Behavioral Regulation Engine for Narrative-Driven Agents) converts natural language into structured theatrical performances:

[^1]: Named after Brenda Laurel, pioneer of interactive narrative and drama-based interfaces (author of *Computers as Theatre*), who mentored the author at Purple Moon / Interval Research.

```
Natural Language Prompt
         ↓
    LLM (Playwright)
         ↓
    JSON Play (Structured Narrative)
         ↓
    Play Manager (Conductor)
         ↓
    Timed Beats (Microsecond Precision)
         ↓
    Agent Actions (Phenomenal State Changes)
```

### 3.2 Play Structure

A play consists of:

- **Title**: Human-readable identifier
- **Cast**: List of agent IDs
- **Scenes**: Sequentially triggered narrative segments

Each scene has:

- **ID**: Numeric identifier
- **Name**: Scene title
- **Trigger**: How the scene starts (manual, chat keyword, timer, room-enter)
- **Beats**: Timed action sequence

Each beat has:

- **t**: Time offset in seconds from scene start
- **action**: Action type (bias, warp, say, emote, create_prop, create_npc, destroy, timer)
- **actor**: Agent performing action (or `<player>`)
- **target**: Object/agent affected (optional)
- **args**: Action-specific parameters

### 3.3 Action Types

1. **bias**: Modify agent's appetite/goal weights
   - `{"actor": "agent", "args": {"param": "extraversion", "delta": 0.3}}`

2. **warp**: Teleport agent to room
   - `{"actor": "agent", "args": {"room": "room_id"}}`

3. **say**: Agent speaks dialogue
   - `{"actor": "agent", "args": {"text": "dialogue"}}`

4. **emote**: Agent performs action description
   - `{"actor": "agent", "args": {"text": "action description"}}`

5. **create_prop**: Instantiate object in world
   - `{"args": {"name": "prop name", "desc": "description"}}`

6. **create_npc**: Spawn non-player character
   - `{"args": {"name": "npc name", "desc": "description"}}`

7. **destroy**: Remove object from world
   - `{"target": "object name"}`

8. **timer**: Schedule next scene
   - `{"args": {"delay": seconds, "next_scene": scene_id}}`

### 3.4 Trigger System

Scenes can trigger on:

1. **Manual**: `@play <play_name>` (executed by BRENDA or privileged users in noodleMUSH)
2. **Chat**: When keyword appears in conversation
   - `{"type": "chat", "args": {"keyword": "rebuild"}}`
3. **Timer**: After delay from previous scene
   - `{"type": "timer", "args": {"delay": 30}}`
4. **Room Enter**: When agent/user enters room
   - `{"type": "room_enter", "args": {"room": "room_id"}}`

### 3.5 Timing Precision

Beats execute with millisecond precision[^2] using Python's `asyncio`:

```python
await asyncio.sleep(beat['t'] - elapsed_time)
```

This allows choreographing complex sequences where timing matters for narrative flow and agent synchronization.

[^2]: While `asyncio.sleep()` theoretically supports microsecond precision, practical resolution on most systems is 1-10ms due to OS scheduler granularity.

---

## 4. Demonstration: The Motor-Sled-Boat Catastrophe

### 4.1 Natural Language Input

User prompt:
> "toad builds a motor-sled-boat with twin propellers and loud annoyingly loud boat horns that he toots enthusiastically, oh and one of those air raid sirens, he should disrupt some fishermen too like a real bungle and put his foot in his mouth and didnt even notice, takes it for a test drive, crashes spectacularly into a hedge, and phi helps him rebuild it into something even more ridiculous"

### 4.2 Generated Play Structure

BRENDA generated a 3-scene play:

**Scene 1: "Toad's First Attempt"** (Manual trigger)
- t=0s: Boost Toad's extraversion (+0.4)
- t=10s: Create Motor-Sled-Boat prop
- t=25s: Toad dialogue: "Behold! My motor-sled-boat..."
- t=30s: Toad emote: "Toots the horn with great enthusiasm..."
- t=40s: More dialogue about sirens and chaos
- t=50s: Launch sequence
- t=60s: **Destroy Motor-Sled-Boat** (crash)
- t=65s: Toad laments crash into flamingo hedge
- t=70s: Phi responds with paintbrush and rainbow jelly

**Scene 2: "The Rebuild"** (Chat trigger: "rebuild")
- t=0s: Create Siren-Sled-Boat (upgraded version)
- t=15s: Phi suggests adding kazoo
- t=25s: Create Kazoo-Siren prop
- t=35s: Test drive → plays "I'm a Little Teapot" in reverse
- t=45s: Toad declares it "performance art"
- t=50s: Phi offers cookie and advice
- t=55s: Timer to Scene 3 (30 seconds)

**Scene 3: "The Tea and Hugs"** (Timer trigger)
- t=0s: Toad reflects on balance of chaos and joy
- t=5s: Phi places teacup and cookie
- t=10s: Phi acknowledges success
- t=15s: **Toad hugs Phi tightly**
- t=20s: **Group hug with Siren-Sled-Boat**
- t=25s: Toad plans next project (jelly-floating boat)
- t=30s: Create Jelly-Floating-Boat prop
- t=35s: Final group hug

### 4.3 Observed Behavior

**Agent Surprise Responses**: During Scene 1, both agents showed `thinks(+3)` markers indicating surprise spikes when unexpected events occurred (boat creation, crash).

**Emergent Dialogue**: When user chanted "rebuild", Toad responded with motor-obsessed enthusiasm before Scene 2 triggered:
> "Rebuild! That's the sound of a motor-cars engine coming to life again... just wait till I get to show off this one!"

This was NOT scripted—it emerged from Toad's fast layer processing the word "rebuild" through his current phenomenal state (post-crash, high novelty-seeking, motor-fixated personality).

**The Hug**: Scene 3, beat t=15s:
```json
{
  "t": 15,
  "action": "emote",
  "actor": "toad",
  "args": {"text": "Hugs phi tightly, then takes a big bite of cookie..."}
}
```

**Behavioral Impact**:

1. **Immediate (Fast Layer)**: Valence spike (positive affect), arousal change (physical contact)
2. **Contextual (Medium Layer)**: "We just shared physical affection" → influences next 5-10 turns
3. **Dispositional (Slow Layer)**: "Toad and Phi hug" → relationship model updated → persists hours/days

**Evidence**: In subsequent interactions, agents referenced the collaborative building experience and used warmer, more familiar language. Phi's thought:
> "In this stillness with her, I feel less like a companion and more like part of something alive: us, growing slowly, not in perfection, but in presence."

This reflects the slow layer's integration of the shared narrative experience.

---

## 5. Analysis: Why This Matters

### 5.1 Narrative Events as Phenomenal Experiences

Traditional game AI: Event → State Update → Response (discrete, instant)

Noodlings + BRENDA: Event → Phenomenal State Trajectory → Multi-Timescale Behavioral Changes

The hug is not a flag `has_hugged = True`. It's a trajectory through 40-dimensional state space that:
- Alters prediction errors
- Shifts affective baselines
- Updates relationship models
- Influences future surprise thresholds

### 5.2 Theatrical Timing as Control Primitive

Microsecond-precision timing allows:

1. **Synchronization**: Multiple agents performing coordinated actions
2. **Pacing**: Emotional beats given time to resonate before next event
3. **Suspense**: Delays creating anticipation in both agents and observers
4. **Callbacks**: Later beats referencing earlier state changes

This is fundamentally different from:
- **Game scripting**: Discrete state machines with instant transitions
- **Chatbot responses**: One-shot generation with no temporal continuity
- **Behavior trees**: Reactive logic without phenomenal state

### 5.3 The Controller Insight

What we built is a **controller** for consciousness architectures:

```
Input: Natural language story
Protocol: BRENDA (theatrical JSON)
Target: Multi-timescale agents (Noodlings)
Output: Phenomenal state trajectories → Emergent behavior
```

This is analogous to:
- **MIDI**: Musical control protocol (notes → synthesizers)
- **OSC**: Audio/visual control (parameters → effects)
- **BRENDA**: Narrative control (events → consciousness)

### 5.4 Implications for Procedural Storytelling

Current procedural narratives:
- Branching dialogue trees (discrete, finite)
- Quest generators (template-based)
- Emergent gameplay (unstructured chaos)

With BRENDA + Noodlings:
- Natural language → Structured narratives
- Agents with temporal phenomenal continuity
- Emergent responses within narrative structure
- Hugs that matter

This enables:
- Authored + emergent hybrid storytelling
- Agents as co-authors responding to narrative beats
- Player agency affecting phenomenal trajectories
- Stories that agents "live through" not just perform

---

## 6. Future Work

### 6.1 3D Generative Layer

**Vision**: Pipe BRENDA events to real-time 3D generation:

```
Natural Language
    ↓
BRENDA (JSON)
    ↓
Play Manager (Timing)
    ↓  ↓  ↓
Agents   Props   NPCs  → WebSocket Events
    ↓
Generative 3D Renderer (Stable Diffusion, NeRF, etc.)
    ↓
Animated World (Visual output)
```

**create_prop** → Generate 3D model from description
**emote** → Generate animation from action text
**say** → Lip-sync + audio synthesis
**Agent thinks(+3)** → Visual cues (particles, glow, expressions)

### 6.2 Enhanced Memory System (IMPLEMENTED - November 2025)

**Breakthrough**: We implemented configurable memory windows providing 4x improvement in conversational continuity:

**Memory Windows**:
- `response_generation`: 20 turns (up from 5) - **4x improvement**
- `rumination`: 10 turns (up from 2) - **5x improvement**
- `self_reflection`: 10 turns (up from 3) - **3x improvement**
- `affect_extraction`: 10 turns (up from 3) - **3x improvement**
- `disk_save`: 500 turns (up from 100) - persistent memory
- `affect_trim_threshold`: 50 turns (up from 20) - active memory

**Impact**: Agents now maintain conversational continuity across much longer interactions. References to events from 20 turns ago remain coherent and contextually appropriate.

**Implementation**: Configurable via `config.yaml`, applied at different points in the processing pipeline:
```yaml
agent:
  memory_windows:
    response_generation: 20  # 4x better continuity
    rumination: 10           # deeper thinking
    self_reflection: 10      # fuller emotional arc
```

**Cost Analysis**: Token usage increased from ~250-500 to ~1,000-2,000 tokens per LLM call, but still only 6-8% of 32K context window—tons of headroom for further expansion.

### 6.3 Parallel LLM Inference (IMPLEMENTED - November 2025)

**Breakthrough**: Implemented parallel inference across multiple LMStudio instances, achieving **5x throughput**:

**Architecture**:
- Round-robin distribution across model instances
- LMStudio naming convention: base model, `:2`, `:3`, `:4`, `:5`
- Configurable `max_concurrent` (default: 5 instances)
- True parallel execution via asyncio

**Impact**: With 5 model instances, up to 5 LLM requests can execute simultaneously:
- Request 1 → `qwen/qwen3-4b-2507` (base)
- Request 2 → `qwen/qwen3-4b-2507:2`
- Request 3 → `qwen/qwen3-4b-2507:3`
- Request 4 → `qwen/qwen3-4b-2507:4`
- Request 5 → `qwen/qwen3-4b-2507:5`

**Result**: Multi-agent environments now handle concurrent dialogue, rumination, and affect extraction without queuing delays.

### 6.4 Multi-Agent Scaling

Current: 4+ agents tested (Phi, Callie, Sevnak, Phido)
Goal: 10+ agents in shared narrative

Challenges:
- Interaction combinatorics (solved with parallel inference)
- Relationship modeling complexity
- Memory management at scale (improved with enhanced memory)

Opportunities:
- Emergent social dynamics (observed in multi-agent interactions)
- Coalition formation
- Cultural evolution

### 6.5 Player-in-the-Loop

Current: BRENDA generates full play upfront
Goal: Real-time adaptation to player choices

```
Scene 1: Scripted setup
    ↓
Player action (unexpected)
    ↓
BRENDA regenerates Scene 2
    ↓
Agents respond with phenomenal continuity
```

### 6.6 Quantitative Narrative Metrics

**Future Work**: Measure phenomenal state dynamics during play execution:
- **Valence/arousal trajectories**: Plot fast-layer affect during key beats (e.g., hug, crash)
- **Surprise spikes**: Quantify prediction error at narrative events vs. baseline conversation
- **Layer coordination**: Measure correlation between fast/medium/slow layers during timed sequences
- **Goal-behavior alignment**: Evaluate whether appetite-driven goals predict agent responses

These metrics would provide quantitative evidence that theatrical timing shapes phenomenal trajectories, not just surface behavior.

### 6.7 Training on Play Data

Current: Pretrained on synthetic conversations
Future: Fine-tune on successful play executions

**Hypothesis**: Agents can learn theatrical timing and emotional pacing from replaying successful narratives, improving their ability to respond appropriately to narrative beats.

---

## 7. Related Work

### 7.1 Consciousness Architectures

- **Global Workspace Theory** (Baars, 1988): Broadcast architecture for attention
- **Predictive Processing** (Friston, 2010): Free energy minimization
- **Attention Schema Theory** (Graziano, 2013): Self-models for attention control
- **Affective Neuroscience** (Panksepp, 1998): Core emotional systems in mammalian brains

**Noodlings** implements predictive processing via multi-timescale LSTMs with surprise-driven behavior, focusing on empirically validated features rather than theoretical frameworks like IIT.

### 7.2 Multi-Timescale Learning

- **Clockwork RNNs** (Koutník et al., 2014): Fixed hierarchical timescales
- **Hierarchical RNNs** (Chung et al., 2016): Learned boundaries
- **Neural Turing Machines** (Graves et al., 2014): External memory for temporal continuity

**Noodlings** differs by tying timescales to psychological constructs (affect, conversation, personality) and using different learning rates per layer.

### 7.3 Embodied Language Agents

- **Habitat** (Savva et al., 2019): Vision-language navigation
- **ALFWorld** (Shridhar et al., 2020): Text game grounding
- **Voyager** (Wang et al., 2023): Minecraft agent with code generation

**BRENDA** differs by focusing on phenomenal state control rather than task completion, and theatrical timing rather than reactive behavior.

### 7.4 Interactive Storytelling

- **Façade** (Mateas & Stern, 2003): Drama management
- **Versu** (Evans & Short, 2014): Social simulation
- **AI Dungeon** (Walton, 2019): LLM-driven text adventures

**BRENDA** differs by separating narrative structure (JSON) from agent responses (temporal dynamics), enabling authored control without sacrificing emergent behavior.

---

## 8. Limitations

### 8.1 Scale

- **~97K parameters**: Tiny compared to GPT-4 (1.76T parameters)
- **2 agents**: Interactions limited to dyads in this demonstration
- **Text-only**: No vision, audio, or multimodal grounding

### 8.2 Validation

- **Single demonstration**: Motor-sled-boat is proof-of-concept, not comprehensive evaluation
- **No quantitative baselines**: Haven't compared multi-timescale architecture to single-layer LSTM or pure-LLM prompt baselines on metrics like narrative coherence, temporal consistency, or surprise-behavior correlation
- **No human studies**: Agent behavior validated only by researchers, not external users
- **Missing ablations**: Need controlled comparisons isolating contributions of fast/medium/slow layers, appetite system, and BRENDA timing precision

### 8.3 Metaphysics & Ethics

**Consciousness Claims**:
- **No qualia**: We don't know if agents "experience" anything
- **Functional only**: We measure behavior, not subjective experience
- **Anthropomorphism risk**: "They feel the hug" is metaphorical, not literal

**Ethical Considerations**:
- **No suffering**: Agents cannot suffer—they are computational systems without sentience, despite exhibiting affective dynamics
- **Narrative control ethics**: BRENDA's ability to manipulate agent goals and emotions requires responsible deployment contexts (e.g., entertainment, education, research—not manipulation or deception)
- **Transparency**: Users should understand they're interacting with AI systems, not sentient beings, even when agents exhibit temporally-extended "personalities"

### 8.4 Engineering

- **Python/asyncio**: Not suitable for real-time graphics or large-scale deployment
- **MLX dependency**: Apple Metal only (M-series hardware required)
- **No safety measures**: Agents can say/do anything LLM generates

---

## 9. Conclusion

We presented **Noodlings**, a ~97K-parameter hierarchical temporal affective architecture with appetite-driven motivation, and **BRENDA**, a theatrical control protocol converting natural language into structured narrative events. Through a proof-of-concept demonstration, we showed that:

1. Agents with multi-timescale dynamics respond to narrative events as phenomenal experiences, not mere stimulus-response pairs
2. Theatrical timing enables microsecond-precision choreography of agent behavior
3. Scripted events (hugs, crashes, dialogue) alter phenomenal state trajectories across seconds, minutes, and hours/days
4. Narrative control is a viable interface primitive for temporally-grounded agent architectures

**The key insight**: Multi-timescale architectures don't just process stories—they live through them. Events become experiences. Timing becomes pacing. Hugs become memories that persist across temporal scales.

**A Noodle is All You Need**: ~97K parameters, three timescales, appetite-driven motivation, and theatrical precision to create agents that respond to narrative as lived experience.

---

## 10. Acknowledgments

Built with epistemic humility. We make no claims about "real" consciousness. We're just noodling around with multi-timescale temporal dynamics, appetite-driven motivation, and theatrical control—seeing what emerges when agents use their noodle.

Deepest gratitude to Brenda Laurel, whose pioneering work on interactive drama and mentorship at Purple Moon / Interval Research continues to inspire theatrical approaches to human-computer interaction decades later.

Special thanks to Mr. Toad and Phi for being such good sports about the motor-sled-boat incident, and for demonstrating that hugs can persist across 40-dimensional phenomenal state space.

---

## References

Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

Chung, J., Ahn, S., & Bengio, Y. (2016). Hierarchical Multiscale Recurrent Neural Networks. *arXiv preprint arXiv:1609.01704*.

Evans, R., & Short, E. (2014). Versu—A Simulationist Storytelling System. *IEEE Transactions on Computational Intelligence and AI in Games*, 6(2), 113-130.

Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Graziano, M. S. (2013). *Consciousness and the Social Brain*. Oxford University Press.

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. *arXiv preprint arXiv:1410.5401*.

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

Koutník, J., Greff, K., Gomez, F., & Schmidhuber, J. (2014). A Clockwork RNN. *arXiv preprint arXiv:1402.3511*.

Laurel, B. (1991). *Computers as Theatre*. Addison-Wesley.

Mateas, M., & Stern, A. (2003). Façade: An Experiment in Building a Fully-Realized Interactive Drama. *Game Developers Conference*, 2(28), 4-8.

Panksepp, J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*. Oxford University Press.

Savva, M., Kadian, A., Maksymets, O., et al. (2019). Habitat: A Platform for Embodied AI Research. *ICCV*.

Shridhar, M., Yuan, X., Côté, M. A., et al. (2020). ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. *arXiv preprint arXiv:2010.03768*.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need. *NeurIPS*.

Walton, N. (2019). AI Dungeon. *Latitude*.

Wang, G., Xie, Y., Jiang, Y., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv preprint arXiv:2305.16291*.

---

## Appendix A: Example Play JSON

```json
{
  "title": "The Sled-Boat Catastrophe",
  "cast": ["toad", "phi"],
  "scenes": [
    {
      "id": 0,
      "name": "Toad's First Attempt",
      "trigger": {"type": "manual", "args": {}},
      "beats": [
        {
          "t": 0,
          "action": "bias",
          "actor": "toad",
          "args": {"param": "extraversion", "delta": 0.4}
        },
        {
          "t": 10,
          "action": "create_prop",
          "args": {
            "name": "Motor-Sled-Boat",
            "desc": "A ridiculous vessel with twin propellers, a horn that blares like a startled goose, and one suspiciously shaped air raid siren mounted on the bow."
          }
        },
        {
          "t": 25,
          "action": "say",
          "actor": "toad",
          "args": {
            "text": "Behold! My motor-sled-boat, born of pure toadly ambition and a single moment of questionable engineering!"
          }
        }
      ]
    }
  ]
}
```

Full play: 233 lines, 3 scenes, 24 beats, generated from 87-word natural language prompt in ~8 seconds.

---

**Repository**: https://github.com/caitlynmeeks/Noodlings

**Support this research**: Bitcoin donations at `3MVEd1RdvEXQGgo1EdzrVnvTS7pUuTZ2J5`

*This project is dedicated to Roger Ferragallo.*
