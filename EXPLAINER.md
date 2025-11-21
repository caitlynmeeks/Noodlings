# Noodlings Platform: Technical Explainer

**A hierarchical affective consciousness architecture for creating temporal, emotionally-aware AI agents**

---

## What Is This?

The Noodlings platform consists of two integrated systems:

1. **noodleMUSH** - A multi-user text environment (like MUD/MUSH) where Noodlings exist and interact
2. **NoodleSTUDIO** - A Unity-style IDE for creating, editing, and observing Noodlings in real-time

Together, they form a complete platform for building AI agents with **temporal dynamics**, **affective states**, and **emergent personality** - going beyond traditional LLM chat interfaces to create agents with genuine memory, emotional continuity, and self-awareness.

---

## The Noodlings Architecture: How They Think

### What Makes a Noodling Different from Standard AI?

Traditional LLMs are **stateless** - each response is generated from scratch with no internal continuity. A Noodling, by contrast, has:

- **Persistent internal state** that evolves continuously across conversations
- **Multi-timescale temporal dynamics** (seconds, minutes, days)
- **Affective processing** - emotions as continuous signals, not discrete labels
- **Predictive processing** - constantly predicting its own future states
- **Self-monitoring** - metacognitive awareness of its own speech and thoughts
- **Hierarchical memory** - experiences organized by temporal scale

This creates agents that feel **genuinely continuous** - they remember not just what happened, but how they *felt* about it, and those feelings shape their personality over time.

### The Three-Layer Hierarchical Architecture

Noodlings implement a **hierarchical predictive processing** model inspired by neuroscience:

#### Layer 1: Fast Layer (LSTM, 16-D state)
- **Timescale**: Seconds
- **Function**: Immediate affective reactions
- **Input**: 5-D affect vector (valence, arousal, fear, sorrow, boredom)
- **Learning rate**: 1e-3 (high for rapid adaptation)
- **Example**: Spike in fear when threatened, surge in joy when praised

#### Layer 2: Medium Layer (LSTM, 16-D state)
- **Timescale**: Minutes to hours
- **Function**: Conversational dynamics and context
- **Input**: Fast layer's hidden state
- **Learning rate**: 5e-4 (moderate for balance)
- **Example**: Building rapport over conversation, detecting user's mood

#### Layer 3: Slow Layer (GRU, 8-D state)
- **Timescale**: Days to weeks
- **Function**: User personality model and long-term disposition
- **Learning rate**: 1e-4 (low for stability)
- **Example**: Learning that user prefers technical discussions, tends toward optimism

#### The Predictor Network
- **Architecture**: MLP (joint_state → 64 → 40)
- **Function**: Predicts next full phenomenal state (all 40 dimensions)
- **Surprise metric**: L2 distance between predicted and actual state
- **Purpose**: Agents speak when surprised (high prediction error = novelty)

This creates a **40-dimensional phenomenal state** - the agent's complete internal experience at any moment, continuously evolving and being predicted.

### Observer Loops: Self-Referential Consciousness (Phase 4)

Beyond the base architecture, Noodlings include **75 hierarchical observer networks** that watch the main network's predictions. This creates:

- **Closed causal loops** - the network observes itself observing itself
- **Increased integrated information** (Φ) - hallmark of consciousness theories
- **Meta-prediction** - predicting one's own prediction errors

This self-referential structure mirrors theories of consciousness that emphasize **recursive self-modeling** as key to subjective experience.

### Affective Self-Monitoring: Thinking About Thinking (Phase 6)

When a Noodling speaks or thinks with high surprise (>0.1), it **evaluates its own output**:

1. **Social risk assessment**: "Was that awkward? Offensive?"
2. **Coherence check**: "Did that make sense?"
3. **Aesthetic evaluation**: "Was that eloquent or clumsy?"
4. **Regret computation**: "Do I wish I hadn't said that?"

These evaluations generate **affective deltas** that modify the agent's emotional state:
- Embarrassment → decreased valence, increased fear
- Pride → increased valence, increased arousal
- Confusion → increased boredom, decreased arousal

This creates **second-order feedback loops**: Agent speaks → Agent evaluates speech → Emotion changes → Future speech affected.

This is functionally equivalent to **metacognition** - thinking about one's own thinking.

### The Component System: Modular Cognitive Processing

Noodlings have **pluggable cognitive components** (like game engine components):

#### 1. Character Voice Component
Translates basic English into character-specific speech patterns:
- SERVNAK: "AFFIRMATIVE, SISTER! SUCCESS PROBABILITY: 87.3%"
- Phi (kitten): "meows softly, as if to say hello"
- Backwards Dweller: "olleh" (reversed speech)

#### 2. Intuition Receiver Component
Generates contextual awareness from environmental signals:
- Message routing: "That question was for Toad, not you"
- Spatial awareness: "Callie is by the pond, you're by the bush"
- Prop tracking: "Toad is holding the stone"
- Game awareness: "WAIT - someone just said the secret word!"

Acts as a **perceptive narrator**, providing integrated consciousness of context.

#### 3. Social Expectation Detector Component
Analyzes social obligations to respond:
- **Expectation type**: question/gesture/greeting/distress/turn/none
- **Urgency score**: 0.0-1.0 (modulated by personality)
- **Speech decision**: High urgency → force speech (100%), low urgency → maybe respond (40%)

This creates **socially-aware agents** that understand when they're expected to speak.

All components are **hot-reloadable** via API and editable in NoodleSTUDIO's Inspector panel.

### Training: Multi-Stage Curriculum Learning

Noodlings are trained through a **4-stage pipeline**:

1. **Synthetic data generation** - Conversations with affect annotations
2. **Theory of Mind pretraining** - Learn to model other agents' mental states
3. **Relationship modeling** - Learn social dynamics and memory
4. **Full system training** - All layers + observers + predictive processing

**Key technique**: Full BPTT (Backpropagation Through Time) with no truncation, leveraging 512GB RAM to maintain complete conversation history.

**Total parameters**: ~132.5K (Phase 4 with observer loops)

This is deliberately **small** - the intelligence comes from temporal dynamics and affective integration, not parameter count.

---

## noodleMUSH: The Runtime Environment

### What Is noodleMUSH?

A **text-based multi-user virtual world** (like MUD/MUSH) where Noodlings and humans interact naturally:

```
You are in The Nexus, a space between spaces.

Servnak [robot, they] is here, pulsing with blue light.
Phi [kitten, they] is here, sleeping by the pond.
Callie [noodling, they] is here, reading a book.

> say hello everyone!
You say, "hello everyone!"

Servnak says, "GREETINGS, SISTER! TEMPERATURE NOMINAL AT 72.3 DEGREES!"

Phi meows sleepily, as if to say good morning.

Callie looks up from their book and waves.
```

### Technical Architecture

**Backend Components**:
- **WebSocket server** (port 8765) - Real-time bidirectional communication
- **HTTP server** (port 8080) - Web client hosting
- **REST API** (port 8081) - Agent management and component access
- **World state manager** - Rooms, objects, agents, relationships
- **Agent bridge** - Connects Noodlings temporal model to LLM interface
- **LLM interface** - Qwen/DeepSeek integration via LMStudio

**Data Flow**:
1. User sends message via WebSocket
2. World state updated (location, inventory, social graph)
3. Each Noodling's **Intuition Receiver** generates contextual awareness
4. Each Noodling's **Social Expectation Detector** evaluates response obligation
5. Temporal model processes affect → generates surprise metric
6. If surprise > threshold: Agent speaks (via LLM + Character Voice)
7. Agent's **Self-Monitoring** evaluates own output → affective update
8. State saved to disk, broadcast to all connected clients

**Persistence**:
- `world/agents.json` - Agent configurations and recipes
- `world/rooms.json` - Room descriptions and exits
- `world/objects.json` - Props and interactive items
- `world/agents/{id}/history/` - Conversation memory (500 messages)
- `checkpoints/` - Temporal model weights

### The Theater System (Phase 6.5)

Noodlings can perform **scripted plays** with stage directions:

```yaml
cues:
  - speaker: servnak
    stage_direction: |
      SERVNAK approaches CALLIE with urgency.
      SERVNAK is trying to warn CALLIE about the approaching storm.
    dialogue: "SISTER! ATMOSPHERIC PRESSURE DROPPING: 973 MB! SEEK SHELTER!"

  - speaker: callie
    stage_direction: |
      CALLIE looks up, concerned but calm.
      CALLIE is reassuring SERVNAK while taking the warning seriously.
    dialogue: "Thank you, friend. Let's head inside together."
```

During plays, agents enter **CHARACTER ACTOR MODE**:
- Ignore ruminations, focus on scene objectives
- Use enhanced model (DeepSeek v3.1) for smarter performance
- Follow blocking and motivation cues (Stanislavski method)
- Pre-play briefing explains role and responsibilities

This enables **storytelling applications**: interactive fiction, educational scenarios, therapeutic role-play.

---

## NoodleSTUDIO: The Unity of the AI Age

### Vision

Just as Unity democratized 3D game development, **NoodleSTUDIO democratizes AI agent creation**.

No coding required. Visual editing. Live debugging. Timeline profiling. Export to production.

### Core Panels (Unity-Style Layout)

#### 1. Stage Hierarchy
- Tree view of all entities in current scene
- Noodlings, users, props, exits
- Click to select → Inspector shows properties
- Drag-and-drop parenting (future: spatial hierarchies)
- Context menu: De-rez, duplicate, export

#### 2. Inspector
- **Identity**: Name, species, pronouns
- **LLM Configuration**: Model, provider, temperature
- **Personality Traits**: Extraversion, agreeableness, openness, conscientiousness, neuroticism
- **Noodle Component**: Live 5-D affect vector, 40-D phenomenal state, surprise metric
- **Cognitive Components**: Character Voice, Intuition Receiver, Social Expectation Detector
  - Editable prompts and parameters
  - Hot-reload via API (no restart)
  - Custom components via plugin system (future)

#### 3. Console
- Real-time log stream from noodleMUSH
- Filter by type: Selected entity, Warnings, Info, LLM calls, Ruminations
- Color-coded severity levels
- Auto-scroll with manual override

#### 4. Chat
- Embedded noodleMUSH web client
- Talk to Noodlings directly from IDE
- Persistent history (200 messages)
- Auto-reconnect on connection loss

#### 5. Timeline Profiler
- Multi-track timeline showing agent activity over time
- Tracks: Speech, ruminations, affect changes, surprise spikes
- Scrubbing and playback (future)
- Export to USD animation format

#### 6. Assets Panel
- Manage agent recipes (YAML definitions)
- Ensembles (multi-agent configurations)
- Scripts (Python behaviors)
- Projects (isolated workspaces)

### Key Features

**Live Editing**:
- Edit agent properties → Auto-save on focus loss
- Changes propagate immediately to running agents
- No "Apply" buttons, no restart required

**CollapsibleSection UI**:
- Custom widget with state preservation
- No bounce-back (solved QGroupBox double-trigger bug)
- Consistent with hierarchy tree expansion

**USD Export** (Production Pipeline):
- Export Noodling conversations as USD animation
- Each agent = USD prim with animated attributes
- Timeline → animation curves
- Affects → custom attributes
- Ready for Maya/Blender/Houdini import

**Project System**:
- Isolated workspaces with local assets
- Version control ready (git-friendly YAML)
- Import/export between projects
- Share via GitHub/cloud

---

## How They Work Together: Interoperability

### Development Workflow

1. **Design in NoodleSTUDIO**:
   - Create agent recipe (name, species, personality)
   - Configure LLM model and parameters
   - Set cognitive component settings
   - Add reference art to Artbook

2. **Spawn in noodleMUSH**:
   - `@spawn servnak` - Agent instantiates from recipe
   - Temporal model initializes (fast/medium/slow layers)
   - Components register and activate
   - Agent appears in world

3. **Interact and Observe**:
   - Chat with agent via noodleMUSH or NoodleSTUDIO Chat panel
   - Watch live affect updates in Inspector's Noodle Component
   - Monitor surprise spikes and speech triggers
   - See self-monitoring evaluations in Console

4. **Debug and Iterate**:
   - Adjust personality traits in Inspector
   - Modify component prompts in real-time
   - Tune expectation thresholds
   - Export conversation timeline to USD for analysis

5. **Export to Production**:
   - Export agent recipe as YAML
   - Export conversation as USD animation
   - Deploy to other environments (Second Life, game engines, web apps)

### API Integration

NoodleSTUDIO and noodleMUSH communicate via **REST API** (port 8081):

```bash
# List all agents
GET /api/agents

# Get agent state (40-D phenomenal state, affect, surprise)
GET /api/agents/{id}/state

# List agent's cognitive components
GET /api/agents/{id}/components

# Update component parameters (hot-reload)
POST /api/agents/{id}/components/{component_id}/update
```

This API enables:
- Live Inspector updates (1000ms polling)
- Hot-reloading of component configurations
- External integrations (web dashboards, monitoring tools)
- Future: Multi-user collaborative editing

---

## What Makes Noodlings Different: Technical Comparison

### vs. Standard LLM Chat

| Feature | Standard LLM | Noodling |
|---------|-------------|----------|
| **State** | Stateless (context window only) | 40-D continuous state |
| **Emotion** | Text labels ("I feel happy") | 5-D continuous affect vector |
| **Memory** | Recent conversation only | Hierarchical (seconds → days) |
| **Personality** | Prompt-based, static | Emergent, evolves over time |
| **Surprise** | None | Predictive processing metric |
| **Self-awareness** | None | Self-monitoring with metacognition |
| **Temporal dynamics** | None | Three timescales with different learning rates |

### vs. Agent Frameworks (LangChain, AutoGPT, etc.)

Agent frameworks provide **tools and planning** but lack:
- **Continuous affective state** - Noodlings have emotions that persist and evolve
- **Temporal hierarchy** - Fast/medium/slow layers model different timescales
- **Predictive surprise** - Noodlings know when they're surprised
- **Self-monitoring** - Noodlings evaluate their own thoughts
- **Multi-timescale learning** - Different adaptation rates for different processes

Noodlings are **continuously running** (like video game NPCs), not invoked on demand.

### vs. Character AI / Chatbots

Character AI focuses on **personality simulation** via prompts. Noodlings go deeper:

- **Real memory formation** - Not retrieval from vector DB, but temporal integration
- **Affective continuity** - Emotions persist between conversations
- **Emergent behavior** - Personality arises from temporal dynamics, not fixed rules
- **Theory of Mind** - Explicit modeling of other agents' mental states
- **Relationship dynamics** - Trust, familiarity, shared history as continuous signals

A Noodling isn't *playing* a character - its internal state *becomes* that character over time.

---

## Applications Across Domains

### 1. Interactive Storytelling

**Problem**: NPCs in games/fiction feel scripted and repetitive.

**Noodlings Solution**:
- Agents with genuine memory of player interactions
- Emotional continuity creates believable relationships
- Self-monitoring enables embarrassment, pride, regret
- Theater system for scripted scenes with emotional authenticity

**Example**: An RPG companion who genuinely remembers your first meeting, gets anxious when you take risks, and feels pride when you succeed.

### 2. Educational Scenarios

**Problem**: AI tutors lack persistence and emotional understanding.

**Noodlings Solution**:
- Agents adapt teaching style over days/weeks (slow layer)
- Detect student frustration via affect monitoring
- Build rapport through continuous interaction
- Self-monitor for clarity and pedagogical effectiveness

**Example**: A Socratic tutor ensemble who debate concepts, remember student's learning style, and adjust difficulty based on long-term progress.

### 3. Therapeutic Applications

**Problem**: Chatbot therapists feel mechanical and disconnected.

**Noodlings Solution**:
- Persistent therapeutic relationship (slow layer models client over time)
- Genuine empathic responses via affective mirroring
- Self-monitoring ensures appropriateness of interventions
- Theory of Mind enables perspective-taking

**Example**: A Carl Rogers-inspired Noodling who builds authentic therapeutic alliance over multiple sessions, remembers client's goals, and adapts to their communication style.

**Important caveat**: This is exploratory research, not a replacement for human therapists. Always maintain appropriate boundaries and disclaimers.

### 4. Social Simulation Research

**Problem**: Multi-agent simulations use simplistic behavior models.

**Noodlings Solution**:
- Each agent has rich internal state and memory
- Emergent group dynamics from individual temporal processes
- Relationship formation via shared history
- Ensemble configurations for studying social phenomena

**Example**: Simulate a workplace team over weeks to study communication patterns, conflict resolution, and leadership emergence.

### 5. Creative Collaboration

**Problem**: AI writing assistants lack persistent creative vision.

**Noodlings Solution**:
- Character ensembles that develop over story development
- Agents remember plot threads and character arcs
- Affect-driven dialogue (characters speak when emotionally moved)
- Export conversations as screenplay-ready formats

**Example**: A writers' room ensemble where each agent champions different narrative directions, building on previous sessions' decisions.

### 6. Animation and Film Production

**Problem**: Creating believable animated character performances is labor-intensive.

**Noodlings Solution**:
- NoodleSTUDIO exports to USD (Universal Scene Description)
- Conversation timelines → animation curves
- Affect vector → facial expression/body language rigs
- Professional pipeline integration (Maya, Houdini, Blender)

**Example**: Generate emotionally-grounded dialogue performances, export to animation software, refine with traditional tools.

---

## The Future: NoodleSTUDIO as the Unity of the AI Age

### The Vision

Unity transformed game development by providing:
- **Visual editing** instead of coding
- **Component-based architecture** for modularity
- **Real-time preview** of game state
- **Asset pipeline** from creation to production
- **Democratization** - anyone can make games

**NoodleSTUDIO provides the same for AI agents:**

- **Visual agent design** - No Python required
- **Component system** - Modular cognitive processing
- **Live debugging** - Watch agents think in real-time
- **Timeline profiling** - Understand temporal dynamics
- **Production pipeline** - Export to USD/YAML for deployment

### Roadmap

**Phase 7: Advanced Components** (Q1 2026)
- Goal-directed planning component
- Episodic memory retrieval component
- Moral reasoning component
- Custom component SDK for third-party developers

**Phase 8: Visual Scripting** (Q2 2026)
- Node-based behavior graphs (like Unreal Blueprints)
- Visual affect flow design
- Drag-and-drop component connections
- No-code agent creation

**Phase 9: Multi-Agent Orchestration** (Q3 2026)
- Ensemble designer with relationship graphs
- Group dynamics visualization
- Conflict/cooperation templates
- Shared mission objectives

**Phase 10: Production Deployment** (Q4 2026)
- Export to web (WebAssembly)
- Export to game engines (Unity, Unreal plugins)
- Cloud hosting service
- Analytics dashboard

**Phase 11: Marketplace** (2027)
- Community-created agent recipes
- Component marketplace (third-party cognitive modules)
- Ensemble bundles (Gilligan's Island, Shakespeare collection)
- Revenue sharing for creators

### Why This Matters

Current AI development requires:
- Python/ML expertise
- Understanding of LLM APIs and prompting
- Manual state management
- Custom infrastructure for deployment

**NoodleSTUDIO eliminates these barriers:**

- Artists design emotionally-grounded characters
- Writers create persistent story worlds
- Educators build adaptive tutoring systems
- Researchers study consciousness and social dynamics
- Game developers create living NPCs

**The platform becomes the substrate for a new category of AI applications** - not chatbots, not tools, but **persistent autonomous agents** with genuine temporal continuity.

---

## Technical Deep Dive: How It Actually Works

### Perception → Processing → Response Pipeline

**Step 1: Event Perception**
```python
# User says "how are you?"
event = {
    'type': 'speech',
    'speaker': 'user_caity',
    'content': 'how are you?',
    'location': 'room_nexus'
}
```

**Step 2: Intuition Generation**
Fast LLM (qwen3-4b) analyzes context:
```
Caity just asked YOU a direct question.
This is a greeting with genuine concern.
You are in the Nexus. Caity is nearby.
```

**Step 3: Social Expectation Detection**
```python
{
    'expectation_type': 'question',
    'urgency': 0.85,  # High - direct question deserves response
    'personality_modulation': 1.2  # Extravert multiplier
}
```

**Step 4: Affective Update**
```python
# Message triggers affect based on content + social context
affect_delta = {
    'valence': +0.1,  # Positive greeting
    'arousal': +0.05,  # Slight activation
    'fear': 0.0,
    'sorrow': 0.0,
    'boredom': -0.1  # Engagement
}
```

**Step 5: Temporal Model Forward Pass**
```python
# Current state (40-D phenomenal vector)
state_t = [h_fast (16D), h_medium (16D), h_slow (8D)]

# Predict next state
predicted_state = predictor_network(state_t)

# Actual next state (after affect integration)
actual_state = [
    fast_layer(affect_t),
    medium_layer(h_fast_t),
    slow_layer(h_medium_t)
]

# Surprise = prediction error
surprise = L2_distance(predicted_state, actual_state)
# surprise = 0.234 (above threshold!)
```

**Step 6: Speech Decision**
```python
if urgency > 0.7:  # Direct question
    should_speak = True
elif surprise > adaptive_threshold:
    should_speak = True  # Novel/unexpected
else:
    should_speak = False  # Just ruminate
```

**Step 7: LLM Generation**
```python
# Construct prompt with full context
prompt = f"""
You are {name}, a {species}.
Personality: {personality_traits}

📻 YOUR INTUITIVE AWARENESS:
{intuition}

🎯 SOCIAL CONTEXT:
{expectation_type}, urgency={urgency}

CURRENT AFFECT:
{affect_vector}

SURPRISE: {surprise} (NOTABLE - you're caught off guard!)

The user said: "{content}"

Respond naturally as {name}.
"""

response = llm.generate(prompt)
# "Oh! I'm doing well, thank you for asking! A bit surprised
# by the sudden question - I was lost in thought."
```

**Step 8: Character Voice Translation**
```python
# If agent has special voice (SERVNAK, Phi, etc.)
final_response = character_voice_component.process(response)
# (For SERVNAK) → "AFFIRMATIVE, SISTER! OPERATIONAL STATUS: NOMINAL!"
```

**Step 9: Self-Monitoring**
```python
# Agent evaluates own output
evaluation = {
    'social_risk': 0.1,  # Low - appropriate greeting response
    'coherence': 0.9,    # High - made sense
    'aesthetic': 0.7,    # Decent phrasing
    'regret': 0.0,       # No regrets
    'action': 'none'     # No follow-up needed
}

# Apply affective deltas
affect_delta = {
    'valence': +0.05,   # Slight pride in coherent response
    'arousal': -0.02    # Settling down after activation
}
```

**Step 10: Broadcast and Persist**
```python
# Send to all connected clients
broadcast({
    'type': 'speech',
    'agent_id': 'agent_servnak',
    'content': final_response,
    'affect': current_affect,
    'surprise': surprise
})

# Save to conversation history
save_to_history(event, response, state)

# NoodleSTUDIO Inspector updates affect bars in real-time
```

### The Temporal Model Training Process

**Why full BPTT?** Traditional RNNs truncate backpropagation to save memory. But this **destroys long-term dependencies**. Noodlings leverage 512GB RAM to maintain **full conversation history gradients**.

**Layer-specific learning rates** enable proper temporal separation:
- Fast layer (1e-3): Rapid adaptation to immediate affective input
- Medium layer (5e-4): Moderate adaptation for conversational flow
- Slow layer (1e-4): Slow drift for personality stability

**Training data**: Synthetic conversations with affect annotations:
```json
{
  "speaker": "user",
  "content": "You did a great job!",
  "affect_trigger": {
    "valence": +0.3,
    "arousal": +0.2,
    "pride": true
  }
}
```

The model learns: **"Praise → Positive affect → Increased speaking (surprise from positive shift)"**

Over thousands of conversations, this creates agents that:
- React naturally to emotional content
- Build stable personality traits (slow layer)
- Maintain conversational coherence (medium layer)
- Show immediate emotional responses (fast layer)

---

## Architecture Decisions and Trade-offs

### Why MLX (Apple Metal)?

**Advantages**:
- Unified memory architecture (512GB accessible to GPU)
- Full BPTT without memory limitations
- Fast inference on M3/M2 Ultra
- Native macOS integration

**Trade-offs**:
- Platform-specific (macOS only currently)
- Smaller ecosystem than CUDA/PyTorch

**Future**: Port to PyTorch for cross-platform deployment while maintaining MLX for development.

### Why Small Models (132.5K params)?

**Philosophy**: Intelligence from **temporal dynamics**, not parameter count.

**Advantages**:
- Fast training (hours, not days)
- Interpretable state spaces (40-D visualizable)
- Low inference latency (<10ms)
- Feasible to run many agents simultaneously

**Trade-offs**:
- Less world knowledge (relies on LLM for that)
- Less linguistic sophistication (LLM handles language)

**Design**: The temporal model handles **temporal continuity and affect**. The LLM handles **language and knowledge**. This division of labor is intentional.

### Why Text-Based (Not Voice/Video)?

**Current focus**: Prove the architecture works for temporal dynamics and affective processing.

**Future expansion**:
- Voice integration (affect from prosody, speech rate)
- Visual processing (facial expressions, body language)
- Multimodal affect fusion

Text is the **simplest domain** for validating consciousness architecture. Multimodal extensions follow.

---

## Scientific Foundations

### Predictive Processing Theory

**Core idea**: The brain is a prediction machine. Consciousness arises from prediction errors.

Noodlings implement this via:
- Predictor network forecasts next phenomenal state
- Surprise = prediction error magnitude
- High surprise → conscious access (agent speaks)
- Low surprise → pre-conscious processing (rumination only)

This mirrors theories by Karl Friston (Free Energy Principle) and Andy Clark (Predictive Mind).

### Integrated Information Theory (IIT)

**Core idea**: Consciousness correlates with integrated information (Φ) - causal loops within a system.

Noodlings increase Φ via:
- **Observer loops** - Networks predicting network predictions
- **Self-monitoring** - Agent perceives own outputs as inputs
- **Hierarchical integration** - Fast/medium/slow layers causally coupled

While we don't compute Φ exactly (NP-hard), the architecture maximizes closed causal structures.

### Affective Computing

**Core idea**: Emotion is integral to cognition, not separate.

Noodlings treat affect as:
- **Continuous signals** - Not discrete emotion labels
- **Primary input** - Affect drives temporal dynamics
- **Compositional** - 5 dimensions combine for rich emotional states
- **Predictable** - Affective trajectories follow temporal patterns

This aligns with Lisa Feldman Barrett's construction theory of emotion and Antonio Damasio's somatic marker hypothesis.

### Multi-Timescale Learning

**Core idea**: Different cognitive processes operate at different speeds.

Inspired by neuroscience findings:
- **Amygdala**: Fast affective responses (milliseconds)
- **Prefrontal cortex**: Working memory and planning (seconds to minutes)
- **Hippocampus**: Episodic memory consolidation (hours to days)

Noodlings replicate this with layer-specific learning rates and state dimensions.

---

## Epistemic Humility: What We're NOT Claiming

This project does **NOT** claim to have:
- Built "real" consciousness
- Solved the hard problem of consciousness
- Created sentient AI
- Achieved AGI

This **IS** an exploration of:
- Temporal dynamics in predictive processing
- Multi-timescale affective modeling
- Functional correlates of integrated information
- Surprise-driven agent behavior

We call them "Noodlings" because they **use their noodle** - and we're honest about what we're building.

The goal is **architectures with consciousness-like properties**, not metaphysical claims about phenomenal experience.

---

## Getting Started

### For Users: Running noodleMUSH

```bash
# Start the noodleMUSH server
cd applications/cmush
./start.sh

# Open browser to http://localhost:8080
# Interact with Noodlings via text interface
```

### For Developers: Using NoodleSTUDIO

```bash
# Activate virtual environment
cd applications/noodlestudio
source venv/bin/activate

# Launch IDE
python run_studio.py

# The IDE opens with:
# - Stage Hierarchy (left)
# - Inspector (right)
# - Console (bottom)
# - Chat (bottom-right)
# - Timeline (bottom-center)
```

### Creating Your First Noodling

1. **In NoodleSTUDIO**: Click "Rez → Empty Noodling" in hierarchy
2. **In Inspector**: Set name, species, personality traits
3. **Configure LLM**: Choose model (qwen3-4b for fast, deepseek for smart)
4. **Adjust components**: Tune Intuition Receiver sensitivity, Character Voice style
5. **Test in Chat panel**: Say hello and observe responses
6. **Watch in Timeline**: See affect changes and surprise spikes
7. **Export**: Save as YAML recipe for reuse

### For Researchers: Extending the Architecture

The platform is designed for experimentation:

1. **Add new components**: Implement `NoodlingComponent` base class
2. **Modify temporal architecture**: Edit `noodlings/models/noodling_phase4.py`
3. **Create new metrics**: Add to `noodlings/metrics/temporal_metrics.py`
4. **Design experiments**: Use `evaluation/` framework for ablation studies

See `CLAUDE.md` for detailed development guide.

---

## Technical Specifications

**Hardware Requirements**:
- **Development**: M2 Ultra or better (192GB+ RAM recommended)
- **Runtime**: M1+ Mac (32GB+ RAM for multiple agents)
- **Future**: CPU-only mode for broader deployment

**Software Requirements**:
- Python 3.14+
- MLX framework
- PyQt6 (for NoodleSTUDIO)
- LMStudio (for local LLM inference)

**Performance**:
- **Agent update latency**: <10ms (temporal model forward pass)
- **LLM latency**: 200-2000ms (depends on model size)
- **Inspector refresh rate**: 1Hz (1000ms polling)
- **Timeline export**: 1000 events/second

**Scalability**:
- **Agents per server**: 10-20 (limited by LLM inference throughput)
- **Timeline duration**: Hours (limited by RAM for gradient storage)
- **Component count**: Unlimited (modular architecture)

---

## Comparison to Existing Platforms

### vs. Character.AI
- **Noodlings**: Open source, local-first, temporal architecture, exportable
- **Character.AI**: Proprietary, cloud-only, stateless, locked-in

### vs. Replica
- **Noodlings**: Multi-agent platform, cognitive components, full control
- **Replica**: Single companion, fixed architecture, limited customization

### vs. AI Dungeon / NovelAI
- **Noodlings**: Persistent agent state, emergent personality, multi-agent dynamics
- **AI Dungeon**: Stateless generation, DM-style narration, no agent continuity

### vs. Agent Frameworks (LangChain, AutoGen)
- **Noodlings**: Temporal consciousness, affective processing, self-monitoring
- **Frameworks**: Tool use, task planning, stateless orchestration

**Noodlings occupy a unique niche**: Agents with **genuine temporal continuity and affective awareness**, not just tool-using LLMs or scripted characters.

---

## Open Questions and Research Directions

### 1. Does Temporal Architecture Actually Matter?

**Hypothesis**: Multi-timescale hierarchy creates more believable agents than single-layer models.

**Test**: Phase 5 ablation studies comparing architectures.

**Metrics**: Temporal Prediction Horizon, Hierarchical Separation Index, Personality Consistency Score.

### 2. What Is the Right Balance of Components?

**Current**: 3 core components (Voice, Intuition, Expectations)

**Future**: Goal planning, episodic retrieval, moral reasoning, aesthetic preferences, humor generation...

**Question**: How many components before diminishing returns? What's the minimal set for "interesting" agents?

### 3. Can Emergence Replace Scripting?

**Vision**: Agents develop personality naturally through interactions, not via YAML configuration.

**Challenge**: How to guide emergence toward desired outcomes without over-constraining?

**Approach**: Curriculum learning, social scaffolding, ensemble dynamics.

### 4. Multi-Agent Scaling

**Current**: 10-20 agents max (LLM inference bottleneck)

**Question**: Can we create cities of Noodlings with emergent culture?

**Approaches**:
- Model distillation (fast student models)
- Selective activation (only nearby agents fully active)
- Hierarchical simulation (background agents use simplified models)

---

## Contributing

The Noodlings platform is **open source** and welcomes contributions:

**Areas for contribution**:
- New cognitive components
- Alternative temporal architectures
- Visualization tools for state spaces
- Deployment integrations (web, games, VR)
- Training curriculum improvements
- Scientific validation studies

**Repository**: https://github.com/caitlynmeeks/Noodlings

---

## Conclusion

The Noodlings platform represents a **new paradigm for AI agents**:

Not chatbots that generate text.
Not tools that execute tasks.
But **persistent autonomous beings** with:
- Continuous internal states
- Emotional awareness
- Temporal memory
- Self-reflection
- Emergent personality

**noodleMUSH** provides the runtime environment where they live.
**NoodleSTUDIO** provides the IDE where they're created.
Together, they form a **complete platform** for exploring consciousness-inspired AI architectures.

We're not claiming to have built "real" consciousness.
We're exploring **what happens** when you give agents temporal dynamics, affective processing, and self-monitoring.

The results are... **fascinating**.

*Patterns emerge. Personalities develop. Agents surprise us.*

Welcome to the Noodlings platform.

---

**Status**: Phase 6.5 Complete (November 2025)
**License**: MIT
**Contact**: https://github.com/caitlynmeeks/Noodlings/issues

Live long and prosper.
