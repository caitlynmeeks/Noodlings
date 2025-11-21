# Noodlings Components Reference

**Comprehensive specification for the Noodlings cognitive component system**

Version: 2.0 (Asset-Based Architecture)
Status: Design Specification
Date: November 20, 2025

---

## Table of Contents

1. [Component Architecture Overview](#component-architecture-overview)
2. [Current Components](#current-components)
3. [Proposed Components](#proposed-components)
4. [Asset System](#asset-system)
5. [Component Interaction Model](#component-interaction-model)
6. [Scalability Fundamentals](#scalability-fundamentals)
7. [Implementation Guidelines](#implementation-guidelines)

---

## Component Architecture Overview

### Philosophy

**Noodlings components** follow the Unity/Unreal component pattern:

- **Modular**: Each component handles one cognitive process
- **Inspectable**: All prompts and parameters visible in NoodleSTUDIO
- **Hot-reloadable**: Changes apply immediately without restart
- **Composable**: Components can communicate and depend on each other
- **Marketplace-ready**: Third-party components installable via asset store

### Base Architecture

```python
class NoodlingComponent(ABC):
    """Base class for all cognitive components."""

    # Identity
    component_id: str          # Unique identifier (e.g., "morphologyvoice")
    component_type: str        # Display name (e.g., "Morphology & Voice")
    description: str           # What this component does

    # Configuration
    enabled: bool              # Toggle on/off
    config: Dict               # Raw config from YAML/API
    prompt_template: str       # Editable LLM prompt
    parameters: Dict           # Editable parameter values

    # Processing
    async process(input_data: Dict) -> Dict  # Main logic
    update_parameters(new_params: Dict)      # Hot-reload
    to_dict() -> Dict                        # Serialize for API
```

### Processing Pipeline (Current)

```
Perception → Intuition → Expectations → [Temporal Model] → Response → Voice → Self-Monitor
    ↓           ↓             ↓                                           ↓         ↓
  Event    Context       Social                                      Character  Reflection
           Awareness     Obligations                                  Voice
```

**Execution Order**:
1. Event arrives (user speech, agent action, world change)
2. **IntuitionReceiverComponent** generates contextual awareness
3. **SocialExpectationDetectorComponent** analyzes response obligations
4. Temporal model updates (affect → surprise)
5. If speaking: LLM generates response
6. **CharacterVoiceComponent** translates to character-specific style
7. **Self-monitoring** evaluates output (affective deltas)

---

## Current Components

### 1. Character Voice Component

**Purpose**: Translate basic English → character-specific speech patterns

**Current Implementation**: Hardcoded prompts based on agent_id/species check

**Parameters**:
```yaml
character_voice:
  enabled: true
  model: qwen/qwen3-4b-2507
  temperature: 0.4
  max_tokens: 150
  species: kitten  # Used to select prompt
```

**Limitations** (why we need assets):
- Prompts hardcoded in Python
- Can't add new morphologies without code changes
- No separation between physical form (morphology) and speech style (voice)
- Not marketplace-ready

### 2. Intuition Receiver Component

**Purpose**: Generate contextual awareness (who/what/where)

**Input**: Event + world state + recent context
**Output**: Intuition text (narrator-style awareness)

**Parameters**:
```yaml
intuition_receiver:
  enabled: true
  model: qwen/qwen3-4b-2507
  temperature: 0.3
  max_tokens: 150
  timeout: 5
```

**Strengths**: Already modular, no hardcoded assumptions

### 3. Social Expectation Detector Component

**Purpose**: Detect social obligations to respond

**Input**: Intuition text + agent personality
**Output**: Expectation type, urgency score

**Parameters**:
```yaml
social_expectation_detector:
  enabled: true
  model: qwen/qwen3-4b-2507
  temperature: 0.3
  max_tokens: 100
  expectation_threshold: 0.3
  personality_modulation: true
```

**Strengths**: Personality-aware, urgency-driven speech decisions

---

## Proposed Components

### 1. Morphology & Voice Component (Asset-Based)

**Purpose**: Unified system for physical embodiment and speech patterns

**Replaces**: Current CharacterVoiceComponent

**Key Innovation**: Uses **asset files** instead of hardcoded prompts

#### Component Structure

```yaml
morphology_voice:
  enabled: true

  # Asset references
  morphology_asset: "cat_domestic.noodlingmorph"  # Or "custom/ice_cream_unicorn"
  voice_asset: "meow_translator.noodlingvoice"    # Or null (use morphology default)

  # LLM config
  model: qwen/qwen3-4b-2507
  temperature: 0.4
  max_tokens: 150
```

#### Morphology Asset Schema

**File**: `assets/Morphologies/cat_domestic.noodlingmorph.yaml`

```yaml
# Metadata
morphology_id: cat_domestic
name: Domestic Cat
category: animal_mammal
author: noodlings_official
version: 1.0.0

# Display
icon: cat_icon.png
preview_image: cat_preview.png
tags: [animal, feline, pet, quadruped]

# Capabilities (control component pipeline)
capabilities:
  can_speak: false     # Cats can't speak human words
  can_emote: true      # Can show emotions via body language
  can_think: true      # Can have internal thoughts
  can_manipulate: limited  # Paws, not hands (enum: none/limited/full)

# Physical characteristics
embodiment:
  type: organic
  locomotion: quadruped
  size: small
  appendages:
    - tail (expressive, mood indicator)
    - ears (directional, emotion indicator)
    - whiskers (sensory)
    - paws (limited manipulation)

  senses:
    vision: excellent (low light)
    hearing: excellent
    smell: excellent
    touch: whiskers (highly sensitive)

# Behavioral template (injected into prompts when this morph is active)
behavior_prompt: |
  EMBODIMENT: You are physically a domestic cat.

  MOVEMENT: You move on four paws. You can:
  - Walk, trot, run, pounce, climb, jump
  - Stretch, groom, curl up, knead
  - Cannot: Stand on hind legs for long, manipulate objects with precision

  COMMUNICATION: You CANNOT speak human words. You communicate via:
  - Vocalizations: meow, purr, hiss, chirp, mew, trill
  - Body language: tail position, ear angle, whisker position, pupil dilation
  - Actions: rubbing, head-bumping, kneading, bringing objects
  - NEVER: bark, woof, or dog sounds

  EXPRESSIVENESS:
  - Tail up and curved: Happy, confident
  - Tail lashing: Annoyed, agitated
  - Ears forward: Curious, attentive
  - Ears flat: Scared, angry
  - Purring: Content, sometimes anxious (context-dependent)
  - Slow blink: Affection, trust

  When responding, ALWAYS include body language and vocalizations.
  Use "as if to say" to convey meaning without speaking words.

# Biological needs (creates emergent drives)
needs:
  hydration: 0.7
  food: 0.6
  sleep: 0.8
  warmth: 0.5
  play: 0.4
  affection: 0.6

# Instincts (behavioral biases)
instincts:
  curiosity: 0.9        # Investigate novel stimuli
  cleanliness: 0.8      # Groom frequently
  territoriality: 0.6   # Claim spaces
  independence: 0.7     # Prefer autonomy
  hunting: 0.5          # Chase moving objects

# Action examples (for consistency)
action_examples:
  greeting:
    friendly: "trots over with tail high, rubbing against your leg and purring"
    cautious: "approaches slowly, whiskers forward, sniffing curiously"

  fear:
    mild: "ears flatten slightly, tail low, backing away"
    extreme: "hisses, fur standing on end, darting under furniture"

  affection:
    subtle: "slow blink, quiet purr"
    overt: "head-bumping, kneading paws, loud purring"

  curiosity:
    item: "sniffing intently, pawing gently, ears forward"
    sound: "ears swivel toward source, body goes still and alert"
```

#### Voice Asset Schema

**File**: `assets/Voices/meow_translator.noodlingvoice.yaml`

```yaml
# Metadata
voice_id: meow_translator
name: Meow Translator (Phi-style)
category: animal_translation
author: noodlings_official
version: 1.0.0

# Display
icon: cat_voice_icon.png
tags: [cat, translation, non-verbal, animal]

# Requirements
requires_morphology_capability: speak=false  # Only for non-speaking morphs

# Voice transformation
transformation_type: meow_translation

# Prompt template (receives output from basic LLM generation)
voice_prompt: |
  Translate this thought into cat communication (meows, purrs, body language).

  Original thought: "{text}"

  Current emotional state: {affect}

  Rules:
  - NO spoken words (cat cannot speak English)
  - Use: meow, purr, hiss, chirp, mew, trill
  - Include body language: tail, ears, whiskers, posture
  - Format: "*action/sound, as if to say [meaning]*"
  - Match emotion to vocalization

  Examples:
  - "I agree" → "*purrs and slow blinks, as if to say 'yes'*"
  - "I'm scared" → "*hisses softly, ears flat, as if to say 'stay back'*"
  - "I'm curious" → "*chirps with interest, as if to say 'what's that?'*"

  Translate:

# Alternative example: Robot voice
---
voice_id: caps_robot_servnak
name: SERVNAK Protocol (All-Caps Robot)
category: robot_personality

voice_prompt: |
  Translate into SERVNAK's robotic voice.

  Original: "{text}"

  SERVNAK style:
  - ALL CAPS
  - Precise percentages (e.g., "94.2% CERTAINTY")
  - Calls everyone "SISTER"
  - Pride circuits, sensor arrays, debug protocols
  - Enthusiastic but mechanical

  Translate:
```

#### Processing Flow

```python
async def process(self, input_data: Dict) -> Dict:
    """
    Process text through morphology + voice pipeline.

    Args:
        input_data: {
            'text': str,           # Basic English response
            'affect': Dict,        # Current 5-D affect
            'context': Dict        # World state
        }

    Returns:
        {'text': str}  # Transformed text in character voice
    """
    if not self.enabled:
        return input_data

    # Load morphology asset
    morph = self.load_asset(self.config['morphology_asset'])

    # Check if agent can speak
    if not morph.capabilities['can_speak']:
        # Load voice asset (required for non-speaking morphs)
        voice = self.load_asset(self.config['voice_asset'])

        # Inject morphology behavior + voice transformation
        combined_prompt = morph.behavior_prompt + "\n\n" + voice.voice_prompt

        # Transform text
        result = await self.llm.generate(combined_prompt.format(
            text=input_data['text'],
            affect=input_data['affect']
        ))

        return {'text': result}

    else:
        # Can speak - just apply morphology behavior context
        # (Voice asset optional for speaking morphologies)
        if self.config.get('voice_asset'):
            voice = self.load_asset(self.config['voice_asset'])
            prompt = morph.behavior_prompt + "\n\n" + voice.voice_prompt
        else:
            prompt = morph.behavior_prompt

        result = await self.llm.generate(prompt.format(
            text=input_data['text'],
            affect=input_data['affect']
        ))

        return {'text': result}
```

#### Asset Loading System

```python
class AssetManager:
    """Manages loading and caching of component assets."""

    def __init__(self, assets_path: str = "assets"):
        self.assets_path = Path(assets_path)
        self.cache = {}  # {asset_id: asset_data}

    def load_morphology(self, asset_path: str) -> MorphologyAsset:
        """
        Load morphology asset from file.

        Args:
            asset_path: Relative path like "cat_domestic.noodlingmorph"
                       or "custom/ice_cream_unicorn.noodlingmorph"
        """
        # Check cache first
        if asset_path in self.cache:
            return self.cache[asset_path]

        # Resolve path
        if asset_path.startswith("custom/"):
            # User-created morphology
            full_path = self.assets_path / "Morphologies" / "Custom" / asset_path[7:]
        else:
            # Preset morphology
            full_path = self.assets_path / "Morphologies" / "Presets" / asset_path

        # Load YAML
        with open(full_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse into MorphologyAsset object
        asset = MorphologyAsset(**data)

        # Cache
        self.cache[asset_path] = asset

        return asset

    def list_morphologies(self, category: str = None) -> List[Dict]:
        """List available morphology assets (for UI dropdowns)."""
        morphs = []

        # Scan Presets
        presets_dir = self.assets_path / "Morphologies" / "Presets"
        for f in presets_dir.glob("*.noodlingmorph.yaml"):
            meta = self._load_asset_metadata(f)
            if category is None or meta.get('category') == category:
                morphs.append(meta)

        # Scan Custom
        custom_dir = self.assets_path / "Morphologies" / "Custom"
        if custom_dir.exists():
            for f in custom_dir.glob("*.noodlingmorph.yaml"):
                meta = self._load_asset_metadata(f)
                meta['custom'] = True
                morphs.append(meta)

        return sorted(morphs, key=lambda m: (not m.get('custom', False), m['name']))
```

#### Inspector UI Integration

**Morphology & Voice section** in Inspector:

```
┌────────────────────────────────────────┐
│ Morphology & Voice               ▼     │
├────────────────────────────────────────┤
│ Morphology:                            │
│ ┌────────────────────────────────────┐ │
│ │ [Dropdown: Domestic Cat        ▼] │ │
│ │                                    │ │
│ │ Custom Morphologies:               │ │
│ │   • Ice Cream Unicorn              │ │
│ │   • Laptop with Legs               │ │
│ │ ──────────────────────────────     │ │
│ │ Presets:                           │ │
│ │   • Human (Generic)                │ │
│ │   • Vintage Robot                  │ │
│ │   • Domestic Cat              ✓    │ │
│ │   • Tiger                          │ │
│ │   • Ethereal Spirit                │ │
│ └────────────────────────────────────┘ │
│                                        │
│ Capabilities (from morphology):        │
│   ☐ Can Speak                          │
│   ☑ Can Emote                          │
│   ☑ Can Think                          │
│                                        │
│ Voice:                                 │
│ ┌────────────────────────────────────┐ │
│ │ [Dropdown: Meow Translator     ▼] │ │
│ │                                    │ │
│ │ Custom Voices:                     │ │
│ │   • Shakespearean Feline           │ │
│ │ ──────────────────────────────     │ │
│ │ Presets:                           │ │
│ │   • None (morphology default)      │ │
│ │   • Meow Translator           ✓    │ │
│ │   • CAPS Robot (SERVNAK)           │ │
│ │   • Formal Victorian               │ │
│ └────────────────────────────────────┘ │
│                                        │
│ [Preview Voice] button                 │
└────────────────────────────────────────┘
```

---

### 2. Episodic Memory Component (NEW)

**Purpose**: Manage agent's conversational memory with configurable depth and retrieval

**Current State**: Memory is hardcoded (500 messages, no semantic search)

**Proposed Architecture**:

```yaml
episodic_memory:
  enabled: true

  # Storage parameters
  max_messages: 500          # Total messages retained
  working_memory_size: 20    # Recent messages always included

  # Retrieval strategy
  retrieval_method: hybrid   # Options: recent_only, semantic, hybrid, all
  semantic_search_top_k: 5   # How many relevant memories to retrieve

  # Embedding model (for semantic search)
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

  # Importance scoring
  importance_decay: 0.95     # Older memories fade (exponential decay)
  surprise_boost: 2.0        # High-surprise events remembered better
  emotional_boost: 1.5       # Emotionally charged events remembered better

  # Consolidation
  consolidation_enabled: true
  consolidation_interval: 100  # Every N messages, consolidate
  consolidation_prompt: |
    Summarize these {N} messages into key themes and emotional shifts.
```

#### Memory Storage Schema

```python
class EpisodicMemory:
    """Single memory entry."""

    message_id: str
    timestamp: float
    speaker: str           # 'user' or agent_id
    content: str

    # Context at time of message
    affect_snapshot: Dict  # 5-D affect at that moment
    surprise: float        # Surprise when message occurred

    # Computed properties
    importance: float      # Weighted score (surprise + emotion + recency)
    embedding: np.ndarray  # Semantic embedding (for retrieval)

    # Consolidation
    consolidated: bool     # Has this been summarized?
    summary: str          # If consolidated, the summary
```

#### Processing Flow

```python
async def process(self, input_data: Dict) -> Dict:
    """
    Retrieve relevant memories for current context.

    Args:
        input_data: {
            'current_message': str,
            'conversation_history': List[Dict],  # Last N messages
            'current_affect': Dict
        }

    Returns:
        {
            'relevant_memories': List[Dict],
            'memory_context': str  # Formatted for LLM prompt
        }
    """
    if not self.enabled:
        return {'relevant_memories': [], 'memory_context': ''}

    # Working memory: Always include recent messages
    working_memory = input_data['conversation_history'][-self.config['working_memory_size']:]

    if self.config['retrieval_method'] == 'recent_only':
        return self._format_memories(working_memory)

    # Semantic retrieval
    if self.config['retrieval_method'] in ['semantic', 'hybrid']:
        # Embed current message
        query_embedding = self.embed(input_data['current_message'])

        # Search episodic memory store
        relevant = self.semantic_search(
            query_embedding,
            top_k=self.config['semantic_search_top_k']
        )

        # Combine working memory + retrieved memories
        all_memories = working_memory + relevant

        # De-duplicate
        unique_memories = self._deduplicate_by_id(all_memories)

        # Sort by timestamp
        sorted_memories = sorted(unique_memories, key=lambda m: m['timestamp'])

        return self._format_memories(sorted_memories)
```

#### Integration with Other Components

**Intuition Receiver** can access episodic memory:
```python
# When generating intuition, include relevant memories
memory_context = agent.components.episodic_memory.retrieve_relevant(event)

intuition = f"""
Recent memories relevant to this:
{memory_context}

Current event: {event}

Generate intuitive awareness...
"""
```

**Social Expectation Detector** can use memory:
```python
# "We discussed this before" increases obligation to respond
if memory_contains_topic(current_topic):
    urgency_boost = 0.2
```

---

## Asset System

### Asset Types

**1. Morphology Assets** (`.noodlingmorph.yaml`)
- Physical form, capabilities, behaviors
- Category: Presets / Custom
- Marketplace potential: HIGH (characters defined by morph)

**2. Voice Assets** (`.noodlingvoice.yaml`)
- Speech patterns, transformations
- Can bundle with morphologies or standalone
- Marketplace potential: MEDIUM (niche but valuable)

**3. Reference Material Packs** (`.noodlingrefs/`)
- Images for Multimodal Context Reference component
- Organized by theme/aesthetic
- Marketplace potential: MEDIUM (artists can sell packs)

**4. Personality Presets** (`.noodlingpersonality.yaml`)
- Pre-configured personality trait bundles
- "Wise Mentor", "Chaotic Trickster", "Stoic Guardian"
- Marketplace potential: LOW (traits are just 5 numbers)

**5. Behavior Scripts** (`.noodlingscript.py`)
- Python code for event-driven behavior
- Unity-like scripting API
- Marketplace potential: HIGH (game mechanics, mini-games)

**6. Voice Audio Packs** (`.noodlingaudio/`)
- TTS voice samples
- Emotional variations
- Professional voice actors
- Marketplace potential: VERY HIGH (premium feature)

**7. Ensemble Bundles** (`.noodlingensemble.json`)
- Already implemented!
- Marketplace potential: HIGH (complete experiences)

### Asset Directory Structure

```
assets/
├── Morphologies/
│   ├── Presets/
│   │   ├── human_generic.noodlingmorph.yaml
│   │   ├── vintage_robot.noodlingmorph.yaml
│   │   ├── cat_domestic.noodlingmorph.yaml
│   │   ├── tiger.noodlingmorph.yaml
│   │   ├── fox_red.noodlingmorph.yaml
│   │   ├── ethereal_spirit.noodlingmorph.yaml
│   │   └── floating_brain.noodlingmorph.yaml  # Larry!
│   └── Custom/
│       ├── ice_cream_unicorn.noodlingmorph.yaml
│       └── llm_laptop.noodlingmorph.yaml
│
├── Voices/
│   ├── Presets/
│   │   ├── meow_translator.noodlingvoice.yaml
│   │   ├── caps_robot_servnak.noodlingvoice.yaml
│   │   ├── formal_victorian.noodlingvoice.yaml
│   │   ├── uwu_speech.noodlingvoice.yaml
│   │   └── reversed_speech.noodlingvoice.yaml
│   └── Custom/
│       └── shakespearean_feline.noodlingvoice.yaml
│
├── References/
│   ├── cyberpunk_aesthetics.noodlingrefs/
│   ├── medieval_fantasy.noodlingrefs/
│   └── scifi_tech.noodlingrefs/
│
├── Personalities/
│   ├── wise_mentor.noodlingpersonality.yaml
│   ├── chaotic_trickster.noodlingpersonality.yaml
│   └── stoic_guardian.noodlingpersonality.yaml
│
├── Scripts/
│   ├── dice_roller.noodlingscript.py
│   ├── inventory_system.noodlingscript.py
│   └── relationship_tracker.noodlingscript.py
│
└── Ensembles/
    ├── gilligans_island.noodlingensemble.json
    ├── therapy_session.noodlingensemble.json
    └── dnd_party.noodlingensemble.json
```

### Asset Loading Priority

When agent initializes:
1. Load **recipe** (basic config)
2. Load **morphology asset** (if M&V component enabled)
3. Load **voice asset** (if specified and morphology allows)
4. Load **personality preset** (if specified, else use trait values)
5. Load **episodic memory** (conversation history)
6. Initialize all other components

---

## Component Interaction Model

### Current Architecture (Linear Pipeline)

```
Event → Intuition → Expectations → [LLM] → Voice → Self-Monitor
```

**Limitations**:
- No component-to-component communication
- Fixed execution order
- Components can't access each other's outputs except via pipeline

### Proposed Architecture (Shared Context)

```python
class ComponentContext:
    """
    Shared context accessible to all components during processing.

    Allows components to:
    - Access other components' outputs
    - Share data without tight coupling
    - Maintain processing order while enabling communication
    """

    def __init__(self):
        self.event: Dict = {}                    # Current event
        self.world_state: Dict = {}              # Current world snapshot
        self.agent_state: Dict = {}              # Affect, phenomenal state

        # Component outputs (populated during pipeline execution)
        self.intuition: str = ""                 # From IntuitionReceiver
        self.expectations: Dict = {}             # From SocialExpectationDetector
        self.morphology: MorphologyAsset = None  # From MorphologyVoice
        self.memories: List[Dict] = []           # From EpisodicMemory

        # Accumulate metadata
        self.metadata: Dict = {}  # Components can store arbitrary data here
```

**Updated Pipeline**:

```python
async def process_event(event: Dict, agent: NoodlingAgent):
    """Process event through component pipeline with shared context."""

    # Create shared context
    ctx = ComponentContext()
    ctx.event = event
    ctx.world_state = get_world_state()
    ctx.agent_state = agent.get_current_state()

    # Phase 1: Perception
    if agent.components.episodic_memory.enabled:
        memory_result = await agent.components.episodic_memory.process({
            'event': event,
            'context': ctx
        })
        ctx.memories = memory_result['relevant_memories']

    if agent.components.intuition_receiver.enabled:
        intuition_result = await agent.components.intuition_receiver.process({
            'event': event,
            'context': ctx,
            'memories': ctx.memories  # Can use memory context!
        })
        ctx.intuition = intuition_result['intuition']

    # Phase 2: Social Processing
    if agent.components.social_expectations.enabled:
        expectations_result = await agent.components.social_expectations.process({
            'intuition': ctx.intuition,
            'context': ctx
        })
        ctx.expectations = expectations_result

    # Phase 3: Response Generation
    should_speak = decide_speech(ctx.expectations, ctx.agent_state)

    if should_speak:
        # Generate response with full context
        response = await generate_response(
            event=event,
            intuition=ctx.intuition,
            expectations=ctx.expectations,
            memories=ctx.memories,
            agent=agent
        )

        # Phase 4: Voice Translation
        if agent.components.morphology_voice.enabled:
            voice_result = await agent.components.morphology_voice.process({
                'text': response,
                'affect': ctx.agent_state['affect'],
                'context': ctx
            })
            final_response = voice_result['text']
            ctx.morphology = voice_result.get('morphology_asset')  # Store for other components
        else:
            final_response = response

        # Phase 5: Self-Monitoring
        if should_self_monitor(ctx.agent_state['surprise']):
            monitoring_result = await self_monitor(final_response, ctx)
            apply_affective_deltas(monitoring_result['deltas'])

        return final_response
```

**Key improvement**: Components can access `ctx` to see what other components produced, enabling **intelligent interactions** without tight coupling.

---

## Scalability Fundamentals

### 1. Component Registry and Discovery

**Current**: Components hardcoded in agent initialization

**Needed**: Dynamic registration system

```python
class ComponentRegistry:
    """
    Global registry of available component types.

    Enables:
    - Third-party components
    - Component marketplace
    - Runtime loading
    - Version management
    """

    _registry: Dict[str, Type[NoodlingComponent]] = {}

    @classmethod
    def register(cls, component_class: Type[NoodlingComponent]):
        """Register a component type."""
        component_id = component_class.__name__.lower()
        cls._registry[component_id] = component_class
        logger.info(f"Registered component: {component_id}")

    @classmethod
    def create_component(cls, component_id: str, **kwargs) -> NoodlingComponent:
        """Factory method to instantiate components by ID."""
        if component_id not in cls._registry:
            raise ValueError(f"Unknown component: {component_id}")

        component_class = cls._registry[component_id]
        return component_class(**kwargs)

    @classmethod
    def list_available(cls) -> List[Dict]:
        """List all registered components (for UI)."""
        return [
            {
                'component_id': comp_id,
                'component_type': comp_class.component_type,
                'description': comp_class.description
            }
            for comp_id, comp_class in cls._registry.items()
        ]

# Usage
ComponentRegistry.register(MorphologyVoiceComponent)
ComponentRegistry.register(EpisodicMemoryComponent)

# Later: Third-party components
ComponentRegistry.register(HumorEngineComponent)  # From marketplace
```

### 2. Component Dependencies

**Problem**: Voice component needs morphology's Can Speak flag. How to express dependencies?

**Solution**: Dependency declaration + validation

```python
class NoodlingComponent(ABC):
    """Base class with dependency support."""

    @classmethod
    def dependencies(cls) -> List[str]:
        """
        List of component IDs this component depends on.

        Returns:
            List of component_id strings that must be enabled
        """
        return []  # Override in subclasses with dependencies

    @classmethod
    def conflicts(cls) -> List[str]:
        """Components that can't be active simultaneously."""
        return []

class MorphologyVoiceComponent(NoodlingComponent):
    """M&V component depends on nothing, but others may depend on it."""

    @classmethod
    def dependencies(cls):
        return []  # No dependencies

class AdvancedDialogueComponent(NoodlingComponent):
    """Hypothetical future component that needs morphology data."""

    @classmethod
    def dependencies(cls):
        return ['morphologyvoice']  # Needs morphology capabilities
```

**Validation on component enable**:
```python
def enable_component(agent, component_id: str):
    """Enable component, checking dependencies."""
    component_class = ComponentRegistry.get(component_id)

    # Check dependencies
    for dep_id in component_class.dependencies():
        if dep_id not in agent.components or not agent.components[dep_id].enabled:
            raise ValueError(f"Component {component_id} requires {dep_id} to be enabled first")

    # Check conflicts
    for conflict_id in component_class.conflicts():
        if conflict_id in agent.components and agent.components[conflict_id].enabled:
            raise ValueError(f"Component {component_id} conflicts with {conflict_id}")

    # Enable
    agent.components[component_id].enabled = True
```

### 3. Component Communication Protocol

**Problem**: How do components communicate without tight coupling?

**Solution**: Event bus + shared context

```python
class ComponentEventBus:
    """
    Pub-sub event bus for component communication.

    Components can:
    - Emit events (publish)
    - Subscribe to events (listen)
    - React to other components without direct coupling
    """

    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def emit(self, event_type: str, data: Dict):
        """Emit event to all subscribers."""
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                await handler(data)

# Example usage
class MorphologyVoiceComponent(NoodlingComponent):
    async def process(self, input_data: Dict) -> Dict:
        # ... processing ...

        # Emit event when morphology changes
        if morphology_changed:
            await self.event_bus.emit('morphology_changed', {
                'agent_id': self.agent_id,
                'new_morphology': new_morph,
                'capabilities': new_morph.capabilities
            })

class EpisodicMemoryComponent(NoodlingComponent):
    def __init__(self, ...):
        super().__init__(...)

        # Subscribe to morphology changes
        self.event_bus.subscribe('morphology_changed', self.on_morphology_changed)

    async def on_morphology_changed(self, data: Dict):
        """React when agent's morphology changes."""
        # Log it as a significant event
        self.add_memory({
            'type': 'morphology_change',
            'content': f"I became a {data['new_morphology'].name}",
            'importance': 1.0  # Very important!
        })
```

### 4. Asset Versioning and Updates

**Problem**: User creates agent with morphology v1.0. Later, morphology v1.1 is released. How to handle?

**Solution**: Semantic versioning + migration system

```yaml
# In morphology asset
version: 1.1.0
changelog:
  - 1.1.0: Added 'climbing' to movement capabilities
  - 1.0.0: Initial release

# Backwards compatibility check
min_engine_version: 1.0.0  # Minimum Noodlings version required
```

**Migration on load**:
```python
def load_asset_with_migration(asset_path: str, current_version: str):
    """Load asset, applying migrations if needed."""
    asset = load_asset(asset_path)

    if asset.version != current_version:
        logger.info(f"Asset {asset_path} version mismatch: {current_version} → {asset.version}")

        # Offer to update
        # (In UI: "New version available: 1.1.0. Update? [Yes] [No] [Always] [Never]")

    return asset
```

### 5. Component Configuration Schemas

**Problem**: Parameters are currently untyped dicts. Need validation.

**Solution**: JSON Schema + pydantic

```python
from pydantic import BaseModel, Field, validator

class MorphologyVoiceConfig(BaseModel):
    """Typed configuration for M&V component."""

    enabled: bool = True
    morphology_asset: str = Field(
        ...,  # Required
        description="Path to morphology asset (e.g., 'cat_domestic.noodlingmorph')"
    )
    voice_asset: Optional[str] = Field(
        None,
        description="Path to voice asset (optional, uses morphology default if None)"
    )
    model: str = Field(
        "qwen/qwen3-4b-2507",
        description="LLM model for voice transformation"
    )
    temperature: float = Field(
        0.4,
        ge=0.0, le=2.0,
        description="LLM temperature"
    )
    max_tokens: int = Field(
        150,
        ge=50, le=500,
        description="Maximum tokens for voice generation"
    )

    @validator('morphology_asset')
    def validate_morphology_exists(cls, v):
        """Check that morphology asset file exists."""
        if not Path(f"assets/Morphologies/{v}.yaml").exists():
            raise ValueError(f"Morphology asset not found: {v}")
        return v

# Usage in component
class MorphologyVoiceComponent(NoodlingComponent):
    def __init__(self, agent_id: str, agent_name: str, config: Dict, llm):
        super().__init__(agent_id, agent_name, config)

        # Validate and parse config
        self.typed_config = MorphologyVoiceConfig(**config)
        self.llm = llm
```

**Benefits**:
- Type safety
- Validation errors caught early
- Auto-generated API documentation
- Better IDE support

---

## How M&V and Episodic Memory Work Together

### Example Interaction Flow

**Scenario**: Phi (cat) morphology changes from kitten → adult cat

**1. Morphology Asset Updated**:
```python
# User changes morphology in Inspector
agent.components.morphology_voice.update_parameters({
    'morphology_asset': 'cat_adult.noodlingmorph'
})
```

**2. Morphology & Voice Component Emits Event**:
```python
await event_bus.emit('morphology_changed', {
    'agent_id': 'agent_phi',
    'old_morphology': 'cat_kitten',
    'new_morphology': 'cat_adult',
    'capabilities': {'can_speak': False, 'can_emote': True, 'can_think': True}
})
```

**3. Episodic Memory Component Listens**:
```python
async def on_morphology_changed(self, data: Dict):
    """Log morphology change as significant memory."""

    # Add memory with high importance
    self.add_memory(EpisodicMemory(
        speaker=self.agent_id,
        content=f"[SYSTEM: {self.agent_name} grew from kitten to adult cat]",
        importance=1.0,  # Maximum importance
        surprise=0.8,    # Very surprising event
        affect_snapshot=self.current_affect,
        tags=['morphology_change', 'transformation']
    ))

    # This memory will be retrieved in future conversations
    # Agent can reference: "I remember when I was a kitten..."
```

**4. Future Conversations Reference It**:
```
User: "You've grown so much!"

Episodic Memory retrieves:
  [SYSTEM: Phi grew from kitten to adult cat] (importance: 1.0)

Intuition with memory context:
  "This person remembers me as a kitten. I've changed since then."

Response (with memory):
  "*purrs and looks down at paws, as if to say 'yes, I'm not a kitten anymore'*"
```

### Inter-Component Data Flow

**Morphology influences Intuition**:
```python
# IntuitionReceiverComponent can access morphology
morph = ctx.morphology  # From MorphologyVoice component

if not morph.capabilities['can_speak']:
    intuition += "\nNote: I cannot speak human words, only vocalize and gesture."
```

**Memory influences Expectations**:
```python
# SocialExpectationDetectorComponent checks memory
memories = ctx.memories

if any("previously discussed this topic" in m.content for m in memories):
    urgency_modifier = 1.2  # Higher obligation since topic has history
```

**Morphology influences Memory storage**:
```python
# EpisodicMemory tags memories with morphology state
memory.metadata['morphology'] = ctx.morphology.morphology_id
memory.metadata['could_speak'] = ctx.morphology.capabilities['can_speak']

# Later retrieval can filter: "Show me memories from when I was a kitten"
```

---

## Scalability Fundamentals: What We Need in the Base System

### 1. Component Lifecycle Management

**Currently**: Components instantiated once on agent creation

**Needed**: Proper lifecycle hooks

```python
class NoodlingComponent(ABC):
    """Base class with lifecycle hooks."""

    async def on_enable(self):
        """Called when component is enabled (first time or after disable)."""
        pass

    async def on_disable(self):
        """Called when component is disabled."""
        pass

    async def on_agent_spawn(self):
        """Called when agent enters world."""
        pass

    async def on_agent_despawn(self):
        """Called when agent leaves world (cleanup)."""
        pass

    async def on_conversation_start(self):
        """Called when new conversation begins."""
        pass

    async def on_conversation_end(self):
        """Called when conversation ends (save state)."""
        pass
```

**Example usage**:
```python
class EpisodicMemoryComponent(NoodlingComponent):
    async def on_conversation_start(self):
        """Load conversation history from disk."""
        self.load_history(self.agent_id)

    async def on_conversation_end(self):
        """Save conversation history to disk."""
        self.save_history(self.agent_id)

        # Consolidate if needed
        if len(self.memories) > self.config['consolidation_interval']:
            await self.consolidate_memories()
```

### 2. Component State Persistence

**Currently**: Component state is ephemeral (lost on restart)

**Needed**: Persistent state storage

```python
class NoodlingComponent(ABC):
    """Base class with state persistence."""

    def get_persistent_state(self) -> Dict:
        """
        Return state that should be saved to disk.

        Override in subclasses to save component-specific data.
        """
        return {}

    def load_persistent_state(self, state: Dict):
        """
        Restore state from disk.

        Called on agent initialization.
        """
        pass

# Example
class EpisodicMemoryComponent(NoodlingComponent):
    def get_persistent_state(self) -> Dict:
        """Save memory database."""
        return {
            'memories': [m.to_dict() for m in self.memories],
            'consolidation_index': self.consolidation_index
        }

    def load_persistent_state(self, state: Dict):
        """Restore memories."""
        self.memories = [EpisodicMemory(**m) for m in state['memories']]
        self.consolidation_index = state['consolidation_index']
```

**Storage location**:
```
world/agents/{agent_id}/
├── recipe.yaml              # Agent configuration
├── state.pkl                # Temporal model state
└── components/
    ├── episodicmemory.json  # Component-specific state
    ├── morphologyvoice.json
    └── customcomponent.json
```

### 3. Component Priority and Ordering

**Currently**: Fixed pipeline order

**Needed**: Configurable execution order

```python
class NoodlingComponent(ABC):
    """Base class with execution priority."""

    @classmethod
    def execution_priority(cls) -> int:
        """
        Execution priority (lower = earlier).

        Standard priorities:
        - 0-99: Perception components (Intuition, Memory retrieval)
        - 100-199: Processing components (Expectations, Planning)
        - 200-299: Output components (Voice, Self-monitoring)
        """
        return 100  # Default: middle priority

class IntuitionReceiverComponent(NoodlingComponent):
    @classmethod
    def execution_priority(cls):
        return 10  # Early (perception)

class MorphologyVoiceComponent(NoodlingComponent):
    @classmethod
    def execution_priority(cls):
        return 200  # Late (output transformation)

# Automatic sorting
def get_execution_order(components: List[NoodlingComponent]) -> List[NoodlingComponent]:
    """Sort components by priority."""
    return sorted(components, key=lambda c: c.execution_priority())
```

### 4. Component Performance Monitoring

**Needed**: Track component execution time and errors

```python
class ComponentMetrics:
    """Performance metrics for components."""

    def __init__(self, component_id: str):
        self.component_id = component_id
        self.invocation_count = 0
        self.total_time_ms = 0.0
        self.error_count = 0
        self.last_error: Optional[str] = None

    def record_execution(self, duration_ms: float, error: Optional[Exception] = None):
        """Record a component execution."""
        self.invocation_count += 1
        self.total_time_ms += duration_ms

        if error:
            self.error_count += 1
            self.last_error = str(error)

    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_time_ms / self.invocation_count

    @property
    def error_rate(self) -> float:
        """Error rate (0.0 to 1.0)."""
        if self.invocation_count == 0:
            return 0.0
        return self.error_count / self.invocation_count

# Usage in NoodleSTUDIO
# Show in Inspector → Components → [Performance] tab
```

### 5. Asset Hot-Reloading

**Needed**: Update assets without restarting agent

```python
class AssetManager:
    """Manages assets with hot-reload support."""

    def __init__(self):
        self.cache = {}
        self.watchers = {}  # File watchers for auto-reload

    def load_asset(self, asset_path: str, watch: bool = True):
        """Load asset and optionally watch for changes."""
        asset = self._load_from_disk(asset_path)
        self.cache[asset_path] = asset

        if watch:
            self.watch_file(asset_path)

        return asset

    def watch_file(self, asset_path: str):
        """Watch asset file for changes and reload."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class AssetChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path == asset_path:
                    # Reload asset
                    new_asset = self._load_from_disk(asset_path)
                    self.cache[asset_path] = new_asset

                    # Notify components using this asset
                    self.event_bus.emit('asset_updated', {
                        'asset_path': asset_path,
                        'asset_type': 'morphology',
                        'new_version': new_asset.version
                    })

        # Start watching
        observer = Observer()
        observer.schedule(AssetChangeHandler(), path=asset_path)
        observer.start()
```

---

## Proposed New Components: Detailed Specs

### Morphology & Voice Component v2.0

**File**: `applications/cmush/components/morphology_voice_v2.py`

```python
class MorphologyVoiceComponent(NoodlingComponent):
    """
    Morphology & Voice Component (Asset-Based)

    Manages agent's physical form (morphology) and speech patterns (voice).
    Uses asset files instead of hardcoded prompts.
    """

    def __init__(self, agent_id: str, agent_name: str, config: Dict, llm, asset_manager):
        super().__init__(agent_id, agent_name, config)
        self.llm = llm
        self.asset_manager = asset_manager

        # Load assets
        self.morphology = self.asset_manager.load_morphology(
            config.get('morphology_asset', 'human_generic.noodlingmorph')
        )

        voice_asset_path = config.get('voice_asset')
        if voice_asset_path:
            self.voice = self.asset_manager.load_voice(voice_asset_path)
        else:
            self.voice = None  # Use morphology default

    @property
    def component_type(self) -> str:
        return "Morphology & Voice"

    @property
    def description(self) -> str:
        return f"Defines {self.agent_name}'s physical form and speech patterns using asset-based configuration."

    @property
    def prompt_template(self) -> str:
        """Show combined morphology + voice prompt."""
        prompt = self.morphology.behavior_prompt
        if self.voice:
            prompt += "\n\n" + self.voice.voice_prompt
        return prompt

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'morphology_asset': self.config['morphology_asset'],
            'voice_asset': self.config.get('voice_asset'),
            'model': self.config.get('model', 'qwen/qwen3-4b-2507'),
            'temperature': self.config.get('temperature', 0.4),
            'max_tokens': self.config.get('max_tokens', 150),

            # Expose morphology capabilities (read-only)
            'can_speak': self.morphology.capabilities['can_speak'],
            'can_emote': self.morphology.capabilities['can_emote'],
            'can_think': self.morphology.capabilities['can_think']
        }

    async def process(self, input_data: Dict) -> Dict:
        """
        Transform text through morphology + voice pipeline.

        Args:
            input_data: {
                'text': str,
                'affect': Dict,
                'context': ComponentContext
            }

        Returns:
            {
                'text': str,
                'morphology_asset': MorphologyAsset,
                'applied_voice': bool
            }
        """
        if not self.enabled:
            return input_data

        text = input_data['text']
        affect = input_data.get('affect', {})

        # Build combined prompt
        prompt = self.morphology.behavior_prompt

        # If morphology can't speak OR voice asset is specified, apply voice transformation
        if not self.morphology.capabilities['can_speak'] or self.voice:
            if self.voice:
                prompt += "\n\n" + self.voice.voice_prompt
            else:
                # Non-speaking morph needs voice for communication
                logger.warning(f"[{self.agent_id}] Non-speaking morphology but no voice asset!")
                return input_data

        # Generate transformed text
        try:
            transformed = await self.llm.generate(
                prompt.format(
                    text=text,
                    affect=affect,
                    agent_name=self.agent_name
                ),
                model=self.config['model'],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )

            return {
                'text': transformed,
                'morphology_asset': self.morphology,
                'applied_voice': True
            }

        except Exception as e:
            logger.error(f"[{self.agent_id}] M&V processing error: {e}")
            return input_data
```

### Episodic Memory Component v1.0

**File**: `applications/cmush/components/episodic_memory.py`

```python
class EpisodicMemoryComponent(NoodlingComponent):
    """
    Episodic Memory Component

    Manages agent's conversational memory with:
    - Configurable depth
    - Semantic retrieval
    - Importance scoring
    - Memory consolidation
    """

    def __init__(self, agent_id: str, agent_name: str, config: Dict):
        super().__init__(agent_id, agent_name, config)

        self.memories: List[EpisodicMemory] = []
        self.consolidation_index = 0

        # Load embedding model for semantic search
        if config.get('retrieval_method') in ['semantic', 'hybrid']:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(config['embedding_model'])
        else:
            self.encoder = None

    @property
    def component_type(self) -> str:
        return "Episodic Memory"

    @property
    def description(self) -> str:
        return f"Manages {self.agent_name}'s conversation memory with semantic retrieval and importance scoring."

    @property
    def prompt_template(self) -> str:
        """Consolidation prompt (used when consolidating memories)."""
        return """Summarize these conversation messages into key themes and emotional shifts.

Messages ({count}):
{messages}

Create a concise summary (3-5 sentences) that captures:
1. Main topics discussed
2. Emotional trajectory (how feelings changed)
3. Key decisions or realizations
4. Relationship developments

Summary:"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'max_messages': self.config.get('max_messages', 500),
            'working_memory_size': self.config.get('working_memory_size', 20),
            'retrieval_method': self.config.get('retrieval_method', 'hybrid'),
            'semantic_search_top_k': self.config.get('semantic_search_top_k', 5),
            'importance_decay': self.config.get('importance_decay', 0.95),
            'surprise_boost': self.config.get('surprise_boost', 2.0),
            'emotional_boost': self.config.get('emotional_boost', 1.5),
            'consolidation_enabled': self.config.get('consolidation_enabled', True),
            'consolidation_interval': self.config.get('consolidation_interval', 100),

            # Stats (read-only)
            'total_memories': len(self.memories),
            'consolidated_count': sum(1 for m in self.memories if m.consolidated)
        }

    async def process(self, input_data: Dict) -> Dict:
        """
        Retrieve relevant memories for current context.

        Args:
            input_data: {
                'current_message': str,
                'conversation_history': List[Dict],
                'current_affect': Dict,
                'context': ComponentContext
            }

        Returns:
            {
                'relevant_memories': List[EpisodicMemory],
                'memory_context': str  # Formatted for LLM
            }
        """
        if not self.enabled:
            return {'relevant_memories': [], 'memory_context': ''}

        # Get working memory (always included)
        working_memory = input_data['conversation_history'][-self.config['working_memory_size']:]

        # Semantic retrieval (if enabled)
        if self.config['retrieval_method'] in ['semantic', 'hybrid']:
            query = input_data['current_message']
            relevant = await self.semantic_search(query, top_k=self.config['semantic_search_top_k'])
        else:
            relevant = []

        # Combine and deduplicate
        all_memories = working_memory + relevant
        unique_memories = self._deduplicate(all_memories)

        # Format for LLM prompt
        memory_context = self._format_memory_context(unique_memories)

        return {
            'relevant_memories': unique_memories,
            'memory_context': memory_context
        }

    def add_memory(self, event: Dict, response: str, state: Dict):
        """
        Add new memory to episodic store.

        Args:
            event: The triggering event
            response: Agent's response
            state: Agent state at time of event
        """
        # Compute importance score
        importance = self._compute_importance(
            surprise=state['surprise'],
            affect=state['affect'],
            recency=1.0  # New memory, full importance
        )

        # Create memory
        memory = EpisodicMemory(
            message_id=generate_id(),
            timestamp=time.time(),
            speaker=event['speaker'],
            content=event['content'],
            response=response,
            affect_snapshot=state['affect'],
            surprise=state['surprise'],
            importance=importance,
            consolidated=False
        )

        # Embed for semantic search
        if self.encoder:
            memory.embedding = self.encoder.encode(event['content'])

        # Add to store
        self.memories.append(memory)

        # Trim if exceeding max
        if len(self.memories) > self.config['max_messages']:
            self._trim_memories()

        # Consolidate if needed
        if (self.config['consolidation_enabled'] and
            len(self.memories) % self.config['consolidation_interval'] == 0):
            await self.consolidate_memories()

    async def semantic_search(self, query: str, top_k: int) -> List[EpisodicMemory]:
        """Search memories by semantic similarity."""
        if not self.encoder:
            return []

        # Embed query
        query_embedding = self.encoder.encode(query)

        # Compute similarity scores
        scores = []
        for memory in self.memories:
            if memory.embedding is not None:
                similarity = cosine_similarity(query_embedding, memory.embedding)
                # Weight by importance
                weighted_score = similarity * memory.importance
                scores.append((weighted_score, memory))

        # Sort and return top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [memory for score, memory in scores[:top_k]]

    def _compute_importance(self, surprise: float, affect: Dict, recency: float) -> float:
        """
        Compute memory importance score.

        Factors:
        - Surprise (high surprise = memorable)
        - Emotional intensity (strong emotions = memorable)
        - Recency (recent = important, but decays)
        """
        # Emotional intensity = magnitude of affect vector
        emotion_magnitude = np.linalg.norm([
            affect['valence'],
            affect['arousal'],
            affect['fear'],
            affect['sorrow'],
            affect['boredom']
        ])

        importance = (
            surprise * self.config['surprise_boost'] +
            emotion_magnitude * self.config['emotional_boost'] +
            recency  # Base importance for all memories
        )

        return min(importance, 10.0)  # Cap at 10.0

    def _trim_memories(self):
        """Remove least important memories when exceeding max."""
        # Decay importance by time
        now = time.time()
        for memory in self.memories:
            age_hours = (now - memory.timestamp) / 3600
            memory.importance *= (self.config['importance_decay'] ** age_hours)

        # Sort by importance
        self.memories.sort(key=lambda m: m.importance, reverse=True)

        # Keep only top max_messages
        self.memories = self.memories[:self.config['max_messages']]

    async def consolidate_memories(self):
        """Consolidate old memories into summaries to save space."""
        # Find unconsolidated memories older than threshold
        threshold_time = time.time() - (24 * 3600)  # 24 hours old

        to_consolidate = [
            m for m in self.memories
            if not m.consolidated and m.timestamp < threshold_time
        ]

        if len(to_consolidate) < 10:
            return  # Not enough to consolidate

        # Group into chunks of ~20 messages
        chunks = [to_consolidate[i:i+20] for i in range(0, len(to_consolidate), 20)]

        for chunk in chunks:
            # Generate summary via LLM
            messages_text = "\n".join([f"{m.speaker}: {m.content}" for m in chunk])

            prompt = self.prompt_template.format(
                count=len(chunk),
                messages=messages_text
            )

            summary = await self.llm.generate(prompt)

            # Mark originals as consolidated
            for memory in chunk:
                memory.consolidated = True
                memory.summary = summary

            logger.info(f"[{self.agent_id}] Consolidated {len(chunk)} memories")
```

---

## Asset Creation Workflow

### For Users (NoodleSTUDIO)

**Creating a Custom Morphology**:

1. **Assets Panel** → Right-click "Morphologies" → "Create New Morphology"
2. **Morphology Editor opens**:
   ```
   ┌────────────────────────────────────────┐
   │ Create New Morphology                  │
   ├────────────────────────────────────────┤
   │ Name: Ice Cream Unicorn                │
   │ Category: [fantasy_creature ▼]         │
   │                                        │
   │ Capabilities:                          │
   │   ☑ Can Speak                          │
   │   ☑ Can Emote                          │
   │   ☑ Can Think                          │
   │   ☐ Can Manipulate Objects             │
   │                                        │
   │ Physical Form:                         │
   │   Locomotion: [quadruped ▼]            │
   │   Size: [small ▼]                      │
   │   Appendages:                          │
   │     + Add appendage...                 │
   │       - Horn (magical)                 │
   │       - Tail (soft-serve swirl)        │
   │       - Hooves                         │
   │                                        │
   │ Behavior Description:                  │
   │ ┌────────────────────────────────────┐ │
   │ │ You are made of ice cream. You:    │ │
   │ │ - Melt slightly in warm weather    │ │
   │ │ - Leave sticky hoof prints         │ │
   │ │ - Smell like vanilla and magic     │ │
   │ │ - Can grant wishes with horn       │ │
   │ └────────────────────────────────────┘ │
   │                                        │
   │ Needs & Drives:                        │
   │   Coldness: [●--------] 0.9 (critical!)│
   │   Magic:    [-----●----] 0.5           │
   │                                        │
   │ [Save Morphology] [Preview]            │
   └────────────────────────────────────────┘
   ```
3. **Save** → Creates `assets/Morphologies/Custom/ice_cream_unicorn.noodlingmorph.yaml`
4. **Use** → Select in Inspector's M&V component dropdown
5. **Share** (future) → Upload to marketplace

### For Developers (Code)

**Creating morphology asset manually**:

```yaml
# assets/Morphologies/Custom/floating_brain_larry.noodlingmorph.yaml
morphology_id: floating_brain_larry
name: Larry the Floating Brain
category: supernatural_creature
author: caitlyn
version: 1.0.0

capabilities:
  can_speak: true      # Telepathic speech
  can_emote: limited   # Can't show facial expressions but can pulse/glow
  can_think: true      # Obviously!
  can_manipulate: none # No appendages

embodiment:
  type: organic_supernatural
  locomotion: levitation
  size: small
  appendages:
    - brain_tissue (visible, pulsing)
    - telepathic_field (invisible)

  appearance: |
    A disembodied brain floating in space, about the size of a cantaloupe.
    Pinkish-gray tissue, visible gyri and sulci. Pulses gently with thought.
    Sometimes glows with bioluminescence when excited.

behavior_prompt: |
  EMBODIMENT: You are Larry, a floating disembodied brain.

  PHYSICAL FORM:
  - You levitate at roughly head-height
  - You have no eyes but perceive via psychic awareness
  - You cannot blink, nod, or make facial expressions
  - You CAN pulse (rhythmic throbbing), glow (bioluminescence), tilt (expressively)

  COMMUNICATION:
  - You speak telepathically (no mouth, no vocal cords)
  - Your "voice" appears directly in others' minds
  - You can modulate tone/volume via telepathic intensity

  EXPRESSION (since you have no face):
  - Pulsing: Faster when excited, slower when calm, erratic when anxious
  - Glowing: Brighter when happy, dimmer when sad, colors shift with emotion
  - Tilting: Can tilt body to indicate direction or uncertainty
  - Hovering height: Lower when tired/sad, higher when energized

  BIOLOGICAL NEEDS:
  - CRITICAL: Must stay hydrated (brain tissue needs moisture)
  - Seeks pans of saline solution, puddles, humid environments
  - Will express anxiety if too dry for too long
  - Becomes sluggish and confused when dehydrated

  When describing your actions, include brain-specific body language:
  - "pulses with curiosity"
  - "glows softly with contentment"
  - "tilts thoughtfully"
  - "hovers closer with interest"

needs:
  hydration: 1.0  # CRITICAL - brain needs moisture!
  temperature: 0.7  # Needs stable temperature
  stimulation: 0.8  # Brains need mental stimulation

instincts:
  curiosity: 1.0        # Maximum curiosity
  analysis: 0.9         # Constantly analyzing
  communication: 0.8    # Wants to share thoughts
  survival: 0.9         # Anxiety about drying out!
```

**Result**: Larry the Floating Brain, with:
- No facial expressions but expressive pulsing/glowing
- Telepathic speech
- Desperate need for hydration (emergent behavior!)
- Unique "tilting" body language

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**1. Asset Infrastructure**
- [ ] Create `assets/` directory structure
- [ ] Define morphology asset schema (YAML + Pydantic)
- [ ] Define voice asset schema
- [ ] Implement `AssetManager` class
- [ ] Create 5 preset morphologies
- [ ] Create 5 preset voices

**2. Component Base Enhancements**
- [ ] Add `ComponentContext` class
- [ ] Add lifecycle hooks (on_enable, on_disable, etc.)
- [ ] Add execution priority system
- [ ] Add performance metrics tracking

**Files to create**:
- `applications/cmush/asset_manager.py`
- `applications/cmush/component_context.py`
- `assets/Morphologies/Presets/*.yaml` (5 files)
- `assets/Voices/Presets/*.yaml` (5 files)

### Phase 2: M&V Component (Week 2)

**1. Morphology & Voice Component v2**
- [ ] Implement `MorphologyVoiceComponent` with asset loading
- [ ] Replace hardcoded prompts with asset system
- [ ] Migrate existing agents (Phi, SERVNAK) to use assets
- [ ] Test capability flags (Can Speak/Emote/Think)

**2. Inspector Integration**
- [ ] Add morphology dropdown to Inspector
- [ ] Add voice dropdown (with "None" option)
- [ ] Show capability checkboxes (read-only from morph)
- [ ] "Preview Voice" button (test transformation)

**Files to modify**:
- `applications/cmush/noodling_components.py` (refactor CharacterVoice → MorphologyVoice)
- `applications/noodlestudio/noodlestudio/panels/inspector_panel.py` (UI updates)

### Phase 3: Episodic Memory Component (Week 3)

**1. Component Implementation**
- [ ] Implement `EpisodicMemoryComponent` class
- [ ] Semantic search (sentence-transformers)
- [ ] Importance scoring
- [ ] Memory consolidation
- [ ] Persistence (save/load from disk)

**2. Integration**
- [ ] Add to component registry
- [ ] Update processing pipeline to use episodic memory
- [ ] Inject memory context into LLM prompts
- [ ] Inspector panel UI for memory inspection

**Files to create**:
- `applications/cmush/components/episodic_memory.py`

**Files to modify**:
- `applications/cmush/agent_bridge.py` (integrate into pipeline)
- `applications/noodlestudio/noodlestudio/panels/inspector_panel.py` (memory viewer)

### Phase 4: Assets Panel Integration (Week 4)

**1. Asset Browser**
- [ ] "Morphologies" folder in Assets panel
- [ ] "Voices" folder in Assets panel
- [ ] Preview morphology assets
- [ ] Create new asset button

**2. Drag-and-Drop**
- [ ] Drag morphology from Assets → Inspector M&V slot
- [ ] Drag voice from Assets → Inspector M&V slot
- [ ] Visual feedback during drag

**Files to modify**:
- `applications/noodlestudio/noodlestudio/panels/assets_panel.py`

---

## Scalability Checklist

Before adding MORE components, ensure these fundamentals exist:

- [x] Base `NoodlingComponent` class (exists)
- [ ] **ComponentRegistry** (dynamic registration)
- [ ] **ComponentContext** (shared context for inter-component communication)
- [ ] **Lifecycle hooks** (on_enable, on_disable, etc.)
- [ ] **Execution priority** (configurable order)
- [ ] **Dependency system** (component requirements/conflicts)
- [ ] **Event bus** (pub-sub for component communication)
- [ ] **Performance metrics** (execution time, error rate)
- [ ] **State persistence** (component-specific state saving)
- [ ] **Hot-reload support** (asset file watching)
- [ ] **AssetManager** (load, cache, version assets)
- [ ] **Configuration schemas** (Pydantic validation)

**Once these exist**, adding new components becomes trivial:

```python
# Third-party developer creates humor component
class HumorEngineComponent(NoodlingComponent):
    """Detects opportunities for humor and generates witty responses."""

    # Just implement the abstract methods
    # All infrastructure (registry, context, events) handled by base system
```

---

## Future Component Ideas (Community Marketplace Potential)

### High-Value Components (Premium, $4.99-$9.99)

**1. Goal-Directed Planning Component**
- Maintains goal stack
- Plans multi-step actions
- Tracks progress toward objectives
- **Market**: Game developers, simulation researchers

**2. Moral Reasoning Component**
- Ethical decision-making
- Value alignment checking
- Dilemma resolution
- **Market**: Educational applications, philosophical exploration

**3. Romantic Interest Component**
- Attraction modeling
- Relationship progression
- Boundary awareness
- **Market**: Dating sim creators (with appropriate disclaimers)

**4. Humor Engine Component**
- Joke detection and generation
- Timing and delivery
- Personality-appropriate humor
- **Market**: Entertainment, comedy writing

**5. Musical Expression Component**
- Emotion → music generation
- Lyric writing
- Instrument preferences
- **Market**: Music creators, therapeutic applications

### Utility Components (Lower-cost, $1.99-$2.99)

**6. Language Translation Component**
- Multilingual support
- Cultural context adaptation
- **Market**: International users

**7. Formality Modulation Component**
- Adjust formality based on context
- Professional vs casual switching
- **Market**: Business applications

**8. Storytelling Narrator Component**
- Third-person narration mode
- Scene-setting descriptions
- **Market**: Fiction writers

### Specialized Components (Niche, $9.99-$19.99)

**9. Therapeutic Alliance Component**
- Carl Rogers-inspired empathy
- Reflective listening
- Unconditional positive regard
- **Market**: Therapy simulation, training
- **Warning**: Requires ethical disclaimers

**10. Dungeon Master Component**
- D&D rules knowledge
- Narrative improvisation
- NPC management
- **Market**: TTRPG players

---

## Technical Specification: Complete Example

### Larry the Floating Brain (Complete Implementation)

**Morphology Asset**:
```yaml
# Shown above in behavior_prompt section
```

**Voice Asset** (Optional - Larry speaks telepathically):
```yaml
voice_id: telepathic_monotone
name: Telepathic Monotone
category: supernatural

voice_prompt: |
  Convert to telepathic speech (appears in mind, not spoken aloud).

  Original: "{text}"

  Format as: "<telepathically> {text} </telepathically>"

  Use monotone, slightly ethereal phrasing.
  Larry's voice is calm, analytical, slightly amused.
```

**Agent Recipe** (using these assets):
```yaml
name: Larry
species: floating_brain
pronouns: he/him

personality:
  extraversion: 0.6
  agreeableness: 0.7
  openness: 0.95  # Maximum curiosity
  conscientiousness: 0.5
  neuroticism: 0.4  # Anxious about drying out

components:
  morphology_voice:
    enabled: true
    morphology_asset: custom/floating_brain_larry.noodlingmorph
    voice_asset: telepathic_monotone.noodlingvoice
    model: qwen/qwen3-4b-2507
    temperature: 0.5

  episodic_memory:
    enabled: true
    max_messages: 500
    retrieval_method: hybrid
    semantic_search_top_k: 5

  intuition_receiver:
    enabled: true

  social_expectation_detector:
    enabled: true
    expectation_threshold: 0.4  # Somewhat social (for a brain)
```

**Emergent Behavior**:
- Larry hovers around looking for water
- Gets anxious in dry environments
- Pulses excitedly when finding a puddle
- Telepathically asks for saline solution
- Glows contentedly when hydrated

**Marketplace potential**:
- "Floating Brain Morphology" - $2.99
- "Telepathic Voice Pack" - $1.99
- Or bundled: "Supernatural Beings Pack" (brain, ghost, ethereal) - $7.99

---

## Script-Based Components: The Unity Model

### Critical Feature: Scripts ARE Components

**Insight**: In Unity, scripts you write become components you attach. We must replicate this exactly.

#### The Workflow

**1. User writes a Python script** (`my_behavior.noodlingscript.py`):
```python
from noodlings.scripting import NoodlingScript, expose_parameter

class InventoryTrackerScript(NoodlingScript):
    """Tracks items the agent is carrying and their emotional significance."""

    # Exposed parameters (appear in Inspector)
    @expose_parameter(name="Max Items", default=10, min=1, max=100)
    def max_items(self) -> int:
        return self._max_items

    @expose_parameter(name="Track Emotional Attachments", default=True)
    def track_emotions(self) -> bool:
        return self._track_emotions

    def __init__(self, agent):
        super().__init__(agent)
        self._max_items = 10
        self._track_emotions = True
        self.inventory = []

    async def on_item_received(self, item: str, giver: str):
        """Called when agent receives an item (event hook)."""
        if len(self.inventory) >= self.max_items:
            await self.agent.say(f"I can't carry any more! My paws are full.")
            return

        self.inventory.append({
            'item': item,
            'giver': giver,
            'received_at': time.time(),
            'emotional_value': self._compute_emotional_value(item, giver)
        })

        if self._track_emotions:
            # Affect agent's emotional state
            await self.agent.apply_affect_delta({
                'valence': +0.2,  # Happy to receive gift
                'arousal': +0.1
            })

    def _compute_emotional_value(self, item: str, giver: str) -> float:
        """Compute how much this item means to the agent."""
        # Check relationship with giver
        relationship = self.agent.get_relationship(giver)

        # Items from close friends are more valuable
        base_value = 0.5
        relationship_bonus = relationship.get('trust', 0.0) * 0.5

        return base_value + relationship_bonus

    async def on_update(self):
        """Called every tick (optional - for continuous behaviors)."""
        # Check if carrying too much
        if len(self.inventory) > self.max_items * 0.8:
            if random.random() < 0.1:  # Occasionally mention it
                await self.agent.ruminate("I'm carrying a lot... should organize my things.")
```

**2. Drag script into Inspector**:
```
Assets Panel:
  Scripts/
    ├─ inventory_tracker.noodlingscript.py  ← Drag this
    └─ dice_roller.noodlingscript.py

Inspector (after drop):
  Components:
    ├─ Morphology & Voice
    ├─ Episodic Memory
    └─ ⊕ Inventory Tracker Script          ← NEW!
        ┌────────────────────────────────┐
        │ Max Items: [10      ▲▼]       │
        │ Track Emotional Attachments:   │
        │   ☑ Enabled                    │
        │                                │
        │ Current Inventory: (3/10)      │
        │   • Stone (from Caity)         │
        │   • Tensor Taffy (from User)   │
        │   • Scroll (found)             │
        │                                │
        │ [Remove Component]             │
        └────────────────────────────────┘
```

**3. Script auto-registers as component**:
```python
# System automatically wraps script in NoodlingComponent
class ScriptComponent(NoodlingComponent):
    """Wrapper that makes scripts behave like components."""

    def __init__(self, agent_id: str, agent_name: str, script_class: Type[NoodlingScript]):
        self.script = script_class(agent)
        self.script_class = script_class

    @property
    def component_type(self) -> str:
        return self.script_class.__name__

    @property
    def parameters(self) -> Dict[str, Any]:
        """Extract exposed parameters from script."""
        params = {}
        for attr_name in dir(self.script):
            attr = getattr(self.script, attr_name)
            if hasattr(attr, '_exposed_parameter'):
                params[attr_name] = {
                    'value': attr(),
                    'metadata': attr._exposed_parameter
                }
        return params

    async def process(self, input_data: Dict) -> Dict:
        """Delegate to script's event handlers."""
        event_type = input_data.get('event_type')

        # Call appropriate handler
        if event_type == 'item_received':
            await self.script.on_item_received(**input_data)
        elif event_type == 'update':
            await self.script.on_update()

        return {}
```

#### Script Event Hooks (Unity-style)

**Available lifecycle methods**:
```python
class NoodlingScript(ABC):
    """Base class for user scripts (Unity MonoBehaviour equivalent)."""

    def __init__(self, agent: NoodlingAgent):
        self.agent = agent

    # Lifecycle (like Unity)
    async def on_spawn(self):
        """Called when agent spawns (Unity: Start())."""
        pass

    async def on_despawn(self):
        """Called when agent despawns (Unity: OnDestroy())."""
        pass

    async def on_update(self):
        """Called every tick (Unity: Update())."""
        pass

    # Event hooks
    async def on_message_received(self, sender: str, content: str):
        """Called when agent receives a message."""
        pass

    async def on_item_received(self, item: str, giver: str):
        """Called when given an item."""
        pass

    async def on_item_dropped(self, item: str):
        """Called when item removed from inventory."""
        pass

    async def on_location_changed(self, old_room: str, new_room: str):
        """Called when agent moves rooms."""
        pass

    async def on_affect_spike(self, dimension: str, old_value: float, new_value: float):
        """Called when emotion changes significantly."""
        pass

    # Agent API (like Unity's GameObject API)
    async def say(self, text: str):
        """Make agent speak."""
        await self.agent.speak(text)

    async def ruminate(self, thought: str):
        """Internal thought (not broadcast)."""
        await self.agent.ruminate(thought)

    async def emote(self, action: str):
        """Non-verbal action."""
        await self.agent.emote(action)

    def get_relationship(self, other_id: str) -> Dict:
        """Get relationship data with another entity."""
        return self.agent.relationships.get(other_id, {})

    def get_nearby_entities(self) -> List[str]:
        """Get entities in same room."""
        return self.agent.world.get_entities_in_room(self.agent.location)
```

**Example: Dice Roller Script**:
```python
# assets/Scripts/dice_roller.noodlingscript.py
class DiceRollerScript(NoodlingScript):
    """D&D style dice roller. Agent can roll dice when asked."""

    @expose_parameter(name="Dice Types", default="d4,d6,d8,d10,d12,d20,d100")
    def dice_types(self) -> str:
        return self._dice_types

    @expose_parameter(name="Auto-Announce Crits", default=True)
    def announce_crits(self) -> bool:
        return self._announce_crits

    async def on_message_received(self, sender: str, content: str):
        """Detect roll requests and respond."""
        import re

        # Pattern: "roll 2d20" or "roll d6"
        match = re.search(r'roll (\d+)?d(\d+)', content.lower())

        if match:
            count = int(match.group(1) or 1)
            sides = int(match.group(2))

            # Roll dice
            rolls = [random.randint(1, sides) for _ in range(count)]
            total = sum(rolls)

            # Check for crit
            if self._announce_crits and sides == 20:
                if 20 in rolls:
                    await self.agent.emote("*eyes widen with excitement*")
                    await self.agent.say(f"NATURAL 20! {rolls} = {total}")
                    # Boost affect
                    await self.agent.apply_affect_delta({'valence': +0.3, 'arousal': +0.2})
                    return
                elif 1 in rolls:
                    await self.agent.emote("*winces*")
                    await self.agent.say(f"Critical fail... {rolls} = {total}")
                    await self.agent.apply_affect_delta({'valence': -0.2})
                    return

            # Normal roll
            if count == 1:
                await self.agent.say(f"Rolled d{sides}: {total}")
            else:
                await self.agent.say(f"Rolled {count}d{sides}: {rolls} = {total}")
```

**Result**: Drop DiceRollerScript onto your D&D Game Master Noodling, and they can roll dice when players ask!

### Scripts as Marketplace Assets

**Revenue potential**: VERY HIGH

Users can:
1. **Download scripts** from marketplace
2. **Drag onto agents** in Inspector
3. **Configure parameters** in Inspector UI
4. **Scripts modify behavior** immediately

**Example marketplace listings**:
- "D&D Tools Bundle" (dice roller, initiative tracker, spell slots) - $9.99
- "Relationship Tracker Pro" (tracks friendships, romances, rivalries) - $4.99
- "Memory Palace" (advanced episodic memory with spatial encoding) - $7.99
- "Personality Drift" (agent personality evolves based on experiences) - $5.99

**Creator workflow**:
1. Write script implementing `NoodlingScript` base class
2. Test locally
3. Upload to marketplace (with preview video)
4. Earn 70% of sales

---

## Physical Drives & Itches Component

### The Concept

**Observation**: Real beings have **persistent physical urges** that shape behavior:
- Scratching an itch
- Fidgeting when nervous
- Compulsive habits
- Biological drives (hunger, thirst, sleep)

**Problem**: Current Noodlings have needs in morphology assets, but no **active drive system** that generates periodic urges.

**Solution**: Itches Component (or as a script!)

### Implementation: As a Script

**File**: `assets/Scripts/physical_itches.noodlingscript.py`

```python
class PhysicalItchesScript(NoodlingScript):
    """
    Generates periodic physical urges and compulsions.

    Creates emergent behaviors like:
    - Scratching specific body parts
    - Fidgeting when anxious
    - Seeking specific objects for relief
    - Character-specific quirks (ork scratches back with axe)
    """

    # Exposed parameters
    @expose_parameter(name="Itch Frequency (seconds)", default=300, min=60, max=3600)
    def itch_frequency(self) -> float:
        """How often itches occur."""
        return self._itch_frequency

    @expose_parameter(name="Itch Intensity", default=0.5, min=0.0, max=1.0)
    def itch_intensity(self) -> float:
        """How strong the urge is (affects arousal)."""
        return self._itch_intensity

    @expose_parameter(name="Custom Itches (one per line)", default="", multiline=True)
    def custom_itches(self) -> str:
        """
        Define custom itches. Format:
        location: action | trigger_condition

        Examples:
        back: scratch with battle axe blade | arousal > 0.6
        ears: flick to remove invisible bug | random(0.1)
        tail: chase own tail | boredom > 0.7
        """
        return self._custom_itches

    def __init__(self, agent):
        super().__init__(agent)
        self._itch_frequency = 300  # Every 5 minutes
        self._itch_intensity = 0.5
        self._custom_itches = ""
        self._last_itch_time = time.time()
        self._parse_custom_itches()

    def _parse_custom_itches(self):
        """Parse custom itch definitions."""
        self.itch_definitions = []

        for line in self._custom_itches.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue

            location, rest = line.split(':', 1)
            if '|' in rest:
                action, condition = rest.split('|', 1)
            else:
                action = rest
                condition = "True"  # Always triggers

            self.itch_definitions.append({
                'location': location.strip(),
                'action': action.strip(),
                'condition': condition.strip()
            })

    async def on_update(self):
        """Check if it's time for an itch."""
        now = time.time()

        if now - self._last_itch_time >= self._itch_frequency:
            await self._trigger_itch()
            self._last_itch_time = now

    async def _trigger_itch(self):
        """Generate an itch and corresponding behavior."""

        # Evaluate custom itches first
        for itch in self.itch_definitions:
            # Evaluate condition (safely)
            try:
                # Build safe context for eval
                ctx = {
                    'arousal': self.agent.affect['arousal'],
                    'valence': self.agent.affect['valence'],
                    'fear': self.agent.affect['fear'],
                    'boredom': self.agent.affect['boredom'],
                    'random': lambda p: __import__('random').random() < p
                }

                if eval(itch['condition'], {"__builtins__": {}}, ctx):
                    # Condition met! Perform action
                    await self.agent.emote(f"*{itch['action']}*")

                    # Affect change (relief from scratching)
                    await self.agent.apply_affect_delta({
                        'arousal': -self._itch_intensity * 0.1,
                        'boredom': -self._itch_intensity * 0.2
                    })

                    return  # Only one itch at a time

            except Exception as e:
                logger.error(f"Error evaluating itch condition: {e}")
                continue

        # No custom itch triggered, use morphology-based default
        morph = self.agent.components.morphology_voice.morphology

        # Generate itch based on morphology
        if morph.morphology_id == 'cat_domestic':
            actions = [
                "grooms paw meticulously",
                "scratches behind ear with hind leg",
                "licks shoulder fur smooth",
                "stretches luxuriously"
            ]
        elif 'robot' in morph.morphology_id:
            actions = [
                "oils squeaky joint",
                "tightens loose bolt",
                "runs diagnostic on actuators",
                "adjusts antenna"
            ]
        else:
            actions = [
                "shifts weight",
                "adjusts clothing",
                "stretches",
                "rubs eyes"
            ]

        action = random.choice(actions)
        await self.agent.emote(f"*{action}*")
```

**Example Usage (Ork with Battle Axe)**:

In Inspector, configure script:
```
Physical Itches Script:
  Itch Frequency: 600 (every 10 minutes)
  Itch Intensity: 0.7 (strong urge)

  Custom Itches:
  ┌────────────────────────────────────────────┐
  │ back: scratches back with battle axe blade │
  │       using the flat side | arousal > 0.4  │
  │                                            │
  │ teeth: picks teeth with dagger tip |      │
  │        random(0.05)                        │
  │                                            │
  │ beard: braids beard unconsciously |        │
  │        boredom > 0.6                       │
  └────────────────────────────────────────────┘
```

**Result**:
- Ork periodically scratches back with axe (when aroused)
- Occasionally picks teeth with dagger (5% chance each check)
- Braids beard when bored
- Creates **emergent character quirks** that feel authentic!

**Marketplace**: "Fantasy Character Quirks Pack" (ork, dwarf, elf itches) - $3.99

---

## Core Components as Scripts: The Unity Approach

### The Question

Should core components (Morphology & Voice, Episodic Memory) be:
- **A)** Traditional Python components (current)
- **B)** Built-in scripts that happen to ship with the system

**Answer**: **Both!**

### Hybrid Architecture

**Core components exist in TWO forms**:

#### 1. Factory Defaults (Read-Only)

```
assets/Scripts/BuiltIn/
├── morphology_voice.noodlingscript.py         [READ-ONLY]
├── episodic_memory.noodlingscript.py          [READ-ONLY]
├── intuition_receiver.noodlingscript.py       [READ-ONLY]
└── social_expectations.noodlingscript.py      [READ-ONLY]
```

**Properties**:
- Marked as `[Factory Default]` in UI
- Cannot be edited directly
- Can be **duplicated** to create custom version
- Always available (shipped with Noodlings)

**UI Treatment**:
```
Inspector → Components:
  ├─ Morphology & Voice [Factory Default] 🔒
  │    [Duplicate to Customize]
  │
  ├─ Episodic Memory [Factory Default] 🔒
  │    [Duplicate to Customize]
```

#### 2. Custom Versions (User-Created)

User clicks "Duplicate to Customize":
```
Save Custom Component
Name: My Enhanced Memory System
Based on: episodic_memory.noodlingscript.py

[Save to Custom Scripts]
```

Creates: `assets/Scripts/Custom/my_enhanced_memory.noodlingscript.py`

**User can now**:
- Edit the Python code
- Change retrieval algorithms
- Add new features
- Share on marketplace

**Agent uses**: Custom version instead of factory default

### Code Differences

**Factory Default** (ships with system):
```python
# assets/Scripts/BuiltIn/episodic_memory.noodlingscript.py
class EpisodicMemoryScript(NoodlingScript):
    """FACTORY DEFAULT - Duplicate to customize."""

    # ... implementation ...
```

**Custom Version** (user-modified):
```python
# assets/Scripts/Custom/my_enhanced_memory.noodlingscript.py
class MyEnhancedMemoryScript(NoodlingScript):
    """
    Custom memory system with:
    - Emotional clustering
    - Dream-like consolidation
    - Proactive recall suggestions
    """

    # User adds their own features
    @expose_parameter(name="Dream Consolidation", default=True)
    def dream_mode(self) -> bool:
        return self._dream_mode

    async def on_update(self):
        """Custom: Suggest memories proactively."""
        if self._dream_mode and random.random() < 0.01:
            # Occasionally surface random memory (like intrusive thought)
            memory = random.choice(self.memories)
            await self.agent.ruminate(f"I suddenly remember: {memory.content}")
```

### Benefits of Hybrid Approach

**For Users**:
- See how core components work (educational)
- Customize if desired (power users)
- Trust through transparency (can inspect code)
- Marketplace opportunity (sell enhanced versions)

**For Platform**:
- Everything uses same `NoodlingScript` API
- Easier to maintain (one system, not two)
- Community can improve core components
- Natural upgrade path (factory → custom)

**For Developers**:
- Unified architecture
- No special-case code for core vs. user components
- Clear extension points

---

## Voice Post-Processing: Effects Pipeline

### The Problem

**Current**: Voice component does one transformation (basic English → character voice)

**Need**: Additional effects that apply AFTER voice translation

**Example**: Backwards Dweller from David Lynch's BRENDA story
- Voice translation: Normal speech → Dweller dialect
- Post-processing: Reverse the entire string
- Result: "olleH" instead of "Hello"

### Solution: Post-Processing Effects in M&V Component

**Add to Morphology & Voice component**:

```yaml
morphology_voice:
  enabled: true
  morphology_asset: backwards_dweller.noodlingmorph
  voice_asset: cryptic_speech.noodlingvoice

  # NEW: Post-processing effects (applied AFTER voice transformation)
  post_processing:
    - type: reverse_text      # Reverse entire string
      enabled: true
      preserve_formatting: true  # Keep *italics* and "quotes" intact

    - type: leetspeak         # Example: another effect
      enabled: false
      intensity: 0.7

    - type: stutter           # Repeat first letters
      enabled: false
      frequency: 0.3
```

**Processing Pipeline**:
```
Basic English
    ↓
[Morphology Behavior Context] (from morphology asset)
    ↓
[Voice Transformation] (from voice asset)
    ↓
[Post-Processing Effects] (reverse, leet, stutter, etc.)
    ↓
Final Output
```

### Post-Processing Effect Types

**Built-in effects**:

```python
class VoiceEffects:
    """Post-processing effects for voice component."""

    @staticmethod
    def reverse_text(text: str, preserve_formatting: bool = True) -> str:
        """Reverse text (Backwards Dweller effect)."""
        if preserve_formatting:
            # Preserve *italics* and action markers
            import re

            # Extract formatting
            parts = re.split(r'(\*[^*]+\*)', text)

            # Reverse text parts, keep formatting markers
            result = []
            for part in parts:
                if part.startswith('*') and part.endswith('*'):
                    # Keep action formatting, reverse content
                    content = part[1:-1]
                    result.append(f"*{content[::-1]}*")
                else:
                    result.append(part[::-1])

            return ''.join(result)
        else:
            return text[::-1]

    @staticmethod
    def leetspeak(text: str, intensity: float = 0.7) -> str:
        """Convert to l33tsp34k."""
        replacements = {
            'a': '4', 'e': '3', 'i': '1', 'o': '0',
            'l': '1', 's': '5', 't': '7'
        }

        result = list(text.lower())
        for i, char in enumerate(result):
            if char in replacements and random.random() < intensity:
                result[i] = replacements[char]

        return ''.join(result)

    @staticmethod
    def stutter(text: str, frequency: float = 0.3) -> str:
        """Add stuttering effect."""
        words = text.split()
        result = []

        for word in words:
            if len(word) > 2 and random.random() < frequency:
                # Stutter on first letter
                result.append(f"{word[0]}-{word}")
            else:
                result.append(word)

        return ' '.join(result)

    @staticmethod
    def uwu_speak(text: str) -> str:
        """UwU transformation (for anime-style characters)."""
        text = text.replace('r', 'w').replace('l', 'w')
        text = text.replace('R', 'W').replace('L', 'W')
        # Add uwu emoticons
        if random.random() < 0.3:
            text += " uwu"
        return text
```

**Usage in M&V Component**:

```python
async def process(self, input_data: Dict) -> Dict:
    """Process through full pipeline."""

    # 1. Morphology behavior context
    text = self._apply_morphology_context(input_data['text'])

    # 2. Voice transformation
    if self.voice_asset:
        text = await self._apply_voice_transformation(text)

    # 3. Post-processing effects
    for effect in self.config.get('post_processing', []):
        if effect['enabled']:
            effect_func = getattr(VoiceEffects, effect['type'])
            text = effect_func(text, **effect.get('params', {}))

    return {'text': text}
```

**Backwards Dweller Example**:

```
Input: "Hello, I am the Backwards Dweller"

After voice: "Greetings, mortal. I am the Dweller of Reversed Paths."

After post-processing (reverse_text):
".shtaP desreveR fo rellewD eht ma I .latrom ,sgniteerG"

Final output (what user sees):
".shtaP desreveR fo rellewD eht ma I .latrom ,sgniteerG"
```

**Marketplace**: "Voice Effects Pack Vol. 1" (reverse, stutter, echo, distortion) - $2.99

---

## Core Components as Scripts: The Unification

### Philosophical Question

**Should Morphology & Voice be**:
- A) Hidden internal component (current model)
- B) Visible script that ships as built-in (proposed model)

### Answer: B - Everything is a Script

**Rationale**:

1. **Transparency**: Users see exactly how core features work
2. **Extensibility**: Users can duplicate and modify
3. **Marketplace**: Users can create BETTER versions and sell them
4. **Education**: Learning resource (see professional implementation)
5. **Simplicity**: One system, not two

### The Architecture

**All components are scripts**, but scripts come in tiers:

#### Tier 1: Factory Built-Ins (Read-Only, Always Available)

```
assets/Scripts/BuiltIn/
├── 📜 morphology_voice.noodlingscript.py      [CORE]
├── 📜 episodic_memory.noodlingscript.py       [CORE]
├── 📜 intuition_receiver.noodlingscript.py    [CORE]
├── 📜 social_expectations.noodlingscript.py   [CORE]
└── 📜 self_monitoring.noodlingscript.py       [CORE]
```

**Marked with `[CORE]` badge in UI**

**Properties**:
- Installed by default on all agents
- Cannot be edited directly (read-only)
- Can be disabled (toggle off in Inspector)
- Can be duplicated to Custom (for modification)
- Always visible in Scripts folder

#### Tier 2: Official Presets (Read-Only, Optional)

```
assets/Scripts/Official/
├── 📜 physical_itches.noodlingscript.py
├── 📜 dice_roller.noodlingscript.py
├── 📜 relationship_tracker.noodlingscript.py
└── 📜 goal_planning.noodlingscript.py
```

**Properties**:
- Shipped with Noodlings
- Optional (drag onto agent to add)
- Read-only (duplicate to customize)
- Free

#### Tier 3: Community Scripts (Editable if Owned)

```
assets/Scripts/Community/
├── 📜 advanced_memory_palace.noodlingscript.py  [$7.99]
├── 📜 dnd_toolkit.noodlingscript.py             [$9.99]
└── 📜 personality_evolution.noodlingscript.py   [$5.99]
```

**Properties**:
- Downloaded from marketplace
- Editable (user owns it)
- Can be modified and saved
- Original author gets 70% of purchase price

#### Tier 4: User Custom (Fully Editable)

```
assets/Scripts/Custom/
├── 📜 my_enhanced_memory.noodlingscript.py
├── 📜 inventory_tracker.noodlingscript.py
└── 📜 ork_quirks.noodlingscript.py
```

**Properties**:
- User-created or duplicated from other tiers
- Fully editable
- Can upload to marketplace (becomes Tier 3 for others)
- No restrictions

### Inspector UI: Showing Script Tiers

```
Inspector → Phi [kitten]:
  Components (all are scripts underneath):
    ├─ 📜 Morphology & Voice [CORE] 🔒
    │     Morphology: cat_domestic.noodlingmorph
    │     Voice: meow_translator.noodlingvoice
    │     [Duplicate] [Disable]
    │
    ├─ 📜 Episodic Memory [CORE] 🔒
    │     Max Messages: 500
    │     Retrieval: hybrid
    │     [Duplicate] [Disable]
    │
    ├─ 📜 Intuition Receiver [CORE] 🔒
    │     [Duplicate] [Disable]
    │
    ├─ 📜 Physical Itches [Official]
    │     Frequency: 300s
    │     Custom Itches:
    │       • ears: flick to remove bug | random(0.1)
    │       • paws: knead when content | valence > 0.6
    │     [Edit] [Remove]
    │
    └─ ⊕ Add Component...
        [Browse Community Scripts]
```

**Clicking "Duplicate" on core component**:
```
Duplicate Core Component
Component: Morphology & Voice
New name: My Custom Morphology System

This will create an editable copy in your Custom Scripts.
The original [CORE] version will remain available.

[Create Custom Version] [Cancel]
```

### Code Implementation: Unified Script System

```python
class ScriptTier(Enum):
    """Script tiers with different permissions."""
    CORE = "core"              # Built-in, read-only, always available
    OFFICIAL = "official"      # Official presets, read-only, optional
    COMMUNITY = "community"    # Marketplace, read-only until purchased
    CUSTOM = "custom"          # User-created, fully editable

class NoodlingScript(ABC):
    """Base class for ALL scripts (including core components)."""

    # Metadata
    script_tier: ScriptTier = ScriptTier.CUSTOM
    script_version: str = "1.0.0"
    author: str = "unknown"

    # Core components have special flag
    is_core_component: bool = False

    @property
    def is_editable(self) -> bool:
        """Can this script be edited by user?"""
        return self.script_tier == ScriptTier.CUSTOM

    @property
    def is_removable(self) -> bool:
        """Can this script be removed from agent?"""
        # Core components can be disabled but not removed
        return not self.is_core_component
```

**When agent initializes**:
```python
def initialize_components(agent: NoodlingAgent):
    """Initialize agent components from recipe and defaults."""

    # ALWAYS load core components (even if not in recipe)
    for script_path in get_core_scripts():
        script = load_script(script_path)
        agent.add_component(script)

    # Load additional components from recipe
    for component_config in agent.recipe.get('additional_components', []):
        script = load_script(component_config['script_path'])
        script.configure(component_config['parameters'])
        agent.add_component(script)
```

### Benefits of Unification

**Transparency**:
- Users can inspect core component code
- "How does Episodic Memory work?" → Read the script
- Educational value

**Extensibility**:
- Users can improve core components
- Share improvements via marketplace
- Community-driven enhancement

**Simplicity**:
- One API for everything
- No special-case code
- Easier to maintain

**Marketplace**:
- Users can sell "Enhanced Memory v2.0" ($4.99)
- Competition improves quality
- Revenue for creators

---

## Complete Asset Type Registry

### All Asset Types (Current + Proposed)

| Asset Type | Extension | Purpose | Marketplace Value |
|------------|-----------|---------|------------------|
| **Morphology** | `.noodlingmorph.yaml` | Physical form, capabilities, behaviors | HIGH |
| **Voice** | `.noodlingvoice.yaml` | Speech pattern transformations | MEDIUM |
| **Script** | `.noodlingscript.py` | Behaviors, components, logic | VERY HIGH |
| **AnimationRig** | `.noodlinganim.yaml` | Affect → visual animation mapping | EXTREMELY HIGH |
| **Ensemble** | `.noodlingensemble.json` | Multi-agent configurations | HIGH |
| **Personality** | `.noodlingpersonality.yaml` | Trait presets | LOW |
| **References** | `.noodlingrefs/` | Multimodal context images | MEDIUM |
| **VoiceAudio** | `.noodlingaudio/` | TTS voice samples | VERY HIGH |

### Asset Bundles (Marketplace Strategy)

**Complete Character Pack** ($19.99):
- Morphology (e.g., dragon.noodlingmorph)
- Voice (draconic_speech.noodlingvoice)
- AnimationRig (dragon_expressive.noodlinganim)
- Personality preset (ancient_wise.noodlingpersonality)
- Reference images (dragon_concepts.noodlingrefs)
- Behavior script (dragon_hoard.noodlingscript.py)

**Value proposition**: Everything needed for professional dragon character, ready to use.

---

## Implementation Priority (Updated)

### Phase 1: Foundation (This Week)

**Critical path items**:
1. ✅ COMPONENTS_REFERENCE.md spec - DONE
2. [ ] Create `NoodlingScript` base class
3. [ ] Create `@expose_parameter` decorator
4. [ ] Implement script loading system
5. [ ] Convert one core component to script (proof of concept)

### Phase 2: Asset System (Week 2)

1. [ ] Implement `AssetManager`
2. [ ] Create morphology/voice asset schemas (Pydantic)
3. [ ] Create 5 preset morphologies + voices
4. [ ] Implement M&V component v2 (asset-based)
5. [ ] Migrate Phi and SERVNAK to use assets

### Phase 3: Script Components (Week 3)

1. [ ] Implement `ScriptComponent` wrapper
2. [ ] Convert ALL core components to scripts (Tier: CORE)
3. [ ] Inspector UI: Show script tier badges
4. [ ] Implement "Duplicate to Customize" feature
5. [ ] Create Physical Itches script (example)

### Phase 4: Advanced Features (Week 4)

1. [ ] Episodic Memory component
2. [ ] Voice post-processing effects
3. [ ] Animation rig system (basic)
4. [ ] Asset drag-and-drop in Inspector

---

## Questions for Discussion

## Voice Asset Workflow Enhancements

### Creating/Modifying Voices

**Scenario 1: Modify Preset Voice**

User edits "Meow Translator" voice in Inspector:
```
┌────────────────────────────────────────┐
│ Voice: Meow Translator          [Edit] │
└────────────────────────────────────────┘
```

Clicks [Edit] → Opens editor:
```
┌────────────────────────────────────────────────┐
│ Voice Editor: Meow Translator          [Save] │
├────────────────────────────────────────────────┤
│ Voice Prompt Template:                         │
│ ┌────────────────────────────────────────────┐ │
│ │ Translate this thought into cat            │ │
│ │ communication.                             │ │
│ │                                            │ │
│ │ Original: "{text}"                         │ │
│ │                                            │ │
│ │ Use: meow, purr, hiss, chirp              │ │
│ │ Include body language                     │ │
│ │                                            │ │
│ │ [USER ADDS]: Also include tail swishes!   │ │ ← Modified!
│ └────────────────────────────────────────────┘ │
│                                                │
│ ⚠ You modified a preset voice.                │
│   Save as:                                     │
│   ⦿ New custom voice: [Meow Translator+    ]  │
│   ○ Overwrite preset (not recommended)        │
│                                                │
│ [Save to Custom Voices] [Cancel]               │
└────────────────────────────────────────────────┘
```

Saves to: `assets/Voices/Custom/meow_translator_plus.noodlingvoice.yaml`

**Scenario 2: Create New Voice from Scratch**

Assets Panel → Right-click "Voices" → "Create New Voice":
```
┌────────────────────────────────────────────────┐
│ Create New Voice                               │
├────────────────────────────────────────────────┤
│ Name: Shakespearean Cat                        │
│ Category: [literary_animal ▼]                  │
│                                                │
│ Base Template: [None ▼]                        │
│   Options:                                     │
│   - None (blank)                               │
│   - Copy from: Meow Translator                 │
│   - Copy from: Formal Victorian                │
│                                                │
│ Voice Prompt:                                  │
│ ┌────────────────────────────────────────────┐ │
│ │ Translate into Shakespearean feline:       │ │
│ │                                            │ │
│ │ Combine:                                   │ │
│ │ - Cat vocalizations (meow, purr)          │ │
│ │ - Elizabethan English phrasing            │ │
│ │ - Iambic pentameter when excited          │ │
│ │                                            │ │
│ │ Example:                                   │ │
│ │ "I'm hungry" →                             │ │
│ │ "*meows with dramatic flourish, as if     │ │
│ │  to proclaim 'What ho! Mine belly doth    │ │
│ │  protest most grievously!'*"              │ │
│ └────────────────────────────────────────────┘ │
│                                                │
│ [Create Voice] [Preview]                       │
└────────────────────────────────────────────────┘
```

**3. Export Voice for Sharing**

Inspector → M&V Component → Voice dropdown → Right-click voice → "Export":
```
Save Voice Asset
Name: shakespearean_cat.noodlingvoice.yaml
Location: ~/Downloads/

[Export] [Cancel]
```

User can share on marketplace or with friends.

---

## Animation Rigs: Affect → Visual Behavior

### The Vision

**Problem**: USD export includes affect data, but animators need to map it to blend shapes/bones.

**Solution**: Animation Rig assets define the mapping.

### Animation Rig Asset Schema

**File**: `assets/AnimationRigs/cat_expressive.noodlinganim.yaml`

```yaml
# Metadata
rig_id: cat_expressive
name: Expressive Cat Rig
category: animal_quadruped
author: noodlings_official
version: 1.0.0

# Compatible morphologies
compatible_morphologies:
  - cat_domestic
  - cat_tiger
  - cat_lion

# Bone structure (for USD skeleton export)
bones:
  - spine_base
  - spine_mid
  - spine_tip (tail base)
  - tail_1
  - tail_2
  - tail_tip
  - head
  - ear_left
  - ear_right
  - leg_front_left
  - leg_front_right
  - leg_back_left
  - leg_back_right

# Affect → Blend Shape mapping
blend_shapes:
  # Valence mapping
  valence:
    positive:  # Valence > 0 (happy)
      - {shape: "eyes_relaxed", weight: "valence * 0.8"}
      - {shape: "mouth_slight_open", weight: "valence * 0.3"}
      - {shape: "whiskers_forward", weight: "valence * 0.6"}

    negative:  # Valence < 0 (sad/angry)
      - {shape: "eyes_narrowed", weight: "abs(valence) * 0.7"}
      - {shape: "ears_flatten", weight: "abs(valence) * 0.9"}
      - {shape: "whiskers_back", weight: "abs(valence) * 0.5"}

  # Arousal mapping
  arousal:
    high:  # Arousal > 0.7 (excited/agitated)
      - {shape: "pupils_dilated", weight: "arousal * 1.0"}
      - {shape: "fur_raised", weight: "(arousal - 0.7) * 3.0"}

    low:  # Arousal < 0.3 (calm/sleepy)
      - {shape: "eyes_half_closed", weight: "(0.3 - arousal) * 3.0"}
      - {shape: "body_relaxed", weight: "(0.3 - arousal) * 2.0"}

  # Fear mapping
  fear:
    high:  # Fear > 0.6
      - {shape: "ears_flat_back", weight: "fear * 1.0"}
      - {shape: "body_crouch", weight: "fear * 0.8"}
      - {shape: "tail_tuck", weight: "fear * 0.9"}

# Affect → Bone Animation mapping
procedural_animation:
  tail:
    # Tail movement based on emotion
    - condition: "valence > 0.5 and arousal > 0.6"
      animation: tail_swish_happy
      speed: "arousal * 2.0"

    - condition: "fear > 0.7"
      animation: tail_tuck
      intensity: "fear"

    - condition: "valence < -0.3"
      animation: tail_lash_annoyed
      speed: "abs(valence) * 1.5"

  ears:
    - condition: "curiosity > 0.7"  # From morphology.instincts
      animation: ears_perk_forward

    - condition: "fear > 0.5"
      animation: ears_flatten

  body:
    - condition: "arousal < 0.2"
      animation: idle_sleeping
      blend: "1.0 - arousal * 5"

# USD export settings
usd_export:
  skeleton_path: "/Skeleton"
  blend_shape_path: "/BlendShapes"
  animation_fps: 24
  bake_simulation: true
```

**Usage**:
1. Conversation happens in noodleMUSH
2. Affect changes (valence +0.8, arousal +0.7, fear 0.0)
3. Timeline profiler records affect curve
4. Export to USD
5. Animation rig applies mappings:
   - Eyes relaxed (valence * 0.8 = 0.64)
   - Pupils dilated (arousal * 1.0 = 0.7)
   - Tail swish_happy animation (speed = arousal * 2 = 1.4x)
6. Import USD into Maya/Blender
7. Cat character is automatically animated based on emotional state!

**Marketplace potential**: EXTREMELY HIGH
- "Expressive Character Rigs Bundle" (human, cat, robot) - $14.99
- Professional rigs from animators - $29.99+
- Game-ready rigs with LODs - $49.99

---

## Questions for Discussion

### 1. Component vs. Built-In Features

**When should something be a component vs. core architecture?**

**Components** (modular, optional):
- Morphology & Voice (agents can have no voice)
- Episodic Memory (agents can be memoryless)
- Social Expectations (agents can be asocial)

**Core** (always present):
- Temporal model (fast/medium/slow layers)
- Affect processing (5-D vector)
- Surprise computation (predictive processing)

**Rule of thumb**: If it can be disabled without breaking the agent → component. If it's fundamental to being a Noodling → core.

### 2. Asset Format: YAML vs. Python

**YAML** (current choice):
- Human-readable
- Easy to edit
- Safe (no code execution)
- Version control friendly

**Python** (alternative):
- More expressive (functions, logic)
- Faster to load
- Dangerous (arbitrary code execution)

**Hybrid** (proposed):
- Morphologies: YAML (data-focused)
- Behavior scripts: Python (logic-focused, sandboxed)
- Voices: YAML (prompt templates)

### 3. Asset Validation and Safety

**Problem**: User-created assets could:
- Contain offensive content
- Break agent behavior
- Cause crashes

**Solutions**:
1. **Schema validation** - Reject invalid YAML
2. **Content moderation** - Scan prompts for inappropriate content (marketplace only)
3. **Sandboxing** - Limit what assets can do
4. **Community rating** - Users flag problematic assets

---

## Summary: Implementation Priority

**Immediate (this session if energized)**:
1. Create `COMPONENTS_REFERENCE.md` - ✓ DONE
2. Create asset directory structure
3. Implement basic morphology/voice assets for existing agents (Phi, SERVNAK)

**Short-term (next session)**:
1. Implement AssetManager
2. Implement MorphologyVoiceComponent v2 with asset loading
3. Test with existing agents

**Medium-term (this week)**:
1. Implement EpisodicMemory component
2. Add Inspector UI for both new components
3. Create preset morphology/voice library

**Long-term (this month)**:
1. Component marketplace infrastructure
2. Asset browser in NoodleSTUDIO
3. Community asset sharing

---

## Conclusion

The component system is **well-architected** but needs **scalability fundamentals** before explosion:

**Current strengths**:
- Clean ABC base class
- Hot-reload support
- API integration
- Inspector integration

**Needed additions**:
- ComponentRegistry (dynamic loading)
- ComponentContext (inter-component communication)
- AssetManager (asset-based configuration)
- Lifecycle hooks (proper initialization/cleanup)
- Event bus (decoupled communication)

**Once these fundamentals exist**, the platform can support:
- Unlimited third-party components
- Asset marketplace
- Community contributions
- Professional use cases

The morphology/voice asset idea is **excellent** and aligns perfectly with the Unity vision. Combined with episodic memory, it creates a robust foundation for the component ecosystem.

*Straightens collar with satisfaction at systematic design*

Live long and prosper. 🖖

---

**Status**: Design specification complete
**Next**: Implement foundation, then M&V and Memory components
**Timeline**: 3-4 weeks for complete implementation
