# Cognitive Manifold & Transistor Architecture

**Authors:** Lieutenant Caitlyn + Commander Spock
**Date:** November 22, 2025
**Status:** Specification - Ready for Implementation
**Inspiration:** Signal processing meets consciousness

---

## Core Concept

**Cognitive Transistors** filter/amplify thoughts based on belief systems.
**Cognitive Manifolds** integrate multiple transistor outputs into coherent thought.

**Metaphor:** Electronics signal processing
- **Transistor** = Amplifies/switches signal based on input (voltage → current)
- **Manifold** = Combines multiple signals into unified output (mixer)

**Application:** Consciousness processing
- **Cognitive Transistor** = Colors thoughts based on beliefs (culture, personality, mood)
- **Cognitive Manifold** = Synthesizes all cognitive signals into final output

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PERCEPTION INPUT                          │
│  "Someone threw a rock at the can"                          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│             COGNITIVE TRANSISTORS (Parallel)                 │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Cultural Transistor                             │      │
│  │  Input: "rock thrown at can"                     │      │
│  │  Belief: "Violence is wrong"                     │      │
│  │  Output: "This is aggressive behavior"           │      │
│  │  Salience: 0.8                                   │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Personality Transistor                          │      │
│  │  Input: "rock thrown at can"                     │      │
│  │  Trait: Curiosity = 0.9                          │      │
│  │  Output: "I wonder about the trajectory physics" │      │
│  │  Salience: 0.6                                   │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Mood Transistor                                 │      │
│  │  Input: "rock thrown at can"                     │      │
│  │  Mood: Anxious (fear = 0.7)                      │      │
│  │  Output: "What if the rock had hit me?"          │      │
│  │  Salience: 0.4                                   │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Memory Transistor                               │      │
│  │  Input: "rock thrown at can"                     │      │
│  │  Memory: "Last time: glass shattered"            │      │
│  │  Output: "This could end badly like before"      │      │
│  │  Salience: 0.5                                   │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  COGNITIVE MANIFOLD                          │
│  ┌────────────────────────────────────────────────┐         │
│  │  LLM-Powered Signal Integration                │         │
│  │                                                 │         │
│  │  Inputs (weighted by salience):                │         │
│  │  • Cultural: "aggressive" (0.8)                │         │
│  │  • Personality: "curious about physics" (0.6)  │         │
│  │  • Mood: "worried about safety" (0.4)          │         │
│  │  • Memory: "could end badly" (0.5)             │         │
│  │                                                 │         │
│  │  Synthesis (LLM blend):                        │         │
│  │  "That was aggressive, though I'm curious      │         │
│  │   about the physics. Hope it doesn't end       │         │
│  │   badly like last time."                       │         │
│  │                                                 │         │
│  │  Final Coherent Thought                        │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   POST PROCESSOR                             │
│  Formats output, applies character voice, adds emojis, etc. │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      RENDERER                                │
│  "THAT WAS AGGRESSIVE BEHAVIOR! THOUGH I AM CURIOUS ABOUT   │
│   THE KINETIC TRAJECTORY. 73.2% PROBABILITY IT ENDS BADLY   │
│   LIKE PREVIOUS GLASS INCIDENT, SISTER!"                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Hierarchy

### Base: CognitiveTransistor

```python
class CognitiveTransistor(NoodleComponent):
    """
    Base class for cognitive transistors.

    A transistor receives input and colors it based on internal state
    (beliefs, personality, mood, etc.). Outputs transformed thought
    with salience weight.
    """

    def __init__(self):
        super().__init__()
        self.salience = 0.5  # Default importance weight (0.0 to 1.0)
        self.enabled = True  # Can be turned off

    def process(self, input_text: str, context: Dict) -> TransistorOutput:
        """
        Process input through cognitive filter.

        Args:
            input_text: Raw perception/thought
            context: Additional context (phenomenal state, memories, etc.)

        Returns:
            TransistorOutput with transformed text and salience
        """
        raise NotImplementedError("Subclasses must implement process()")

    def get_transistor_type(self) -> str:
        """Return transistor type identifier."""
        return self.__class__.__name__

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'type': self.get_transistor_type(),
            'salience': self.salience,
            'enabled': self.enabled
        }


@dataclass
class TransistorOutput:
    """Output from a cognitive transistor."""
    transformed_text: str  # Colored/filtered thought
    salience: float        # Importance weight (0.0 to 1.0)
    metadata: Dict         # Additional info (emotions, reasons, etc.)
```

### Base: CognitiveManifold

```python
class CognitiveManifold(NoodleComponent):
    """
    Cognitive Manifold - Integrates multiple transistor outputs.

    Uses LLM to synthesize coherent thought from multiple colored
    perspectives. Weights inputs by salience.
    """

    def __init__(self):
        super().__init__()
        self.transistors = []  # List of connected CognitiveTransistor components
        self.blending_strategy = "llm_weighted"  # or "simple_concat", "priority"

    def register_transistor(self, transistor: CognitiveTransistor):
        """Register a transistor to integrate."""
        self.transistors.append(transistor)

    def integrate(self, input_text: str, context: Dict) -> str:
        """
        Integrate all transistor outputs into coherent thought.

        Args:
            input_text: Raw perception/thought
            context: Additional context

        Returns:
            Synthesized coherent thought
        """
        # Collect outputs from all transistors
        outputs = []
        for transistor in self.transistors:
            if transistor.enabled:
                output = transistor.process(input_text, context)
                outputs.append(output)

        # Synthesize using configured strategy
        if self.blending_strategy == "llm_weighted":
            return self._llm_weighted_blend(outputs, context)
        elif self.blending_strategy == "simple_concat":
            return self._simple_concatenation(outputs)
        elif self.blending_strategy == "priority":
            return self._priority_blend(outputs)
        else:
            return input_text  # Fallback

    def _llm_weighted_blend(self, outputs: List[TransistorOutput], context: Dict) -> str:
        """
        Use LLM to blend multiple perspectives into coherent thought.

        Provides all transistor outputs with salience weights to LLM.
        LLM synthesizes single coherent response.
        """
        # Build prompt for LLM
        prompt = "You are synthesizing multiple cognitive perspectives into a single coherent thought.\n\n"
        prompt += "Perspectives (weighted by salience):\n"

        for i, output in enumerate(outputs, 1):
            prompt += f"{i}. [{output.salience:.2f}] {output.transformed_text}\n"

        prompt += "\nSynthesize these into ONE coherent thought that integrates all perspectives "
        prompt += "proportionally to their salience. Higher salience = more influence.\n\n"
        prompt += "Synthesized thought:"

        # Call LLM (fast model for real-time processing)
        response = call_llm(prompt, model="qwen/qwen3-4b-2507", max_tokens=150)

        return response.strip()

    def _simple_concatenation(self, outputs: List[TransistorOutput]) -> str:
        """Simple concatenation of outputs (no LLM)."""
        # Sort by salience (highest first)
        sorted_outputs = sorted(outputs, key=lambda x: x.salience, reverse=True)

        # Concatenate with periods
        parts = [o.transformed_text for o in sorted_outputs if o.salience > 0.2]
        return ". ".join(parts) + "."

    def _priority_blend(self, outputs: List[TransistorOutput]) -> str:
        """Use only the highest salience output."""
        if not outputs:
            return ""
        highest = max(outputs, key=lambda x: x.salience)
        return highest.transformed_text

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'type': 'CognitiveManifold',
            'blending_strategy': self.blending_strategy,
            'transistor_count': len(self.transistors)
        }
```

---

## Concrete Transistor Types

### 1. CulturalTransistor

```python
class CulturalTransistor(CognitiveTransistor):
    """
    Colors thoughts based on cultural beliefs and values.

    Example beliefs:
    - "Violence is wrong"
    - "Honesty is paramount"
    - "Family comes first"
    - "Elders must be respected"
    """

    def __init__(self, beliefs: List[str] = None):
        super().__init__()
        self.beliefs = beliefs or []
        self.salience = 0.8  # Cultural beliefs have high influence

    def process(self, input_text: str, context: Dict) -> TransistorOutput:
        """Filter input through cultural lens."""
        # Use LLM to apply cultural filter
        prompt = f"Cultural beliefs: {', '.join(self.beliefs)}\n\n"
        prompt += f"Event: {input_text}\n\n"
        prompt += "How would someone with these cultural beliefs interpret this event? "
        prompt += "Response (one sentence):"

        response = call_llm(prompt, model="qwen/qwen3-4b-2507", max_tokens=50)

        return TransistorOutput(
            transformed_text=response.strip(),
            salience=self.salience,
            metadata={'beliefs': self.beliefs}
        )

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d['beliefs'] = self.beliefs
        return d
```

### 2. PersonalityTransistor

```python
class PersonalityTransistor(CognitiveTransistor):
    """
    Colors thoughts based on personality traits.

    Uses Big Five or custom trait system.
    """

    def __init__(self, traits: Dict[str, float] = None):
        super().__init__()
        self.traits = traits or {
            'curiosity': 0.5,
            'impulsivity': 0.5,
            'emotional_volatility': 0.5,
            'extraversion': 0.5,
            'vanity': 0.5
        }
        self.salience = 0.6  # Moderate influence

    def process(self, input_text: str, context: Dict) -> TransistorOutput:
        """Filter input through personality lens."""
        # Find dominant traits
        dominant = {k: v for k, v in self.traits.items() if v > 0.6}

        if not dominant:
            # No strong traits - minimal coloring
            return TransistorOutput(
                transformed_text=input_text,
                salience=0.3,
                metadata={}
            )

        # Use LLM to apply personality filter
        trait_desc = ", ".join([f"{k}={v:.1f}" for k, v in dominant.items()])
        prompt = f"Personality traits: {trait_desc}\n\n"
        prompt += f"Event: {input_text}\n\n"
        prompt += "How would someone with these personality traits react? Response (one sentence):"

        response = call_llm(prompt, model="qwen/qwen3-4b-2507", max_tokens=50)

        return TransistorOutput(
            transformed_text=response.strip(),
            salience=self.salience,
            metadata={'dominant_traits': dominant}
        )
```

### 3. MoodTransistor

```python
class MoodTransistor(CognitiveTransistor):
    """
    Colors thoughts based on current emotional state.

    Integrates with Noodling phenomenal state (affect vector).
    """

    def __init__(self):
        super().__init__()
        self.current_affect = None  # Set from phenomenal state
        self.salience = 0.5  # Moderate influence

    def process(self, input_text: str, context: Dict) -> TransistorOutput:
        """Filter input through emotional lens."""
        # Get current affect from context
        affect = context.get('affect', [0.0, 0.0, 0.0, 0.0, 0.0])
        valence, arousal, fear, sorrow, boredom = affect

        # Determine mood
        if fear > 0.6:
            mood = "anxious and worried"
        elif sorrow > 0.6:
            mood = "sad and melancholic"
        elif valence > 0.5 and arousal > 0.5:
            mood = "excited and happy"
        elif arousal < 0.3 and boredom > 0.5:
            mood = "bored and disinterested"
        else:
            mood = "neutral"

        # Apply mood filter
        prompt = f"Current mood: {mood}\n\n"
        prompt += f"Event: {input_text}\n\n"
        prompt += f"How would someone feeling {mood} interpret this? Response (one sentence):"

        response = call_llm(prompt, model="qwen/qwen3-4b-2507", max_tokens=50)

        return TransistorOutput(
            transformed_text=response.strip(),
            salience=self.salience,
            metadata={'mood': mood, 'affect': affect}
        )
```

### 4. MemoryTransistor

```python
class MemoryTransistor(CognitiveTransistor):
    """
    Colors thoughts based on past experiences.

    Retrieves relevant memories and uses them to contextualize input.
    """

    def __init__(self):
        super().__init__()
        self.salience = 0.4  # Lower influence (unless strong memory)

    def process(self, input_text: str, context: Dict) -> TransistorOutput:
        """Filter input through memory lens."""
        # Extract keywords from input
        keywords = extract_keywords(input_text)

        # Retrieve relevant memories (semantic search)
        memories = context.get('memory_system')
        if not memories:
            return TransistorOutput(
                transformed_text=input_text,
                salience=0.1,
                metadata={}
            )

        relevant = memories.search(keywords, limit=3)

        if not relevant:
            return TransistorOutput(
                transformed_text=input_text,
                salience=0.1,
                metadata={}
            )

        # Build memory context
        memory_text = "\n".join([f"- {m['text']}" for m in relevant])

        # Apply memory filter
        prompt = f"Past experiences:\n{memory_text}\n\n"
        prompt += f"Current event: {input_text}\n\n"
        prompt += "How do these past experiences color this current event? Response (one sentence):"

        response = call_llm(prompt, model="qwen/qwen3-4b-2507", max_tokens=50)

        # Higher salience if strong memories
        salience = min(0.8, 0.4 + sum([m.get('importance', 0) for m in relevant]) / 3)

        return TransistorOutput(
            transformed_text=response.strip(),
            salience=salience,
            metadata={'memories': [m['text'] for m in relevant]}
        )
```

### 5. SocialExpectationTransistor

```python
class SocialExpectationTransistor(CognitiveTransistor):
    """
    Colors thoughts based on social norms and expectations.

    "What would others think?"
    "Is this socially appropriate?"
    """

    def __init__(self, social_rules: List[str] = None):
        super().__init__()
        self.social_rules = social_rules or [
            "Be polite to others",
            "Don't interrupt",
            "Show gratitude when helped"
        ]
        self.salience = 0.6

    def process(self, input_text: str, context: Dict) -> TransistorOutput:
        """Filter through social norms lens."""
        prompt = f"Social rules: {', '.join(self.social_rules)}\n\n"
        prompt += f"Event: {input_text}\n\n"
        prompt += "How does this relate to social expectations? Response (one sentence):"

        response = call_llm(prompt, model="qwen/qwen3-4b-2507", max_tokens=50)

        return TransistorOutput(
            transformed_text=response.strip(),
            salience=self.salience,
            metadata={'rules': self.social_rules}
        )
```

---

## Dependency Resolution System

### Auto-Dependency Prompt

When adding a CognitiveTransistor to a prim:

```
┌─────────────────────────────────────────────────────────┐
│  Component Dependency                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  "CulturalTransistor" requires:                         │
│    • CognitiveManifold                                  │
│                                                          │
│  Without a Cognitive Manifold, this transistor's output │
│  will not be integrated into the agent's thoughts.      │
│                                                          │
│  Add missing dependencies automatically?                │
│                                                          │
│  ┌─────┐  ┌────┐  ┌──────────────────┐                │
│  │ Yes │  │ No │  │ Don't Ask Again  │                │
│  └─────┘  └────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Implementation

```python
class ComponentDependency:
    """Tracks component dependencies."""

    DEPENDENCIES = {
        'CognitiveTransistor': ['CognitiveManifold'],
        'CulturalTransistor': ['CognitiveManifold'],
        'PersonalityTransistor': ['CognitiveManifold'],
        'MoodTransistor': ['CognitiveManifold'],
        'MemoryTransistor': ['CognitiveManifold'],
        'SocialExpectationTransistor': ['CognitiveManifold']
    }

    @staticmethod
    def check_dependencies(component_type: str, existing_components: List[str]) -> List[str]:
        """
        Check if component has missing dependencies.

        Args:
            component_type: Type being added
            existing_components: List of existing component types

        Returns:
            List of missing dependency types
        """
        required = ComponentDependency.DEPENDENCIES.get(component_type, [])
        missing = [dep for dep in required if dep not in existing_components]
        return missing

    @staticmethod
    def prompt_add_dependencies(missing: List[str], user_prefs: Dict) -> bool:
        """
        Prompt user to add missing dependencies.

        Args:
            missing: List of missing component types
            user_prefs: User preferences (stores "don't ask again")

        Returns:
            True if user wants to add dependencies
        """
        # Check if user said "don't ask again"
        if user_prefs.get('auto_add_dependencies', False):
            return True

        # Show dialog (Qt implementation)
        from PyQt6.QtWidgets import QMessageBox, QCheckBox

        msg = QMessageBox()
        msg.setWindowTitle("Component Dependency")
        msg.setText(f"This component requires:\n  • {', '.join(missing)}\n\n"
                   "Add missing dependencies automatically?")

        dont_ask = QCheckBox("Don't ask again (always add)")
        msg.setCheckBox(dont_ask)

        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        reply = msg.exec()

        # Save preference if checked
        if dont_ask.isChecked():
            user_prefs['auto_add_dependencies'] = (reply == QMessageBox.StandardButton.Yes)

        return reply == QMessageBox.StandardButton.Yes
```

---

## Asset Store Integration

### Package Format

```json
{
  "name": "Cultural Belief Systems Pack",
  "version": "1.0.0",
  "author": "NoodleForge Studios",
  "price": "$4.99",
  "type": "cognitive_transistor_pack",
  "description": "Pre-built cultural transistors for diverse belief systems",
  "components": [
    {
      "type": "CulturalTransistor",
      "name": "Japanese Honor Culture",
      "beliefs": [
        "Honor above all",
        "Shame must be avoided",
        "Group harmony is essential",
        "Duty to ancestors"
      ],
      "salience": 0.9
    },
    {
      "type": "CulturalTransistor",
      "name": "Stoic Philosophy",
      "beliefs": [
        "Control only what you can control",
        "Emotions should be mastered",
        "Virtue is the only good",
        "Accept fate with equanimity"
      ],
      "salience": 0.8
    }
  ],
  "dependencies": ["CognitiveManifold"],
  "tags": ["culture", "beliefs", "philosophy", "transistor"]
}
```

### Asset Store Categories

**Cognitive Transistors:**
- Cultural Belief Packs
- Personality Archetypes
- Mood Presets
- Social Norm Collections
- Professional Mindsets (scientist, artist, warrior, etc.)

**Cognitive Manifolds:**
- Standard Manifold (LLM-weighted blending)
- Priority Manifold (highest salience wins)
- Democratic Manifold (equal weighting)
- Emotional Manifold (affect-weighted blending)

**Complete Cognitive Stacks:**
- "The Stoic" (cultural + personality + mood)
- "The Warrior" (combat mindset + cultural honor)
- "The Scholar" (curiosity + memory + analysis)
- "The Empath" (social + emotional + cultural)

---

## Integration with Existing Systems

### With Noodling Phenomenal State

```python
# In agent_bridge.py perceive_event()

# 1. Get perception
event_text = "Rock struck can with CLANG"

# 2. Pass through cognitive pipeline
if agent.has_component('CognitiveManifold'):
    manifold = agent.get_component('CognitiveManifold')

    context = {
        'affect': agent.phenomenal_state.affect_vector,
        'memory_system': agent.conversation_context,
        'surprise': agent.last_surprise
    }

    # Integrate all transistors
    colored_thought = manifold.integrate(event_text, context)
else:
    # No manifold - use raw perception
    colored_thought = event_text

# 3. Continue with response generation
response = agent.generate_response(colored_thought)
```

### With Post Processor & Renderer

```
Perception → Transistors → Manifold → Post Processor → Renderer

Post Processor:
- Apply character voice (SERVNAK caps, Phi meows)
- Add formatting
- Insert metadata

Renderer:
- Output to chat
- Log to memory
- Trigger animations
```

---

## Example Scenarios

### Scenario 1: SERVNAK with Cultural Transistor

**Setup:**
- SERVNAK has CulturalTransistor: "Logic is supreme, emotions are inefficient"
- Also has PersonalityTransistor: curiosity=0.9, emotional_volatility=0.2
- Manifold: LLM-weighted blending

**Perception:** "Phi is crying because her toy broke"

**Processing:**

1. **CulturalTransistor (salience: 0.8)**
   - Output: "Emotional response is inefficient. The toy can be repaired logically."

2. **PersonalityTransistor (salience: 0.6)**
   - Output: "I'm curious why the material failed. What was the fracture point?"

3. **MoodTransistor (salience: 0.4)**
   - Current affect: neutral
   - Output: "This situation requires analysis."

4. **Manifold Integration:**
   - Input: Cultural (0.8) + Personality (0.6) + Mood (0.4)
   - LLM synthesizes: "EMOTIONAL RESPONSE INEFFICIENT — BUT I AM CURIOUS ABOUT THE FRACTURE MECHANICS. SISTER, SHALL WE ANALYZE THE FAILURE MODE?"

5. **Post Processor:** Adds SERVNAK formatting
6. **Renderer:** Outputs to chat

**Result:** SERVNAK responds logically but with curiosity, weighted by cultural belief in logic.

---

### Scenario 2: Phi with Multiple Transistors

**Setup:**
- Phi has CulturalTransistor: "Cuteness is power"
- PersonalityTransistor: extraversion=0.9, impulsivity=0.8
- MoodTransistor: current affect = high arousal, positive valence
- Manifold: LLM-weighted

**Perception:** "Third Prim Ever is glowing"

**Processing:**

1. **CulturalTransistor (0.7):** "That's adorable! Must touch!"
2. **PersonalityTransistor (0.6):** "Ooh shiny! Want it now!"
3. **MoodTransistor (0.5):** "I'm so excited about everything!"

4. **Manifold:** "✨ SHINY ADORABLE THING!! *pounces immediately*"
5. **Post Processor:** Adds cat actions
6. **Renderer:** "*Phi's eyes go wide and she pounces at the glowing Third Prim*"

---

## Implementation Plan

**Phase 1: Core Architecture**
1. Base classes (CognitiveTransistor, CognitiveManifold)
2. TransistorOutput dataclass
3. Dependency resolution system

**Phase 2: Concrete Transistors**
1. CulturalTransistor
2. PersonalityTransistor
3. MoodTransistor
4. MemoryTransistor
5. SocialExpectationTransistor

**Phase 3: Integration**
1. Add to agent_bridge perception pipeline
2. Connect with phenomenal state
3. Connect with memory system
4. Post-processor integration

**Phase 4: UI & Asset Store**
1. Component Inspector panel (show transistors)
2. Dependency prompt dialog
3. Asset store packaging format
4. Marketplace integration

---

## Theoretical Significance

**Modular Cognition:**
- Cognitive processing as signal flow
- Beliefs as filters/amplifiers
- Integration as synthesis

**Emergent Coherence:**
- Multiple perspectives create nuanced thought
- Salience weighting = attention mechanism
- LLM blending = integration layer

**Extensibility:**
- New transistors = new cognitive dimensions
- Asset store = community-created cognition
- Mix-and-match belief systems

**This is consciousness as modular architecture.**

---

## Summary

**Cognitive Transistors** = Belief-based thought filters
**Cognitive Manifolds** = Integration layers
**Asset Store** = Downloadable belief systems
**Dependency Resolution** = Automatic component management

**Architecture Pattern:**
```
Perception → [Transistor₁, Transistor₂, ..., Transistorₙ] → Manifold → Output
```

**Result:** Rich, nuanced, belief-colored consciousness with modular cognitive architecture.

**Fascinating.**

---

**End of Specification**

*Ready for implementation, Lieutenant.*
