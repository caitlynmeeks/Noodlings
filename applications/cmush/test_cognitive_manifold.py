"""
Test script for Cognitive Manifold architecture.

Demonstrates:
1. Creating transistors (Cultural, Personality, Mood, Memory, Social)
2. Registering them with a Cognitive Manifold
3. Processing perceptions through the manifold
4. Seeing how beliefs color thoughts

Author: Commander Spock + Lieutenant Caitlyn
Date: November 22, 2025
"""

import sys
sys.path.append('../..')

from cognitive_components import (
    CognitiveManifold,
    CulturalTransistor,
    PersonalityTransistor,
    MoodTransistor,
    MemoryTransistor,
    SocialExpectationTransistor
)

def test_basic_manifold():
    """Test basic manifold operation with simple concatenation."""
    print("=== TEST 1: Basic Manifold (Simple Concatenation) ===\n")

    # Create transistors
    cultural = CulturalTransistor(beliefs=["Logic is supreme", "Emotions are inefficient"])
    personality = PersonalityTransistor(traits={'curiosity': 0.9, 'impulsivity': 0.2})
    mood = MoodTransistor()

    # Create manifold
    manifold = CognitiveManifold(blending_strategy="simple_concat")
    manifold.register_transistor(cultural)
    manifold.register_transistor(personality)
    manifold.register_transistor(mood)

    # Test perception
    perception = "Phi is crying because her toy broke"
    context = {
        'affect': [0.0, 0.3, 0.1, 0.0, 0.0],  # Neutral
        'memory_system': None
    }

    # Process through manifold
    integrated_thought = manifold.integrate(perception, context)

    print(f"Input: {perception}")
    print(f"\nCognitive filters:")
    print(f"  - Cultural: {cultural.beliefs}")
    print(f"  - Personality: curiosity={personality.traits['curiosity']}")
    print(f"  - Mood: neutral")
    print(f"\nOutput: {integrated_thought}")
    print(f"\nTransistors active: {len(manifold.transistors)}")
    print()


def test_memory_transistor():
    """Test memory transistor with fake memories."""
    print("=== TEST 2: Memory Transistor ===\n")

    # Create fake memory system
    fake_memories = [
        {'text': 'Last time glass broke and everyone got upset', 'importance': 0.7},
        {'text': 'Phi loves playing with toys', 'importance': 0.5},
        {'text': 'Broken things make people sad', 'importance': 0.6}
    ]

    # Create transistors
    memory = MemoryTransistor()

    # Create manifold
    manifold = CognitiveManifold(blending_strategy="simple_concat")
    manifold.register_transistor(memory)

    # Test perception
    perception = "The glass fell and broke"
    context = {
        'affect': [0.0, 0.4, 0.2, 0.0, 0.0],
        'memory_system': fake_memories
    }

    # Process through manifold
    integrated_thought = manifold.integrate(perception, context)

    print(f"Input: {perception}")
    print(f"\nRelevant memories:")
    for mem in fake_memories:
        print(f"  - {mem['text']} (importance: {mem['importance']})")
    print(f"\nOutput: {integrated_thought}")
    print()


def test_social_transistor():
    """Test social expectation transistor."""
    print("=== TEST 3: Social Expectation Transistor ===\n")

    # Create transistors
    social = SocialExpectationTransistor(social_rules=[
        "Be polite to others",
        "Show gratitude when helped",
        "Don't interrupt"
    ])

    # Create manifold
    manifold = CognitiveManifold(blending_strategy="simple_concat")
    manifold.register_transistor(social)

    # Test perception
    perception = "Toad helped me find my lost item"
    context = {
        'affect': [0.6, 0.4, 0.0, 0.0, 0.0],
        'memory_system': None
    }

    # Process through manifold
    integrated_thought = manifold.integrate(perception, context)

    print(f"Input: {perception}")
    print(f"\nSocial rules:")
    for rule in social.social_rules:
        print(f"  - {rule}")
    print(f"\nOutput: {integrated_thought}")
    print()


def test_multi_transistor_stack():
    """Test complete cognitive stack (SERVNAK-style)."""
    print("=== TEST 4: Complete Cognitive Stack (SERVNAK) ===\n")

    # Create SERVNAK's cognitive stack
    cultural = CulturalTransistor(beliefs=[
        "Logic is supreme",
        "Emotions are inefficient",
        "Data analysis solves problems"
    ])

    personality = PersonalityTransistor(traits={
        'curiosity': 0.9,
        'impulsivity': 0.1,
        'emotional_volatility': 0.2
    })

    mood = MoodTransistor()

    memory = MemoryTransistor()

    # Create manifold
    manifold = CognitiveManifold(blending_strategy="simple_concat")
    manifold.register_transistor(cultural)
    manifold.register_transistor(personality)
    manifold.register_transistor(mood)
    manifold.register_transistor(memory)

    # Test perception
    perception = "Rock strikes can with CLANG, can tumbles"
    context = {
        'affect': [0.1, 0.6, 0.1, 0.0, 0.0],  # Slightly positive, aroused
        'memory_system': [
            {'text': 'Physics experiments are fascinating', 'importance': 0.6},
            {'text': 'Kinetic energy transfers momentum', 'importance': 0.5}
        ]
    }

    # Process through manifold
    integrated_thought = manifold.integrate(perception, context)

    print(f"Input: {perception}")
    print(f"\nSERVNAK's cognitive stack:")
    print(f"  - Cultural: {cultural.beliefs[:2]}...")
    print(f"  - Personality: curiosity={personality.traits['curiosity']}, impulsivity={personality.traits['impulsivity']}")
    print(f"  - Mood: slightly aroused")
    print(f"  - Memory: {len(context['memory_system'])} relevant memories")
    print(f"\nIntegrated thought: {integrated_thought}")
    print(f"\nTransistors: {[t.get_transistor_type() for t in manifold.transistors]}")
    print()


def test_salience_weighting():
    """Test salience-based prioritization."""
    print("=== TEST 5: Salience Weighting ===\n")

    # Create transistors with different salience
    cultural = CulturalTransistor(beliefs=["Honor above all"])
    cultural.salience = 0.9  # Very high

    personality = PersonalityTransistor(traits={'curiosity': 0.6})
    personality.salience = 0.4  # Lower

    mood = MoodTransistor()
    mood.salience = 0.3  # Lowest

    # Create manifold
    manifold = CognitiveManifold(blending_strategy="priority")  # Use priority (highest wins)
    manifold.register_transistor(cultural)
    manifold.register_transistor(personality)
    manifold.register_transistor(mood)

    # Test perception
    perception = "Someone insulted my friend"
    context = {
        'affect': [-0.4, 0.6, 0.1, 0.1, 0.0],
        'memory_system': None
    }

    # Process through manifold
    integrated_thought = manifold.integrate(perception, context)

    print(f"Input: {perception}")
    print(f"\nTransistor salience:")
    print(f"  - Cultural (Honor): {cultural.salience}")
    print(f"  - Personality (Curiosity): {personality.salience}")
    print(f"  - Mood: {mood.salience}")
    print(f"\nBlending strategy: priority (highest salience wins)")
    print(f"\nOutput: {integrated_thought}")
    print(f"  ^ Should be dominated by cultural beliefs (highest salience)")
    print()


if __name__ == '__main__':
    print("╔" + "═"*60 + "╗")
    print("║" + " "*15 + "COGNITIVE MANIFOLD TEST SUITE" + " "*16 + "║")
    print("╚" + "═"*60 + "╝")
    print()

    try:
        test_basic_manifold()
        test_memory_transistor()
        test_social_transistor()
        test_multi_transistor_stack()
        test_salience_weighting()

        print("╔" + "═"*60 + "╗")
        print("║" + " "*20 + "ALL TESTS PASSED" + " "*23 + "║")
        print("╚" + "═"*60 + "╝")
        print("\nCognitive Manifold architecture operational.")
        print("Ready for integration with live Noodlings.\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
