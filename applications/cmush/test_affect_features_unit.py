#!/usr/bin/env python3
"""
Unit tests for affect-memory integration features.
Tests the three helper methods directly without WebSocket.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the CMUSHConsilienceAgent class
from agent_bridge import CMUSHConsilienceAgent

def test_name_based_memory_triggering():
    """Test _trigger_memories_by_names method"""
    print("\n" + "=" * 70)
    print("TEST 1: Name-Based Memory Triggering")
    print("=" * 70)

    # Create a minimal CMUSHConsilienceAgent instance
    # We'll manually set up conversation_context with memories
    agent = CMUSHConsilienceAgent(agent_id="test_agent", checkpoint_path="dummy.npz")

    # Create mock memories with names
    agent.conversation_context = [
        {
            'text': 'I met Alice at the park yesterday. She was so kind.',
            'identity_salience': 0.8,
            'affect': np.array([0.7, 0.3, 0.1, 0.1, 0.2])  # Happy memory
        },
        {
            'text': 'Bob told me a funny joke',
            'identity_salience': 0.5,
            'affect': np.array([0.9, 0.6, 0.0, 0.0, 0.1])  # Very happy memory
        },
        {
            'text': 'Alice and I used to be such good friends',
            'identity_salience': 0.9,
            'affect': np.array([0.2, 0.2, 0.1, 0.7, 0.3])  # Sad memory
        },
        {
            'text': 'The weather is nice today',
            'identity_salience': 0.2,
            'affect': np.array([0.4, 0.3, 0.0, 0.0, 0.1])  # Neutral memory
        }
    ]

    # Test: Mention "Alice" in text
    test_text = "Have you heard from Alice recently?"
    triggered = agent._trigger_memories_by_names(test_text)

    print(f"Input text: '{test_text}'")
    print(f"Triggered {len(triggered)} memories:")

    for i, mem in enumerate(triggered):
        print(f"  {i+1}. Salience={mem['identity_salience']:.2f}: '{mem['text'][:60]}'")

    # Verify: Should find 2 Alice memories, sorted by salience
    assert len(triggered) == 2, f"Expected 2 memories, got {len(triggered)}"
    assert triggered[0]['identity_salience'] == 0.9, "Highest salience should be first"
    assert triggered[1]['identity_salience'] == 0.8, "Second highest should be second"
    print("✓ PASS: Correctly retrieved and sorted memories by name")

    # Test: No names in text
    test_text2 = "what a nice day"
    triggered2 = agent._trigger_memories_by_names(test_text2)
    print(f"\nInput text: '{test_text2}'")
    print(f"Triggered {len(triggered2)} memories")
    assert len(triggered2) == 0, "Should not trigger memories without names"
    print("✓ PASS: No false positives without names")

    return True


def test_memory_affect_blending():
    """Test _apply_memory_affect method"""
    print("\n" + "=" * 70)
    print("TEST 2: Memory Affect Blending")
    print("=" * 70)

    agent = CMUSHConsilienceAgent(agent_id="test_agent", checkpoint_path="dummy.npz")

    # Current neutral affect
    current_affect = np.array([0.0, 0.3, 0.2, 0.2, 0.3])
    print(f"Current affect: {current_affect}")

    # Memories with strong positive affect
    memories = [
        {
            'affect': np.array([0.9, 0.7, 0.0, 0.0, 0.1]),  # Very happy
            'identity_salience': 0.8
        },
        {
            'affect': np.array([0.8, 0.6, 0.0, 0.1, 0.1]),  # Happy
            'identity_salience': 0.6
        }
    ]

    # Apply memory blending
    blended_affect = agent._apply_memory_affect(memories, current_affect)

    print(f"Blended affect:  {blended_affect}")
    print(f"\nDifference: {blended_affect - current_affect}")

    # Verify: Valence should increase (happy memories)
    assert blended_affect[0] > current_affect[0], "Valence should increase with happy memories"
    print("✓ PASS: Valence increased from happy memories")

    # Verify: Should be 70% current + 30% memory
    # The blended valence should be between current and memory values
    memory_avg_valence = np.average([0.9, 0.8], weights=[0.8**2, 0.6**2])
    expected_valence = 0.7 * current_affect[0] + 0.3 * memory_avg_valence
    print(f"Expected valence: {expected_valence:.3f}, Got: {blended_affect[0]:.3f}")
    assert abs(blended_affect[0] - expected_valence) < 0.01, "Blending ratio should be 70/30"
    print("✓ PASS: Correct 70/30 blending ratio")

    return True


def test_emotional_contagion():
    """Test _detect_emotional_contagion method"""
    print("\n" + "=" * 70)
    print("TEST 3: Emotional Contagion Detection")
    print("=" * 70)

    agent = CMUSHConsilienceAgent(agent_id="test_agent", checkpoint_path="dummy.npz")

    test_cases = [
        ("haha that's so funny!", "laughter", ["valence_boost", "arousal_boost"]),
        ("I'm so scared and anxious", "fear", ["fear_boost", "arousal_boost"]),
        ("I'm crying and heartbroken", "sadness", ["sorrow_boost", "valence_decrease"]),
        ("*yawns* I'm tired", "sleepiness", ["boredom_boost", "arousal_decrease"]),
        ("what a nice day", None, [])  # No contagion
    ]

    for text, expected_type, expected_keys in test_cases:
        result = agent._detect_emotional_contagion(text)

        print(f"\nInput: '{text}'")

        if expected_type is None:
            assert result is None, f"Expected no contagion, got {result}"
            print("  ✓ No contagion detected (correct)")
        else:
            assert result is not None, f"Expected {expected_type} contagion, got None"
            assert result['type'] == expected_type, f"Expected {expected_type}, got {result['type']}"
            print(f"  ✓ Detected: {result['type']}")

            # Check for expected affect keys
            for key in expected_keys:
                assert key in result, f"Expected key '{key}' in result"
                print(f"    - {key}: {result[key]}")

    print("\n✓ PASS: All emotional contagion patterns detected correctly")
    return True


def test_integration():
    """Test all three features working together"""
    print("\n" + "=" * 70)
    print("TEST 4: Integration Test - All Features Together")
    print("=" * 70)

    agent = CMUSHConsilienceAgent(agent_id="test_agent", checkpoint_path="dummy.npz")

    # Set up memories with a sad memory about "Sam"
    agent.conversation_context = [
        {
            'text': 'Sam was my best friend. I miss them so much.',
            'identity_salience': 0.95,
            'affect': np.array([-0.6, 0.2, 0.1, 0.9, 0.4])  # Very sad memory
        }
    ]

    # Current neutral affect
    current_affect = np.array([0.0, 0.3, 0.2, 0.2, 0.3])
    print(f"Starting affect: {current_affect}")

    # Input text mentions "Sam" AND contains "crying" (sadness contagion)
    text = "I heard about Sam and I've been crying all day"

    # 1. Name-based triggering
    triggered_memories = agent._trigger_memories_by_names(text)
    print(f"\n1. Name-based triggering: Found {len(triggered_memories)} memory about Sam")
    assert len(triggered_memories) == 1, "Should trigger 1 memory"

    # 2. Memory affect blending
    affect_after_memory = agent._apply_memory_affect(triggered_memories, current_affect)
    print(f"2. After memory blending: {affect_after_memory}")
    print(f"   Sorrow increased: {affect_after_memory[3] - current_affect[3]:.3f}")

    # 3. Emotional contagion
    contagion = agent._detect_emotional_contagion(text)
    print(f"3. Emotional contagion: {contagion['type']}")

    # Apply contagion
    final_affect = affect_after_memory.copy()
    if 'sorrow_boost' in contagion:
        final_affect[3] = min(1.0, final_affect[3] + contagion['sorrow_boost'])
    if 'valence_decrease' in contagion:
        final_affect[0] = max(-1.0, final_affect[0] - contagion['valence_decrease'])

    print(f"4. Final affect: {final_affect}")
    print(f"   Total sorrow change: {final_affect[3] - current_affect[3]:.3f}")
    print(f"   Total valence change: {final_affect[0] - current_affect[0]:.3f}")

    # Verify: Sorrow should be significantly higher
    assert final_affect[3] > current_affect[3] + 0.3, "Sorrow should increase significantly"
    # Verify: Valence should decrease
    assert final_affect[0] < current_affect[0], "Valence should decrease"

    print("\n✓ PASS: All three features working together correctly!")
    print("  - Memory triggered by name ✓")
    print("  - Memory affect blended into state ✓")
    print("  - Emotional contagion applied ✓")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("AFFECT-MEMORY INTEGRATION - UNIT TESTS")
    print("=" * 70)

    try:
        # Run all tests
        test_name_based_memory_triggering()
        test_memory_affect_blending()
        test_emotional_contagion()
        test_integration()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe three affect-memory features are working correctly:")
        print("  1. Memory → Affect Blending ✓")
        print("  2. Name-Based Memory Triggering ✓")
        print("  3. Social Contagion (laughter, yawning, fear, sadness) ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
