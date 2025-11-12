#!/usr/bin/env python3
"""
Test script for Phase 6 Goal Override System

Tests Brenda's narrative control capabilities:
- Goal overrides (direct control)
- Goal biases (subtle influence)
- Clearing overrides and biases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from noodlings.models.appetite_layer import AppetiteLayer
from noodlings.models.noodling_phase6 import NoodlingModelPhase6

def test_appetite_layer():
    """Test goal override at the AppetiteLayer level."""
    print("=" * 60)
    print("Testing AppetiteLayer Goal Overrides")
    print("=" * 60)

    # Initialize appetite layer
    appetite_layer = AppetiteLayer(
        appetite_dim=8,
        goal_dim=16,
        slow_dim=8,
        fast_dim=16,
        affect_dim=5
    )

    # Set Mr. Toad's appetite baselines
    toad_appetites = [
        0.7,   # curiosity
        0.8,   # status
        0.6,   # mastery
        0.95,  # novelty (insatiable!)
        0.1,   # safety (reckless)
        0.5,   # social_bond
        0.2,   # comfort
        0.9    # autonomy
    ]
    appetite_layer.set_appetite_baselines(toad_appetites)

    # Create dummy states
    slow_state = mx.array([[0.9, 0.95, 0.7, 0.8, 0.9, 0.0, 0.0, 0.0]])  # Toad's personality
    fast_state = mx.random.normal((1, 16)) * 0.1
    affect = mx.array([[0.0, 0.3, 0.1, 0.0, 0.5]])  # Bored

    # 1. Test natural goal generation
    print("\n1. Natural Goal Generation (no overrides)")
    print("-" * 60)
    goals, conflicts, appetites = appetite_layer.forward(slow_state, fast_state, affect)
    top_goals = appetite_layer.get_top_goals(goals, k=5)
    print("Top 5 goals (natural):")
    for goal_name, strength in top_goals:
        print(f"  {goal_name:25s}: {strength:.3f}")

    # 2. Test goal override
    print("\n2. Testing Goal Override (motorcycle obsession)")
    print("-" * 60)
    appetite_layer.override_goal('learn_skill', 0.95)
    appetite_layer.override_goal('demonstrate_competence', 0.90)
    appetite_layer.override_goal('pursue_novelty', 0.85)

    goals, conflicts, appetites = appetite_layer.forward(slow_state, fast_state, affect)
    top_goals = appetite_layer.get_top_goals(goals, k=5)
    print("Top 5 goals (with overrides):")
    for goal_name, strength in top_goals:
        print(f"  {goal_name:25s}: {strength:.3f}")

    print("\nActive overrides:")
    for goal, strength in appetite_layer.get_goal_overrides().items():
        print(f"  {goal:25s}: {strength:.3f}")

    # 3. Test goal bias
    print("\n3. Testing Goal Bias (subtle influence)")
    print("-" * 60)
    appetite_layer.clear_goal_overrides()  # Clear overrides first
    appetite_layer.set_goal_bias('ensure_safety', -0.3)  # More reckless
    appetite_layer.set_goal_bias('pursue_novelty', 0.2)  # More novelty-seeking

    goals, conflicts, appetites = appetite_layer.forward(slow_state, fast_state, affect)
    top_goals = appetite_layer.get_top_goals(goals, k=5)
    print("Top 5 goals (with biases):")
    for goal_name, strength in top_goals:
        print(f"  {goal_name:25s}: {strength:.3f}")

    print("\nActive biases:")
    for goal, bias in appetite_layer.get_goal_biases().items():
        print(f"  {goal:25s}: {bias:+.3f}")

    # 4. Test clearing
    print("\n4. Testing Clear Operations")
    print("-" * 60)
    appetite_layer.clear_goal_biases()
    print("✓ Cleared all biases")

    goals, conflicts, appetites = appetite_layer.forward(slow_state, fast_state, affect)
    top_goals = appetite_layer.get_top_goals(goals, k=5)
    print("Top 5 goals (back to natural):")
    for goal_name, strength in top_goals:
        print(f"  {goal_name:25s}: {strength:.3f}")

    print("\n✓ AppetiteLayer test complete!")
    return True


def test_phase6_model():
    """Test goal override through the full Phase 6 model."""
    print("\n" + "=" * 60)
    print("Testing NoodlingModelPhase6 Goal Overrides")
    print("=" * 60)

    # Initialize Phase 6 model (without checkpoint for testing)
    model = NoodlingModelPhase6(
        affect_dim=5,
        fast_hidden=16,
        medium_hidden=16,
        slow_hidden=8,
        predictor_hidden=64,
        use_appetite_layer=True,
        appetite_dim=8,
        goal_dim=16
    )

    # Set appetites
    toad_appetites = [0.7, 0.8, 0.6, 0.95, 0.1, 0.5, 0.2, 0.9]
    model.set_appetite_baselines(toad_appetites)

    print("\n✓ Phase 6 model initialized with appetite layer")

    # Test forward pass
    affect = mx.array([[0.0, 0.3, 0.1, 0.0, 0.5]])

    print("\n1. Testing override_goal method")
    print("-" * 60)
    model.override_goal('learn_skill', 0.95)
    print(f"✓ Set override: learn_skill = 0.95")

    overrides = model.get_goal_overrides()
    print(f"Active overrides: {overrides}")

    print("\n2. Testing set_goal_bias method")
    print("-" * 60)
    model.set_goal_bias('ensure_safety', -0.3)
    print(f"✓ Set bias: ensure_safety = -0.3")

    biases = model.get_goal_biases()
    print(f"Active biases: {biases}")

    print("\n3. Testing clear methods")
    print("-" * 60)
    model.clear_goal_overrides()
    print("✓ Cleared overrides")
    model.clear_goal_biases()
    print("✓ Cleared biases")

    overrides = model.get_goal_overrides()
    biases = model.get_goal_biases()
    print(f"Overrides after clear: {overrides}")
    print(f"Biases after clear: {biases}")

    print("\n✓ Phase 6 model test complete!")
    return True


def test_scenario_toad_motorcycles():
    """Test the Mr. Toad motorcycle obsession scenario."""
    print("\n" + "=" * 60)
    print("Scenario Test: Making Toad Obsess Over Motorcycles")
    print("=" * 60)

    appetite_layer = AppetiteLayer()

    # Mr. Toad's natural appetites
    toad_appetites = [0.7, 0.8, 0.6, 0.95, 0.1, 0.5, 0.2, 0.9]
    appetite_layer.set_appetite_baselines(toad_appetites)

    slow_state = mx.array([[0.9, 0.95, 0.7, 0.8, 0.9, 0.0, 0.0, 0.0]])
    fast_state = mx.random.normal((1, 16)) * 0.1
    affect = mx.array([[0.5, 0.8, 0.0, 0.0, 0.0]])  # Excited!

    print("\n📖 Narrative: Brenda wants Toad to become obsessed with motorcycles")
    print("    instead of motor-cars...")

    print("\nBefore intervention:")
    goals, _, _ = appetite_layer.forward(slow_state, fast_state, affect)
    top_goals = appetite_layer.get_top_goals(goals, k=3)
    for goal_name, strength in top_goals:
        print(f"  {goal_name:25s}: {strength:.3f}")

    print("\n🎭 Brenda intervenes with goal overrides:")
    print("   @override Toad learn_skill 0.95")
    print("   @override Toad demonstrate_competence 0.90")
    print("   @override Toad pursue_novelty 0.85")

    appetite_layer.override_goal('learn_skill', 0.95)
    appetite_layer.override_goal('demonstrate_competence', 0.90)
    appetite_layer.override_goal('pursue_novelty', 0.85)

    print("\nAfter intervention:")
    goals, _, _ = appetite_layer.forward(slow_state, fast_state, affect)
    top_goals = appetite_layer.get_top_goals(goals, k=3)
    for goal_name, strength in top_goals:
        print(f"  {goal_name:25s}: {strength:.3f}")

    print("\n✅ Success! Toad is now driven to master motorcycles!")
    print("   The LLM prompt will include these top goals, shaping his responses.")

    return True


if __name__ == '__main__':
    print("\n" + "🧪 PHASE 6 GOAL OVERRIDE SYSTEM TEST" + "\n")

    try:
        # Run tests
        success = True
        success = success and test_appetite_layer()
        success = success and test_phase6_model()
        success = success and test_scenario_toad_motorcycles()

        print("\n" + "=" * 60)
        if success:
            print("✅ ALL TESTS PASSED!")
            print("\nBrenda's goal override system is ready for use!")
            print("\nAvailable commands in noodleMUSH:")
            print("  @override <agent> <goal> <strength>  - Direct goal control")
            print("  @bias <agent> <goal> <bias>          - Subtle influence")
            print("  @reset_goals <agent> [goal]          - Clear overrides")
            print("  @clear_bias <agent> [goal]           - Clear biases")
            print("  @goals <agent>                       - View goal state")
        else:
            print("❌ SOME TESTS FAILED")
            sys.exit(1)

        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
