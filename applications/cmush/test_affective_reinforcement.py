# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.test_affective_reinforcement
# PURPOSE:  Test affect-based reward shaping system
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Test script for Affective Reinforcement system

Validates that comedy markers boost affect and mysticism markers reduce it.

Usage:
    python test_affective_reinforcement.py
"""

import numpy as np
from affective_reinforcement import ComedyAffectiveReinforcement

def test_comedy_reward():
    """Test that comedy markers boost valence and reduce boredom."""
    reinforcement = ComedyAffectiveReinforcement(enabled=True, intensity=1.0)

    # Start with neutral affect
    neutral_affect = np.array([0.0, 0.3, 0.1, 0.1, 0.5])  # [valence, arousal, fear, sorrow, boredom]

    # Comedy text (multiple markers)
    comedy_text = "*HONK* Oh no! *trips over coat, feathers EXPLODE everywhere* YES! *waddles desperately* Bread?!"

    # Apply reinforcement
    modulated = reinforcement.modulate_affect(
        text=comedy_text,
        current_affect=neutral_affect,
        context={'agent_id': 'test_geese'}
    )

    print("=== COMEDY REWARD TEST ===")
    print(f"Original affect:   valence={neutral_affect[0]:.2f}, boredom={neutral_affect[4]:.2f}")
    print(f"After comedy:      valence={modulated[0]:.2f}, boredom={modulated[4]:.2f}")
    print(f"Valence change:    {modulated[0] - neutral_affect[0]:+.2f} (should be positive)")
    print(f"Boredom change:    {modulated[4] - neutral_affect[4]:+.2f} (should be negative)")
    print()

    assert modulated[0] > neutral_affect[0], "Comedy should increase valence!"
    assert modulated[4] < neutral_affect[4], "Comedy should decrease boredom!"
    print(" Comedy reward working correctly\n")


def test_mysticism_penalty():
    """Test that mysticism markers reduce valence and increase boredom."""
    reinforcement = ComedyAffectiveReinforcement(enabled=True, intensity=1.0)

    # Start with positive affect
    positive_affect = np.array([0.5, 0.4, 0.1, 0.1, 0.2])

    # Mysticism text (multiple markers)
    mysticism_text = "The quiet part of me listens to the stillness. The air holds its breath. I feel the gentle calm."

    # Apply reinforcement
    modulated = reinforcement.modulate_affect(
        text=mysticism_text,
        current_affect=positive_affect,
        context={'agent_id': 'test_geese'}
    )

    print("=== MYSTICISM PENALTY TEST ===")
    print(f"Original affect:   valence={positive_affect[0]:.2f}, boredom={positive_affect[4]:.2f}")
    print(f"After mysticism:   valence={modulated[0]:.2f}, boredom={modulated[4]:.2f}")
    print(f"Valence change:    {modulated[0] - positive_affect[0]:+.2f} (should be negative)")
    print(f"Boredom change:    {modulated[4] - positive_affect[4]:+.2f} (should be positive)")
    print()

    assert modulated[0] < positive_affect[0], "Mysticism should decrease valence!"
    assert modulated[4] > positive_affect[4], "Mysticism should increase boredom!"
    print(" Mysticism penalty working correctly\n")


def test_neutral_text():
    """Test that neutral text doesn't trigger reinforcement."""
    reinforcement = ComedyAffectiveReinforcement(enabled=True, intensity=1.0)

    neutral_affect = np.array([0.3, 0.4, 0.1, 0.1, 0.3])
    neutral_text = "Hello there. How are you today?"

    modulated = reinforcement.modulate_affect(
        text=neutral_text,
        current_affect=neutral_affect,
        context={'agent_id': 'test_agent'}
    )

    print("=== NEUTRAL TEXT TEST ===")
    print(f"Original affect:   valence={neutral_affect[0]:.2f}, boredom={neutral_affect[4]:.2f}")
    print(f"After neutral:     valence={modulated[0]:.2f}, boredom={modulated[4]:.2f}")
    print(f"Change:            {np.linalg.norm(modulated - neutral_affect):.4f} (should be ~0)")
    print()

    assert np.allclose(modulated, neutral_affect), "Neutral text shouldn't change affect!"
    print(" Neutral text handling correct\n")


def test_intensity_scaling():
    """Test that intensity parameter scales the reinforcement."""
    # Low intensity
    low_intensity = ComedyAffectiveReinforcement(enabled=True, intensity=0.5)
    high_intensity = ComedyAffectiveReinforcement(enabled=True, intensity=2.0)

    affect = np.array([0.0, 0.3, 0.1, 0.1, 0.5])
    comedy_text = "*HONK* *trips* *waddles*"

    modulated_low = low_intensity.modulate_affect(comedy_text, affect.copy(), {'agent_id': 'test'})
    modulated_high = high_intensity.modulate_affect(comedy_text, affect.copy(), {'agent_id': 'test'})

    print("=== INTENSITY SCALING TEST ===")
    print(f"Original valence:   {affect[0]:.2f}")
    print(f"Low intensity:      {modulated_low[0]:.2f}")
    print(f"High intensity:     {modulated_high[0]:.2f}")
    print()

    assert modulated_low[0] < modulated_high[0], "Higher intensity should cause bigger changes!"
    print(" Intensity scaling working correctly\n")


def test_statistics():
    """Test that reinforcement statistics are tracked."""
    reinforcement = ComedyAffectiveReinforcement(enabled=True, intensity=1.0)

    affect = np.array([0.0, 0.3, 0.1, 0.1, 0.5])

    # Apply multiple reinforcements
    reinforcement.modulate_affect("*HONK* *trip*", affect.copy(), {'agent_id': 'test'})
    reinforcement.modulate_affect("The quiet stillness", affect.copy(), {'agent_id': 'test'})
    reinforcement.modulate_affect("*waddle* *fumble*", affect.copy(), {'agent_id': 'test'})

    stats = reinforcement.get_statistics()

    print("=== STATISTICS TEST ===")
    print(f"Comedy events:      {stats['comedy_events']}")
    print(f"Mysticism events:   {stats['mysticism_events']}")
    print(f"Total events:       {stats['total_events']}")
    print()

    assert stats['comedy_events'] == 2, "Should have 2 comedy events"
    assert stats['mysticism_events'] == 1, "Should have 1 mysticism event"
    assert stats['total_events'] == 3, "Should have 3 total events"
    print(" Statistics tracking working correctly\n")


if __name__ == '__main__':
    print("Testing Affective Reinforcement System")
    print("=" * 50)
    print()

    try:
        test_comedy_reward()
        test_mysticism_penalty()
        test_neutral_text()
        test_intensity_scaling()
        test_statistics()

        print("=" * 50)
        print("ALL TESTS PASSED ")
        print()
        print("The affective reinforcement system is operational.")
        print("Comedy characters will learn to WANT comedy through")
        print("positive affect feedback, not external constraint.")
        print()
        print("Next: Test with live geese in noodleMUSH!")

    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
