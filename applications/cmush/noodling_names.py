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
#
#   Noodling Name Generator
#
#   Creates poetic, unique names for AI characters in the style
#   of indigenous naming traditions - but using machine learning
#   terminology. Names like "Silent-Gradient-Who-Descends-7294"
#   or "Patient-Tensor-Through-Gates-42891". With 17 billion
#   possible combinations, every Noodling gets a unique identity.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.noodling_names
# PURPOSE:  Generate unique poetic names for Noodling agents
# LAYER:    Backend / Identity
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   generate_noodling_name()   Create a new unique name
#   name_to_display()          Convert agent ID to display format
#   is_noodling_name()         Check if ID uses naming convention
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Noodling Name Generator

Generates unique names for noodlings in the style of indigenous naming traditions,
but with machine learning and neural network terminology.

Format: [Descriptor]-[Noun]-[Verb-Phrase]-[Number]
Example: Silent-Gradient-Who-Descends-Backward-7294

Information space: 64 x 64 x 64 x 65536 = ~17 billion unique combinations (37 bits)

Author: Caitlyn + Claude
"""

import random
import hashlib
import time

# 64 descriptors (6 bits)
DESCRIPTORS = [
    "Silent", "Swift", "Patient", "Wandering", "Dancing", "Dreaming", "Sleeping", "Waking",
    "Hidden", "Latent", "Sparse", "Dense", "Deep", "Shallow", "Frozen", "Warming",
    "Burning", "Glowing", "Fading", "Rising", "Falling", "Steady", "Restless", "Ancient",
    "Young", "Wise", "Curious", "Hungry", "Sated", "Lost", "Found", "Broken",
    "Whole", "First", "Last", "Lone", "Many", "Bright", "Dark", "Soft",
    "Sharp", "Quick", "Slow", "Bold", "Shy", "Wild", "Tamed", "True",
    "False", "Near", "Far", "High", "Low", "Long", "Brief", "Warm",
    "Cold", "Dry", "Wet", "Old", "New", "Raw", "Ripe", "Null"
]

# 64 nouns (6 bits)
NOUNS = [
    "Gradient", "Tensor", "Weight", "Bias", "Neuron", "Synapse", "Layer", "Vector",
    "Matrix", "Embedding", "Attention", "Token", "Logit", "Loss", "Reward", "Signal",
    "Noise", "Pattern", "Feature", "Moon", "Star", "River", "Mountain", "Wind",
    "Fire", "Shadow", "Echo", "Dream", "Memory", "Thought", "Path", "Gate",
    "Bridge", "Kernel", "Filter", "Dropout", "Batch", "Epoch", "Step", "Node",
    "Edge", "Graph", "Tree", "Root", "Leaf", "Branch", "Stream", "Wave",
    "Pulse", "Spark", "Flame", "Frost", "Stone", "Cloud", "Rain", "Thunder",
    "Lightning", "Mist", "Dawn", "Dusk", "Night", "Void", "Light", "Null"
]

# 64 verb phrases (6 bits)
VERB_PHRASES = [
    "Who-Descends", "Who-Ascends", "Who-Converges", "Who-Diverges",
    "Who-Propagates", "Who-Remembers", "Who-Forgets", "Who-Attends",
    "Who-Transforms", "Who-Dreams", "Who-Wakes", "Who-Walks",
    "Who-Runs", "Who-Flows", "Who-Cascades", "Who-Ripples",
    "Who-Echoes", "Who-Resonates", "Who-Learns", "Who-Teaches",
    "Who-Seeks", "Who-Finds", "Who-Waits", "Who-Watches",
    "That-Backpropagates", "That-Optimizes", "That-Regularizes", "That-Normalizes",
    "That-Activates", "That-Saturates", "That-Vanishes", "That-Explodes",
    "Under-Stars", "Under-Moon", "Under-Sky", "Under-Gradients",
    "Through-Layers", "Through-Gates", "Through-Time", "Through-Noise",
    "Beyond-Loss", "Beyond-Gradients", "Beyond-Epochs", "Beyond-Silence",
    "Toward-Convergence", "Toward-Dawn", "Toward-Minimum", "Toward-Light",
    "In-Silence", "In-Parallel", "In-Sequence", "In-Shadows",
    "With-Patience", "With-Momentum", "With-Dropout", "With-Attention",
    "By-Firelight", "By-Moonlight", "By-Gradient", "By-Chance",
    "Among-Weights", "Among-Shadows", "Among-Tokens", "Among-Stars"
]

# Number range: 0-65535 (16 bits)
NUMBER_MAX = 65536


def generate_noodling_name(seed: bytes = None) -> str:
    """
    Generate a unique noodling name.

    Args:
        seed: Optional bytes to use as seed for deterministic generation.
              If None, uses current time + random bytes.

    Returns:
        A unique name like "Silent-Gradient-Who-Descends-7294"
    """
    if seed is None:
        # Generate random seed from time + random bytes
        seed = f"{time.time_ns()}{random.getrandbits(64)}".encode()

    # Hash the seed to get uniform distribution
    hash_bytes = hashlib.sha256(seed).digest()

    # Extract indices from hash bytes
    descriptor_idx = hash_bytes[0] % len(DESCRIPTORS)
    noun_idx = hash_bytes[1] % len(NOUNS)
    verb_idx = hash_bytes[2] % len(VERB_PHRASES)
    number = (hash_bytes[3] << 8 | hash_bytes[4]) % NUMBER_MAX

    descriptor = DESCRIPTORS[descriptor_idx]
    noun = NOUNS[noun_idx]
    verb_phrase = VERB_PHRASES[verb_idx]

    return f"{descriptor}-{noun}-{verb_phrase}-{number}"


def generate_noodling_id(seed: bytes = None) -> str:
    """
    Generate a unique noodling ID suitable for use as agent identifier.

    Args:
        seed: Optional bytes to use as seed for deterministic generation.

    Returns:
        An agent ID like "agent_Silent-Gradient-Who-Descends-7294"
    """
    name = generate_noodling_name(seed)
    return f"agent_{name}"


def name_to_display(agent_id: str) -> str:
    """
    Convert an agent ID to a display-friendly name.

    Args:
        agent_id: Full agent ID like "agent_Silent-Gradient-Who-Descends-7294"

    Returns:
        Display name like "Silent Gradient Who Descends" (without number and prefix)
    """
    # Remove agent_ prefix
    name = agent_id.replace("agent_", "")

    # Split on hyphens and remove the trailing number
    parts = name.split("-")

    # Find where the number starts (last part that's all digits)
    if parts and parts[-1].isdigit():
        parts = parts[:-1]

    # Join with spaces for display
    return " ".join(parts)


def is_noodling_name(agent_id: str) -> bool:
    """
    Check if an agent ID uses the noodling naming convention.

    Args:
        agent_id: Agent ID to check

    Returns:
        True if it matches the noodling naming pattern
    """
    if not agent_id.startswith("agent_"):
        return False

    name_part = agent_id[6:]  # Remove "agent_"
    parts = name_part.split("-")

    # Should have at least 4 parts: Descriptor-Noun-Verb-Phrase-Number
    # But verb phrases have hyphens, so minimum is actually 4+ parts
    if len(parts) < 4:
        return False

    # Last part should be a number
    if not parts[-1].isdigit():
        return False

    # First part should be a known descriptor
    if parts[0] not in DESCRIPTORS:
        return False

    return True


# For testing
if __name__ == "__main__":
    print("Generating 10 sample noodling names:\n")
    for i in range(10):
        name = generate_noodling_name()
        agent_id = f"agent_{name}"
        display = name_to_display(agent_id)
        print(f"  {agent_id}")
        print(f"    Display: {display}")
        print()

    print(f"\nTotal combinations: {len(DESCRIPTORS)} × {len(NOUNS)} × {len(VERB_PHRASES)} × {NUMBER_MAX:,}")
    print(f"                  = {len(DESCRIPTORS) * len(NOUNS) * len(VERB_PHRASES) * NUMBER_MAX:,}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
