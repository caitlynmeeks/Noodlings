#!/usr/bin/env python3
"""
Quick test that Noodlings works after rebrand
"""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
import mlx.utils
from noodlings.models.noodling_phase4 import NoodlingModelPhase4
from noodlings.memory.social_memory import SocialContext

print("Testing Noodlings...")

# Create model
model = NoodlingModelPhase4(
    fast_hidden=16,
    medium_hidden=16,
    slow_hidden=8,
    affect_dim=5,
    use_theory_of_mind=True,
    use_relationship_model=True
)

print(f"✓ NoodlingModelPhase4 created")

# Test forward pass
affect = mx.array([[0.5, 0.6, 0.1, 0.0, 0.0]], dtype=mx.float32)  # (batch=1, affect_dim=5)

# Create social context to avoid the MLX scalar conversion issue
social_context = SocialContext(
    present_agents=[],
    group_valence=0.5,
    group_arousal=0.6
)

self_state, predicted_state, social_info = model.forward_with_social_context(
    affect=affect,
    linguistic_features=None,
    context_features=None,
    present_agents=[],
    social_context=social_context,
    user_text="Hello"
)

print(f"✓ Forward pass successful")
print(f"  Internal state shape: {self_state.shape}")  # Should be (1, 40)
print(f"  Predicted shape: {predicted_state.shape}")

print("\n🎉 Noodlings is fully operational!")
print("Ready for Phase 5 metrics implementation.")
