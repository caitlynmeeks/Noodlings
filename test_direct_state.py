#!/usr/bin/env python3
"""
Direct test of temporal model state retrieval.
Bypasses all server infrastructure to test the core issue.
"""

import os
import sys

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

from noodlings.api import NoodlingAgent
import mlx.core as mx
import numpy as np

print("=" * 70)
print("DIRECT STATE TEST")
print("=" * 70)

# Create agent with checkpoint
checkpoint = "models/checkpoints/best_checkpoint.npz"
print(f"\nInitializing NoodlingAgent with checkpoint: {checkpoint}")

agent = NoodlingAgent(
    checkpoint_path=checkpoint,
    config={
        'memory_capacity': 100,
        'surprise_threshold': 0.0001,
        'use_vae': False,
        'max_agents': 10,
        'use_phase6': False
    }
)

print("✓ Agent initialized")

# Check model states directly
print("\n--- Checking model states BEFORE perception ---")
print(f"model.h_fast: {type(agent.model.h_fast)}")
print(f"  shape: {agent.model.h_fast.shape if agent.model.h_fast is not None else 'None'}")
print(f"  value: {agent.model.h_fast if agent.model.h_fast is not None else 'None'}")

print(f"\nmodel.h_medium: {type(agent.model.h_medium)}")
print(f"  shape: {agent.model.h_medium.shape if agent.model.h_medium is not None else 'None'}")

print(f"\nmodel.h_slow: {type(agent.model.h_slow)}")
print(f"  shape: {agent.model.h_slow.shape if agent.model.h_slow is not None else 'None'}")

# Try calling get_states()
print("\n--- Calling agent.get_states() ---")
states = agent.get_states()
print(f"fast: {type(states['fast'])}, value: {states['fast']}")
print(f"medium: {type(states['medium'])}, value: {states['medium']}")
print(f"slow: {type(states['slow'])}, value: {states['slow']}")

# Now trigger a perception
print("\n--- Triggering perception event ---")
affect = mx.array([0.5, 0.5, 0.2, 0.1, 0.3])
result = agent.perceive(
    affect_vector=affect,
    agent_id="test",
    user_text="Hello world"
)

print(f"✓ Perception complete, surprise: {result['surprise']:.4f}")

# Check states AFTER perception
print("\n--- Checking model states AFTER perception ---")
print(f"model.h_fast: {type(agent.model.h_fast)}")
print(f"  shape: {agent.model.h_fast.shape if agent.model.h_fast is not None else 'None'}")
if agent.model.h_fast is not None:
    print(f"  first 3 values: {agent.model.h_fast.squeeze()[:3]}")

print(f"\nmodel.h_medium: {type(agent.model.h_medium)}")
print(f"  shape: {agent.model.h_medium.shape if agent.model.h_medium is not None else 'None'}")
if agent.model.h_medium is not None:
    print(f"  first 3 values: {agent.model.h_medium.squeeze()[:3]}")

print(f"\nmodel.h_slow: {type(agent.model.h_slow)}")
print(f"  shape: {agent.model.h_slow.shape if agent.model.h_slow is not None else 'None'}")
if agent.model.h_slow is not None:
    print(f"  first 3 values: {agent.model.h_slow.squeeze()[:3]}")

# Try calling get_states() again
print("\n--- Calling agent.get_states() AFTER perception ---")
states = agent.get_states()
print(f"fast: {type(states['fast'])}")
if states['fast'] is not None:
    print(f"  shape: {states['fast'].shape}")
    print(f"  first 3 values: {states['fast'][:3]}")
else:
    print(f"  value: None")

print(f"\nmedium: {type(states['medium'])}")
if states['medium'] is not None:
    print(f"  shape: {states['medium'].shape}")
    print(f"  first 3 values: {states['medium'][:3]}")
else:
    print(f"  value: None")

print(f"\nslow: {type(states['slow'])}")
if states['slow'] is not None:
    print(f"  shape: {states['slow'].shape}")
    print(f"  first 3 values: {states['slow'][:3]}")
else:
    print(f"  value: None")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
