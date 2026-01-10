# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Test script for Noodle API Phase 1 implementation.
#
#   Tests: 1. ScriptContext includes noodle property 2. Model...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.test_noodle_api
# PURPOSE:  Test script for Noodle API Phase 1 implementation.
# LAYER:    Studio / Application
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   test_script_context_has_noodle_api(), test_models_api_accessible(), test_neural_api_accessible()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import sys
import time
from noodlestudio.core.scripted_facet import ScriptContext, ScriptedFacet


def test_script_context_has_noodle_api():
    """Test 1: ScriptContext includes noodle API."""
    print("Test 1: ScriptContext includes noodle API")
    print("-" * 50)

    context = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="test_agent",
        agent_name="Test Agent",
        agent_species="test"
    )

    # Check that noodle API exists
    assert hasattr(context, '_noodle_api'), "ScriptContext missing _noodle_api"
    assert context._noodle_api is not None, "Noodle API not initialized"

    print(f"  ✓ Noodle API initialized: {type(context._noodle_api).__name__}")

    # Check that to_dict includes noodle
    context_dict = context.to_dict()
    assert 'noodle' in context_dict, "to_dict() missing 'noodle' key"

    print(f"  ✓ to_dict() includes 'noodle' key")
    print()


def test_models_api_accessible():
    """Test 2: ModelsAPI accessible."""
    print("Test 2: ModelsAPI accessible")
    print("-" * 50)

    context = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="test_agent",
        agent_name="Test Agent",
        agent_species="test"
    )

    # Access models API
    models_api = context._noodle_api.models
    assert models_api is not None, "Models API not accessible"

    print(f"  ✓ Models API type: {type(models_api).__name__}")

    # Test get_all_labels
    labels = models_api.get_all_labels()
    print(f"  ✓ get_all_labels() returned: {list(labels.keys())}")

    # Test get_label
    small_assignment = models_api.get_label("SMALL")
    print(f"  ✓ SMALL label: {small_assignment['provider']} / {small_assignment['model']}")

    # Test list_providers
    providers = models_api.list_providers()
    print(f"  ✓ Available providers: {[p['id'] for p in providers]}")

    print()


def test_neural_api_accessible():
    """Test 3: NeuralAPI accessible."""
    print("Test 3: NeuralAPI accessible")
    print("-" * 50)

    context = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="test_agent",
        agent_name="Test Agent",
        agent_species="test"
    )

    # Access neural API
    neural_api = context._noodle_api.neural
    assert neural_api is not None, "Neural API not accessible"

    print(f"  ✓ Neural API type: {type(neural_api).__name__}")

    # Test create_network
    network = neural_api.create_network("TestNetwork")
    assert network is not None, "Failed to create network"

    print(f"  ✓ Created network: {type(network).__name__}")

    # Test node creation (this might fail if definitions not available, that's OK)
    try:
        node_id = network.create_node("LSTM", hidden_dim=32, position=[100, 200])
        if node_id:
            print(f"  ✓ Created LSTM node: {node_id[:8]}...")
        else:
            print(f"  ⚠ Node creation returned None (node definitions may not be loaded)")
    except Exception as e:
        print(f"  ⚠ Node creation failed (expected if node definitions not available): {e}")

    print()


def test_agents_api_accessible():
    """Test 4: AgentsAPI accessible."""
    print("Test 4: AgentsAPI accessible")
    print("-" * 50)

    context = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="test_agent",
        agent_name="Test Agent",
        agent_species="test"
    )

    # Access agents API
    agents_api = context._noodle_api.agents
    assert agents_api is not None, "Agents API not accessible"

    print(f"  ✓ Agents API type: {type(agents_api).__name__}")

    # Test list_all (will be empty initially)
    agents = agents_api.list_all()
    print(f"  ✓ list_all() returned: {agents}")

    print()


def test_javascript_context_dict():
    """Test 5: JavaScript context dict includes noodle placeholders."""
    print("Test 5: JavaScript context dict structure")
    print("-" * 50)

    context = ScriptContext(
        cycle=1,
        timestamp=time.time(),
        agent_id="test_agent",
        agent_name="Test Agent",
        agent_species="test"
    )

    context_dict = context.to_dict()

    # Check noodle structure
    assert 'noodle' in context_dict, "Missing 'noodle' key"
    noodle = context_dict['noodle']

    assert 'models' in noodle, "Missing 'models' in noodle"
    assert 'neural' in noodle, "Missing 'neural' in noodle"
    assert 'agents' in noodle, "Missing 'agents' in noodle"

    print(f"  ✓ noodle.models keys: {list(noodle['models'].keys())}")
    print(f"  ✓ noodle.neural keys: {list(noodle['neural'].keys())}")
    print(f"  ✓ noodle.agents keys: {list(noodle['agents'].keys())}")

    print()


def run_all_tests():
    """Run all Phase 1 tests."""
    print("=" * 50)
    print("NOODLE API PHASE 1 TESTS")
    print("=" * 50)
    print()

    tests = [
        test_script_context_has_noodle_api,
        test_models_api_accessible,
        test_neural_api_accessible,
        test_agents_api_accessible,
        test_javascript_context_dict
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
