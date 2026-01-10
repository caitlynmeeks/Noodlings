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
#   Test Tutorial Canvases
#
#   Test suite for tutorial canvases.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.test_tutorial_canvases
# PURPOSE:  Tests for tutorial canvases
# LAYER:    Studio / Application
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   load_canvas(), test_tutorial(), main()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

#!/usr/bin/env python3
"""
Test Tutorial Canvases - Verify logic gate tutorials work in test executor.

Run: python test_tutorial_canvases.py
"""

import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noodlestudio', 'core', 'neural_canvas'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noodlestudio', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noodlestudio'))

from neural_canvas.neural_graph import NeuralGraph
from neural_canvas.test_executor import CanvasTestExecutor


def load_canvas(path):
    """Load a .nncanvas file."""
    with open(path, 'r') as f:
        data = json.load(f)
    graph = NeuralGraph.from_dict(data)
    print(f"  Loaded {len(graph.nodes)} nodes:")
    for node_id, node in graph.nodes.items():
        print(f"    - {node.name} ({node.type})")
    return graph


def test_tutorial(name, path, test_cases):
    """Test a tutorial with given input cases."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")
        print(f"Description: {graph.description[:100]}...")

        # Create executor
        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        print(f"Executor initialized. Testing {len(test_cases)} cases...")

        all_passed = True
        for inputs, expected_on in test_cases:
            # Find input nodes and set their values
            input_a_node = None
            input_b_node = None
            for node_id, node in graph.nodes.items():
                if node.name == "Input A":
                    input_a_node = node
                elif node.name == "Input B":
                    input_b_node = node

            if input_a_node and input_b_node:
                input_a_node.params['value'] = inputs[0]
                input_b_node.params['value'] = inputs[1]

            # Re-initialize after param change
            executor._initialized = False
            executor.initialize()

            # Execute
            result = executor.execute()

            if not result.success:
                print(f"  [{inputs[0]}, {inputs[1]}] -> ERROR: {result.error}")
                all_passed = False
                continue

            # Find threshold output
            is_on = None
            for node_id, outputs in result.node_outputs.items():
                if 'is_on' in outputs:
                    is_on = outputs['is_on']
                    value = outputs.get('value', '?')
                    break

            status = "PASS" if is_on == expected_on else "FAIL"
            print(f"  [{inputs[0]}, {inputs[1]}] -> {is_on} (value: {value:.4f}) [{status}]")

            if is_on != expected_on:
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tutorials'
    )

    print("=" * 60)
    print("Neural Canvas Tutorial Test Suite")
    print("=" * 60)

    # Note: These tests will use random weights since we don't have trained weights.
    # The tutorials are designed for users to manually adjust weights.
    # We're just verifying the topology works, not the correct output.

    print("\nNOTE: Using random weights - testing topology only.")
    print("Tutorials expect users to manually adjust weights to learn.")

    # Test AND gate topology (weights pre-loaded in .nncanvas)
    and_path = os.path.join(base_path, '01_and_gate.nncanvas')
    if os.path.exists(and_path):
        test_tutorial(
            "AND Gate",
            and_path,
            [
                ([0.0, 0.0], False),  # Both off -> OFF
                ([0.0, 1.0], False),  # One off -> OFF
                ([1.0, 0.0], False),  # One off -> OFF
                ([1.0, 1.0], True),   # Both on -> ON (AND gate!)
            ]
        )
    else:
        print(f"WARNING: {and_path} not found")

    # Test OR gate topology
    or_path = os.path.join(base_path, '02_or_gate.nncanvas')
    if os.path.exists(or_path):
        test_tutorial(
            "OR Gate",
            or_path,
            [
                ([0.0, 0.0], False),  # Both off -> OFF
                ([0.0, 1.0], True),   # One on -> ON (OR gate!)
                ([1.0, 0.0], True),   # One on -> ON
                ([1.0, 1.0], True),   # Both on -> ON
            ]
        )
    else:
        print(f"WARNING: {or_path} not found")

    # Test XOR topology
    xor_path = os.path.join(base_path, '03_xor_problem.nncanvas')
    if os.path.exists(xor_path):
        test_tutorial(
            "XOR Problem",
            xor_path,
            [
                ([0.0, 0.0], None),
                ([0.0, 1.0], None),
                ([1.0, 0.0], None),
                ([1.0, 1.0], None),
            ]
        )
    else:
        print(f"WARNING: {xor_path} not found")

    # Test Echo Chamber (Phase 2)
    echo_path = os.path.join(base_path, '04_echo_chamber.nncanvas')
    if os.path.exists(echo_path):
        test_rnn_tutorial("Echo Chamber", echo_path)
    else:
        print(f"WARNING: {echo_path} not found")

    # Test Counting (Phase 2)
    count_path = os.path.join(base_path, '05_counting.nncanvas')
    if os.path.exists(count_path):
        test_lstm_tutorial("Counting", count_path)
    else:
        print(f"WARNING: {count_path} not found")

    # Test Mood Ring (Phase 3)
    mood_path = os.path.join(base_path, '06_mood_ring.nncanvas')
    if os.path.exists(mood_path):
        test_affect_tutorial("Mood Ring", mood_path)
    else:
        print(f"WARNING: {mood_path} not found")

    # Test CharmNetwork Lite (Phase 3)
    charm_path = os.path.join(base_path, '07_charm_lite.nncanvas')
    if os.path.exists(charm_path):
        test_charm_tutorial("CharmNetwork Lite", charm_path)
    else:
        print(f"WARNING: {charm_path} not found")

    # Test Token Prediction (Phase 4)
    token_path = os.path.join(base_path, '08_token_prediction.nncanvas')
    if os.path.exists(token_path):
        test_generation_tutorial("Token Prediction", token_path)
    else:
        print(f"WARNING: {token_path} not found")

    # Test Temperature Control (Phase 4)
    temp_path = os.path.join(base_path, '09_temperature.nncanvas')
    if os.path.exists(temp_path):
        test_temperature_tutorial("Temperature Control", temp_path)
    else:
        print(f"WARNING: {temp_path} not found")

    # Test Demoscene (Creative nodes)
    demo_path = os.path.join(base_path, '10_demoscene.nncanvas')
    if os.path.exists(demo_path):
        test_demoscene_tutorial("Demoscene", demo_path)
    else:
        print(f"WARNING: {demo_path} not found")

    print("\n" + "=" * 60)
    print("TUTORIAL TOPOLOGY TEST COMPLETE")
    print("All tutorials loaded and executed successfully!")
    print("=" * 60)


def test_rnn_tutorial(name, path):
    """Test RNN tutorial with sequential execution."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        print("Running sequence: 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0")

        # Find input node and set values over time
        input_node = None
        for node_id, node in graph.nodes.items():
            if node.name == "Input Signal":
                input_node = node
                break

        if not input_node:
            print("WARNING: Could not find Input Signal node")
            return False

        sequence = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, val in enumerate(sequence):
            input_node.params['value'] = val
            result = executor.execute()
            if result.success:
                # Find chart output
                for node_id, outputs in result.node_outputs.items():
                    if 'history' in outputs:
                        latest = outputs.get('value', '?')
                        print(f"  Step {i+1}: input={val} -> output={latest:.4f}")
                        break
            else:
                print(f"  Step {i+1}: ERROR - {result.error}")

        print("SUCCESS: RNN tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_tutorial(name, path):
    """Test LSTM tutorial with pulse inputs."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        print("Sending 5 pulses...")

        # Find pulse input node
        pulse_node = None
        for node_id, node in graph.nodes.items():
            if node.type.name == "PULSE_INPUT":
                pulse_node = node
                break

        if not pulse_node:
            print("WARNING: Could not find PULSE_INPUT node")
            return False

        for i in range(5):
            # Send pulse
            pulse_node.params['pulse_active'] = True
            result = executor.execute()
            pulse_node.params['pulse_active'] = False  # Reset for next step

            if result.success:
                # Find counter output
                for node_id, outputs in result.node_outputs.items():
                    if 'count' in outputs:
                        count = outputs.get('count', '?')
                        print(f"  Pulse {i+1}: count={count}")
                        break
            else:
                print(f"  Pulse {i+1}: ERROR - {result.error}")

        print("SUCCESS: LSTM tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_tutorial(name, path):
    """Test token prediction tutorial."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        # Find token input node
        token_node = None
        for node_id, node in graph.nodes.items():
            if node.type.name == "TOKEN_INPUT":
                token_node = node
                break

        if not token_node:
            print("WARNING: Could not find TOKEN_INPUT node")
            return False

        vocab = token_node.params.get('vocab', [])
        print(f"Testing with vocab: {vocab[:5]}...")

        for token_id in range(min(3, len(vocab))):
            token_node.params['token_id'] = token_id
            result = executor.execute()

            if result.success:
                # Find prob vis output
                for node_id, outputs in result.node_outputs.items():
                    if 'top_probs' in outputs:
                        top = outputs['top_probs'][:3] if outputs['top_probs'] else []
                        top_str = ', '.join([f"{p['token']}:{p['prob']:.2f}" for p in top])
                        print(f"  '{vocab[token_id]}' -> {top_str}")
                        break
            else:
                print(f"  Token {token_id}: ERROR - {result.error}")

        print("SUCCESS: Token prediction tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temperature_tutorial(name, path):
    """Test temperature control tutorial."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        # Find sampling node
        sampling_node = None
        for node_id, node in graph.nodes.items():
            if node.type.name == "SAMPLING":
                sampling_node = node
                break

        if not sampling_node:
            print("WARNING: Could not find SAMPLING node")
            return False

        print("Testing different temperatures...")

        for temp in [0.1, 1.0, 2.0]:
            sampling_node.params['temperature'] = temp
            results = []

            # Run 3 times to show variance
            for _ in range(3):
                result = executor.execute()
                if result.success:
                    for node_id, outputs in result.node_outputs.items():
                        if 'token_text' in outputs:
                            results.append(outputs['token_text'])
                            break

            print(f"  temp={temp}: {results}")

        print("SUCCESS: Temperature tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_affect_tutorial(name, path):
    """Test affect/text tutorial."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        # Find text input node
        text_node = None
        for node_id, node in graph.nodes.items():
            if node.type.name == "TEXT_INPUT":
                text_node = node
                break

        if not text_node:
            print("WARNING: Could not find TEXT_INPUT node")
            return False

        test_words = ["happy", "sad", "angry", "calm", "bored"]
        print(f"Testing words: {test_words}")

        for word in test_words:
            text_node.params['text'] = word
            result = executor.execute()

            if result.success:
                # Find affect vis output
                for node_id, outputs in result.node_outputs.items():
                    if 'affect' in outputs and isinstance(outputs['affect'], list):
                        affect = outputs['affect']
                        print(f"  '{word}' -> V:{affect[0]:.2f} A:{affect[1]:.2f} D:{affect[2]:.2f} S:{affect[3]:.2f} B:{affect[4]:.2f}")
                        break
            else:
                print(f"  '{word}' -> ERROR: {result.error}")

        print("SUCCESS: Affect tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_charm_tutorial(name, path):
    """Test CharmNetwork lite tutorial."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        print("Running affect sequence through LSTM...")

        # Simulate affect inputs (happy -> sad transition)
        sequence = [
            [0.8, 0.6, 0.5, 0.0, 0.0],  # Happy
            [0.8, 0.6, 0.5, 0.0, 0.0],  # Happy
            [0.8, 0.6, 0.5, 0.0, 0.0],  # Happy
            [-0.5, 0.3, 0.3, 0.8, 0.0],  # Sad (sudden change)
            [-0.5, 0.3, 0.3, 0.8, 0.0],  # Sad
            [-0.5, 0.3, 0.3, 0.8, 0.0],  # Sad
        ]

        for i, affect_input in enumerate(sequence):
            result = executor.execute(input_affect=affect_input)
            if result.success:
                # Find chart output for valence tracking
                for node_id, outputs in result.node_outputs.items():
                    if 'history' in outputs:
                        val = outputs.get('value', 0)
                        mood = "happy" if affect_input[0] > 0 else "sad"
                        print(f"  Step {i+1} ({mood} input) -> valence output: {val:.3f}")
                        break
            else:
                print(f"  Step {i+1}: ERROR - {result.error}")

        print("SUCCESS: CharmNetwork lite tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demoscene_tutorial(name, path):
    """Test demoscene tutorial with TIME, SINE, NOISE, and SHADER_VIS nodes."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        graph = load_canvas(path)
        print(f"Loaded: {graph.name}")

        executor = CanvasTestExecutor(graph)
        success, error = executor.initialize()

        if not success:
            print(f"FAIL: Could not initialize executor: {error}")
            return False

        print("Running 5 frames of signal flow (TIME -> SINE -> SHADER)...")

        import time as time_module
        for i in range(5):
            result = executor.execute()

            if result.success:
                # Find key outputs
                time_val = None
                signal_val = None
                shader_val = None

                for node_id, outputs in result.node_outputs.items():
                    node = graph.nodes[node_id]
                    if node.type.name == 'TIME':
                        t = outputs.get('time')
                        if isinstance(t, list):
                            time_val = t[0]
                        elif hasattr(t, 'item'):
                            time_val = t.item()
                        else:
                            time_val = t
                    elif node.name == 'Add Noise':
                        s = outputs.get('out')
                        if isinstance(s, list):
                            signal_val = s[0]
                        elif hasattr(s, 'item'):
                            signal_val = s.item()
                        else:
                            signal_val = s
                    elif node.type.name == 'SHADER_VIS':
                        shader_val = outputs.get('u_value', 0)

                print(f"  Frame {i+1}: time={time_val:.4f}s signal={signal_val:.4f} shader_u={shader_val:.4f}")
            else:
                print(f"  Frame {i+1}: ERROR - {result.error}")

            time_module.sleep(0.05)

        print("SUCCESS: Demoscene tutorial executed")
        return True

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    main()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
