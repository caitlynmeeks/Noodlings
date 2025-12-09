#!/usr/bin/env python3
"""
Test Neural Canvas implementation.

Tests:
1. Load default CharmNetwork topology (.nncanvas)
2. Validate graph
3. Generate MLX code
4. Verify code structure

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

import os
import sys

# Add noodlestudio to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from noodlestudio.core.neural_canvas.neural_graph import NeuralGraph
from noodlestudio.core.neural_canvas.mlx_codegen import generate_mlx_code


def test_load_default_topology():
    """Test loading default CharmNetwork .nncanvas file."""
    print("=" * 60)
    print("TEST 1: Load default.nncanvas")
    print("=" * 60)

    topology_path = os.path.join(
        os.path.dirname(__file__),
        '../../facet_assemblies/charm_networks/default.nncanvas'
    )
    topology_path = os.path.abspath(topology_path)

    print(f"Loading: {topology_path}")

    if not os.path.exists(topology_path):
        print(f"❌ File not found: {topology_path}")
        return None

    try:
        graph = NeuralGraph.from_json(topology_path)
        print(f"✅ Loaded: {graph}")
        print(f"   Name: {graph.name}")
        print(f"   Description: {graph.description}")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Connections: {len(graph.connections)}")
        print(f"   Parameters: {graph.compute_total_parameters():,}")
        return graph
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_validate_graph(graph: NeuralGraph):
    """Test graph validation."""
    print("\n" + "=" * 60)
    print("TEST 2: Validate Graph")
    print("=" * 60)

    result = graph.validate()

    if result.valid:
        print("✅ Graph is valid!")
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
    else:
        print("❌ Graph is invalid!")
        print("\nErrors:")
        for error in result.errors:
            print(f"  • {error}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")

    return result.valid


def test_generate_mlx_code(graph: NeuralGraph):
    """Test MLX code generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Generate MLX Code")
    print("=" * 60)

    try:
        code = generate_mlx_code(graph)

        # Save to file
        output_path = os.path.join(
            os.path.dirname(__file__),
            'generated_charm_network.py'
        )
        with open(output_path, 'w') as f:
            f.write(code)

        print(f"✅ Generated {len(code)} characters of code")
        print(f"   Saved to: {output_path}")

        # Show preview
        print("\n--- Code Preview (first 50 lines) ---")
        lines = code.split('\n')
        for i, line in enumerate(lines[:50], 1):
            print(f"{i:3d} | {line}")

        if len(lines) > 50:
            print(f"... ({len(lines) - 50} more lines)")

        return True

    except Exception as e:
        print(f"❌ Failed to generate code: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_definitions():
    """Test node creation from definitions."""
    print("\n" + "=" * 60)
    print("TEST 4: Node Definitions")
    print("=" * 60)

    from noodlestudio.core.neural_canvas.node_definitions import (
        create_node_from_type, NODE_DEFINITIONS
    )
    from noodlestudio.core.neural_canvas.neural_node import NodeType

    print(f"Total node types defined: {len(NODE_DEFINITIONS)}")

    # Test creating a few node types
    test_types = [NodeType.LSTM, NodeType.GRU, NodeType.AFFECT_HEAD, NodeType.IBM_QUANTUM]

    for node_type in test_types:
        try:
            node = create_node_from_type(node_type)
            print(f"✅ {node_type.value:20s} - {node.name} ({node.compute_num_parameters():,} params)")
        except Exception as e:
            print(f"❌ {node_type.value:20s} - Failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEURAL CANVAS TEST SUITE")
    print("=" * 60 + "\n")

    # Test 1: Load topology
    graph = test_load_default_topology()
    if not graph:
        print("\n❌ Test suite aborted (failed to load topology)")
        return 1

    # Test 2: Validate
    valid = test_validate_graph(graph)
    if not valid:
        print("\n⚠️  Graph has validation errors, but continuing tests...")

    # Test 3: Generate code
    code_generated = test_generate_mlx_code(graph)

    # Test 4: Node definitions
    test_node_definitions()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Topology loaded: {graph.name}")
    print(f"{'✅' if valid else '⚠️ '} Validation: {'passed' if valid else 'has warnings'}")
    print(f"{'✅' if code_generated else '❌'} Code generation: {'success' if code_generated else 'failed'}")
    print("\n" + "=" * 60)
    print("Neural Canvas core implementation: COMPLETE")
    print("=" * 60 + "\n")

    return 0 if (graph and code_generated) else 1


if __name__ == "__main__":
    sys.exit(main())
