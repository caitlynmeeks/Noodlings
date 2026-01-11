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
#   Neural Canvas Facet - Execute visual neural networks in cognitive cycles.
#
#   Bridges the Facets system (cognitive architecture) with NeuralCanvas
#   (visual neural network editor). Load .nncanvas files and run them
#   as facets in a FacetAssembly.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.neural_canvas_facet
# PURPOSE:  Neural Canvas Facet - Execute .nncanvas networks as facets
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NeuralCanvasFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import asyncio
import time
from typing import Dict, Any, Optional, List

from .neural_canvas.neural_graph import NeuralGraph
from .neural_canvas.test_executor import CanvasTestExecutor, TestResult


class NeuralCanvasFacet:
    """
    Facet that executes a NeuralGraph from a .nncanvas file.

    This bridges:
    - Facets system (cognitive architecture, execution cycles)
    - NeuralCanvas (visual neural network editor)

    The .nncanvas file defines the network architecture.
    This facet loads it and runs inference during cognition cycles.

    Usage in a FacetAssembly YAML:
        facets:
          - id: charm_processor
            name: "Charm Network"
            type: NeuralCanvasFacet
            nncanvas_path: "charm_networks/default.nncanvas"
            inputs:
              - name: affect_in
                type: input
            outputs:
              - name: affect_out
                type: output
    """

    def __init__(
        self,
        facet_id: str,
        name: str,
        nncanvas_path: str,
        project_root: Optional[str] = None
    ):
        """
        Initialize NeuralCanvasFacet.

        Args:
            facet_id: Unique identifier for this facet
            name: Display name
            nncanvas_path: Path to .nncanvas file (relative to project or absolute)
            project_root: Project root directory for resolving relative paths
        """
        self.id = facet_id
        self.name = name
        self.nncanvas_path = nncanvas_path
        self.project_root = project_root or os.getcwd()

        # Resolve full path
        if not os.path.isabs(nncanvas_path):
            self.full_path = os.path.join(self.project_root, nncanvas_path)
        else:
            self.full_path = nncanvas_path

        # Graph and executor
        self.graph: Optional[NeuralGraph] = None
        self.executor: Optional[CanvasTestExecutor] = None
        self._initialized = False

        # Execution statistics
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
        self._last_result: Optional[TestResult] = None

        # Execution lock for async safety
        self._execution_lock = asyncio.Lock()

        # Load graph if file exists
        self._load_graph()

    def _load_graph(self):
        """Load NeuralGraph from .nncanvas file."""
        if not self.nncanvas_path:
            print(f"[NeuralCanvasFacet] No nncanvas_path configured for '{self.name}'")
            return

        if not os.path.exists(self.full_path):
            print(f"[NeuralCanvasFacet] WARNING: File not found: {self.full_path}")
            return

        try:
            self.graph = NeuralGraph.from_json(self.full_path)
            self.executor = CanvasTestExecutor(self.graph)
            self._initialized = False  # Will init on first execute

            print(f"[NeuralCanvasFacet] Loaded: {self.nncanvas_path}")
            print(f"  Graph: '{self.graph.name}' - "
                  f"{len(self.graph.nodes)} nodes, "
                  f"{len(self.graph.connections)} connections, "
                  f"{self.graph.compute_total_parameters()} params")

        except Exception as e:
            print(f"[NeuralCanvasFacet] ERROR loading {self.full_path}: {e}")
            self.graph = None
            self.executor = None

    def reload_graph(self):
        """Reload graph from file (for hot reload during development)."""
        print(f"[NeuralCanvasFacet] Reloading: {self.nncanvas_path}")
        self._load_graph()
        self._initialized = False

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the neural network with given inputs.

        Args:
            inputs: Dict mapping input pad names to values.
                    For affect-based networks, expects 'affect' or 'input' key
                    with a 5-D affect vector [valence, arousal, dominance, sorrow, boredom]

        Returns:
            Dict mapping output pad names to computed values.
            On error, returns {'error': error_message}
        """
        async with self._execution_lock:
            start_time = time.time()

            if not self.executor:
                return {"error": f"No graph loaded from {self.nncanvas_path}"}

            # Initialize executor if needed
            if not self._initialized:
                success, error = self.executor.initialize()
                if not success:
                    return {"error": f"Failed to initialize graph: {error}"}
                self._initialized = True

            try:
                # Extract affect input from inputs dict
                affect_input = None

                # Try common input names
                for key in ['affect', 'input', 'affect_in', 'x', 'in']:
                    if key in inputs:
                        value = inputs[key]
                        # Convert to list if needed
                        if hasattr(value, 'tolist'):
                            affect_input = value.flatten().tolist()
                        elif isinstance(value, list):
                            affect_input = value
                        elif isinstance(value, (int, float)):
                            # Single value - wrap in neutral affect
                            affect_input = [float(value), 0.5, 0.5, 0.0, 0.0]
                        break

                # Run inference
                result = self.executor.execute(input_affect=affect_input)
                self._last_result = result

                # Track stats
                elapsed = time.time() - start_time
                self.execution_count += 1
                self.total_execution_time += elapsed
                self.last_execution_time = elapsed

                if not result.success:
                    return {"error": result.error or "Execution failed"}

                # Return outputs
                outputs = dict(result.outputs)
                outputs['_node_outputs'] = result.node_outputs
                outputs['_execution_time_ms'] = result.execution_time_ms

                return outputs

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"error": str(e)}

    def execute_sync(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute().

        Use this when calling from non-async code.
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.execute(inputs))
                return future.result()
        except RuntimeError:
            # No running loop - create one
            return asyncio.run(self.execute(inputs))

    def reset_states(self):
        """Reset all hidden states (between conversations)."""
        if self.executor:
            self.executor.reset_states()
            print(f"[NeuralCanvasFacet] States reset for '{self.name}'")

    def get_input_pads(self) -> List[str]:
        """Get list of input pad names from graph."""
        if not self.graph:
            return ["input"]  # Default

        input_nodes = self.graph.get_input_nodes()
        if input_nodes:
            # Return names of input nodes
            return [node.name.lower().replace(' ', '_') for node in input_nodes]

        return ["input"]

    def get_output_pads(self) -> List[str]:
        """Get list of output pad names from graph."""
        if not self.graph:
            return ["output"]  # Default

        output_nodes = self.graph.get_output_nodes()
        if output_nodes:
            return [node.name.lower().replace(' ', '_') for node in output_nodes]

        return ["output"]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,  # Neural network, not LLM
            'avg_tokens': 0,
            'total_time': self.total_execution_time,
            'avg_time': (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            ),
            'last_time': self.last_execution_time,
            'graph_path': self.nncanvas_path,
            'graph_loaded': self.graph is not None,
            'graph_name': self.graph.name if self.graph else None,
            'node_count': len(self.graph.nodes) if self.graph else 0,
            'param_count': self.graph.compute_total_parameters() if self.graph else 0
        }

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get token usage (always 0 for NeuralCanvasFacet - neural computation).

        Included for API consistency with LLM facets.
        """
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0
        }

    def get_last_result(self) -> Optional[TestResult]:
        """Get the last execution result."""
        return self._last_result

    def validate_graph(self) -> Dict[str, Any]:
        """
        Validate the loaded graph.

        Returns:
            Dict with 'valid' bool and 'errors'/'warnings' lists
        """
        if not self.graph:
            return {
                'valid': False,
                'errors': [f"No graph loaded from {self.nncanvas_path}"],
                'warnings': []
            }

        result = self.graph.validate()
        return {
            'valid': result.valid,
            'errors': result.errors,
            'warnings': result.warnings
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize facet configuration."""
        return {
            'id': self.id,
            'name': self.name,
            'type': 'NeuralCanvasFacet',
            'nncanvas_path': self.nncanvas_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], project_root: Optional[str] = None) -> 'NeuralCanvasFacet':
        """Deserialize from dict."""
        return cls(
            facet_id=data['id'],
            name=data['name'],
            nncanvas_path=data.get('nncanvas_path', ''),
            project_root=project_root
        )

    def __repr__(self) -> str:
        status = "loaded" if self.graph else "not loaded"
        return f"NeuralCanvasFacet('{self.name}', path='{self.nncanvas_path}', {status})"


# ═══════════════════════════════════════════════════════════
# Factory function for FacetExecutor
# ═══════════════════════════════════════════════════════════

def create_neural_canvas_facet(
    facet_id: str,
    name: str,
    nncanvas_path: str,
    project_root: Optional[str] = None
) -> NeuralCanvasFacet:
    """
    Factory function to create a NeuralCanvasFacet.

    Used by FacetExecutor when instantiating NeuralCanvasFacet type facets.
    """
    return NeuralCanvasFacet(
        facet_id=facet_id,
        name=name,
        nncanvas_path=nncanvas_path,
        project_root=project_root
    )


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Test NeuralCanvasFacet."""

    # Test with a sample .nncanvas file
    test_path = "../../../../test_data/sample.nncanvas"

    print("=== NeuralCanvasFacet Test ===\n")

    facet = NeuralCanvasFacet(
        facet_id="test_neural",
        name="Test Neural",
        nncanvas_path=test_path
    )

    print(f"Facet: {facet}")
    print(f"Stats: {facet.get_execution_stats()}")

    if facet.graph:
        print(f"\nValidation: {facet.validate_graph()}")

        # Test execution
        import asyncio
        result = asyncio.run(facet.execute({'affect': [0.5, 0.6, 0.4, 0.1, 0.2]}))
        print(f"\nExecution result: {result}")
    else:
        print("\nNo graph loaded - create a test .nncanvas file to test execution")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
