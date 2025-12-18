"""
Noodle API - Unified scriptable interface to Noodlings systems.

Main entry point for ScriptedFacet access to:
- Models/providers (context.noodle.models)
- Neural Canvas (context.noodle.neural)
- Agent facet assemblies (context.noodle.agents)
- UUID-based entity lookup (context.noodle.get_by_uuid)

This API extends the existing ScriptContext with system configuration
capabilities, enabling ScriptedFacets to programmatically modify the
entire Noodlings architecture.

Author: Commander Spock + Cadet Caity
Date: December 10, 2025
"""

from typing import Dict, Any, Optional
from .models_api import ModelsAPI
from .neural_api import NeuralAPI
from .agents_api import AgentsAPI
from .quantum_api import QuantumAPI


class NoodleAPI:
    """
    Unified Noodlings scripting API.

    Provides JavaScript-accessible interface to all system components.
    Available in ScriptedFacets via context.noodle

    Example (JavaScript in ScriptedFacet):
        function process(inputs, context) {
            // Change model for LARGE label
            context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");

            // Modify neural topology
            var network = context.noodle.neural.get_network("default");
            var lstm = network.create_node("LSTM", {hidden_dim: 64});

            // Reconfigure facet assembly
            var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
            var facet = assembly.get_facet("CHARM_NET");
            facet.set_property("model", "LARGE");

            return {modified: true};
        }
    """

    def __init__(
        self,
        model_label_manager=None,
        provider_manager=None
    ):
        """
        Initialize Noodle API.

        Args:
            model_label_manager: ModelLabelManager instance (optional, lazy init)
            provider_manager: ProviderManager instance (optional, lazy init)
        """
        # Sub-API instances
        self._models_api = None
        self._neural_api = NeuralAPI()
        self._agents_api = AgentsAPI()
        self._quantum_api = QuantumAPI()

        # Manager references (lazy initialization)
        self._model_label_manager = model_label_manager
        self._provider_manager = provider_manager

        # UUID registry (future enhancement)
        self._uuid_registry: Dict[str, Any] = {}

    @property
    def models(self) -> ModelsAPI:
        """
        Access Models API.

        Returns:
            ModelsAPI instance

        Example (JavaScript):
            var assignment = context.noodle.models.get_label("SMALL");
        """
        if self._models_api is None:
            # Lazy initialization
            if self._model_label_manager is None:
                from noodlestudio.core.model_label_manager import get_model_label_manager
                self._model_label_manager = get_model_label_manager()

            if self._provider_manager is None:
                from noodlestudio.core.provider_manager import ProviderManager
                self._provider_manager = ProviderManager()

            self._models_api = ModelsAPI(
                self._model_label_manager,
                self._provider_manager
            )

        return self._models_api

    @property
    def neural(self) -> NeuralAPI:
        """
        Access Neural Canvas API.

        Returns:
            NeuralAPI instance

        Example (JavaScript):
            var network = context.noodle.neural.create_network("MyNet");
        """
        return self._neural_api

    @property
    def agents(self) -> AgentsAPI:
        """
        Access Agents API.

        Returns:
            AgentsAPI instance

        Example (JavaScript):
            var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
        """
        return self._agents_api

    @property
    def quantum(self) -> QuantumAPI:
        """
        Access Quantum API.

        Provides quantum computation simulation for ScriptedFacets:
        - Qubit measurement with simulated quantum randomness
        - Schrodinger's Cat experiment helper
        - Neural Canvas quantum node execution
        - Entanglement simulation

        Returns:
            QuantumAPI instance

        Example (JavaScript):
            // Measure a qubit
            var q = context.noodle.quantum.measure_qubit();

            // Schrodinger's Cat experiment
            var cat = context.noodle.quantum.schrodingers_cat();
            if (cat.is_alive) {
                console.log("The cat lives!");
            }

            // Execute a quantum canvas
            var result = context.noodle.quantum.execute_canvas(
                "tutorials/08_schrodingers_cat.nncanvas"
            );
        """
        return self._quantum_api

    def get_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get any entity by UUID (future enhancement).

        Args:
            uuid: Universal unique identifier

        Returns:
            Entity dict with {type, properties, methods} or None

        Example (JavaScript):
            var entity = context.noodle.get_by_uuid("550e8400-...");
            console.log(entity.type);  // "LLMFacet", "LSTMNode", etc.
        """
        # TODO: Implement UUID registry
        return self._uuid_registry.get(uuid)

    def register_entity(self, uuid: str, entity: Any):
        """
        Register entity in UUID registry (internal use).

        Args:
            uuid: Entity UUID
            entity: Entity object
        """
        self._uuid_registry[uuid] = entity

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with sub-API method names as placeholders

        This is used by ScriptContext to inject the API into JavaScript.
        """
        return {
            'models': self.models.to_dict(),
            'neural': self.neural.to_dict(),
            'agents': self.agents.to_dict(),
            'quantum': self.quantum.to_dict(),
            'get_by_uuid': '__noodle_get_by_uuid__'
        }


# Global singleton instance
_noodle_api_instance = None


def get_noodle_api(
    model_label_manager=None,
    provider_manager=None
) -> NoodleAPI:
    """
    Get global NoodleAPI singleton.

    Args:
        model_label_manager: ModelLabelManager (optional)
        provider_manager: ProviderManager (optional)

    Returns:
        NoodleAPI instance
    """
    global _noodle_api_instance

    if _noodle_api_instance is None:
        _noodle_api_instance = NoodleAPI(
            model_label_manager=model_label_manager,
            provider_manager=provider_manager
        )

    return _noodle_api_instance
