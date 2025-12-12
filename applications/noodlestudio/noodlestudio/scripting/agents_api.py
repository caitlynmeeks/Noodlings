"""
Agents API - Scriptable interface to agent facet assemblies.

Provides JavaScript-accessible methods for:
- Getting agent facet assemblies
- Modifying facets
- Connecting facets
- Setting facet properties
- Saving assemblies

Part of the unified Noodlings scripting API (context.noodle.agents).

Author: Commander Spock + Cadet Caity
Date: December 10, 2025
"""

from typing import Dict, List, Optional, Any


class FacetProxy:
    """
    Proxy object for a single facet.

    Provides JavaScript-friendly interface to Facet.
    """

    def __init__(self, facet, assembly):
        """
        Initialize facet proxy.

        Args:
            facet: Facet instance
            assembly: Parent FacetAssembly
        """
        self._facet = facet
        self._assembly = assembly

    def get_property(self, name: str) -> Any:
        """
        Get facet property value.

        Args:
            name: Property name (e.g., "model", "temperature", "prompt")

        Returns:
            Property value or None

        Example (JavaScript):
            var model = facet.get_property("model");  // "LARGE"
        """
        return self._facet.properties.get(name)

    def set_property(self, name: str, value: Any) -> bool:
        """
        Set facet property value.

        Args:
            name: Property name
            value: New value

        Returns:
            True if set successfully

        Example (JavaScript):
            facet.set_property("model", "LARGE");
            facet.set_property("temperature", 0.9);
        """
        try:
            self._facet.properties[name] = value
            return True
        except:
            return False

    def get_all_properties(self) -> Dict[str, Any]:
        """
        Get all facet properties.

        Returns:
            Dict of {property_name: value}

        Example (JavaScript):
            var props = facet.get_all_properties();
            console.log(props.model);      // "LARGE"
            console.log(props.temperature); // 0.7
        """
        return dict(self._facet.properties)

    def get_type(self) -> str:
        """
        Get facet type.

        Returns:
            Type name (e.g., "LLMFacet", "ScriptedFacet", "CharmNetworkFacet")
        """
        return self._facet.facet_type

    def get_id(self) -> str:
        """
        Get facet ID.

        Returns:
            Unique facet identifier
        """
        return self._facet.id

    def get_name(self) -> str:
        """
        Get facet name.

        Returns:
            Human-readable facet name
        """
        return self._facet.name


class FacetAssemblyProxy:
    """
    Proxy object for a facet assembly.

    Provides JavaScript-friendly interface to FacetAssembly.
    """

    def __init__(self, assembly):
        """
        Initialize assembly proxy.

        Args:
            assembly: FacetAssembly instance
        """
        self._assembly = assembly

    def get_facet(self, facet_id: str) -> Optional[FacetProxy]:
        """
        Get facet by ID.

        Args:
            facet_id: Facet ID

        Returns:
            FacetProxy or None

        Example (JavaScript):
            var facet = assembly.get_facet("CHARM_NET");
            facet.set_property("model", "LARGE");
        """
        for facet in self._assembly.facets:
            if facet.id == facet_id:
                return FacetProxy(facet, self._assembly)
        return None

    def get_facet_by_name(self, name: str) -> Optional[FacetProxy]:
        """
        Get facet by name.

        Args:
            name: Facet name (e.g., "Red's Mind")

        Returns:
            FacetProxy or None
        """
        for facet in self._assembly.facets:
            if facet.name == name:
                return FacetProxy(facet, self._assembly)
        return None

    def list_facets(self) -> List[Dict[str, str]]:
        """
        List all facets in assembly.

        Returns:
            List of {id, name, type} dicts

        Example (JavaScript):
            var facets = assembly.list_facets();
            // [
            //   {id: "CHARM_NET", name: "CharmNetwork", type: "CharmNetworkFacet"},
            //   {id: "MIND", name: "Red's Mind", type: "LLMFacet"}
            // ]
        """
        result = []
        for facet in self._assembly.facets:
            result.append({
                'id': facet.id,
                'name': facet.name,
                'type': facet.facet_type
            })
        return result

    def add_facet(self, facet_type: str, name: str, **properties) -> Optional[str]:
        """
        Add new facet to assembly.

        Args:
            facet_type: Facet type (e.g., "LLMFacet", "ScriptedFacet")
            name: Facet name
            **properties: Initial properties

        Returns:
            Facet ID if created, None on failure

        Example (JavaScript):
            var facet_id = assembly.add_facet("LLMFacet", "Custom Reasoner", {
                model: "LARGE",
                temperature: 0.8
            });
        """
        try:
            from noodlestudio.core.facet_system import Facet
            import uuid

            facet_id = f"CUSTOM_{name.upper().replace(' ', '_')}"
            facet = Facet(
                id=facet_id,
                name=name,
                facet_type=facet_type,
                properties=dict(properties)
            )

            self._assembly.facets.append(facet)
            return facet_id
        except:
            return None

    def remove_facet(self, facet_id: str) -> bool:
        """
        Remove facet from assembly.

        Args:
            facet_id: Facet ID to remove

        Returns:
            True if removed successfully
        """
        try:
            self._assembly.facets = [
                f for f in self._assembly.facets
                if f.id != facet_id
            ]
            # Also remove connections involving this facet
            self._assembly.connections = [
                c for c in self._assembly.connections
                if c.from_facet != facet_id and c.to_facet != facet_id
            ]
            return True
        except:
            return False

    def connect(self, from_facet: str, from_pad: str, to_facet: str, to_pad: str) -> bool:
        """
        Connect two facets.

        Args:
            from_facet: Source facet ID
            from_pad: Source pad name
            to_facet: Target facet ID
            to_pad: Target pad name

        Returns:
            True if connected successfully

        Example (JavaScript):
            assembly.connect("CHARM_NET", "affect_valence", "MIND", "affect");
        """
        try:
            from noodlestudio.core.facet_system import FacetConnection

            conn = FacetConnection(
                from_facet=from_facet,
                from_pad=from_pad,
                to_facet=to_facet,
                to_pad=to_pad
            )
            self._assembly.connections.append(conn)
            return True
        except:
            return False

    def disconnect(self, from_facet: str, from_pad: str, to_facet: str, to_pad: str) -> bool:
        """
        Disconnect two facets.

        Args:
            from_facet: Source facet ID
            from_pad: Source pad name
            to_facet: Target facet ID
            to_pad: Target pad name

        Returns:
            True if disconnected successfully
        """
        try:
            self._assembly.connections = [
                c for c in self._assembly.connections
                if not (c.from_facet == from_facet and
                       c.from_pad == from_pad and
                       c.to_facet == to_facet and
                       c.to_pad == to_pad)
            ]
            return True
        except:
            return False

    def save(self, filepath: str) -> bool:
        """
        Save assembly to YAML file.

        Args:
            filepath: Path to save file

        Returns:
            True if saved successfully

        Example (JavaScript):
            assembly.save("modified_red.yaml");
        """
        try:
            self._assembly.save_to_file(filepath)
            return True
        except:
            return False


class AgentsAPI:
    """
    Scriptable interface to agent facet assemblies.

    Available to JavaScript via context.noodle.agents
    """

    def __init__(self):
        """Initialize Agents API."""
        self._assemblies: Dict[str, Any] = {}  # agent_id -> FacetAssembly

    def get(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent info by ID.

        Args:
            agent_id: Agent ID (e.g., "red-fire-anklebiter")

        Returns:
            Dict with {id, name, species, assembly} or None

        Example (JavaScript):
            var agent = context.noodle.agents.get("red-fire-anklebiter");
            var assembly = agent.assembly;
        """
        if agent_id in self._assemblies:
            return {
                'id': agent_id,
                'assembly': FacetAssemblyProxy(self._assemblies[agent_id])
            }
        return None

    def get_assembly(self, agent_id: str) -> Optional[FacetAssemblyProxy]:
        """
        Get facet assembly for agent.

        Args:
            agent_id: Agent ID

        Returns:
            FacetAssemblyProxy or None

        Example (JavaScript):
            var assembly = context.noodle.agents.get_assembly("red-fire-anklebiter");
            var charm_facet = assembly.get_facet("CHARM_NET");
            charm_facet.set_property("model", "LARGE");
        """
        if agent_id in self._assemblies:
            return FacetAssemblyProxy(self._assemblies[agent_id])
        return None

    def load_assembly(self, agent_id: str, filepath: str) -> Optional[FacetAssemblyProxy]:
        """
        Load assembly from YAML file.

        Args:
            agent_id: Agent ID to register assembly under
            filepath: Path to .yaml file

        Returns:
            FacetAssemblyProxy or None

        Example (JavaScript):
            var assembly = context.noodle.agents.load_assembly("test", "test.yaml");
        """
        try:
            from noodlestudio.core.facet_system import FacetAssembly
            assembly = FacetAssembly.load_from_file(filepath)
            self._assemblies[agent_id] = assembly
            return FacetAssemblyProxy(assembly)
        except:
            return None

    def register_assembly(self, agent_id: str, assembly) -> bool:
        """
        Register an existing assembly (internal use).

        Args:
            agent_id: Agent ID
            assembly: FacetAssembly instance

        Returns:
            True if registered
        """
        self._assemblies[agent_id] = assembly
        return True

    def list_all(self) -> List[str]:
        """
        List all registered agent IDs.

        Returns:
            List of agent IDs

        Example (JavaScript):
            var agents = context.noodle.agents.list_all();
            // ["red-fire-anklebiter", "test-agent"]
        """
        return list(self._assemblies.keys())

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with method names as keys
        """
        return {
            'get': '__agents_get__',
            'get_assembly': '__agents_get_assembly__',
            'load_assembly': '__agents_load_assembly__',
            'list_all': '__agents_list_all__'
        }
