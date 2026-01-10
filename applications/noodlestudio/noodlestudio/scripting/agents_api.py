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
#   Agents API - Scriptable interface to agent facet assemblies.
#
#   Provides JavaScript-accessible methods for: - Getting age...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.agents_api
# PURPOSE:  Agents Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetProxy, FacetAssemblyProxy, AgentsAPI
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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
        except Exception:
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

    def get_prompt(self) -> str:
        """Get facet prompt."""
        return self._facet.prompt

    def set_prompt(self, prompt: str) -> bool:
        """Set facet prompt."""
        try:
            self._facet.prompt = prompt
            return True
        except Exception:
            return False

    def get_model(self) -> str:
        """Get LLM model label (SMALL/MEDIUM/LARGE)."""
        return self._facet.model

    def set_model(self, model: str) -> bool:
        """Set LLM model label."""
        try:
            self._facet.model = model
            return True
        except Exception:
            return False

    def get_temperature(self) -> float:
        """Get sampling temperature."""
        return self._facet.temperature

    def set_temperature(self, temp: float) -> bool:
        """Set sampling temperature."""
        try:
            self._facet.temperature = temp
            return True
        except Exception:
            return False

    def get_position(self) -> Dict[str, float]:
        """Get visual position {x, y}."""
        return dict(self._facet.position)

    def set_position(self, x: float, y: float) -> bool:
        """Set visual position."""
        try:
            self._facet.position = {'x': x, 'y': y}
            return True
        except Exception:
            return False

    def get_inputs(self) -> List[Dict[str, Any]]:
        """
        Get all input pads.

        Returns:
            List of {name, description, required, connected_to}

        Example (JavaScript):
            var inputs = facet.get_inputs();
            // [{name: "context", description: "...", required: true, connected_to: ["CHARM_NET.out"]}]
        """
        result = []
        for pad in self._facet.input_pads:
            result.append({
                'name': pad.name,
                'description': pad.description,
                'required': pad.required,
                'connected_to': list(pad.connected_to)
            })
        return result

    def get_outputs(self) -> List[Dict[str, Any]]:
        """
        Get all output pads.

        Returns:
            List of {name, description, connected_to}
        """
        result = []
        for pad in self._facet.output_pads:
            result.append({
                'name': pad.name,
                'description': pad.description,
                'connected_to': list(pad.connected_to)
            })
        return result

    def is_enabled(self) -> bool:
        """Check if facet is enabled."""
        return getattr(self._facet, 'enabled', True)

    def set_enabled(self, enabled: bool) -> bool:
        """Enable or disable facet (skipped during execution if disabled)."""
        try:
            self._facet.enabled = enabled
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full facet data as dictionary.

        Returns:
            Complete facet representation

        Example (JavaScript):
            var data = facet.to_dict();
            console.log(JSON.stringify(data, null, 2));
        """
        return self._facet.to_dict()


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

    def get_facets_by_type(self, facet_type: str) -> List[FacetProxy]:
        """
        Get all facets of a specific type (like Unity's GetComponents<T>).

        Args:
            facet_type: Type name (e.g., "LLMFacet", "ScriptedFacet")

        Returns:
            List of FacetProxy objects

        Example (JavaScript):
            var llm_facets = assembly.get_facets_by_type("LLMFacet");
            llm_facets.forEach(function(f) {
                f.set_temperature(0.9);  // Set all LLM facets to same temp
            });
        """
        result = []
        for facet in self._assembly.facets:
            if facet.facet_type == facet_type:
                result.append(FacetProxy(facet, self._assembly))
        return result

    def find_facets(self, predicate: dict) -> List[FacetProxy]:
        """
        Find facets matching criteria.

        Args:
            predicate: Dict of field->value to match
                      Supports: type, name, model, enabled

        Returns:
            List of matching FacetProxy objects

        Example (JavaScript):
            var large_llms = assembly.find_facets({type: "LLMFacet", model: "LARGE"});
        """
        result = []
        for facet in self._assembly.facets:
            match = True
            if 'type' in predicate and facet.facet_type != predicate['type']:
                match = False
            if 'name' in predicate and facet.name != predicate['name']:
                match = False
            if 'model' in predicate and facet.model != predicate['model']:
                match = False
            if 'enabled' in predicate:
                enabled = getattr(facet, 'enabled', True)
                if enabled != predicate['enabled']:
                    match = False
            if match:
                result.append(FacetProxy(facet, self._assembly))
        return result

    def get_connections(self) -> List[Dict[str, str]]:
        """
        Get all connections in the assembly.

        Returns:
            List of {from_facet, from_pad, to_facet, to_pad}

        Example (JavaScript):
            var conns = assembly.get_connections();
            conns.forEach(function(c) {
                console.log(c.from_facet + "." + c.from_pad + " -> " +
                           c.to_facet + "." + c.to_pad);
            });
        """
        result = []
        for conn in self._assembly.connections:
            result.append({
                'from_facet': conn.from_facet,
                'from_pad': conn.from_pad,
                'to_facet': conn.to_facet,
                'to_pad': conn.to_pad
            })
        return result

    def get_connections_from(self, facet_id: str) -> List[Dict[str, str]]:
        """
        Get all connections originating from a facet.

        Args:
            facet_id: Source facet ID

        Returns:
            List of connection dicts
        """
        result = []
        for conn in self._assembly.connections:
            if conn.from_facet == facet_id:
                result.append({
                    'from_facet': conn.from_facet,
                    'from_pad': conn.from_pad,
                    'to_facet': conn.to_facet,
                    'to_pad': conn.to_pad
                })
        return result

    def get_connections_to(self, facet_id: str) -> List[Dict[str, str]]:
        """
        Get all connections going to a facet.

        Args:
            facet_id: Target facet ID

        Returns:
            List of connection dicts
        """
        result = []
        for conn in self._assembly.connections:
            if conn.to_facet == facet_id:
                result.append({
                    'from_facet': conn.from_facet,
                    'from_pad': conn.from_pad,
                    'to_facet': conn.to_facet,
                    'to_pad': conn.to_pad
                })
        return result

    def duplicate_facet(self, facet_id: str, new_name: Optional[str] = None) -> Optional[str]:
        """
        Duplicate a facet (like Unity's Instantiate).

        Args:
            facet_id: ID of facet to clone
            new_name: Optional name for the clone

        Returns:
            New facet ID or None

        Example (JavaScript):
            var clone_id = assembly.duplicate_facet("MIND", "Red's Backup Mind");
        """
        try:
            from noodlestudio.core.facet_system import Facet, FacetPad, PadType
            import uuid

            # Find original
            original = None
            for facet in self._assembly.facets:
                if facet.id == facet_id:
                    original = facet
                    break

            if not original:
                return None

            # Create clone with new ID and position offset
            new_id = f"{original.id}_CLONE_{str(uuid.uuid4())[:8]}"
            clone = Facet(
                id=new_id,
                name=new_name or f"{original.name} (Clone)",
                facet_type=original.facet_type,
                prompt=original.prompt,
                model=original.model,
                temperature=original.temperature,
                max_tokens=original.max_tokens,
                position={'x': original.position.get('x', 0) + 50, 'y': original.position.get('y', 0) + 50}
            )

            # Copy pads
            clone.input_pads = [
                FacetPad(name=p.name, pad_type=PadType.INPUT, description=p.description)
                for p in original.input_pads
            ]
            clone.output_pads = [
                FacetPad(name=p.name, pad_type=PadType.OUTPUT, description=p.description)
                for p in original.output_pads
            ]

            self._assembly.facets.append(clone)
            return new_id
        except Exception as e:
            print(f"[AgentsAPI] duplicate_facet error: {e}")
            return None

    def get_facet_count(self) -> int:
        """Get total number of facets."""
        return len(self._assembly.facets)

    def get_connection_count(self) -> int:
        """Get total number of connections."""
        return len(self._assembly.connections)

    def get_incoming(self) -> Optional[FacetProxy]:
        """Get the INCOMING facet (entry point)."""
        for facet in self._assembly.facets:
            if facet.id == 'incoming' or facet.name == 'INCOMING':
                return FacetProxy(facet, self._assembly)
        return None

    def get_outgoing(self) -> Optional[FacetProxy]:
        """Get the OUTGOING facet (exit point)."""
        for facet in self._assembly.facets:
            if facet.id == 'outgoing' or facet.name == 'OUTGOING':
                return FacetProxy(facet, self._assembly)
        return None

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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
