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
#   Facet System - Node-based cognitive architecture
#
#   A facet is a discrete cognitive transformation - like a l...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.facet_system
# PURPOSE:  Facet System - Node-based cognitive architecture
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PadType, FacetPad, FacetConnection, Facet, FacetAssembly
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import uuid
import time

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# Import ChannelsConfig for assembly channel configuration
try:
    from ..runtime.channels import ChannelsConfig
    CHANNELS_AVAILABLE = True
except ImportError:
    CHANNELS_AVAILABLE = False
    ChannelsConfig = None


class PadType(Enum):
    """Type of connection pad."""
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class FacetPad:
    """Connection point on a facet."""
    name: str                    # Pad identifier (e.g., "context", "awareness", "affect")
    pad_type: PadType           # INPUT or OUTPUT
    description: str = ""        # Human-readable description
    required: bool = True        # Is this pad required to be connected?

    # Runtime connection tracking
    connected_to: List[str] = field(default_factory=list)  # List of "facet_id.pad_name"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'type': self.pad_type.value,
            'description': self.description,
            'required': self.required
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FacetPad':
        """Deserialize from dictionary."""
        return FacetPad(
            name=data['name'],
            pad_type=PadType(data['type']),
            description=data.get('description', ''),
            required=data.get('required', True)
        )


@dataclass
class FacetConnection:
    """Connection between two facet pads."""
    from_facet: str      # Source facet ID
    from_pad: str        # Source pad name
    to_facet: str        # Destination facet ID
    to_pad: str          # Destination pad name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'from': f"{self.from_facet}.{self.from_pad}",
            'to': f"{self.to_facet}.{self.to_pad}"
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FacetConnection':
        """Deserialize from dictionary."""
        from_parts = data['from'].split('.')
        to_parts = data['to'].split('.')
        return FacetConnection(
            from_facet=from_parts[0],
            from_pad='.'.join(from_parts[1:]),
            to_facet=to_parts[0],
            to_pad='.'.join(to_parts[1:])
        )


@dataclass
class Facet:
    """
    A single cognitive facet - transforms information like a lens.

    Each facet:
    - Has a unique UUID
    - Contains a prompt that defines its transformation
    - Has input and output pads for connections
    - Can be positioned visually in the editor
    - Tracks token usage and execution statistics
    """
    id: str                              # Unique UUID identifier
    name: str                            # Display name
    facet_type: str                      # Type (IntuitionFacet, EmotionFacet, etc.)
    prompt: str                          # LLM prompt defining transformation

    # LLM parameters
    model: str = "SMALL"   # Which LLM label to use (SMALL/MEDIUM/LARGE)
    temperature: float = 0.7             # Sampling temperature
    max_tokens: int = 150                # Max output length
    delivery: str = "buffered"           # Delivery mode: buffered, stream_animated, stream_raw
    model_prefix: Optional[str] = None  # Per-facet prefix override (None = use label default)

    # Scripting API - Dynamic salience control
    salience_script: Optional[str] = None  # JavaScript code for continuous salience computation

    # NeuralCanvasFacet - Path to .nncanvas file (for NeuralCanvasFacet type only)
    nncanvas_path: Optional[str] = None  # Path to .nncanvas file (relative to project)

    # Connection pads
    input_pads: List[FacetPad] = field(default_factory=list)
    output_pads: List[FacetPad] = field(default_factory=list)

    # Visual editor metadata
    position: Dict[str, float] = field(default_factory=lambda: {'x': 0, 'y': 0})
    color: str = "#64B5F6"              # Visual color (for editor only)

    # Runtime state
    enabled: bool = True
    locked: bool = False                # If True, facet cannot be edited

    # Execution statistics (not serialized to YAML)
    _execution_count: int = field(default=0, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _total_execution_time: float = field(default=0.0, repr=False)
    _last_token_count: int = field(default=0, repr=False)
    _last_execution_time: float = field(default=0.0, repr=False)
    _last_output: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML export."""
        result = {
            'id': self.id,
            'name': self.name,
            'type': self.facet_type,
            'prompt': self.prompt,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'inputs': [pad.to_dict() for pad in self.input_pads],
            'outputs': [pad.to_dict() for pad in self.output_pads],
            'position': self.position,
            'color': self.color,
            'enabled': self.enabled,
            'locked': self.locked
        }
        # Only include salience_script if it exists
        if self.salience_script:
            result['salience_script'] = self.salience_script
        # Include nncanvas_path for NeuralCanvasFacet types
        if self.nncanvas_path:
            result['nncanvas_path'] = self.nncanvas_path
        # Only include delivery when non-default (keeps existing YAMLs clean)
        if self.delivery and self.delivery != "buffered":
            result['delivery'] = self.delivery
        # Only include model_prefix when explicitly set (None = use label default)
        if self.model_prefix is not None:
            result['model_prefix'] = self.model_prefix
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Facet':
        """Deserialize from dictionary."""
        return Facet(
            id=data['id'],
            name=data['name'],
            facet_type=data['type'],
            prompt=data.get('prompt', ''),  # NeuralCanvasFacet may not have prompt
            model=data.get('model', 'SMALL'),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 150),
            delivery=data.get('delivery', 'buffered'),
            model_prefix=data.get('model_prefix'),  # None = use label default
            salience_script=data.get('salience_script'),  # Load salience script if present
            nncanvas_path=data.get('nncanvas_path'),  # Load nncanvas_path for NeuralCanvasFacet
            input_pads=[FacetPad.from_dict(p) for p in data.get('inputs', [])],
            output_pads=[FacetPad.from_dict(p) for p in data.get('outputs', [])],
            position=data.get('position', {'x': 0, 'y': 0}),
            color=data.get('color', '#64B5F6'),
            enabled=data.get('enabled', True),
            locked=data.get('locked', False)
        )

    def add_input_pad(self, name: str, description: str = "", required: bool = True):
        """Add an input pad to this facet."""
        self.input_pads.append(FacetPad(
            name=name,
            pad_type=PadType.INPUT,
            description=description,
            required=required
        ))

    def add_output_pad(self, name: str, description: str = ""):
        """Add an output pad to this facet."""
        self.output_pads.append(FacetPad(
            name=name,
            pad_type=PadType.OUTPUT,
            description=description
        ))

    def record_execution(self, token_count: int, execution_time: float, outputs: Dict[str, Any]):
        """
        Record execution statistics for this facet.

        Args:
            token_count: Number of tokens used by LLM (0 for non-LLM facets)
            execution_time: Time taken in seconds
            outputs: Output values generated
        """
        self._execution_count += 1
        self._total_tokens += token_count
        self._total_execution_time += execution_time
        self._last_token_count = token_count
        self._last_execution_time = execution_time
        self._last_output = outputs

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            'last_tokens': self._last_token_count,
            'total_tokens': self._total_tokens,
            'execution_count': self._execution_count,
            'avg_tokens': (
                self._total_tokens / self._execution_count
                if self._execution_count > 0 else 0
            )
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        return {
            'execution_count': self._execution_count,
            'total_tokens': self._total_tokens,
            'avg_tokens': (
                self._total_tokens / self._execution_count
                if self._execution_count > 0 else 0
            ),
            'total_time': self._total_execution_time,
            'avg_time': (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0 else 0
            ),
            'last_tokens': self._last_token_count,
            'last_time': self._last_execution_time
        }

    def get_last_output(self) -> Optional[Dict[str, Any]]:
        """Get last output values."""
        return self._last_output

    def reset_stats(self):
        """Reset execution statistics."""
        self._execution_count = 0
        self._total_tokens = 0
        self._total_execution_time = 0.0
        self._last_token_count = 0
        self._last_execution_time = 0.0
        self._last_output = None

    @staticmethod
    def generate_uuid() -> str:
        """Generate a new UUID for facet ID."""
        return str(uuid.uuid4())

    def get_editable_fields(self, cognition_paused: bool = False) -> List[Dict[str, Any]]:
        """
        Return list of fields that can be edited in the UI.

        Args:
            cognition_paused: If True, output fields become editable

        Each field has:
        - name: Display name
        - key: Property key
        - value: Current value
        - type: 'text', 'number', 'dropdown'
        - read_only: Boolean
        - preview: Short preview text (first ~50 chars for display in node)
        """
        fields = []

        # Input field (last received input - read-only)
        if self._last_output:
            input_preview = str(self._last_output.get('in', ''))[:50]
            if len(str(self._last_output.get('in', ''))) > 50:
                input_preview += '...'
            fields.append({
                'name': 'Input',
                'key': 'last_input',
                'value': str(self._last_output.get('in', '')),
                'type': 'text',
                'read_only': True,
                'preview': input_preview
            })

        # Processing prompt (editable)
        prompt_preview = self.prompt[:50]
        if len(self.prompt) > 50:
            prompt_preview += '...'
        fields.append({
            'name': 'Processing Prompt',
            'key': 'prompt',
            'value': self.prompt,
            'type': 'text',
            'read_only': False,
            'preview': prompt_preview
        })

        # Output field (read-only by default, editable when cognition paused)
        if self._last_output:
            output_str = str(self._last_output)[:50]
            if len(str(self._last_output)) > 50:
                output_str += '...'
            fields.append({
                'name': 'Output',
                'key': 'last_output',
                'value': str(self._last_output),
                'type': 'text',
                'read_only': not cognition_paused,  # Editable when paused
                'preview': output_str
            })

        return fields


@dataclass
class FacetAssembly:
    """
    A complete cognitive architecture - network of connected facets.

    Like a Unity prefab, can be saved/loaded as YAML and shared.
    """
    name: str                            # Assembly name
    version: str = "1.0.0"              # Version for compatibility
    description: str = ""                # Human-readable description
    author: str = ""                     # Creator name
    tags: List[str] = field(default_factory=list)  # Searchable tags

    # The network
    facets: List[Facet] = field(default_factory=list)
    connections: List[FacetConnection] = field(default_factory=list)

    # Channel configuration (subscribe/publish)
    channels: Optional[Any] = None  # ChannelsConfig when available

    # Metadata
    created_date: str = ""
    modified_date: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML export."""
        result = {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'tags': self.tags,
            'created': self.created_date,
            'modified': self.modified_date,
            'facets': [f.to_dict() for f in self.facets],
            'connections': [c.to_dict() for c in self.connections]
        }

        # Include channels if configured
        if self.channels:
            if hasattr(self.channels, 'to_dict'):
                channels_data = self.channels.to_dict()
            else:
                channels_data = self.channels
            if channels_data:
                result['channels'] = channels_data

        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FacetAssembly':
        """Deserialize from dictionary."""
        # Parse channels if present
        channels = None
        channels_data = data.get('channels')
        if channels_data:
            if CHANNELS_AVAILABLE and ChannelsConfig:
                channels = ChannelsConfig.from_dict(channels_data)
            else:
                # Fallback: store as raw dict if ChannelsConfig not available
                channels = channels_data

        return FacetAssembly(
            name=data['name'],
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            author=data.get('author', ''),
            tags=data.get('tags', []),
            created_date=data.get('created', ''),
            modified_date=data.get('modified', ''),
            facets=[Facet.from_dict(f) for f in data.get('facets', [])],
            connections=[FacetConnection.from_dict(c) for c in data.get('connections', [])],
            channels=channels
        )

    def save_yaml(self, filepath: str):
        """Save assembly to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load_yaml(filepath: str) -> 'FacetAssembly':
        """Load assembly from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return FacetAssembly.from_dict(data)

    def get_facet(self, facet_id: str) -> Optional[Facet]:
        """Get facet by ID."""
        for facet in self.facets:
            if facet.id == facet_id:
                return facet
        return None

    def validate(self) -> List[str]:
        """
        Validate assembly structure.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for INCOMING and OUTGOING nodes
        has_incoming = any(f.id == 'INCOMING' for f in self.facets)
        has_outgoing = any(f.id == 'OUTGOING' for f in self.facets)

        if not has_incoming:
            errors.append("Missing INCOMING node")
        if not has_outgoing:
            errors.append("Missing OUTGOING node")

        # Check for disconnected required pads
        for facet in self.facets:
            for pad in facet.input_pads:
                if pad.required and not pad.connected_to:
                    errors.append(f"Facet '{facet.name}' has unconnected required input '{pad.name}'")

        # Check for cycles (would cause deadlock)
        if self._has_cycle():
            errors.append("Assembly contains cycle - would cause deadlock")

        return errors

    def _has_cycle(self) -> bool:
        """Detect cycles in the facet graph using DFS."""
        # Build adjacency list
        graph = {f.id: [] for f in self.facets}
        for conn in self.connections:
            graph[conn.from_facet].append(conn.to_facet)

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for facet in self.facets:
            if facet.id not in visited:
                if dfs(facet.id):
                    return True

        return False

    def get_subscribe_channels(self) -> List[str]:
        """Get list of channels this assembly subscribes to."""
        if not self.channels:
            return []
        if hasattr(self.channels, 'subscribe'):
            return self.channels.subscribe
        if isinstance(self.channels, dict):
            return self.channels.get('subscribe', [])
        return []

    def get_publish_channels(self) -> List[str]:
        """Get list of channels this assembly can publish to."""
        if not self.channels:
            return []
        if hasattr(self.channels, 'publish'):
            return self.channels.publish
        if isinstance(self.channels, dict):
            return self.channels.get('publish', [])
        return []

    def can_publish_to(self, channel: str) -> bool:
        """Check if this assembly is allowed to publish to a channel."""
        return channel in self.get_publish_channels()

    def subscribes_to(self, channel: str) -> bool:
        """Check if this assembly subscribes to a channel."""
        return channel in self.get_subscribe_channels()


def create_default_assembly() -> FacetAssembly:
    """
    Create a simple default assembly for testing.

    Flow: INCOMING → Intuition → OUTGOING
    """
    assembly = FacetAssembly(
        name="Simple Test Assembly",
        description="Minimal assembly for testing",
        author="Noodlings System"
    )

    # Generate UUIDs for nodes
    incoming_id = Facet.generate_uuid()
    intuition_id = Facet.generate_uuid()
    outgoing_id = Facet.generate_uuid()

    # INCOMING node (special)
    incoming = Facet(
        id=incoming_id,
        name="INCOMING",
        facet_type="SpecialNode",
        prompt="",
        position={'x': 100, 'y': 300}
    )
    incoming.add_output_pad("out", "Raw incoming context")
    assembly.facets.append(incoming)

    # Simple intuition facet
    intuition = Facet(
        id=intuition_id,
        name="Intuition",
        facet_type="IntuitionFacet",
        prompt="Analyze the contextual awareness and provide intuitive insights.",
        position={'x': 400, 'y': 300}
    )
    intuition.add_input_pad("context", "Incoming context")
    intuition.add_output_pad("awareness", "Intuitive awareness")
    assembly.facets.append(intuition)

    # OUTGOING node (special)
    outgoing = Facet(
        id=outgoing_id,
        name="OUTGOING",
        facet_type="SpecialNode",
        prompt="",
        position={'x': 700, 'y': 300}
    )
    outgoing.add_input_pad("in", "Final response")
    assembly.facets.append(outgoing)

    # Connections
    assembly.connections.append(FacetConnection(incoming_id, "out", intuition_id, "context"))
    assembly.connections.append(FacetConnection(intuition_id, "awareness", outgoing_id, "in"))

    return assembly


if __name__ == "__main__":
    # Test: Create and save example assembly
    assembly = create_default_assembly()
    print("Created assembly:", assembly.name)
    print("Facets:", len(assembly.facets))
    print("Connections:", len(assembly.connections))

    # Validate
    errors = assembly.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Assembly is valid!")

    # Save to YAML
    assembly.save_yaml("/tmp/test_assembly.yaml")
    print("Saved to /tmp/test_assembly.yaml")

    # Load back
    loaded = FacetAssembly.load_yaml("/tmp/test_assembly.yaml")
    print(f"Loaded assembly: {loaded.name}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
