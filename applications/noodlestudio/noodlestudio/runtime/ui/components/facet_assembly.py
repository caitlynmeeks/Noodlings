# ──────────────────────────────────────────────────────────────
#   Facet Assembly UI Component
#
#   Attaches facet assembly logic to UI Canvas.
#   This is an invisible component that runs assemblies.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.facet_assembly
# PURPOSE:  Facet assembly integration for UI Canvas
# LAYER:    Runtime / UI / Components
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..component import UIComponent, register_component


@dataclass
class InputBinding:
    """Maps an assembly input pad to a UI component property."""
    pad_name: str  # Assembly input pad name (e.g., "text")
    source: str  # UI component.property (e.g., "text_input.value")

    def to_dict(self) -> Dict[str, str]:
        return {"pad": self.pad_name, "source": self.source}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputBinding':
        return cls(
            pad_name=data.get("pad", ""),
            source=data.get("source", "")
        )


@dataclass
class OutputBinding:
    """Maps an assembly output pad to a UI component property."""
    pad_name: str  # Assembly output pad name (e.g., "response")
    target: str  # UI component.property (e.g., "response_label.text")

    def to_dict(self) -> Dict[str, str]:
        return {"pad": self.pad_name, "target": self.target}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputBinding':
        return cls(
            pad_name=data.get("pad", ""),
            target=data.get("target", "")
        )


@register_component
class FacetAssembly(UIComponent):
    """
    UI Canvas component that attaches facet assembly logic.

    This is an invisible component that provides cognitive functionality
    to UI interfaces. It appears as a small gear icon in the canvas editor
    but has no visual representation at runtime.

    Properties:
        assembly_path: Path to .yaml assembly file (relative to project)
        auto_run: Run assembly when UI loads
        run_on_event: Event that triggers execution (e.g., "button.onClick")
        input_bindings: Map assembly inputs to UI component values
        output_bindings: Map assembly outputs to UI component properties

    Usage in ui.yaml:
        FacetAssembly:
          name: sentiment_analyzer
          assembly: assemblies/sentiment.yaml
          auto_run: false
          input_bindings:
            - pad: text
              source: text_input.value
          output_bindings:
            - pad: sentiment
              target: mood_indicator.color
            - pad: response
              target: response_label.text

    Triggering via events:
        Button:
          name: analyze_btn
          events:
            onClick:
              action: run_assembly
              target: sentiment_analyzer
    """

    component_type = "FacetAssembly"

    def __init__(self, name: str = ""):
        super().__init__(name)

        # Assembly configuration
        self.assembly_path: str = ""
        self.auto_run: bool = False

        # Bindings
        self.input_bindings: List[InputBinding] = []
        self.output_bindings: List[OutputBinding] = []

        # Visual representation (invisible at runtime, icon in editor)
        self.geometry.width = 32
        self.geometry.height = 32
        self.visible = True  # Visible in editor, invisible at runtime

        # Runtime state (not serialized)
        self._is_running: bool = False
        self._last_result: Optional[Dict[str, Any]] = None
        self._execution_count: int = 0

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Serialize FacetAssembly-specific properties."""
        if self.assembly_path:
            data["assembly"] = self.assembly_path
        if self.auto_run:
            data["auto_run"] = self.auto_run
        if self.input_bindings:
            data["input_bindings"] = [b.to_dict() for b in self.input_bindings]
        if self.output_bindings:
            data["output_bindings"] = [b.to_dict() for b in self.output_bindings]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FacetAssembly':
        """Create FacetAssembly from dictionary."""
        component = cls(name=data.get("name", ""))
        component._apply_base_properties(data)

        component.assembly_path = data.get("assembly", "")
        component.auto_run = data.get("auto_run", False)

        # Parse input bindings
        for binding_data in data.get("input_bindings", []):
            component.input_bindings.append(InputBinding.from_dict(binding_data))

        # Parse output bindings
        for binding_data in data.get("output_bindings", []):
            component.output_bindings.append(OutputBinding.from_dict(binding_data))

        return component

    def add_input_binding(self, pad_name: str, source: str) -> None:
        """Add an input binding."""
        self.input_bindings.append(InputBinding(pad_name, source))

    def add_output_binding(self, pad_name: str, target: str) -> None:
        """Add an output binding."""
        self.output_bindings.append(OutputBinding(pad_name, target))

    def remove_input_binding(self, pad_name: str) -> None:
        """Remove an input binding by pad name."""
        self.input_bindings = [b for b in self.input_bindings if b.pad_name != pad_name]

    def remove_output_binding(self, pad_name: str) -> None:
        """Remove an output binding by pad name."""
        self.output_bindings = [b for b in self.output_bindings if b.pad_name != pad_name]

    def get_input_sources(self) -> Dict[str, str]:
        """Get input pad -> source mapping."""
        return {b.pad_name: b.source for b in self.input_bindings}

    def get_output_targets(self) -> Dict[str, str]:
        """Get output pad -> target mapping."""
        return {b.pad_name: b.target for b in self.output_bindings}

    @property
    def is_running(self) -> bool:
        """Check if assembly is currently executing."""
        return self._is_running

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        """Get the last execution result."""
        return self._last_result

    @property
    def execution_count(self) -> int:
        """Get total execution count."""
        return self._execution_count


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
