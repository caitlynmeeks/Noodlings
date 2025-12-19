"""
Context Builder - Generates Rich Agent Context from Events

This is the bridge between event-sourced reality and the cognitive architecture.
It builds the narrative context that gets fed to an agent's LLM, giving them
a rich, perspectival understanding of their world.

The context builder produces text like:
    "You stand near the hearth in The Nexus. The flames dance with
    unusual awareness today, casting your shadow long across worn
    flagstones. Warmth seeps into you.

    Servnak works at the cluttered bench across the room, muttering
    over some mechanism. The distance feels deliberate.

    Recently, you entered slowly from the garden. Servnak glanced up
    but said nothing. The silence since has grown thick."

This is what enables genuine storytelling - agents understand MEANING,
not coordinates.

Author: Caitlyn + Claude
Date: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
import yaml
import os

from .event import Event, EventType
from .event_store import EventStore
from .projections import (
    project_situation,
    project_experience,
    Situation,
    Experience,
    Presence
)


@dataclass
class StageDefinition:
    """
    The eternal aspects of a stage - loaded from stage.yaml.

    This is the container that changes slowly (through construction/destruction).
    """

    id: str
    name: str
    essence: str  # The poetic description of what this place IS

    # Named meaningful locations within the stage
    anchors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Each anchor: {essence, qualities, sensory}

    # Features (things that exist in this stage)
    features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Each feature: {essence, anchor, sensory, affordances}

    # Zones (soft attention regions)
    zones: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Each zone: {anchor, radius, falloff, atmosphere}

    @classmethod
    def from_yaml(cls, path: str) -> 'StageDefinition':
        """Load stage definition from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        return cls(
            id=data.get('id', Path(path).stem),
            name=data.get('name', 'Unknown Stage'),
            essence=data.get('essence', ''),
            anchors=data.get('anchors', {}),
            features=data.get('features', {}),
            zones=data.get('zones', {})
        )

    @classmethod
    def empty(cls, stage_id: str, name: str = "Unknown Stage") -> 'StageDefinition':
        """Create an empty stage definition."""
        return cls(id=stage_id, name=name, essence="")


@dataclass
class AgentContext:
    """
    The complete context package for an agent.

    This is what gets serialized and fed to the LLM.
    """

    # Who this context is for
    entity_id: str
    stage_id: str
    timestamp: datetime

    # The narrative text (main context)
    narrative: str

    # Structured data (for programmatic access)
    current_anchor: Optional[str] = None
    current_state: Optional[str] = None
    others_present: List[str] = field(default_factory=list)
    recent_event_count: int = 0
    tension_level: float = 0.0

    # Raw components (for debugging/advanced use)
    situation: Optional[Situation] = None
    experience: Optional[Experience] = None

    def __str__(self) -> str:
        """String representation is the narrative."""
        return self.narrative


class ContextBuilder:
    """
    Builds rich narrative context for agents from the event stream.

    Usage:
        builder = ContextBuilder(event_store, stages_path="world/stages")

        # Build context for an agent
        context = builder.build_context("red", "the_nexus")
        print(context.narrative)

        # Or get just the text
        text = builder.build_context_text("red", "the_nexus")
    """

    def __init__(
        self,
        event_store: EventStore,
        stages_path: Optional[str] = None
    ):
        """
        Initialize the context builder.

        Args:
            event_store: The event store to build context from
            stages_path: Path to directory containing stage YAML files
        """
        self.event_store = event_store
        self.stages_path = stages_path
        self._stage_cache: Dict[str, StageDefinition] = {}

    def get_stage_definition(self, stage_id: str) -> StageDefinition:
        """Get stage definition, loading from disk if needed."""
        if stage_id not in self._stage_cache:
            if self.stages_path:
                # Try to load from YAML
                yaml_path = os.path.join(self.stages_path, stage_id, "stage.yaml")
                if os.path.exists(yaml_path):
                    self._stage_cache[stage_id] = StageDefinition.from_yaml(yaml_path)
                else:
                    self._stage_cache[stage_id] = StageDefinition.empty(stage_id)
            else:
                self._stage_cache[stage_id] = StageDefinition.empty(stage_id)

        return self._stage_cache[stage_id]

    def build_context(
        self,
        entity_id: str,
        stage_id: str,
        window: timedelta = timedelta(minutes=10),
        max_events: int = 10
    ) -> AgentContext:
        """
        Build complete context for an agent.

        Args:
            entity_id: The agent to build context for
            stage_id: The stage they're in
            window: How far back to look for events
            max_events: Maximum recent events to include

        Returns:
            AgentContext with narrative and structured data
        """
        # Get projections
        situation = project_situation(self.event_store, stage_id, window)
        experience = project_experience(self.event_store, entity_id, stage_id, window)

        # Get stage definition
        stage_def = self.get_stage_definition(stage_id)

        # Build the narrative
        narrative = self._build_narrative(
            entity_id, stage_def, situation, experience, max_events
        )

        # Calculate tension level
        tension_level = 0.0
        if situation.tensions:
            tension_level = sum(t.intensity for t in situation.tensions) / len(situation.tensions)

        return AgentContext(
            entity_id=entity_id,
            stage_id=stage_id,
            timestamp=datetime.utcnow(),
            narrative=narrative,
            current_anchor=experience.current_anchor,
            current_state=experience.current_state,
            others_present=[
                eid for eid in situation.presences.keys()
                if eid != entity_id
            ],
            recent_event_count=len(experience.witnessed_events),
            tension_level=tension_level,
            situation=situation,
            experience=experience
        )

    def build_context_text(
        self,
        entity_id: str,
        stage_id: str,
        window: timedelta = timedelta(minutes=10),
        max_events: int = 10
    ) -> str:
        """Build context and return just the narrative text."""
        context = self.build_context(entity_id, stage_id, window, max_events)
        return context.narrative

    def _build_narrative(
        self,
        entity_id: str,
        stage_def: StageDefinition,
        situation: Situation,
        experience: Experience,
        max_events: int
    ) -> str:
        """Build the narrative text from components."""
        sections = []

        # Section 1: Where you are
        location_text = self._build_location_section(
            entity_id, stage_def, situation, experience
        )
        if location_text:
            sections.append(location_text)

        # Section 2: Who else is here
        others_text = self._build_others_section(
            entity_id, situation, experience
        )
        if others_text:
            sections.append(others_text)

        # Section 3: What's been happening
        events_text = self._build_events_section(
            entity_id, experience, max_events
        )
        if events_text:
            sections.append(events_text)

        # Section 4: Tensions/atmosphere
        atmosphere_text = self._build_atmosphere_section(
            situation
        )
        if atmosphere_text:
            sections.append(atmosphere_text)

        return "\n\n".join(sections)

    def _build_location_section(
        self,
        entity_id: str,
        stage_def: StageDefinition,
        situation: Situation,
        experience: Experience
    ) -> str:
        """Build the 'where you are' section."""
        lines = []

        # Stage name and essence
        if stage_def.essence:
            lines.append(f"You are in {stage_def.name}.")
            lines.append(stage_def.essence.strip())
        else:
            lines.append(f"You are in {stage_def.name}.")

        # Specific anchor location
        anchor = experience.current_anchor
        if anchor and anchor in stage_def.anchors:
            anchor_def = stage_def.anchors[anchor]
            if 'essence' in anchor_def:
                lines.append("")
                lines.append(f"You are at the {anchor}. {anchor_def['essence']}")

            # Sensory details
            if 'sensory' in anchor_def:
                sensory = anchor_def['sensory']
                if isinstance(sensory, dict):
                    details = []
                    if 'visual' in sensory:
                        details.append(sensory['visual'])
                    if 'sound' in sensory:
                        details.append(sensory['sound'])
                    if details:
                        lines.append(" ".join(details))

        # Features at this anchor
        if anchor:
            for feat_name, feat_def in stage_def.features.items():
                if feat_def.get('anchor') == anchor:
                    if 'essence' in feat_def:
                        lines.append(f"The {feat_name}: {feat_def['essence']}")

        return "\n".join(lines)

    def _build_others_section(
        self,
        entity_id: str,
        situation: Situation,
        experience: Experience
    ) -> str:
        """Build the 'who else is here' section."""
        lines = []

        for other_id, presence in situation.presences.items():
            if other_id == entity_id:
                continue

            # Get relationship info
            relation = situation.relation_between(entity_id, other_id)

            # Build description
            other_name = other_id.replace('_', ' ').title()

            # Location description
            if relation:
                if relation.quality:
                    loc_text = f"{other_name} is {relation.relation} you, {relation.quality}"
                else:
                    loc_text = f"{other_name} is {relation.relation} you"
            elif presence.anchor:
                loc_text = f"{other_name} is at the {presence.anchor}"
            else:
                loc_text = f"{other_name} is nearby"

            # Add state/manner
            if presence.state:
                loc_text += f", appearing {presence.state}"

            lines.append(loc_text + ".")

            # Add any interpretation from experience
            if other_id in experience.awareness_of:
                awareness = experience.awareness_of[other_id]
                if 'interpretation' in awareness:
                    lines.append(f"({awareness['interpretation']})")

        return "\n".join(lines)

    def _build_events_section(
        self,
        entity_id: str,
        experience: Experience,
        max_events: int
    ) -> str:
        """Build the 'what's been happening' section."""
        events = experience.witnessed_events[-max_events:]

        if not events:
            return ""

        lines = ["Recently:"]

        for event in events:
            narration = event.narrate(perspective=entity_id)
            if narration:
                lines.append(f"- {narration}")

        return "\n".join(lines)

    def _build_atmosphere_section(
        self,
        situation: Situation
    ) -> str:
        """Build the atmosphere/tension section."""
        lines = []

        # Atmosphere
        if 'mood' in situation.atmosphere:
            mood = situation.atmosphere['mood']
            if mood and mood != 'neutral':
                lines.append(f"The atmosphere feels {mood}.")

        # Tensions
        for tension in situation.tensions:
            if tension.intensity > 0.4:  # Only mention notable tensions
                lines.append(tension.description)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_global_builder: Optional[ContextBuilder] = None


def get_context_builder() -> Optional[ContextBuilder]:
    """Get the global context builder if initialized."""
    return _global_builder


def init_context_builder(
    event_store: EventStore,
    stages_path: Optional[str] = None
) -> ContextBuilder:
    """Initialize the global context builder."""
    global _global_builder
    _global_builder = ContextBuilder(event_store, stages_path)
    return _global_builder


def build_agent_context(
    entity_id: str,
    stage_id: str,
    event_store: Optional[EventStore] = None,
    window: timedelta = timedelta(minutes=10)
) -> str:
    """
    Convenience function to build agent context.

    Uses global builder if available, otherwise creates temporary one.
    """
    global _global_builder

    if _global_builder:
        return _global_builder.build_context_text(entity_id, stage_id, window)

    if event_store is None:
        from .event_store import get_event_store
        event_store = get_event_store()

    builder = ContextBuilder(event_store)
    return builder.build_context_text(entity_id, stage_id, window)


__all__ = [
    "StageDefinition",
    "AgentContext",
    "ContextBuilder",
    "get_context_builder",
    "init_context_builder",
    "build_agent_context",
]
