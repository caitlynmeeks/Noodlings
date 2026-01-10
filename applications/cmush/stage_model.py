# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Stage Model - Zone-Based Spatial System
#
#   Virtual worlds need to know who is near whom. This module
#   uses "zones" (like rooms in a house) instead of complex 3D
#   physics. Asking "who's near Red?" just checks zone membership.
#   It's USD-compatible (same format as Pixar films), supports
#   containment (Red is IN the fireplace), and tags (this chair
#   is "sittable"). Works for both text MUDs and 3D visualization.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.stage_model
# PURPOSE:  Zone-based spatial model for proximity queries
# LAYER:    Backend / World Model
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Entity                Base class for objects, users, agents
#   Stage                 Zone-based spatial container
#   StageQuery            Spatial query utilities
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Stage Model - Zone-based spatial model for noodleMUSH.

USD-compatible hierarchical scene graph with zone-based proximity.
Perfect for 2-20 entities, no octree overhead.

Philosophy:
- Zone-based (coarse): Cheap "who's near me?" queries
- Hierarchical containment: USD-style parent-child (Red IN fireplace)
- Tag-based affordances: 'sittable', 'warm', 'container'
- Text + 3D ready: Same data structure for MUSH and Studio
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import json


@dataclass
class Entity:
    """
    Base class for everything in the stage.
    Objects, users, agents - all use this.
    """
    entity_id: str
    entity_type: str  # 'object', 'user', 'agent'
    name: str

    # Spatial
    zone: Optional[str] = None  # Current zone
    parent: Optional[str] = None  # Hierarchical containment (USD-style)
    position: Dict[str, float] = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'z': 0.0})
    rotation: Dict[str, float] = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'z': 0.0})

    # Semantic
    tags: List[str] = field(default_factory=list)  # ['sittable', 'warm', 'container', ...]
    state: Dict[str, any] = field(default_factory=dict)  # Flexible key-value state

    # Relationships
    contains: List[str] = field(default_factory=list)  # Child entity IDs (if container)
    occupied_by: Optional[str] = None  # For sittable/usable objects

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'name': self.name,
            'zone': self.zone,
            'parent': self.parent,
            'position': self.position,
            'rotation': self.rotation,
            'tags': self.tags,
            'state': self.state,
            'contains': self.contains,
            'occupied_by': self.occupied_by
        }

    @staticmethod
    def from_dict(data: dict) -> 'Entity':
        """Deserialize from dict."""
        return Entity(
            entity_id=data['entity_id'],
            entity_type=data['entity_type'],
            name=data['name'],
            zone=data.get('zone'),
            parent=data.get('parent'),
            position=data.get('position', {'x': 0.0, 'y': 0.0, 'z': 0.0}),
            rotation=data.get('rotation', {'x': 0.0, 'y': 0.0, 'z': 0.0}),
            tags=data.get('tags', []),
            state=data.get('state', {}),
            contains=data.get('contains', []),
            occupied_by=data.get('occupied_by')
        )


class Stage:
    """
    Zone-based spatial model for text MUD + 3D visualization.
    USD-compatible, no octree overhead.
    """

    def __init__(self, stage_id: str, name: str):
        self.stage_id = stage_id
        self.name = name
        self.description = ""

        # Entity registry (all objects, users, agents)
        self.entities: Dict[str, Entity] = {}

        # Spatial zones (coarse proximity grouping)
        self.zones: Dict[str, List[str]] = {}  # {zone_name: [entity_ids]}

        # Zone adjacency (for movement/visibility)
        self.zone_graph: Dict[str, List[str]] = {}  # {zone_name: [adjacent_zones]}

    def add_entity(self, entity: Entity) -> None:
        """Add entity to stage and register in zone."""
        self.entities[entity.entity_id] = entity

        if entity.zone:
            if entity.zone not in self.zones:
                self.zones[entity.zone] = []
            if entity.entity_id not in self.zones[entity.zone]:
                self.zones[entity.zone].append(entity.entity_id)

    def remove_entity(self, entity_id: str) -> Optional[Entity]:
        """Remove entity from stage and zone."""
        entity = self.entities.pop(entity_id, None)
        if entity and entity.zone:
            if entity.zone in self.zones:
                try:
                    self.zones[entity.zone].remove(entity_id)
                except ValueError:
                    pass
        return entity

    def move_entity(self, entity_id: str, new_zone: str) -> bool:
        """Move entity to a new zone."""
        entity = self.entities.get(entity_id)
        if not entity:
            return False

        # Remove from old zone
        if entity.zone and entity.zone in self.zones:
            try:
                self.zones[entity.zone].remove(entity_id)
            except ValueError:
                pass

        # Add to new zone
        entity.zone = new_zone
        if new_zone not in self.zones:
            self.zones[new_zone] = []
        if entity_id not in self.zones[new_zone]:
            self.zones[new_zone].append(entity_id)

        return True

    def to_dict(self) -> dict:
        """Serialize stage to dict for JSON storage."""
        return {
            'stage_id': self.stage_id,
            'name': self.name,
            'description': self.description,
            'entities': {eid: e.to_dict() for eid, e in self.entities.items()},
            'zones': self.zones,
            'zone_graph': self.zone_graph
        }

    @staticmethod
    def from_dict(data: dict) -> 'Stage':
        """Deserialize stage from dict."""
        stage = Stage(data['stage_id'], data['name'])
        stage.description = data.get('description', '')
        stage.zones = data.get('zones', {})
        stage.zone_graph = data.get('zone_graph', {})

        # Deserialize entities
        for entity_data in data.get('entities', {}).values():
            entity = Entity.from_dict(entity_data)
            stage.entities[entity.entity_id] = entity

        return stage


class StageQuery:
    """Spatial queries for Context Intelligence / agent prompts."""

    @staticmethod
    def get_nearby_entities(
        stage: Stage,
        entity_id: str,
        include_adjacent: bool = False
    ) -> List[str]:
        """
        Get all entities in same zone (optionally adjacent zones).

        Args:
            stage: The stage to query
            entity_id: The observer entity
            include_adjacent: Include entities in adjacent zones?

        Returns:
            List of entity IDs (excluding the observer)
        """
        entity = stage.entities.get(entity_id)
        if not entity or not entity.zone:
            return []

        nearby = list(stage.zones.get(entity.zone, []))

        if include_adjacent:
            for adj_zone in stage.zone_graph.get(entity.zone, []):
                nearby.extend(stage.zones.get(adj_zone, []))

        # Remove self and deduplicate
        nearby_set = set(nearby) - {entity_id}
        return list(nearby_set)

    @staticmethod
    def describe_scene(stage: Stage, observer_id: str, include_adjacent: bool = False) -> str:
        """
        Generate narrative scene description for agent/user.

        Args:
            stage: The stage to describe
            observer_id: The entity observing the scene
            include_adjacent: Include adjacent zones in description?

        Returns:
            Formatted scene description string
        """
        observer = stage.entities.get(observer_id)
        if not observer:
            return f"Location: {stage.name}\n(Observer not found in stage)"

        zone_name = observer.zone

        nearby_ids = StageQuery.get_nearby_entities(stage, observer_id, include_adjacent)
        nearby = [stage.entities[eid] for eid in nearby_ids if eid in stage.entities]

        # Categorize
        users = [e for e in nearby if e.entity_type == 'user']
        agents = [e for e in nearby if e.entity_type == 'agent']
        objects = [e for e in nearby if e.entity_type == 'object']

        # Build description
        desc = f"Location: {stage.name}"
        if zone_name:
            desc += f" ({zone_name})"

        if users or agents:
            people = [e.name for e in users + agents]
            desc += f"\nPresent: {', '.join(people)}"

        if objects:
            obj_list = []
            for obj in objects:
                tags_str = ', '.join(obj.tags) if obj.tags else ''
                obj_list.append(f"{obj.name} ({tags_str})" if tags_str else obj.name)
            desc += f"\nObjects: {', '.join(obj_list)}"

        return desc

    @staticmethod
    def find_by_tag(stage: Stage, zone_name: str, tag: str) -> List[Entity]:
        """
        Find all entities in zone with specific tag.

        Args:
            stage: The stage to search
            zone_name: Zone to search in
            tag: Tag to match (e.g., 'sittable', 'warm')

        Returns:
            List of matching entities
        """
        zone_entities = stage.zones.get(zone_name, [])
        matches = []

        for eid in zone_entities:
            entity = stage.entities.get(eid)
            if entity and tag in entity.tags:
                matches.append(entity)

        return matches

    @staticmethod
    def find_sittable(stage: Stage, zone_name: str) -> List[Entity]:
        """Find available sittable objects in zone."""
        sittable = StageQuery.find_by_tag(stage, zone_name, 'sittable')
        # Filter to only unoccupied
        return [e for e in sittable if not e.occupied_by]

    @staticmethod
    def find_containers(stage: Stage, zone_name: str) -> List[Entity]:
        """Find container objects in zone."""
        return StageQuery.find_by_tag(stage, zone_name, 'container')

    @staticmethod
    def get_occupants(stage: Stage, zone_name: str) -> Dict[str, List[Entity]]:
        """
        Get all occupants in a zone, categorized by type.

        Returns:
            Dict with keys 'users', 'agents', 'objects'
        """
        zone_entities = stage.zones.get(zone_name, [])

        result = {
            'users': [],
            'agents': [],
            'objects': []
        }

        for eid in zone_entities:
            entity = stage.entities.get(eid)
            if entity:
                result[f"{entity.entity_type}s"].append(entity)

        return result

    @staticmethod
    def count_people_in_zone(stage: Stage, zone_name: str) -> int:
        """Count users and agents in zone (for 1-on-1 detection)."""
        zone_entities = stage.zones.get(zone_name, [])
        count = 0

        for eid in zone_entities:
            entity = stage.entities.get(eid)
            if entity and entity.entity_type in ['user', 'agent']:
                count += 1

        return count

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
