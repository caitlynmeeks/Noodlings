"""
USD Exporter for noodleMUSH

Export stages to Pixar USD format (.usda ASCII) for animation studio pipelines.

USD Terminology:
- Stage: The composed scene (what's rendered)
- Prim: Basic scene object (rooms, Noodlings, objects)
- Typed Schema: Defines prim type (we define custom "Noodling" schema)
- Layer: A .usda/.usdc file

Exports:
- Stage hierarchy (rooms, Noodlings, objects as prims)
- Prim properties (descriptions, personality, affect states)
- Relationships (location, holding, following)
- Time-sampled affect data (for animation)

Compatible with:
- Maya, Houdini, Blender (USD support)
- Unreal Engine, Unity (USD import)
- Pixar's USD tools

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import yaml


class USDExporter:
    """
    Export noodleMUSH stages to USD format.

    Creates custom typed schema for Noodlings with kindling properties.
    Each Noodling, room, prim, etc. becomes a Prim in the Stage.
    """

    def __init__(self):
        pass

    def export_stage(self, world_data: Dict, output_path: Path):
        """
        Export noodleMUSH world as USD Stage.

        Args:
            world_data: Dictionary with rooms, users, noodlings, objects
            output_path: Path to .usda layer file to create
        """
        lines = []

        # USD layer header
        lines.append('#usda 1.0')
        lines.append('(')
        lines.append('    defaultPrim = "Stage"')
        lines.append('    doc = """noodleMUSH Stage - Kindled Noodling Prims"""')
        lines.append('    metersPerUnit = 1')
        lines.append('    upAxis = "Y"')
        lines.append(')')
        lines.append('')

        # Define custom "Noodling" typed schema
        lines.append('# Custom Typed Schema: Noodling')
        lines.append('# Defines kindling properties for Noodling prims')
        lines.append('class "NoodlingSchema" (')
        lines.append('    customData = {')
        lines.append('        string className = "Noodling"')
        lines.append('        string schemaType = "singleApply"')
        lines.append('    }')
        lines.append(') {')
        lines.append('    # Identity')
        lines.append('    string species')
        lines.append('    string description')
        lines.append('    ')
        lines.append('    # LLM Configuration')
        lines.append('    string llm_provider')
        lines.append('    string llm_model')
        lines.append('    ')
        lines.append('    # Personality Traits (Big Five + extras)')
        lines.append('    float extraversion')
        lines.append('    float curiosity')
        lines.append('    float impulsivity')
        lines.append('    float emotional_volatility')
        lines.append('    ')
        lines.append('    # 5-D Affect Vector')
        lines.append('    float affect_valence')
        lines.append('    float affect_arousal')
        lines.append('    float affect_fear')
        lines.append('    float affect_sorrow')
        lines.append('    float affect_boredom')
        lines.append('}')
        lines.append('')

        # Stage root (contains all prims)
        lines.append('def Xform "Stage" (')
        lines.append('    kind = "assembly"')
        lines.append(') {')

        # Export rooms as prims
        rooms = world_data.get('rooms', {})
        for room_id, room_data in rooms.items():
            lines.extend(self._export_room_prim(room_id, room_data, indent=1))

        # Export Noodlings as prims with custom schema
        noodlings = world_data.get('noodlings', [])
        for noodling in noodlings:
            lines.extend(self._export_noodling_prim(noodling, indent=1))

        # Export users as prims
        users = world_data.get('users', [])
        for user in users:
            lines.extend(self._export_user_prim(user, indent=1))

        # Export objects as prims
        objects = world_data.get('objects', {})
        for obj_id, obj_data in objects.items():
            lines.extend(self._export_object_prim(obj_id, obj_data, indent=1))

        lines.append('}')
        lines.append('')

        # Write layer to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"USD stage exported to layer: {output_path}")

    def _export_room_prim(self, room_id: str, room_data: Dict, indent: int = 0) -> List[str]:
        """Export a room as USD Xform prim."""
        ind = '    ' * indent
        lines = []

        name = room_data.get('name', room_id)
        desc = room_data.get('description', '')

        lines.append(f'{ind}def Xform "{room_id}" (')
        lines.append(f'{ind}    customData = {{')
        lines.append(f'{ind}        string type = "room"')
        lines.append(f'{ind}    }}')
        lines.append(f'{ind}) {{')
        lines.append(f'{ind}    string name = "{name}"')
        lines.append(f'{ind}    string description = "{desc}"')

        # Exits as relationships
        exits = room_data.get('exits', {})
        if exits:
            lines.append(f'{ind}    string[] exits = {json.dumps(list(exits.keys()))}')

        lines.append(f'{ind}}}')
        lines.append('')

        return lines

    def _export_noodling_prim(self, noodling_data: Dict, indent: int = 0) -> List[str]:
        """Export a Noodling prim with kindling properties (uses NoodlingSchema)."""
        ind = '    ' * indent
        lines = []

        noodling_id = noodling_data.get('id', 'unknown')
        name = noodling_data.get('name', noodling_id)
        species = noodling_data.get('species', 'noodling')
        desc = noodling_data.get('description', '')

        lines.append(f'{ind}def "Noodlings/{noodling_id}" (')
        lines.append(f'{ind}    prepend apiSchemas = ["NoodlingSchema"]')
        lines.append(f'{ind}    customData = {{')
        lines.append(f'{ind}        string type = "noodling"')
        lines.append(f'{ind}    }}')
        lines.append(f'{ind}) {{')
        lines.append(f'{ind}    string name = "{name}"')
        lines.append(f'{ind}    string species = "{species}"')
        lines.append(f'{ind}    string description = """{desc}"""')

        # Kindling properties
        personality = noodling_data.get('personality', {})
        lines.append(f'{ind}    float extraversion = {personality.get("extraversion", 0.5)}')
        lines.append(f'{ind}    float curiosity = {personality.get("curiosity", 0.5)}')
        lines.append(f'{ind}    float impulsivity = {personality.get("impulsivity", 0.5)}')
        lines.append(f'{ind}    float emotional_volatility = {personality.get("emotional_volatility", 0.5)}')

        # LLM config
        lines.append(f'{ind}    string llm_provider = "{noodling_data.get("llm_provider", "local")}"')
        lines.append(f'{ind}    string llm_model = "{noodling_data.get("llm_model", "")}"')

        # Current affect state (if available)
        affect = noodling_data.get('affect', {})
        if affect:
            lines.append(f'{ind}    # Current 5-D affect state')
            lines.append(f'{ind}    float affect:valence = {affect.get("valence", 0.0)}')
            lines.append(f'{ind}    float affect:arousal = {affect.get("arousal", 0.0)}')
            lines.append(f'{ind}    float affect:fear = {affect.get("fear", 0.0)}')
            lines.append(f'{ind}    float affect:sorrow = {affect.get("sorrow", 0.0)}')
            lines.append(f'{ind}    float affect:boredom = {affect.get("boredom", 0.0)}')

        lines.append(f'{ind}}}')
        lines.append('')

        return lines

    def _export_user_prim(self, user_data: Dict, indent: int = 0) -> List[str]:
        """Export a user (Noodler/human) as prim."""
        ind = '    ' * indent
        lines = []

        user_id = user_data.get('id', 'unknown')
        username = user_data.get('username', user_id)

        lines.append(f'{ind}def "Users/{user_id}" (')
        lines.append(f'{ind}    customData = {{')
        lines.append(f'{ind}        string type = "user"')
        lines.append(f'{ind}    }}')
        lines.append(f'{ind}) {{')
        lines.append(f'{ind}    string username = "{username}"')
        lines.append(f'{ind}    string description = "{user_data.get("description", "")}"')
        lines.append(f'{ind}}}')
        lines.append('')

        return lines

    def _export_object_prim(self, obj_id: str, obj_data: Dict, indent: int = 0) -> List[str]:
        """Export an object as prim."""
        ind = '    ' * indent
        lines = []

        name = obj_data.get('name', obj_id)

        lines.append(f'{ind}def "Objects/{obj_id}" (')
        lines.append(f'{ind}    customData = {{')
        lines.append(f'{ind}        string type = "object"')
        lines.append(f'{ind}    }}')
        lines.append(f'{ind}) {{')
        lines.append(f'{ind}    string name = "{name}"')
        lines.append(f'{ind}    string description = "{obj_data.get("description", "")}"')
        lines.append(f'{ind}}}')
        lines.append('')

        return lines

    def export_timeline(self, session_data: Dict, output_path: Path):
        """
        Export timeline/profiler data as time-sampled USD.

        This creates animated affect states that studios can use!

        Args:
            session_data: SessionProfiler data
            output_path: Path to .usda file
        """
        lines = []

        lines.append('#usda 1.0')
        lines.append('(')
        lines.append('    startTimeCode = 0')
        lines.append(f'    endTimeCode = {session_data.get("duration", 100)}')
        lines.append('    timeCodesPerSecond = 24')
        lines.append('    framesPerSecond = 24')
        lines.append(')')
        lines.append('')

        lines.append('def "Timeline" {')

        # Export each Noodling's affect timeline
        timelines = session_data.get('timelines', {})
        for noodling_id, events in timelines.items():
            lines.append(f'    def "Noodlings/{noodling_id}" {{')
            lines.append(f'        # Time-sampled 5-D affect vector')

            # Time-sampled attributes
            if events:
                timestamps = [e['timestamp'] for e in events]
                valences = [e['affect']['valence'] for e in events]
                arousals = [e['affect']['arousal'] for e in events]
                fears = [e['affect']['fear'] for e in events]

                lines.append(f'        float affect:valence.timeSamples = {{')
                for i, (t, v) in enumerate(zip(timestamps, valences)):
                    lines.append(f'            {t}: {v},')
                lines.append(f'        }}')

                lines.append(f'        float affect:arousal.timeSamples = {{')
                for i, (t, v) in enumerate(zip(timestamps, arousals)):
                    lines.append(f'            {t}: {v},')
                lines.append(f'        }}')

                lines.append(f'        float affect:fear.timeSamples = {{')
                for i, (t, v) in enumerate(zip(timestamps, fears)):
                    lines.append(f'            {t}: {v},')
                lines.append(f'        }}')

            lines.append(f'    }}')

        lines.append('}')

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"USD timeline exported to {output_path}")
