"""
USD Importer for noodleMUSH

Import USD layer files (.usda) and parse Noodling prims.

USD Terminology:
- Stage: The composed scene
- Prim: Basic scene object
- Layer: A .usda/.usdc file
- Schema: Defines prim properties

This importer reads .usda ASCII files without requiring the full USD library.
For production use with binary .usdc files, install the official USD Python package.

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from pathlib import Path
from typing import Dict, List, Any
import re


class USDImporter:
    """
    Import USD layer files and extract Noodling prims.

    Lightweight ASCII .usda parser (no USD library required for basic imports).
    """

    def __init__(self):
        pass

    def import_layer(self, layer_path: Path) -> Dict[str, Any]:
        """
        Import USD layer file.

        Args:
            layer_path: Path to .usda or .usdc file

        Returns:
            Dictionary with parsed prims:
            {
                'noodlings': [list of Noodling prims],
                'rooms': [list of room prims],
                'objects': [list of object prims],
                'users': [list of user prims]
            }
        """
        if layer_path.suffix == '.usdc':
            raise NotImplementedError("Binary .usdc import requires USD Python library. Use .usda (ASCII) instead.")

        with open(layer_path, 'r') as f:
            usd_content = f.read()

        # Parse prims from USD content
        result = {
            'noodlings': self._extract_noodling_prims(usd_content),
            'rooms': self._extract_room_prims(usd_content),
            'objects': self._extract_object_prims(usd_content),
            'users': self._extract_user_prims(usd_content)
        }

        print(f"Imported USD layer: {layer_path}")
        print(f"  - {len(result['noodlings'])} Noodling prims")
        print(f"  - {len(result['rooms'])} Room prims")
        print(f"  - {len(result['objects'])} Object prims")
        print(f"  - {len(result['users'])} User prims")

        return result

    def _extract_noodling_prims(self, content: str) -> List[Dict]:
        """Extract Noodling prims (those with NoodlingSchema)."""
        noodlings = []

        # Find all prims with NoodlingSchema
        pattern = r'def\s+"Noodlings/([^"]+)"[^{]*\{([^}]+)\}'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            noodling_id = match.group(1)
            prim_content = match.group(2)

            # Extract properties
            noodling = {
                'id': noodling_id,
                'name': self._extract_property(prim_content, 'name'),
                'species': self._extract_property(prim_content, 'species'),
                'description': self._extract_property(prim_content, 'description'),
                'personality': {
                    'extraversion': float(self._extract_property(prim_content, 'extraversion') or 0.5),
                    'curiosity': float(self._extract_property(prim_content, 'curiosity') or 0.5),
                    'impulsivity': float(self._extract_property(prim_content, 'impulsivity') or 0.5),
                    'emotional_volatility': float(self._extract_property(prim_content, 'emotional_volatility') or 0.5),
                }
            }

            noodlings.append(noodling)

        return noodlings

    def _extract_room_prims(self, content: str) -> List[Dict]:
        """Extract room prims."""
        rooms = []

        pattern = r'def\s+Xform\s+"([^"]+)"[^{]*customData[^}]*string\s+type\s*=\s*"room"[^}]*\}[^{]*\{([^}]+)\}'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            room_id = match.group(1)
            prim_content = match.group(2)

            room = {
                'id': room_id,
                'name': self._extract_property(prim_content, 'name'),
                'description': self._extract_property(prim_content, 'description'),
            }

            rooms.append(room)

        return rooms

    def _extract_object_prims(self, content: str) -> List[Dict]:
        """Extract object prims."""
        objects = []

        pattern = r'def\s+"Objects/([^"]+)"[^{]*\{([^}]+)\}'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            obj_id = match.group(1)
            prim_content = match.group(2)

            obj = {
                'id': obj_id,
                'name': self._extract_property(prim_content, 'name'),
                'description': self._extract_property(prim_content, 'description'),
            }

            objects.append(obj)

        return objects

    def _extract_user_prims(self, content: str) -> List[Dict]:
        """Extract user prims."""
        users = []

        pattern = r'def\s+"Users/([^"]+)"[^{]*\{([^}]+)\}'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            user_id = match.group(1)
            prim_content = match.group(2)

            user = {
                'id': user_id,
                'username': self._extract_property(prim_content, 'username'),
                'description': self._extract_property(prim_content, 'description'),
            }

            users.append(user)

        return users

    def _extract_property(self, content: str, prop_name: str) -> str:
        """Extract a property value from prim content."""
        pattern = rf'{prop_name}\s*=\s*"([^"]*)"'
        match = re.search(pattern, content)
        return match.group(1) if match else ''
