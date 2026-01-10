#!/usr/bin/env python3
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
#   Recipe to Prefab Converter
#
#   Converts old-style YAML recipe files into the newer .prefab
#   format. Prefabs have unique reverse-DNS identifiers (like
#   com.noodlings.characters.red), metadata sections, and better
#   organization. Run this to upgrade your character recipes.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.convert_recipes_to_prefabs
# PURPOSE:  Migrate recipes/*.yaml to prefabs/*.prefab format
# LAYER:    Backend / Migration Tool
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   convert_recipe_to_prefab()  Transform single recipe to prefab
#   generate_prefab_id()        Create unique reverse-DNS ID
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Convert legacy YAML recipes to .prefab format.

Migrates recipes/*.yaml to prefabs/*.prefab with:
- Unique reverse-DNS identifiers
- Metadata section
- Preserved cognitive_components configuration

Author: Caitlyn + Claude
"""

import yaml
import sys
import uuid
from pathlib import Path
from datetime import datetime
from prefab_loader import PrefabLoader

def generate_prefab_id(recipe_filename: str) -> str:
    """
    Generate unique prefab ID from recipe filename.

    Args:
        recipe_filename: e.g., "red_fire_anklebiter.yaml"

    Returns:
        Unique ID: e.g., "com.noodlings.characters.red_fire_anklebiter"
    """
    stem = Path(recipe_filename).stem  # Remove .yaml

    # Category mapping based on name patterns
    if any(word in stem for word in ['fire', 'anklebiter']):
        category = "characters"
    elif any(word in stem for word in ['stranger', 'mysterious']):
        category = "npcs"
    elif any(word in stem for word in ['phi', 'kitten', 'cat', 'dog', 'fox']):
        category = "creatures"
    elif any(word in stem for word in ['servnak', 'robot']):
        category = "robots"
    elif 'test' in stem or 'example' in stem:
        category = "test"
    else:
        category = "characters"  # Default

    return f"com.noodlings.{category}.{stem}"


def convert_recipe_to_prefab(recipe_path: Path) -> Dict:
    """
    Convert legacy recipe YAML to prefab format.

    Args:
        recipe_path: Path to .yaml recipe file

    Returns:
        Prefab data dict
    """
    with open(recipe_path, 'r') as f:
        recipe_data = yaml.safe_load(f)

    # Generate unique ID
    prefab_id = generate_prefab_id(recipe_path.name)

    # Build metadata section
    metadata = {
        'id': prefab_id,
        'uuid': str(uuid.uuid4()),  # Unique instance identifier
        'name': recipe_data.get('name', recipe_path.stem),
        'version': '1.0.0',
        'created': datetime.now().strftime('%Y-%m-%d'),
        'modified': datetime.now().strftime('%Y-%m-%d'),
        'author': 'Garcia River Forest Research Station',
        'description': recipe_data.get('description', ''),
        'tags': []
    }

    # Auto-generate tags from species and name
    if 'species' in recipe_data:
        metadata['tags'].append(recipe_data['species'])

    # Build character section
    character = {
        'species': recipe_data.get('species', 'unknown'),
        'pronoun': recipe_data.get('pronoun', 'they'),
        'age': recipe_data.get('age', 'unknown'),
        'description': recipe_data.get('description', ''),
        'identity_prompt': recipe_data.get('identity_prompt', ''),
        'language_mode': recipe_data.get('language_mode', 'verbal'),
        'enlightenment': recipe_data.get('enlightenment', False)
    }

    # Build prefab structure
    prefab = {
        'metadata': metadata,
        'character': character,
        'personality': recipe_data.get('personality', {}),
        'appetites': recipe_data.get('appetites', {}),
        'cognitive_components': recipe_data.get('cognitive_components', {}),
        'llm': recipe_data.get('llm', {}),
        'constraints': recipe_data.get('constraints', {})
    }

    # Add affective_reinforcement if present
    if 'affective_reinforcement' in recipe_data:
        prefab['affective_reinforcement'] = recipe_data['affective_reinforcement']

    return prefab


def main():
    """Convert all recipes to prefabs."""
    recipes_dir = Path("recipes")
    loader = PrefabLoader("prefabs")

    if not recipes_dir.exists():
        print(f"ERROR: Recipes directory not found: {recipes_dir}")
        return 1

    recipe_files = list(recipes_dir.glob("*.yaml"))
    if not recipe_files:
        print("No recipe files found to convert")
        return 0

    print(f"Converting {len(recipe_files)} recipes to prefabs...")
    print("=" * 60)

    converted = 0
    skipped = 0
    errors = 0

    for recipe_path in recipe_files:
        try:
            print(f"\nProcessing: {recipe_path.name}")

            # Convert
            prefab_data = convert_recipe_to_prefab(recipe_path)
            prefab_id = prefab_data['metadata']['id']

            # Validate
            is_valid, validation_errors = loader.validate(prefab_data)
            if not is_valid:
                print(f"  VALIDATION FAILED:")
                for err in validation_errors:
                    print(f"    - {err}")
                errors += 1
                continue

            # Check if already exists
            existing = loader.load(prefab_id)
            if existing:
                print(f"  SKIPPED: Prefab already exists ({prefab_id})")
                skipped += 1
                continue

            # Save
            loader.save(prefab_id, prefab_data)
            print(f"  SUCCESS: Created {prefab_id}")
            converted += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print("\n" + "=" * 60)
    print(f"Conversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped} (already exist)")
    print(f"  Errors:    {errors}")

    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
