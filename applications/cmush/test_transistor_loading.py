#!/usr/bin/env python3
"""
Test transistor loading system - verify Unity-style pattern works.
"""

import yaml
import logging
from cognitive_components import COMPONENT_REGISTRY

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_transistor_loading():
    """Test loading transistors from recipe using Unity pattern."""

    # Load mysterious_stranger recipe
    with open('recipes/mysterious_stranger.yaml', 'r') as f:
        recipe = yaml.safe_load(f)

    cognitive_components = recipe.get('cognitive_components', {})

    if not cognitive_components:
        logger.error("No cognitive_components found in recipe!")
        return False

    logger.info(f"Found {len(cognitive_components)} components in recipe")

    # Try to instantiate each transistor using Unity pattern
    success_count = 0
    for component_name, component_config in cognitive_components.items():
        transistor_type = component_config.get('type')

        if not transistor_type:
            logger.warning(f"Component '{component_name}' missing 'type'")
            continue

        transistor_class = COMPONENT_REGISTRY.get(transistor_type)
        if not transistor_class:
            logger.error(f"Unknown transistor type '{transistor_type}'")
            continue

        try:
            # Unity-style factory method
            transistor = transistor_class.from_config(component_config)
            logger.info(f" Created {transistor_type} (salience={transistor.salience:.2f})")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ Failed to create {transistor_type}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nResult: {success_count}/{len(cognitive_components)} transistors loaded successfully")
    return success_count == len(cognitive_components)

if __name__ == '__main__':
    success = test_transistor_loading()
    exit(0 if success else 1)
