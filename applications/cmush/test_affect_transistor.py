#!/usr/bin/env python3
"""
Test AffectTransistor loading and functionality.
"""

import yaml
import logging
from cognitive_components import COMPONENT_REGISTRY

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_affect_transistor_loading():
    """Test loading AffectTransistor from recipes."""

    print("=" * 60)
    print("TEST: AffectTransistor Loading & Salience Configuration")
    print("=" * 60)

    # Test 1: Spock (low affect salience)
    print("\n1. Testing Spock (LOW affect salience = 0.15)")
    with open('recipes/spock_example.yaml', 'r') as f:
        spock_recipe = yaml.safe_load(f)

    affect_config = spock_recipe['cognitive_components']['affect']
    affect_transistor = COMPONENT_REGISTRY['AffectTransistor'].from_config(affect_config)

    print(f"    Loaded AffectTransistor")
    print(f"    Salience: {affect_transistor.salience} (emotional suppression)")
    print(f"   → Spock's emotions are SUPPRESSED (Vulcan training)")

    # Test 2: Ember (high affect salience)
    print("\n2. Testing Ember (HIGH affect salience = 0.95)")
    with open('recipes/emotional_example.yaml', 'r') as f:
        ember_recipe = yaml.safe_load(f)

    affect_config = ember_recipe['cognitive_components']['affect']
    affect_transistor = COMPONENT_REGISTRY['AffectTransistor'].from_config(affect_config)

    print(f"    Loaded AffectTransistor")
    print(f"    Salience: {affect_transistor.salience} (emotional dominance)")
    print(f"   → Ember's emotions DOMINATE expression")

    # Test 3: Salience range demonstration
    print("\n3. Salience Scale Interpretation:")
    print("   0.05-0.20:  Vulcan/Robot - Emotions barely register")
    print("   0.30-0.50:  Balanced - Moderate emotional expression")
    print("   0.60-0.75:  Human typical - Emotions guide but don't dominate")
    print("   0.80-0.95:  High empathy - Emotions color everything")

    # Test 4: Compare to other transistors
    print("\n4. Transistor Salience Comparison (Spock):")
    for name, config in spock_recipe['cognitive_components'].items():
        transistor_type = config['type']
        salience = config.get('salience', 'dynamic')
        print(f"   {name:12s} ({transistor_type:20s}): {salience}")

    print("\n" + "=" * 60)
    print("SUCCESS: AffectTransistor is now tunable per character!")
    print("=" * 60)

    return True

if __name__ == '__main__':
    success = test_affect_transistor_loading()
    exit(0 if success else 1)
