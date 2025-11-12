#!/usr/bin/env python3
"""Quick import test before starting cMUSH"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../consilience_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

print("Testing imports...")
print("-" * 50)

try:
    print("1. TrainingDataCollector...", end=" ")
    from training_data_collector import TrainingDataCollector
    collector = TrainingDataCollector('../../training/data/cmush_real')
    print(f"✓ (stats: {collector.get_stats()})")
except Exception as e:
    print(f"✗ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("2. Server imports (world, auth)...", end=" ")
    from world import World
    from auth import AuthManager
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 50)
print("✓ All imports successful!")
print("\nNow you can start cMUSH with:")
print("  ./start.sh")
print("\nOr test manually with:")
print("  python3 server.py")
