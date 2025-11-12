#!/usr/bin/env python3
"""
Setup verification script for cMUSH

Checks that all dependencies and prerequisites are available.

Author: cMUSH Project
Date: October 2025
"""

import sys
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.10+)")
        return False

def check_module(module_name, package_name=None):
    """Check if a Python module is available."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (pip install {package_name})")
        return False

def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (not found)")
        return False

def check_directory(path, description):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}/")
        return True
    else:
        print(f"⚠ {description}: {path}/ (will be created)")
        return True  # Not critical

def main():
    """Run all checks."""
    print("cMUSH Setup Verification")
    print("=" * 50)

    all_good = True

    # Python version
    print("\n1. Python Environment")
    all_good &= check_python_version()

    # Python packages
    print("\n2. Python Dependencies")
    all_good &= check_module('websockets')
    all_good &= check_module('aiohttp')
    all_good &= check_module('yaml', 'pyyaml')
    all_good &= check_module('mlx')
    all_good &= check_module('numpy')

    # Core files
    print("\n3. Core Files")
    all_good &= check_file('config.yaml', 'Configuration')
    all_good &= check_file('server.py', 'Server')
    all_good &= check_file('world.py', 'World manager')
    all_good &= check_file('commands.py', 'Command parser')
    all_good &= check_file('agent_bridge.py', 'Agent bridge')
    all_good &= check_file('llm_interface.py', 'LLM interface')
    all_good &= check_file('auth.py', 'Authentication')
    all_good &= check_file('web/index.html', 'Web client')

    # Consilience core
    print("\n4. Consilience Core")
    core_path = '../../consilience_core'
    all_good &= check_file(f'{core_path}/api.py', 'API wrapper')
    all_good &= check_file(f'{core_path}/consilience_phase4.py', 'Phase 4 model')

    # Directories
    print("\n5. Directories")
    check_directory('world', 'World data')
    check_directory('logs', 'Logs')
    check_directory('world/agents', 'Agent state')

    # Checkpoint
    print("\n6. Model Checkpoint")
    checkpoint_path = '../../consilience_core/checkpoints_phase4/best_checkpoint.npz'
    checkpoint_exists = check_file(checkpoint_path, 'Phase 4 checkpoint')
    if not checkpoint_exists:
        print("   Note: Train Phase 4 model first or update config.yaml path")
        # Don't mark as critical failure

    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("✓ All checks passed! Ready to run cMUSH.")
        print("\nNext steps:")
        print("  1. Ensure LMStudio (or Ollama) is running")
        print("  2. Initialize world: python3 init_world.py")
        print("  3. Start server: ./start.sh")
        print("  4. Open browser: http://localhost:8080")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nInstall dependencies:")
        print("  pip3 install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
