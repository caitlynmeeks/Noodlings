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
#   Setup Verification Script
#
#   A quick diagnostic tool that checks if your computer has
#   everything needed to run noodleMUSH. It looks for Python
#   version, required libraries, configuration files, and model
#   checkpoints - then reports what is ready and what needs to
#   be installed or configured. Run this first when setting up.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.check_setup
# PURPOSE:  Verify dependencies and prerequisites before running
# LAYER:    Backend / Utility
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   main()              Run all verification checks
#   check_python_version()  Verify Python 3.10+
#   check_module()      Test if a Python package is installed
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Setup verification script for cMUSH

Checks that all dependencies and prerequisites are available.

Author: Caitlyn + Claude
Date: October 2025
"""

import sys
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f" Python {version.major}.{version.minor}.{version.micro}")
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
        print(f" {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (pip install {package_name})")
        return False

def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f" {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (not found)")
        return False

def check_directory(path, description):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f" {description}: {path}/")
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

    # Noodlings package
    print("\n4. Noodlings Package")
    all_good &= check_file('../../noodlings/api.py', 'Noodlings API')
    all_good &= check_file('../../noodlings/models/noodling_phase4.py', 'Phase 4 model')

    # Directories
    print("\n5. Directories")
    check_directory('world', 'World data')
    check_directory('logs', 'Logs')
    check_directory('world/agents', 'Agent state')

    # Checkpoint
    print("\n6. Model Checkpoint")
    checkpoint_path = '../../models/checkpoints/best_checkpoint.npz'
    checkpoint_exists = check_file(checkpoint_path, 'Phase 4 checkpoint')
    if not checkpoint_exists:
        print("   Note: Train Phase 4 model first or update config.yaml path")
        # Don't mark as critical failure

    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print(" All checks passed! Ready to run cMUSH.")
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
