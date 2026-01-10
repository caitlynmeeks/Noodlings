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
#   Noodlings Scripting Runtime
#
#   This is the scripting system that lets you write custom
#   behaviors for objects in the world. It works like Unity or
#   Second Life scripting - you write Python code that responds
#   to events (someone clicks an object, enters a room, etc.)
#
#   The key classes are:
#     - NoodleScript: Base class for all scripts (like MonoBehaviour)
#     - Noodlings: API for spawning characters and sending messages
#     - Debug: Logging utilities (Debug.Log, Debug.LogError)
#     - Prim: Any object in the world (like GameObject)
#
#   Scripts run on the SERVER, not in the editor. This means:
#     - No cheating (server authoritative)
#     - Persistent state (survives server restarts)
#     - All players see the same behavior
#
# ──────────────────────────────────────────────────────────────
# MODULE:   noodlings_scripting
# PURPOSE:  Package init for server-side scripting runtime
# LAYER:    Scripting
# ──────────────────────────────────────────────────────────────
#
# EXPORTS:
#   NoodleScript      Base class for all scripts
#   ScriptExecutor    Compiles and runs scripts
#   Noodlings         API for spawning/finding entities
#   Debug             Logging utilities
#   Vector3           3D vector math
#   Transform         Position/rotation/scale
#   Prim              World entity reference
#   NoodleComponent   Access Noodling consciousness state
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Noodlings Scripting Runtime

Server-authoritative scripting system for noodling.

This package provides:
- NoodleScript base class (event-driven callbacks)
- Script executor (sandbox, lifecycle management)
- Noodlings API (Rez, Find, SendMessage, etc.)
- Debug utilities

Architecture:
- Scripts are Python source code stored in world state
- Backend (cmush) executes scripts server-side
- Studio (noodlestudio) provides editor UI and uploads to backend
- Clean separation: Studio = editor, Backend = runtime

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from .noodle_script import NoodleScript
from .noodlings_api import Noodlings, Debug, Vector3, Transform, Prim
from .script_executor import ScriptExecutor
from .noodle_component import NoodleComponent

__version__ = "1.0.0"

__all__ = [
    'NoodleScript',
    'ScriptExecutor',
    'Noodlings',
    'Debug',
    'Vector3',
    'Transform',
    'Prim',
    'NoodleComponent',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
