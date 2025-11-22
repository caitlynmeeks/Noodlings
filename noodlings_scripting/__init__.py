"""
Noodlings Scripting Runtime

Server-authoritative scripting system for kindled beings.

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
