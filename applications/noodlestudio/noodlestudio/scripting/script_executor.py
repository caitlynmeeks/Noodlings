"""
Script Executor - Runs NoodleScripts in sandbox

Executes Python scripts with Noodlings API available.
Like Unity's script runtime!

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from typing import Dict, Any, Optional
import traceback
from .noodle_script import NoodleScript
from .noodlings_api import Noodlings, Debug, Transform, Prim


class ScriptExecutor:
    """
    Execute NoodleScripts safely in sandbox.

    Provides Noodlings API to scripts.
    Handles event dispatch.
    """

    def __init__(self):
        self.scripts: Dict[str, NoodleScript] = {}  # prim_id -> script instance
        self.script_classes: Dict[str, type] = {}  # script_name -> class

    def compile_script(self, script_name: str, script_code: str) -> bool:
        """
        Compile Python script and register class.

        Args:
            script_name: Name of script
            script_code: Python source code

        Returns:
            True if compiled successfully
        """
        try:
            # Create namespace with Noodlings API
            namespace = {
                'NoodleScript': NoodleScript,
                'Noodlings': Noodlings,
                'Debug': Debug,
                'Transform': Transform,
                'Prim': Prim,
                'Vector3': Vector3,
            }

            # Execute script to define class
            exec(script_code, namespace)

            # Find the NoodleScript subclass
            for name, obj in namespace.items():
                if (isinstance(obj, type) and
                    issubclass(obj, NoodleScript) and
                    obj is not NoodleScript):
                    self.script_classes[script_name] = obj
                    Debug.Log(f"Compiled script: {script_name}")
                    return True

            Debug.LogError(f"No NoodleScript class found in {script_name}")
            return False

        except Exception as e:
            Debug.LogError(f"Script compile error: {e}")
            traceback.print_exc()
            return False

    def attach_script(self, prim_id: str, script_name: str) -> bool:
        """
        Attach script to a prim.

        Args:
            prim_id: Prim to attach to
            script_name: Script class name

        Returns:
            True if attached successfully
        """
        if script_name not in self.script_classes:
            Debug.LogError(f"Script not found: {script_name}")
            return False

        try:
            # Instantiate script
            script_class = self.script_classes[script_name]
            script_instance = script_class()

            # Set up references
            script_instance.prim = Prim(prim_id)
            script_instance.transform = Transform(prim_id)

            # Store instance
            self.scripts[prim_id] = script_instance

            # Call Start()
            script_instance.Start()

            Debug.Log(f"Attached script '{script_name}' to {prim_id}")
            return True

        except Exception as e:
            Debug.LogError(f"Script attach error: {e}")
            traceback.print_exc()
            return False

    def trigger_event(self, prim_id: str, event_name: str, *args) -> Any:
        """
        Trigger event callback on script.

        Args:
            prim_id: Prim that owns the script
            event_name: Event name (OnClick, OnUse, etc.)
            args: Event arguments

        Returns:
            Return value from event callback
        """
        if prim_id not in self.scripts:
            return None

        script = self.scripts[prim_id]

        if not script.enabled:
            return None

        try:
            # Check if script has this event handler
            if hasattr(script, event_name):
                handler = getattr(script, event_name)
                return handler(*args)

        except Exception as e:
            Debug.LogError(f"Script event error ({event_name}): {e}")
            traceback.print_exc()

        return None

    def update_all(self):
        """Call Update() on all active scripts (expensive!)."""
        for prim_id, script in self.scripts.items():
            if script.enabled:
                try:
                    script.Update()
                except Exception as e:
                    Debug.LogError(f"Script update error ({prim_id}): {e}")

    def remove_script(self, prim_id: str):
        """Remove script from prim."""
        if prim_id in self.scripts:
            del self.scripts[prim_id]
            Debug.Log(f"Removed script from {prim_id}")


# Import Vector3 for namespace
from .noodlings_api import Vector3


# Global executor instance (like Unity's runtime)
SCRIPT_EXECUTOR = ScriptExecutor()
