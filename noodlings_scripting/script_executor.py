"""
Script Executor - Server-Side Runtime

Executes NoodleScripts in server backend with:
- Compilation and validation
- Event dispatch
- State persistence
- Sandbox (future: RestrictedPython)

Architecture:
- Scripts stored as Python source in world state
- Compiled on load/upload
- Instance state persisted separately
- Backend injects API implementations

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from typing import Dict, Any, Optional, Callable
import traceback
import logging

from .noodle_script import NoodleScript
from .noodlings_api import Noodlings, Debug, Transform, Prim, Vector3

logger = logging.getLogger(__name__)


class ScriptExecutor:
    """
    Execute NoodleScripts server-side.

    Handles:
    - Script compilation (Python source → class)
    - Script attachment (class → prim instance)
    - Event dispatch (trigger callbacks)
    - State persistence (save/restore instance variables)
    """

    def __init__(self):
        """Initialize script executor."""
        self.script_classes: Dict[str, type] = {}  # script_name → class
        self.script_instances: Dict[str, NoodleScript] = {}  # prim_id → instance

        logger.info("ScriptExecutor initialized")

    def compile_script(self, script_name: str, script_code: str) -> bool:
        """
        Compile Python script source to class.

        Args:
            script_name: Script identifier (e.g., "AnklebiterVendingMachine")
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
                    logger.info(f"✅ Compiled script: {script_name}")
                    return True

            logger.error(f"❌ No NoodleScript subclass found in {script_name}")
            return False

        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {script_name}: {e}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"❌ Error compiling {script_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    def attach_script(
        self,
        prim_id: str,
        script_name: str,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attach script instance to a prim.

        Args:
            prim_id: Prim/object ID (obj_001, agent_foo, etc.)
            script_name: Script class name
            initial_state: Optional state to restore (from world persistence)

        Returns:
            True if attached successfully
        """
        if script_name not in self.script_classes:
            logger.error(f"❌ Script class not found: {script_name}")
            return False

        try:
            # Instantiate script
            script_class = self.script_classes[script_name]
            script_instance = script_class()

            # Set up prim reference
            script_instance.prim = Prim(prim_id)
            script_instance.transform = Transform(prim_id)

            # Restore state if provided
            if initial_state:
                script_instance.SetState(initial_state)
                logger.debug(f"Restored state for {prim_id}: {initial_state}")

            # Store instance
            self.script_instances[prim_id] = script_instance

            # Call Start()
            try:
                script_instance.Start()
                logger.info(f"✅ Attached '{script_name}' to {prim_id}")
            except Exception as e:
                logger.error(f"❌ Error in {script_name}.Start(): {e}")
                logger.error(traceback.format_exc())
                # Keep script attached but log error

            return True

        except Exception as e:
            logger.error(f"❌ Error attaching {script_name} to {prim_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def detach_script(self, prim_id: str) -> Optional[Dict[str, Any]]:
        """
        Detach script from prim and return its state.

        Args:
            prim_id: Prim ID

        Returns:
            Script state dictionary (for persistence), or None
        """
        if prim_id not in self.script_instances:
            return None

        script = self.script_instances[prim_id]

        # Get state for persistence
        state = script.GetState()

        # Remove instance
        del self.script_instances[prim_id]
        logger.info(f"🗑️  Detached script from {prim_id}")

        return state

    def trigger_event(self, prim_id: str, event_name: str, *args) -> Any:
        """
        Trigger event callback on script.

        Args:
            prim_id: Prim that owns the script
            event_name: Event name (OnClick, OnUse, OnHear, etc.)
            args: Event arguments

        Returns:
            Return value from event callback (if any)
        """
        if prim_id not in self.script_instances:
            return None

        script = self.script_instances[prim_id]

        if not script.enabled:
            return None

        try:
            # Check if script has this event handler
            if hasattr(script, event_name):
                handler = getattr(script, event_name)
                result = handler(*args)
                return result
            else:
                # Script doesn't implement this event (normal, not an error)
                return None

        except Exception as e:
            logger.error(f"❌ Error in {prim_id}.{event_name}(): {e}")
            logger.error(f"   Args: {args}")
            logger.error(traceback.format_exc())
            return None

    def get_script_state(self, prim_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state of script for persistence.

        Args:
            prim_id: Prim ID

        Returns:
            State dictionary, or None if no script attached
        """
        if prim_id not in self.script_instances:
            return None

        script = self.script_instances[prim_id]
        return script.GetState()

    def update_all(self):
        """
        Call Update() on all active scripts.

        WARNING: Expensive! Only call if needed.
        Most scripts should use event callbacks, not Update().
        """
        for prim_id, script in list(self.script_instances.items()):
            if script.enabled:
                try:
                    script.Update()
                except Exception as e:
                    logger.error(f"❌ Error in {prim_id}.Update(): {e}")
                    logger.error(traceback.format_exc())

    def list_scripts(self) -> Dict[str, str]:
        """
        List all attached scripts.

        Returns:
            Dictionary of {prim_id: script_class_name}
        """
        return {
            prim_id: script.__class__.__name__
            for prim_id, script in self.script_instances.items()
        }

    def get_stats(self) -> Dict[str, int]:
        """
        Get executor statistics.

        Returns:
            Dictionary with counts
        """
        return {
            'compiled_classes': len(self.script_classes),
            'attached_instances': len(self.script_instances),
        }
