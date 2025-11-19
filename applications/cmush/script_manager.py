"""
Script Manager - Backend Integration

Integrates noodlings_scripting runtime with cmush backend.

Responsibilities:
- Load scripts from example_scripts/ and world state
- Compile and attach scripts to prims
- Inject backend API implementations (Noodlings.Rez → actual spawning)
- Route world events to script callbacks
- Persist script state to world state

Server-authoritative model:
- Scripts execute in cmush backend (not Studio)
- Studio uploads scripts via API (future)
- Scripts can directly manipulate world state

Author: Caitlyn + Claude
Date: November 18, 2025
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add noodlings_scripting to path
script_runtime_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, script_runtime_path)

from noodlings_scripting import (
    ScriptExecutor,
    NoodleScript,
    Noodlings,
    Debug,
    Prim
)

logger = logging.getLogger(__name__)


class ScriptManager:
    """
    Manages NoodleScript execution in cmush backend.

    Bridges:
    - World state ↔ Script storage
    - Agent manager ↔ Noodlings.Rez()
    - Event system ↔ Script callbacks
    """

    def __init__(self, world, agent_manager):
        """
        Initialize script manager.

        Args:
            world: World instance (for state access)
            agent_manager: AgentManager instance (for rezzing agents)
        """
        self.world = world
        self.agent_manager = agent_manager
        self.executor = ScriptExecutor()

        # Scripts directory (~/Noodlings/PROJECT/Scripts/ - user's project)
        # TODO: Make project name configurable (for now: DefaultProject)
        noodlings_root = Path.home() / "Noodlings"
        self.project_name = "DefaultProject"
        self.project_dir = noodlings_root / self.project_name
        self.scripts_dir = self.project_dir / "Scripts"

        # Ensure project structure exists
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ScriptManager initialized")
        logger.info(f"📜 Scripts directory: {self.scripts_dir}")

        # Inject backend API implementations
        self._inject_backend_api()

        # Load scripts
        self._load_example_scripts()
        self._load_world_scripts()

    def _inject_backend_api(self):
        """Inject backend implementations into Noodlings API."""

        # Store reference to script_manager for closures
        script_mgr = self

        # Sync wrapper for async Rez() - schedules rezzing for next event loop
        def rez_impl(recipe: str, room: str) -> Optional[str]:
            # Create task and store for retrieval
            task = asyncio.create_task(script_mgr._backend_rez_noodling(recipe, room))
            # Store task reference (scripts will get placeholder for now)
            script_mgr._pending_rezzes = getattr(script_mgr, '_pending_rezzes', [])
            script_mgr._pending_rezzes.append(task)
            # Return placeholder prim ID (will be resolved async)
            return f"pending_rez_{len(script_mgr._pending_rezzes)}"

        # Noodlings.RezPrim() → create object (sync)
        def rez_prim_impl(prim_type: str, name: str, room: str) -> Optional[str]:
            return script_mgr._backend_rez_prim(prim_type, name, room)

        # Noodlings.Find() → lookup entity (sync)
        def find_impl(prim_id: str) -> Optional[Prim]:
            return script_mgr._backend_find(prim_id)

        # Noodlings.SendMessage() → agent perception (sync for now)
        def send_message_impl(target_id: str, message: str):
            script_mgr._backend_send_message(target_id, message)

        # Noodlings.Broadcast() → room broadcast (sync for now)
        def broadcast_impl(room_id: str, message: str):
            script_mgr._backend_broadcast_sync(room_id, message)

        # Inject implementations
        Noodlings.SetBackend(
            rez_impl=rez_impl,
            rez_prim_impl=rez_prim_impl,
            find_impl=find_impl,
            send_message_impl=send_message_impl,
            broadcast_impl=broadcast_impl
        )

        # Debug logging callback
        def log_callback(level: str, message: str):
            if level == 'INFO':
                logger.info(f"[Script] {message}")
            elif level == 'WARNING':
                logger.warning(f"[Script] {message}")
            elif level == 'ERROR':
                logger.error(f"[Script] {message}")

        Debug.SetLogCallback(log_callback)

        logger.info("✅ Backend API injected into Noodlings runtime")

    def _load_example_scripts(self):
        """Load and compile all scripts from example_scripts/."""
        if not self.scripts_dir.exists():
            logger.warning(f"Scripts directory not found: {self.scripts_dir}")
            return

        loaded_count = 0
        for script_file in self.scripts_dir.glob("*.py"):
            script_name = script_file.stem

            # Skip __init__.py
            if script_name.startswith("__"):
                continue

            try:
                with open(script_file, 'r') as f:
                    script_code = f.read()

                success = self.executor.compile_script(script_name, script_code)
                if success:
                    loaded_count += 1

            except Exception as e:
                logger.error(f"Error loading script {script_name}: {e}")

        logger.info(f"📜 Loaded {loaded_count} example scripts")

    def _load_world_scripts(self):
        """Load scripts attached to prims in world state."""
        attached_count = 0

        for obj_id, obj in self.world.objects.items():
            script_data = obj.get('script')
            if not script_data or not script_data.get('name'):
                continue

            script_name = script_data['name']
            script_state = script_data.get('state', {})

            # Attach script with persisted state
            success = self.executor.attach_script(obj_id, script_name, script_state)
            if success:
                attached_count += 1

        logger.info(f"🔗 Attached {attached_count} scripts from world state")

    # ===== BACKEND API IMPLEMENTATIONS =====

    async def _backend_rez_noodling(self, recipe: str, room: str) -> Optional[str]:
        """Backend implementation of Noodlings.Rez()."""
        logger.info(f"🔥 Script rezzing Noodling: {recipe} in {room}")

        try:
            # Use agent_manager to spawn
            result = await self.agent_manager.spawn_agent(
                user_id="script_system",
                recipe_name=recipe,
                room_id=room,
                fresh_memory=False
            )

            if result.get('success'):
                agent_id = result.get('agent_id')
                logger.info(f"✅ Rezzed {recipe} as {agent_id}")
                return agent_id
            else:
                logger.error(f"❌ Failed to rez {recipe}: {result.get('message')}")
                return None

        except Exception as e:
            logger.error(f"Error rezzing {recipe}: {e}")
            return None

    def _backend_rez_prim(self, prim_type: str, name: str, room: str) -> Optional[str]:
        """Backend implementation of Noodlings.RezPrim()."""
        logger.info(f"🎁 Script rezzing prim: {name} ({prim_type}) in {room}")

        try:
            obj_id = self.world.create_object(
                name=name,
                description=f"A {prim_type} created by a script.",
                owner="script_system",
                location=room,
                portable=True,
                takeable=True,
                obj_type=prim_type
            )

            logger.info(f"✅ Rezzed prim {name} as {obj_id}")
            return obj_id

        except Exception as e:
            logger.error(f"Error rezzing prim {name}: {e}")
            return None

    def _backend_find(self, prim_id: str) -> Optional[Prim]:
        """Backend implementation of Noodlings.Find()."""
        # Check if it's an object
        obj = self.world.get_object(prim_id)
        if obj:
            return Prim(
                prim_id=prim_id,
                prim_type=obj.get('type', 'prim'),
                room=obj.get('location')
            )

        # Check if it's an agent
        agent = self.world.agents.get(prim_id)
        if agent:
            return Prim(
                prim_id=prim_id,
                prim_type='noodling',
                room=agent.get('current_room')
            )

        # Check if it's a user
        user = self.world.users.get(prim_id)
        if user:
            return Prim(
                prim_id=prim_id,
                prim_type='user',
                room=user.get('current_room')
            )

        return None

    def _backend_send_message(self, target_id: str, message: str):
        """Backend implementation of Noodlings.SendMessage()."""
        logger.info(f"📨 Script sending to {target_id}: {message}")
        # TODO: Inject into agent perception
        # For now, just log it

    def _backend_broadcast_sync(self, room_id: str, message: str):
        """Backend implementation of Noodlings.Broadcast() (sync version)."""
        logger.info(f"📢 Script broadcasting to {room_id}: {message}")
        # TODO: Broadcast via server
        # For now, just log it

    # ===== EVENT ROUTING =====

    def on_hear(self, prim_id: str, speaker: str, message: str):
        """Route OnHear event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnHear', speaker, message)

    def on_click(self, prim_id: str, clicker: str):
        """Route OnClick event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnClick', clicker)

    def on_use(self, prim_id: str, user: str):
        """Route OnUse event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnUse', user)

    def on_take(self, prim_id: str, taker: str):
        """Route OnTake event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnTake', taker)

    def on_drop(self, prim_id: str, dropper: str):
        """Route OnDrop event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnDrop', dropper)

    def on_enter(self, prim_id: str, entity: str):
        """Route OnEnter event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnEnter', entity)

    def on_exit(self, prim_id: str, entity: str):
        """Route OnExit event to prim's script."""
        self.executor.trigger_event(prim_id, 'OnExit', entity)

    # ===== BROADCAST EVENTS TO ROOM =====

    def broadcast_hear_to_room(self, room_id: str, speaker: str, message: str):
        """Broadcast OnHear to all scripted prims in room."""
        room = self.world.get_room(room_id)
        if not room:
            return

        for obj_id in room.get('objects', []):
            obj = self.world.get_object(obj_id)
            if obj and obj.get('script') and obj['script'].get('name'):
                self.on_hear(obj_id, speaker, message)

    def broadcast_enter_to_room(self, room_id: str, entity: str):
        """Broadcast OnEnter to all scripted prims in room."""
        room = self.world.get_room(room_id)
        if not room:
            return

        for obj_id in room.get('objects', []):
            obj = self.world.get_object(obj_id)
            if obj and obj.get('script') and obj['script'].get('name'):
                self.on_enter(obj_id, entity)

    def broadcast_exit_to_room(self, room_id: str, entity: str):
        """Broadcast OnExit to all scripted prims in room."""
        room = self.world.get_room(room_id)
        if not room:
            return

        for obj_id in room.get('objects', []):
            obj = self.world.get_object(obj_id)
            if obj and obj.get('script') and obj['script'].get('name'):
                self.on_exit(obj_id, entity)

    # ===== SCRIPT LIFECYCLE =====

    def attach_script(self, prim_id: str, script_name: str, script_code: Optional[str] = None) -> bool:
        """
        Attach script to prim.

        Args:
            prim_id: Object ID
            script_name: Script class name
            script_code: Optional Python source (if uploading new script)

        Returns:
            True if successful
        """
        # Compile script if code provided
        if script_code:
            success = self.executor.compile_script(script_name, script_code)
            if not success:
                return False

        # Attach to executor
        success = self.executor.attach_script(prim_id, script_name)
        if not success:
            return False

        # Update world state
        obj = self.world.get_object(prim_id)
        if obj:
            obj['script'] = {
                'name': script_name,
                'code': script_code,
                'state': {},
                'version': 1,
                'compiled': True
            }
            self.world.save_all()

        return True

    def detach_script(self, prim_id: str):
        """Detach script from prim."""
        # Get final state
        state = self.executor.detach_script(prim_id)

        # Update world state
        obj = self.world.get_object(prim_id)
        if obj:
            obj['script'] = None
            self.world.save_all()

        logger.info(f"🗑️  Detached script from {prim_id}")

    def save_all_script_states(self):
        """Save all script states to world persistence."""
        for prim_id in list(self.executor.script_instances.keys()):
            state = self.executor.get_script_state(prim_id)
            if state:
                obj = self.world.get_object(prim_id)
                if obj and obj.get('script'):
                    obj['script']['state'] = state

        self.world.save_all()
        logger.debug("💾 Saved all script states")

    def get_stats(self) -> Dict[str, Any]:
        """Get script manager statistics."""
        executor_stats = self.executor.get_stats()
        return {
            **executor_stats,
            'scripts_dir': str(self.scripts_dir),
        }
