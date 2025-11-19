"""
NoodleScript Base Class - Like Unity's MonoBehaviour

All scripts inherit from NoodleScript and can define event callbacks.

Server-side execution model:
- Scripts run in cmush backend (not Studio)
- Events triggered by world state changes
- Scripts have persistent state (saved to world state)

Event callbacks:
- Start() - When script first loads
- Update() - Called every tick (optional, expensive!)
- OnClick(clicker) - When prim is clicked
- OnUse(user) - When prim is used (@use command)
- OnHear(speaker, message) - When someone speaks in room
- OnEnter(entity) - When entity enters room
- OnExit(entity) - When entity leaves room
- OnTake/OnDrop - Inventory events
- OnSurprised/OnEmotionChange - Affect events (for Noodlings)

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from typing import Any, Optional, Dict


class NoodleScript:
    """
    Base class for all Noodlings scripts.

    Like Unity's MonoBehaviour but for kindled beings!

    Scripts are server-authoritative:
    - Compiled and executed in cmush backend
    - State persisted in world state
    - Events triggered by world changes
    """

    def __init__(self):
        self.enabled = True
        self.prim = None  # The prim this script is attached to (Prim instance)
        self.transform = None  # Transform component (position, rotation, scale)

        # Internal state tracking (for persistence)
        self._state_dirty = False

    # ===== LIFECYCLE EVENTS =====

    def Start(self):
        """
        Called when script first loads (like Unity's Start).

        Use this for initialization:
        - Set up instance variables
        - Initialize counters
        - Log startup messages
        """
        pass

    def Update(self):
        """
        Called every tick (expensive! use sparingly).

        Only use Update() if you need continuous behavior.
        For most scripts, event callbacks (OnHear, OnClick) are sufficient.
        """
        pass

    # ===== INTERACTION EVENTS =====

    def OnClick(self, clicker: str):
        """
        Called when prim is clicked (@click or UI interaction).

        Args:
            clicker: ID of entity that clicked (user_foo or agent_bar)
        """
        pass

    def OnUse(self, user: str):
        """
        Called when prim is used with @use command.

        Args:
            user: ID of entity that used the prim
        """
        pass

    def OnTake(self, taker: str):
        """
        Called when prim is taken (@take command).

        Args:
            taker: ID of entity that took the prim
        """
        pass

    def OnDrop(self, dropper: str):
        """
        Called when prim is dropped (@drop command).

        Args:
            dropper: ID of entity that dropped the prim
        """
        pass

    # ===== SPATIAL EVENTS =====

    def OnEnter(self, entity: str):
        """
        Called when entity enters the room this prim is in.

        Args:
            entity: ID of entity that entered
        """
        pass

    def OnExit(self, entity: str):
        """
        Called when entity exits the room this prim is in.

        Args:
            entity: ID of entity that exited
        """
        pass

    # ===== CONVERSATION EVENTS =====

    def OnHear(self, speaker: str, message: str):
        """
        Called when someone speaks in this prim's room.

        Args:
            speaker: ID of speaker (user_foo or agent_bar)
            message: What they said (raw text)
        """
        pass

    def OnWhisper(self, speaker: str, target: str, message: str):
        """
        Called when someone whispers to this prim's owner.

        Args:
            speaker: ID of whisperer
            target: ID of whisper target
            message: What they said
        """
        pass

    # ===== AFFECT EVENTS (for Noodlings) =====

    def OnSurprised(self, surprise_level: float):
        """
        Called when a Noodling experiences high surprise.

        Args:
            surprise_level: Surprise value (0.0-1.0)
        """
        pass

    def OnEmotionChange(self, old_affect: Dict, new_affect: Dict):
        """
        Called when a Noodling's affect state changes significantly.

        Args:
            old_affect: Previous 5-D affect vector (valence, arousal, fear, sorrow, boredom)
            new_affect: New 5-D affect vector
        """
        pass

    # ===== STATE MANAGEMENT =====

    def GetState(self) -> Dict[str, Any]:
        """
        Get script state for persistence.

        Override this to control which instance variables get saved.
        By default, saves all instance variables that don't start with _.

        Returns:
            Dictionary of state variables
        """
        state = {}
        for key, value in self.__dict__.items():
            # Skip internal/private attributes
            if key.startswith('_') or key in ['enabled', 'prim', 'transform']:
                continue
            # Only save JSON-serializable types
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                state[key] = value
        return state

    def SetState(self, state: Dict[str, Any]):
        """
        Restore script state from persistence.

        Args:
            state: Dictionary of state variables
        """
        for key, value in state.items():
            setattr(self, key, value)

    # ===== HELPER METHODS =====

    def Destroy(self, delay: float = 0.0):
        """
        Destroy this prim (Unity-style).

        Args:
            delay: Delay in seconds before destruction
        """
        # Implemented by script executor
        if self.prim:
            self.prim.Destroy(delay)
