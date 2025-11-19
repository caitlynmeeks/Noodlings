"""
NoodleScript Base Class - Like Unity's MonoBehaviour

All scripts inherit from NoodleScript and can define event callbacks.

Event callbacks:
- OnClick(clicker) - When prim is clicked
- OnUse(user) - When prim is used (@use command)
- OnEnter(entity) - When entity enters room
- OnExit(entity) - When entity leaves room
- Start() - When script first loads
- Update() - Called every tick (optional, expensive!)

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from typing import Any, Optional


class NoodleScript:
    """
    Base class for all Noodlings scripts.

    Like Unity's MonoBehaviour but for kindled beings!
    """

    def __init__(self):
        self.enabled = True
        self.prim = None  # The prim this script is attached to
        self.transform = None  # Transform component (position, etc.)

    # ===== LIFECYCLE EVENTS =====

    def Start(self):
        """Called when script first loads (like Unity)."""
        pass

    def Update(self):
        """Called every tick (expensive! use sparingly)."""
        pass

    # ===== INTERACTION EVENTS =====

    def OnClick(self, clicker: str):
        """
        Called when prim is clicked.

        Args:
            clicker: ID of entity that clicked (user or noodling)
        """
        pass

    def OnUse(self, user: str):
        """
        Called when prim is used (@use command).

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
        Called when entity enters this room.

        Args:
            entity: ID of entity that entered
        """
        pass

    def OnExit(self, entity: str):
        """
        Called when entity exits this room.

        Args:
            entity: ID of entity that exited
        """
        pass

    # ===== CONVERSATION EVENTS =====

    def OnHear(self, speaker: str, message: str):
        """
        Called when someone speaks in this room.

        Args:
            speaker: ID of speaker
            message: What they said
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

    # ===== AFFECT EVENTS =====

    def OnSurprised(self, surprise_level: float):
        """
        Called when Noodling experiences high surprise.

        Args:
            surprise_level: Surprise value (0.0-1.0)
        """
        pass

    def OnEmotionChange(self, old_affect: dict, new_affect: dict):
        """
        Called when Noodling's affect state changes significantly.

        Args:
            old_affect: Previous 5-D affect vector
            new_affect: New 5-D affect vector
        """
        pass

    # ===== HELPER METHODS =====

    def Destroy(self, delay: float = 0.0):
        """Destroy this prim (Unity-style)."""
        if delay > 0.0:
            # TODO: Schedule destruction
            pass
        else:
            # TODO: Immediate destruction
            pass
