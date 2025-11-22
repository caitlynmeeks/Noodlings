"""
Noodle Component - Consciousness State Access

Provides scripts access to Noodling phenomenal states, affect vectors,
and event hooks for state changes.

Usage:
    noodle = Noodlings.Find("agent_servnak")
    noodleComp = noodle.GetComponent("Noodle")

    affect = noodleComp.GetCurrentAffect()  # [val, aro, fear, sor, bor]
    surprise = noodleComp.GetSurprise()

    # Event hooks
    noodleComp.OnAffectChanged(lambda affect: Debug.Log(f"Affect: {affect}"))
    noodleComp.OnSurpriseSpike(lambda surprise: Debug.Log(f"Surprise: {surprise}"))

Author: Caitlyn + Claude (Spock Mode)
Date: November 21, 2025
"""

from typing import List, Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NoodleComponent:
    """
    Component for accessing Noodling consciousness states.

    Provides read access to:
    - 5-D Affect Vector (valence, arousal, fear, sorrow, boredom)
    - Full 40-D Phenomenal State (fast 16-D + medium 16-D + slow 8-D)
    - Surprise levels
    - Event hooks for state changes
    """

    # Backend implementation (injected by script_manager)
    _get_state_impl: Optional[Callable] = None

    def __init__(self, agent_id: str):
        """
        Initialize Noodle component for an agent.

        Args:
            agent_id: Agent identifier (e.g., "agent_servnak")
        """
        self.agent_id = agent_id
        self.enabled = True

        # Event callbacks
        self._on_affect_changed_callbacks: List[Callable] = []
        self._on_surprise_spike_callbacks: List[Callable] = []
        self._on_speech_callbacks: List[Callable] = []
        self._on_thought_callbacks: List[Callable] = []

        # Last known state (for change detection)
        self._last_affect: Optional[List[float]] = None
        self._last_surprise: Optional[float] = None

    @staticmethod
    def SetBackend(get_state_impl: Callable):
        """
        Inject backend implementation (called by script_manager).

        Args:
            get_state_impl: Function(agent_id) -> state_dict
        """
        NoodleComponent._get_state_impl = get_state_impl

    def GetCurrentAffect(self) -> Optional[List[float]]:
        """
        Get current 5-D affect vector.

        Returns:
            [valence, arousal, fear, sorrow, boredom] or None if unavailable

        Example:
            affect = noodleComp.GetCurrentAffect()
            valence = affect[0]  # -1.0 to 1.0
            arousal = affect[1]  # 0.0 to 1.0
        """
        if not NoodleComponent._get_state_impl:
            logger.error("NoodleComponent backend not initialized")
            return None

        try:
            state = NoodleComponent._get_state_impl(self.agent_id)
            fast_state = state.get('fast')

            if fast_state is not None and len(fast_state) >= 5:
                # First 5 dimensions of fast state are affect vector
                affect = list(fast_state[:5])
                return affect
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting affect for {self.agent_id}: {e}")
            return None

    def GetPhenomenalState(self) -> Optional[Dict[str, Any]]:
        """
        Get full phenomenal state (40-D hierarchical states).

        Returns:
            Dictionary with keys:
            - 'fast': 16-D fast layer state (includes 5-D affect)
            - 'medium': 16-D medium layer state
            - 'slow': 8-D slow layer state
            - 'surprise': Current surprise level
            - 'surprise_threshold': Speech threshold
            - 'step': Number of timesteps

        Example:
            state = noodleComp.GetPhenomenalState()
            fast = state['fast']  # 16-D numpy array
            surprise = state['surprise']
        """
        if not NoodleComponent._get_state_impl:
            logger.error("NoodleComponent backend not initialized")
            return None

        try:
            return NoodleComponent._get_state_impl(self.agent_id)
        except Exception as e:
            logger.error(f"Error getting phenomenal state for {self.agent_id}: {e}")
            return None

    def GetSurprise(self) -> Optional[float]:
        """
        Get current surprise level (prediction error).

        Returns:
            Surprise value (0.0 = no surprise, higher = more surprising)

        Example:
            surprise = noodleComp.GetSurprise()
            if surprise > 0.5:
                Debug.Log(f"{self.agent_id} is very surprised!")
        """
        if not NoodleComponent._get_state_impl:
            logger.error("NoodleComponent backend not initialized")
            return None

        try:
            state = NoodleComponent._get_state_impl(self.agent_id)
            return state.get('surprise', 0.0)
        except Exception as e:
            logger.error(f"Error getting surprise for {self.agent_id}: {e}")
            return None

    def GetSurpriseThreshold(self) -> Optional[float]:
        """
        Get surprise threshold for speech triggering.

        Returns:
            Threshold value (agent speaks when surprise exceeds this)
        """
        if not NoodleComponent._get_state_impl:
            logger.error("NoodleComponent backend not initialized")
            return None

        try:
            state = NoodleComponent._get_state_impl(self.agent_id)
            return state.get('surprise_threshold', 0.3)
        except Exception as e:
            logger.error(f"Error getting surprise threshold for {self.agent_id}: {e}")
            return None

    # Event Hook Registration

    def OnAffectChanged(self, callback: Callable[[List[float]], None]):
        """
        Register callback for when affect vector changes.

        Args:
            callback: Function(affect_vector) called when affect changes

        Example:
            def affect_changed(affect):
                Debug.Log(f"New affect: val={affect[0]:.2f}")

            noodleComp.OnAffectChanged(affect_changed)
        """
        self._on_affect_changed_callbacks.append(callback)
        logger.info(f"Registered OnAffectChanged callback for {self.agent_id}")

    def OnSurpriseSpike(self, callback: Callable[[float], None]):
        """
        Register callback for when surprise exceeds threshold.

        Args:
            callback: Function(surprise_value) called on surprise spike

        Example:
            noodleComp.OnSurpriseSpike(lambda s: Debug.Log(f"Surprise spike: {s}"))
        """
        self._on_surprise_spike_callbacks.append(callback)
        logger.info(f"Registered OnSurpriseSpike callback for {self.agent_id}")

    def OnSpeech(self, callback: Callable[[str], None]):
        """
        Register callback for when Noodling speaks.

        Args:
            callback: Function(speech_text) called when agent speaks

        Example:
            noodleComp.OnSpeech(lambda text: Debug.Log(f"Said: {text}"))
        """
        self._on_speech_callbacks.append(callback)
        logger.info(f"Registered OnSpeech callback for {self.agent_id}")

    def OnThought(self, callback: Callable[[str], None]):
        """
        Register callback for when Noodling has internal thought.

        Args:
            callback: Function(thought_text) called on rumination

        Example:
            noodleComp.OnThought(lambda t: Debug.Log(f"Thinks: {t}"))
        """
        self._on_thought_callbacks.append(callback)
        logger.info(f"Registered OnThought callback for {self.agent_id}")

    # Internal methods (called by backend)

    def _fire_affect_changed(self, new_affect: List[float]):
        """Internal: Fire affect changed event to all callbacks."""
        for callback in self._on_affect_changed_callbacks:
            try:
                callback(new_affect)
            except Exception as e:
                logger.error(f"Error in OnAffectChanged callback: {e}")

    def _fire_surprise_spike(self, surprise: float):
        """Internal: Fire surprise spike event."""
        for callback in self._on_surprise_spike_callbacks:
            try:
                callback(surprise)
            except Exception as e:
                logger.error(f"Error in OnSurpriseSpike callback: {e}")

    def _fire_speech(self, text: str):
        """Internal: Fire speech event."""
        for callback in self._on_speech_callbacks:
            try:
                callback(text)
            except Exception as e:
                logger.error(f"Error in OnSpeech callback: {e}")

    def _fire_thought(self, text: str):
        """Internal: Fire thought event."""
        for callback in self._on_thought_callbacks:
            try:
                callback(text)
            except Exception as e:
                logger.error(f"Error in OnThought callback: {e}")

    def __repr__(self):
        return f"NoodleComponent({self.agent_id})"
