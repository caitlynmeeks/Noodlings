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
#   Noodle Component
#
#   This is the bridge between scripts and Noodling internals.
#   When you have a Noodling character, you can attach this
#   component to read its emotional state:
#
#     noodle = Noodlings.Find("agent_phi")
#     noodleComp = noodle.GetComponent("Noodle")
#
#     affect = noodleComp.GetCurrentAffect()
#     # Returns [valence, arousal, fear, sorrow, boredom]
#     # valence: -1 (sad) to +1 (happy)
#     # arousal: 0 (calm) to 1 (excited)
#
#   You can also subscribe to events:
#     noodleComp.OnSurpriseSpike(lambda s: print(f"Surprised: {s}"))
#     noodleComp.OnSpeech(lambda text: print(f"Said: {text}"))
#
#   This lets scripts react to what Noodlings are feeling - for
#   example, playing sound effects when they're surprised, or
#   triggering animations when their mood changes.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   noodlings_scripting.noodle_component
# PURPOSE:  Script access to Noodling consciousness state
# LAYER:    Scripting
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NoodleComponent  Unity-style component for reading Noodling state
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
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
        # Register architecture event callbacks
        self._on_register_filled_callbacks: List[Callable] = []
        self._on_all_registers_ready_callbacks: List[Callable] = []
        self._on_cycle_start_callbacks: List[Callable] = []
        self._on_cycle_end_callbacks: List[Callable] = []
        self._on_register_error_callbacks: List[Callable] = []

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

    # ===== REGISTER ARCHITECTURE API =====

    # Backend implementation for register queries (injected by script_manager)
    _get_register_state_impl: Optional[Callable] = None
    _get_register_output_impl: Optional[Callable] = None
    _check_registers_ready_impl: Optional[Callable] = None

    @staticmethod
    def SetRegisterBackend(get_register_state, get_register_output, check_registers_ready):
        """
        Inject backend implementations for register queries.

        Args:
            get_register_state: Function(agent_id, transistor_type) -> state_str
            get_register_output: Function(agent_id, transistor_type) -> output_dict
            check_registers_ready: Function(agent_id) -> bool
        """
        NoodleComponent._get_register_state_impl = get_register_state
        NoodleComponent._get_register_output_impl = get_register_output
        NoodleComponent._check_registers_ready_impl = check_registers_ready

    def GetRegisterState(self, transistor_type: str) -> Optional[str]:
        """
        Get current state of a transistor register.

        Args:
            transistor_type: Transistor type (e.g., "IntuitionTransistor", "AffectTransistor")

        Returns:
            State string: "empty", "computing", "ready", "error", or None if unavailable

        Example:
            state = noodleComp.GetRegisterState("IntuitionTransistor")
            if state == "ready":
                Debug.Log("Intuition register is loaded!")
        """
        if not NoodleComponent._get_register_state_impl:
            logger.error("Register backend not initialized")
            return None

        try:
            return NoodleComponent._get_register_state_impl(self.agent_id, transistor_type)
        except Exception as e:
            logger.error(f"Error getting register state for {transistor_type}: {e}")
            return None

    def GetRegisterOutput(self, transistor_type: str) -> Optional[Dict[str, Any]]:
        """
        Get stored output from a transistor register.

        Args:
            transistor_type: Transistor type

        Returns:
            Dictionary with keys: 'text', 'salience', 'metadata', or None if empty

        Example:
            output = noodleComp.GetRegisterOutput("AffectTransistor")
            if output:
                Debug.Log(f"Affect: {output['text']}")
        """
        if not NoodleComponent._get_register_output_impl:
            logger.error("Register backend not initialized")
            return None

        try:
            return NoodleComponent._get_register_output_impl(self.agent_id, transistor_type)
        except Exception as e:
            logger.error(f"Error getting register output for {transistor_type}: {e}")
            return None

    def AreAllRegistersReady(self) -> bool:
        """
        Check if all enabled transistor registers are ready for integration.

        Returns:
            True if all enabled registers in READY state, False otherwise

        Example:
            if noodleComp.AreAllRegistersReady():
                Debug.Log("All registers loaded! Can integrate now.")
        """
        if not NoodleComponent._check_registers_ready_impl:
            logger.error("Register backend not initialized")
            return False

        try:
            return NoodleComponent._check_registers_ready_impl(self.agent_id)
        except Exception as e:
            logger.error(f"Error checking registers ready: {e}")
            return False

    # Register event hooks

    def OnRegisterFilled(self, callback: Callable[[str, str, Dict], None]):
        """
        Register callback for when a transistor register is filled.

        Args:
            callback: Function(transistor_type, cycle_id, output_dict) called when register ready

        Example:
            def on_filled(transistor, cycle, output):
                Debug.Log(f"{transistor} filled: {output['text'][:50]}")
            noodleComp.OnRegisterFilled(on_filled)
        """
        self._on_register_filled_callbacks.append(callback)
        logger.info(f"Registered OnRegisterFilled callback for {self.agent_id}")

    def OnAllRegistersReady(self, callback: Callable[[str], None]):
        """
        Register callback for when all transistor registers are ready.

        Args:
            callback: Function(cycle_id) called when all enabled registers READY

        Example:
            noodleComp.OnAllRegistersReady(lambda cycle: Debug.Log(f"Cycle {cycle[:8]} ready!"))
        """
        self._on_all_registers_ready_callbacks.append(callback)
        logger.info(f"Registered OnAllRegistersReady callback for {self.agent_id}")

    def OnCycleStart(self, callback: Callable[[str, str], None]):
        """
        Register callback for when cognition cycle starts.

        Args:
            callback: Function(cycle_id, perception_text) called at cycle start

        Example:
            noodleComp.OnCycleStart(lambda cid, text: Debug.Log(f"Cycle {cid[:8]}: {text}"))
        """
        self._on_cycle_start_callbacks.append(callback)
        logger.info(f"Registered OnCycleStart callback for {self.agent_id}")

    def OnCycleEnd(self, callback: Callable[[str, str], None]):
        """
        Register callback for when cognition cycle ends.

        Args:
            callback: Function(cycle_id, response_text) called after registers cleared

        Example:
            noodleComp.OnCycleEnd(lambda cid, resp: Debug.Log(f"Cycle {cid[:8]} done: {resp}"))
        """
        self._on_cycle_end_callbacks.append(callback)
        logger.info(f"Registered OnCycleEnd callback for {self.agent_id}")

    def OnRegisterError(self, callback: Callable[[str, str, str], None]):
        """
        Register callback for when a transistor register fails.

        Args:
            callback: Function(transistor_type, cycle_id, error_msg) called on error

        Example:
            noodleComp.OnRegisterError(lambda t, c, e: Debug.Log(f"ERROR {t}: {e}"))
        """
        self._on_register_error_callbacks.append(callback)
        logger.info(f"Registered OnRegisterError callback for {self.agent_id}")

    # Internal fire methods (called by backend)

    def _fire_register_filled(self, transistor_type: str, cycle_id: str, output: Dict):
        """Internal: Fire register filled event."""
        for callback in self._on_register_filled_callbacks:
            try:
                callback(transistor_type, cycle_id, output)
            except Exception as e:
                logger.error(f"Error in OnRegisterFilled callback: {e}")

    def _fire_all_registers_ready(self, cycle_id: str):
        """Internal: Fire all registers ready event."""
        for callback in self._on_all_registers_ready_callbacks:
            try:
                callback(cycle_id)
            except Exception as e:
                logger.error(f"Error in OnAllRegistersReady callback: {e}")

    def _fire_cycle_start(self, cycle_id: str, perception_text: str):
        """Internal: Fire cycle start event."""
        for callback in self._on_cycle_start_callbacks:
            try:
                callback(cycle_id, perception_text)
            except Exception as e:
                logger.error(f"Error in OnCycleStart callback: {e}")

    def _fire_cycle_end(self, cycle_id: str, response_text: str):
        """Internal: Fire cycle end event."""
        for callback in self._on_cycle_end_callbacks:
            try:
                callback(cycle_id, response_text)
            except Exception as e:
                logger.error(f"Error in OnCycleEnd callback: {e}")

    def _fire_register_error(self, transistor_type: str, cycle_id: str, error_msg: str):
        """Internal: Fire register error event."""
        for callback in self._on_register_error_callbacks:
            try:
                callback(transistor_type, cycle_id, error_msg)
            except Exception as e:
                logger.error(f"Error in OnRegisterError callback: {e}")

    def __repr__(self):
        return f"NoodleComponent({self.agent_id})"

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
