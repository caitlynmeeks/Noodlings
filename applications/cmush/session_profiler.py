"""
Session Profiler - Comprehensive logging for NoodleScope 2.0

Logs phenomenal states, events, metrics for real-time visualization and
@Kimmie interpretation. Data format designed for:
- Timeline scrubbing
- HSI (Hierarchical Separation Index) calculation
- Human-readable interpretation by @Kimmie
- Steve DiPaola demo (cause-and-effect storytelling)

Author: Caitlyn (with Claude's help)
Created: 2025-11-14
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SessionProfiler:
    """
    Logs detailed phenomenal state data for visualization and interpretation.

    Each timestep records:
    - Full 40-D phenomenal state (fast 16-D, medium 16-D, slow 8-D)
    - 5-D affect vector (valence, arousal, fear, sorrow, boredom)
    - Surprise, prediction error, speech events
    - HSI metrics (layer variance ratios)
    - Cheap thrills / mysticism scores
    - Context (what's happening in the world)
    """

    def __init__(self, session_id: str, output_dir: str = "profiler_sessions"):
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-agent timeline data
        self.agent_timelines: Dict[str, List[Dict[str, Any]]] = {}

        # Per-agent variance buffers for HSI calculation
        self.variance_buffers: Dict[str, Dict[str, List[float]]] = {}
        self.variance_window = 100  # Calculate HSI over last 100 timesteps

        # Session metadata
        self.session_start = time.time()
        self.session_metadata = {
            "session_id": session_id,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agents": []
        }

        logger.info(f"SessionProfiler initialized: {session_id}")

    def log_timestep(
        self,
        agent_id: str,
        phenomenal_state: np.ndarray,  # 40-D: fast(16) + medium(16) + slow(8)
        affect: np.ndarray,  # 5-D: valence, arousal, fear, sorrow, boredom
        surprise: float,
        speech_threshold: float,
        did_speak: bool,
        utterance: Optional[str] = None,
        prediction_error: float = 0.0,
        cheap_thrills_score: float = 0.0,
        mysticism_penalty: float = 0.0,
        event_context: Optional[str] = None,
        conversation_context: Optional[List] = None,
        **kwargs  # Additional metadata
    ):
        """
        Log a single timestep for an agent.

        Args:
            agent_id: Agent identifier
            phenomenal_state: Full 40-D state vector
            affect: 5-D affect vector
            surprise: Current surprise value
            speech_threshold: Threshold for speech
            did_speak: Whether agent spoke this timestep
            utterance: What the agent said (if any)
            prediction_error: L2 prediction error
            cheap_thrills_score: LLM embodiment score (0-10)
            mysticism_penalty: Philosophical language penalty
            event_context: What happened in the world
            conversation_context: Full conversation context given to LLM
            **kwargs: Additional metadata
        """

        # Initialize agent timeline if needed
        if agent_id not in self.agent_timelines:
            self.agent_timelines[agent_id] = []
            self.variance_buffers[agent_id] = {
                'fast': [],
                'medium': [],
                'slow': []
            }
            self.session_metadata['agents'].append(agent_id)
            logger.info(f"[SessionProfiler] Initialized timeline for {agent_id}")

        logger.info(f"[SessionProfiler] log_timestep called for {agent_id} - current timeline length: {len(self.agent_timelines[agent_id])}")

        # Extract layer components
        fast_state = phenomenal_state[:16].tolist() if len(phenomenal_state) >= 16 else []
        medium_state = phenomenal_state[16:32].tolist() if len(phenomenal_state) >= 32 else []
        slow_state = phenomenal_state[32:40].tolist() if len(phenomenal_state) >= 40 else []

        # Update variance buffers for HSI calculation
        if len(fast_state) == 16:
            self.variance_buffers[agent_id]['fast'].append(np.var(fast_state))
        if len(medium_state) == 16:
            self.variance_buffers[agent_id]['medium'].append(np.var(medium_state))
        if len(slow_state) == 8:
            self.variance_buffers[agent_id]['slow'].append(np.var(slow_state))

        # Trim buffers to window size
        for layer in ['fast', 'medium', 'slow']:
            if len(self.variance_buffers[agent_id][layer]) > self.variance_window:
                self.variance_buffers[agent_id][layer].pop(0)

        # Calculate HSI (Hierarchical Separation Index)
        hsi_metrics = self._calculate_hsi(agent_id)

        # Calculate layer velocities (rate of change)
        layer_velocities = self._calculate_layer_velocities(agent_id, fast_state, medium_state, slow_state)

        # Create timestep record
        timestamp = time.time() - self.session_start
        record = {
            # Timing
            "timestamp": timestamp,
            "wall_time": time.strftime("%H:%M:%S"),

            # Phenomenal state
            "phenomenal_state": {
                "fast": fast_state,
                "medium": medium_state,
                "slow": slow_state,
                "full": phenomenal_state.tolist() if hasattr(phenomenal_state, 'tolist') else list(phenomenal_state)
            },

            # Affect
            "affect": {
                "valence": float(affect[0]) if len(affect) > 0 else 0.0,
                "arousal": float(affect[1]) if len(affect) > 1 else 0.0,
                "fear": float(affect[2]) if len(affect) > 2 else 0.0,
                "sorrow": float(affect[3]) if len(affect) > 3 else 0.0,
                "boredom": float(affect[4]) if len(affect) > 4 else 0.0,
            },

            # Surprise and prediction
            "surprise": float(surprise),
            "speech_threshold": float(speech_threshold),
            "prediction_error": float(prediction_error),

            # Speech
            "did_speak": did_speak,
            "utterance": utterance,

            # Behavior metrics
            "cheap_thrills_score": float(cheap_thrills_score),
            "mysticism_penalty": float(mysticism_penalty),

            # Hierarchical metrics
            "hsi": hsi_metrics,
            "layer_velocities": layer_velocities,

            # Context
            "event": event_context,
            "conversation_context": conversation_context if conversation_context else [],

            # Additional metadata
            "metadata": kwargs
        }

        self.agent_timelines[agent_id].append(record)

    def _calculate_hsi(self, agent_id: str) -> Dict[str, float]:
        """
        Calculate Hierarchical Separation Index.

        HSI measures whether layers operate at different timescales.
        Good HSI: slow layer is 10-100x more stable than fast layer.

        Returns:
            Dict with HSI metrics
        """
        buffers = self.variance_buffers[agent_id]

        if len(buffers['fast']) < 10 or len(buffers['slow']) < 10:
            return {"hsi_slow_fast": 0.0, "hsi_medium_fast": 0.0, "status": "warming_up"}

        fast_var = np.mean(buffers['fast'])
        medium_var = np.mean(buffers['medium'])
        slow_var = np.mean(buffers['slow'])

        # Avoid division by zero
        if fast_var < 1e-10:
            return {"hsi_slow_fast": 0.0, "hsi_medium_fast": 0.0, "status": "no_variance"}

        hsi_slow_fast = slow_var / fast_var
        hsi_medium_fast = medium_var / fast_var

        # Assess HSI health
        status = "good"
        if hsi_slow_fast > 0.5:
            status = "poor_separation"  # Slow moving too fast
        elif hsi_slow_fast < 0.001:
            status = "too_stable"  # Slow not moving enough

        return {
            "hsi_slow_fast": float(hsi_slow_fast),
            "hsi_medium_fast": float(hsi_medium_fast),
            "fast_variance": float(fast_var),
            "medium_variance": float(medium_var),
            "slow_variance": float(slow_var),
            "status": status
        }

    def _calculate_layer_velocities(
        self,
        agent_id: str,
        fast_state: List[float],
        medium_state: List[float],
        slow_state: List[float]
    ) -> Dict[str, float]:
        """
        Calculate rate of change for each layer.

        Returns:
            Dict with velocity (L2 norm of change) for each layer
        """
        timeline = self.agent_timelines[agent_id]
        if len(timeline) < 2:
            return {"fast": 0.0, "medium": 0.0, "slow": 0.0}

        prev = timeline[-1]['phenomenal_state']

        fast_vel = 0.0
        medium_vel = 0.0
        slow_vel = 0.0

        if len(prev['fast']) == len(fast_state):
            fast_vel = np.linalg.norm(np.array(fast_state) - np.array(prev['fast']))
        if len(prev['medium']) == len(medium_state):
            medium_vel = np.linalg.norm(np.array(medium_state) - np.array(prev['medium']))
        if len(prev['slow']) == len(slow_state):
            slow_vel = np.linalg.norm(np.array(slow_state) - np.array(prev['slow']))

        return {
            "fast": float(fast_vel),
            "medium": float(medium_vel),
            "slow": float(slow_vel)
        }

    def get_timeline_segment(
        self,
        agent_id: str,
        start_time: float,
        end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Get timeline segment for @Kimmie interpretation.

        Args:
            agent_id: Agent to query
            start_time: Start timestamp (seconds since session start)
            end_time: End timestamp

        Returns:
            List of timestep records in range
        """
        if agent_id not in self.agent_timelines:
            return []

        return [
            record for record in self.agent_timelines[agent_id]
            if start_time <= record['timestamp'] <= end_time
        ]

    def export_session(self, filename: Optional[str] = None):
        """
        Export complete session to JSON file for @Kimmie and NoodleScope.

        Args:
            filename: Optional filename (default: session_id.json)
        """
        if filename is None:
            filename = f"{self.session_id}.json"

        output_path = self.output_dir / filename

        session_data = {
            "metadata": self.session_metadata,
            "duration": time.time() - self.session_start,
            "timelines": self.agent_timelines
        }

        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Session exported: {output_path}")
        return str(output_path)

    def get_realtime_feed(self, agent_id: str, last_n: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent timeline data for real-time NoodleScope display.

        Args:
            agent_id: Agent to query
            last_n: Number of recent timesteps to return

        Returns:
            List of recent timestep records
        """
        if agent_id not in self.agent_timelines:
            return []

        return self.agent_timelines[agent_id][-last_n:]
