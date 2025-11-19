"""
Session Data Loader - The Krugerrand Profiler Backend

Loads session profiler data from JSON files or live API.
Designed for Logic Pro-style timeline visualization in Qt.

Author: Caitlyn + Claude
Date: November 17, 2025
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TimelineEvent:
    """Single event in a Noodling's timeline."""
    timestamp: float
    wall_time: str

    # 5-D Affect Vector
    valence: float
    arousal: float
    fear: float
    sorrow: float
    boredom: float

    # Phenomenal State (40-D)
    fast_state: List[float]
    medium_state: List[float]
    slow_state: List[float]
    full_state: List[float]

    # Surprise & Metrics
    surprise: float
    speech_threshold: float
    hsi_slow_fast: float
    hsi_medium_fast: float

    # FACS/Body Language
    facs_codes: List[tuple]  # [(code, description), ...]
    body_codes: List[tuple]
    expression_description: str

    # Speech/Thought/Action
    did_speak: bool
    utterance: Optional[str]
    event_type: str
    responding_to: str

    # Context
    event_context: str
    conversation_context: List[Dict]


@dataclass
class SessionData:
    """Complete session data for all Noodlings."""
    session_id: str
    start_time: str
    duration: float
    noodlings: List[str]  # List of Noodling IDs
    timelines: Dict[str, List[TimelineEvent]]  # noodling_id -> events


class SessionLoader:
    """
    Loads session profiler data from JSON or live API.

    The Krugerrand Profiler Backend - worth its weight in gold!
    """

    def __init__(self, api_base: str = "http://localhost:8081/api"):
        self.api_base = api_base

    def load_live_session(self) -> Optional[SessionData]:
        """
        Load current live session from API.

        Returns:
            SessionData or None if error
        """
        try:
            url = f"{self.api_base}/profiler/live-session"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            return self._parse_session_data(data)

        except Exception as e:
            print(f"Error loading live session: {e}")
            return None

    def load_session_file(self, filepath: Path) -> Optional[SessionData]:
        """
        Load session from JSON file.

        Args:
            filepath: Path to session JSON file

        Returns:
            SessionData or None if error
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            return self._parse_session_data(data)

        except Exception as e:
            print(f"Error loading session file {filepath}: {e}")
            return None

    def _parse_session_data(self, data: Dict) -> SessionData:
        """Parse raw JSON into SessionData structure."""
        metadata = data.get('metadata', {})
        timelines_raw = data.get('timelines', {})

        # Parse each Noodling's timeline
        timelines = {}
        for noodling_id, events in timelines_raw.items():
            timeline_events = []

            for event_data in events:
                # Parse affect
                affect = event_data.get('affect', {})

                # Parse phenomenal state (can be dict or list)
                phenom_raw = event_data.get('phenomenal_state', {})
                if isinstance(phenom_raw, list):
                    # Empty list or raw array - create empty dict
                    phenom = {'fast': [], 'medium': [], 'slow': [], 'full': phenom_raw}
                else:
                    phenom = phenom_raw

                # Parse FACS/body codes
                facs_raw = event_data.get('facs_codes', [])
                facs_codes = []
                if facs_raw:
                    for code_data in facs_raw:
                        if isinstance(code_data, list) and len(code_data) >= 2:
                            facs_codes.append((code_data[0], code_data[1]))
                        elif isinstance(code_data, tuple):
                            facs_codes.append(code_data)

                body_raw = event_data.get('body_codes', [])
                body_codes = []
                if body_raw:
                    for code_data in body_raw:
                        if isinstance(code_data, list) and len(code_data) >= 2:
                            body_codes.append((code_data[0], code_data[1]))
                        elif isinstance(code_data, tuple):
                            body_codes.append(code_data)

                # Parse HSI
                hsi = event_data.get('hsi', {})

                event = TimelineEvent(
                    timestamp=event_data.get('timestamp', 0.0),
                    wall_time=event_data.get('wall_time', ''),

                    # 5-D Affect
                    valence=affect.get('valence', 0.0),
                    arousal=affect.get('arousal', 0.0),
                    fear=affect.get('fear', 0.0),
                    sorrow=affect.get('sorrow', 0.0),
                    boredom=affect.get('boredom', 0.0),

                    # Phenomenal state
                    fast_state=phenom.get('fast', []),
                    medium_state=phenom.get('medium', []),
                    slow_state=phenom.get('slow', []),
                    full_state=phenom.get('full', []),

                    # Metrics
                    surprise=event_data.get('surprise', 0.0),
                    speech_threshold=event_data.get('speech_threshold', 0.0),
                    hsi_slow_fast=hsi.get('hsi_slow_fast', 0.0),
                    hsi_medium_fast=hsi.get('hsi_medium_fast', 0.0),

                    # FACS/Body
                    facs_codes=facs_codes,
                    body_codes=body_codes,
                    expression_description=event_data.get('expression_description', ''),

                    # Speech
                    did_speak=event_data.get('did_speak', False),
                    utterance=event_data.get('utterance'),
                    event_type=event_data.get('event_type', 'unknown'),
                    responding_to=event_data.get('responding_to', ''),

                    # Context
                    event_context=event_data.get('event', ''),
                    conversation_context=event_data.get('conversation_context', [])
                )

                timeline_events.append(event)

            timelines[noodling_id] = timeline_events

        return SessionData(
            session_id=metadata.get('session_id', 'unknown'),
            start_time=metadata.get('start_time', ''),
            duration=data.get('duration', 0.0),
            noodlings=list(timelines.keys()),
            timelines=timelines
        )

    def list_session_files(self, sessions_dir: Path) -> List[Path]:
        """
        List all session JSON files in directory.

        Args:
            sessions_dir: Directory containing session files

        Returns:
            List of paths to session JSON files
        """
        if not sessions_dir.exists():
            return []

        return sorted(sessions_dir.glob("cmush_session_*.json"), reverse=True)
