"""
Audio API - Scripting interface for real-time audio.

Provides context.noodle.audio in ScriptedFacets with Unity-like API:

    // TTS/STT Events
    context.noodle.audio.onTranscriptionReady((text) => {
        context.log("User said: " + text);
    });

    // TTS Control
    context.noodle.audio.speak("Hello!");
    context.noodle.audio.listen();
    context.noodle.audio.interrupt();

    // Spatial Audio (NEW)
    context.noodle.audio.attachSource("footsteps", {
        clip: "sounds/step.ogg",
        volume: 0.5,
        spatial: true,
        maxDistance: 10
    });
    context.noodle.audio.play("footsteps");
    context.noodle.audio.playAt("explosion", [10, 0, 5], {volume: 1.0});

    // Polling
    if (context.noodle.audio.isSpeaking) {
        // wait...
    }
    var lastText = context.noodle.audio.lastTranscription;

Model Labels:
    - AUDIO_IN: Speech-to-text model (Whisper)
    - AUDIO_OUT: Text-to-speech model (ElevenLabs, etc.)

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
Extended: December 21, 2025 - Added spatial audio emitter support
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AudioEmitter:
    """
    Spatial audio emitter attached to an entity.

    Used for sound effects, ambient audio, footsteps, etc.
    """
    id: str
    clip: Optional[str] = None
    volume: float = 1.0
    pitch: float = 1.0
    spatial: bool = True
    loop: bool = False
    autoplay: bool = False
    playing: bool = False
    paused: bool = False

    # Spatial properties
    offset: tuple = (0, 0, 0)  # Local offset from entity
    ref_distance: float = 1.0
    max_distance: float = 50.0
    rolloff: float = 1.0

    # Cone (directional audio)
    cone_inner_angle: float = 360.0
    cone_outer_angle: float = 360.0
    cone_outer_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JavaScript-compatible dict."""
        return {
            'id': self.id,
            'clip': self.clip,
            'volume': self.volume,
            'pitch': self.pitch,
            'spatial': self.spatial,
            'loop': self.loop,
            'autoplay': self.autoplay,
            'playing': self.playing,
            'paused': self.paused,
            'offset': list(self.offset),
            'refDistance': self.ref_distance,
            'maxDistance': self.max_distance,
            'rolloff': self.rolloff,
            'coneInnerAngle': self.cone_inner_angle,
            'coneOuterAngle': self.cone_outer_angle,
            'coneOuterGain': self.cone_outer_gain
        }


@dataclass
class AudioAPIState:
    """
    Snapshot of audio state for JavaScript access.

    Updated at sync points.
    """
    is_listening: bool = False
    is_speaking: bool = False
    is_transcribing: bool = False
    last_transcription: str = ""
    partial_transcription: str = ""
    state: str = "IDLE"

    # Pending commands (from JS to audio facet)
    pending_speak: Optional[str] = None
    pending_listen: bool = False
    pending_stop_listening: bool = False
    pending_interrupt: bool = False

    # Configuration
    voice: str = "default"
    sensitivity: float = 0.5

    # Spatial audio emitters
    emitters: Dict[str, AudioEmitter] = field(default_factory=dict)

    # Pending spatial commands
    pending_emitter_updates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JavaScript-compatible dict."""
        return {
            'isListening': self.is_listening,
            'isSpeaking': self.is_speaking,
            'isTranscribing': self.is_transcribing,
            'lastTranscription': self.last_transcription,
            'partialTranscription': self.partial_transcription,
            'state': self.state,
            'voice': self.voice,
            'sensitivity': self.sensitivity
        }


class AudioAPI:
    """
    Audio scripting API for context.noodle.audio.

    Provides both synchronous state access and async command queuing.
    Commands are processed at sync points by AudioStreamFacet.

    Example (JavaScript in ScriptedFacet):
        function process(inputs, context) {
            // Check state
            if (context.noodle.audio.isSpeaking) {
                return {waiting: true};
            }

            // React to transcription
            var text = context.noodle.audio.lastTranscription;
            if (text.includes("hello")) {
                context.noodle.audio.speak("Hi there!");
            }

            // Start listening
            context.noodle.audio.listen();

            return {processed: true};
        }
    """

    def __init__(self):
        """Initialize Audio API."""
        self._state = AudioAPIState()
        self._audio_facet = None  # Reference to AudioStreamFacet

        # Event handlers (JavaScript callbacks)
        self._event_handlers: Dict[str, List[Callable]] = {
            'transcription_ready': [],
            'speech_start': [],
            'speech_end': [],
            'listening_start': [],
            'listening_end': []
        }

        # Pending events to emit to JS (collected during sync)
        self._pending_events: List[Dict] = []

    # ========== Facet Connection ==========

    def set_audio_facet(self, facet):
        """
        Connect to AudioStreamFacet instance.

        Called by FacetExecutor when AudioStreamFacet is instantiated.
        """
        self._audio_facet = facet

        # Subscribe to facet events
        facet.on("transcription_ready", self._on_transcription_ready)
        facet.on("speech_start", self._on_speech_start)
        facet.on("speech_end", self._on_speech_end)
        facet.on("listening_start", self._on_listening_start)
        facet.on("listening_end", self._on_listening_end)

        logger.info("[AudioAPI] Connected to AudioStreamFacet")

    # ========== State Properties (Polling) ==========

    @property
    def is_listening(self) -> bool:
        """Check if audio is listening for input."""
        return self._state.is_listening

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is playing."""
        return self._state.is_speaking

    @property
    def is_transcribing(self) -> bool:
        """Check if transcription is in progress."""
        return self._state.is_transcribing

    @property
    def last_transcription(self) -> str:
        """Get last completed transcription."""
        return self._state.last_transcription

    @property
    def partial_transcription(self) -> str:
        """Get current partial transcription (real-time)."""
        return self._state.partial_transcription

    @property
    def state(self) -> str:
        """Get current audio state (IDLE, LISTENING, TRANSCRIBING, SPEAKING)."""
        return self._state.state

    # ========== Control Methods ==========

    def speak(self, text: str, voice: Optional[str] = None,
              speed: float = 1.0, interrupt: bool = False):
        """
        Queue text for TTS synthesis.

        Args:
            text: Text to speak
            voice: Voice ID (uses default if None)
            speed: Playback speed (1.0 = normal)
            interrupt: If True, interrupt current speech
        """
        self._state.pending_speak = text

        if voice:
            self._state.voice = voice

        logger.info(f"[AudioAPI] Queued speak: {text[:50]}...")

    def listen(self):
        """Start listening for audio input."""
        self._state.pending_listen = True
        logger.info("[AudioAPI] Queued listen start")

    def stop_listening(self):
        """Stop listening and process accumulated audio."""
        self._state.pending_stop_listening = True
        logger.info("[AudioAPI] Queued listen stop")

    def interrupt(self):
        """Interrupt current TTS playback."""
        self._state.pending_interrupt = True
        logger.info("[AudioAPI] Queued interrupt")

    # ========== Spatial Audio Emitters ==========

    def attach_source(self, emitter_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Attach a new audio source to this entity.

        Args:
            emitter_id: Unique ID for this emitter
            config: Emitter configuration:
                - clip: Audio file path
                - volume: 0.0 to 1.0 (default 1.0)
                - spatial: Enable 3D positioning (default True)
                - loop: Loop playback (default False)
                - maxDistance: Max audible distance (default 50)
                - refDistance: Reference distance for attenuation (default 1)
                - offset: Local position offset [x, y, z]
        """
        config = config or {}

        emitter = AudioEmitter(
            id=emitter_id,
            clip=config.get('clip'),
            volume=config.get('volume', 1.0),
            pitch=config.get('pitch', 1.0),
            spatial=config.get('spatial', True),
            loop=config.get('loop', False),
            autoplay=config.get('autoplay', False),
            ref_distance=config.get('refDistance', 1.0),
            max_distance=config.get('maxDistance', 50.0),
            rolloff=config.get('rolloff', 1.0),
            offset=tuple(config.get('offset', [0, 0, 0]))
        )

        self._state.emitters[emitter_id] = emitter
        self._state.pending_emitter_updates.append({
            'action': 'attach',
            'emitter': emitter.to_dict()
        })

        logger.info(f"[AudioAPI] Attached emitter: {emitter_id}")

    def remove_source(self, emitter_id: str):
        """
        Remove an audio emitter.

        Args:
            emitter_id: ID of emitter to remove
        """
        if emitter_id in self._state.emitters:
            del self._state.emitters[emitter_id]
            self._state.pending_emitter_updates.append({
                'action': 'remove',
                'emitter_id': emitter_id
            })
            logger.info(f"[AudioAPI] Removed emitter: {emitter_id}")

    def play(self, emitter_id: str):
        """
        Play sound from an attached emitter.

        Args:
            emitter_id: ID of emitter to play
        """
        if emitter_id in self._state.emitters:
            self._state.emitters[emitter_id].playing = True
            self._state.emitters[emitter_id].paused = False
            self._state.pending_emitter_updates.append({
                'action': 'play',
                'emitter_id': emitter_id
            })
            logger.info(f"[AudioAPI] Playing emitter: {emitter_id}")

    def play_at(self, clip: str, position: List[float],
                config: Optional[Dict[str, Any]] = None):
        """
        Play a one-shot sound at a world position.

        Not attached to any entity - fire and forget.

        Args:
            clip: Audio file path
            position: World position [x, y, z]
            config: Playback options:
                - volume: 0.0 to 1.0
                - maxDistance: Max audible distance
        """
        config = config or {}
        self._state.pending_emitter_updates.append({
            'action': 'play_at',
            'clip': clip,
            'position': list(position),
            'volume': config.get('volume', 1.0),
            'maxDistance': config.get('maxDistance', 50.0)
        })
        logger.info(f"[AudioAPI] One-shot at {position}: {clip}")

    def stop(self, emitter_id: str):
        """
        Stop playback of an emitter.

        Args:
            emitter_id: ID of emitter to stop
        """
        if emitter_id in self._state.emitters:
            self._state.emitters[emitter_id].playing = False
            self._state.emitters[emitter_id].paused = False
            self._state.pending_emitter_updates.append({
                'action': 'stop',
                'emitter_id': emitter_id
            })
            logger.info(f"[AudioAPI] Stopped emitter: {emitter_id}")

    def pause(self, emitter_id: str):
        """
        Pause playback of an emitter.

        Args:
            emitter_id: ID of emitter to pause
        """
        if emitter_id in self._state.emitters:
            self._state.emitters[emitter_id].paused = True
            self._state.pending_emitter_updates.append({
                'action': 'pause',
                'emitter_id': emitter_id
            })
            logger.info(f"[AudioAPI] Paused emitter: {emitter_id}")

    def resume(self, emitter_id: str):
        """
        Resume playback of a paused emitter.

        Args:
            emitter_id: ID of emitter to resume
        """
        if emitter_id in self._state.emitters:
            self._state.emitters[emitter_id].paused = False
            self._state.pending_emitter_updates.append({
                'action': 'resume',
                'emitter_id': emitter_id
            })
            logger.info(f"[AudioAPI] Resumed emitter: {emitter_id}")

    def set_volume(self, emitter_id: str, volume: float):
        """
        Set volume of an emitter.

        Args:
            emitter_id: ID of emitter
            volume: 0.0 to 1.0
        """
        if emitter_id in self._state.emitters:
            self._state.emitters[emitter_id].volume = max(0.0, min(1.0, volume))
            self._state.pending_emitter_updates.append({
                'action': 'set_volume',
                'emitter_id': emitter_id,
                'volume': self._state.emitters[emitter_id].volume
            })

    def set_pitch(self, emitter_id: str, pitch: float):
        """
        Set pitch of an emitter.

        Args:
            emitter_id: ID of emitter
            pitch: Pitch multiplier (1.0 = normal, 0.5 = octave down, 2.0 = octave up)
        """
        if emitter_id in self._state.emitters:
            self._state.emitters[emitter_id].pitch = max(0.1, min(4.0, pitch))
            self._state.pending_emitter_updates.append({
                'action': 'set_pitch',
                'emitter_id': emitter_id,
                'pitch': self._state.emitters[emitter_id].pitch
            })

    def is_playing(self, emitter_id: str) -> bool:
        """
        Check if an emitter is currently playing.

        Args:
            emitter_id: ID of emitter

        Returns:
            True if playing, False otherwise
        """
        if emitter_id in self._state.emitters:
            emitter = self._state.emitters[emitter_id]
            return emitter.playing and not emitter.paused
        return False

    def get_emitter(self, emitter_id: str) -> Optional[AudioEmitter]:
        """
        Get an emitter by ID.

        Args:
            emitter_id: ID of emitter

        Returns:
            AudioEmitter or None if not found
        """
        return self._state.emitters.get(emitter_id)

    def list_emitters(self) -> List[str]:
        """
        List all attached emitter IDs.

        Returns:
            List of emitter IDs
        """
        return list(self._state.emitters.keys())

    # ========== Configuration ==========

    def set_voice(self, voice: str):
        """Set TTS voice."""
        self._state.voice = voice
        if self._audio_facet:
            self._audio_facet.set_voice(voice)

    def set_sensitivity(self, threshold: float):
        """Set VAD sensitivity (0-1)."""
        self._state.sensitivity = max(0.0, min(1.0, threshold))
        if self._audio_facet:
            self._audio_facet.set_sensitivity(threshold)

    # ========== Event Handlers ==========

    def on_transcription_ready(self, callback: Callable):
        """
        Subscribe to transcription ready event.

        Args:
            callback: Function(text: str) -> None
        """
        self._event_handlers['transcription_ready'].append(callback)

    def on_speech_start(self, callback: Callable):
        """Subscribe to speech start event."""
        self._event_handlers['speech_start'].append(callback)

    def on_speech_end(self, callback: Callable):
        """Subscribe to speech end event."""
        self._event_handlers['speech_end'].append(callback)

    def on_listening_start(self, callback: Callable):
        """Subscribe to listening start event."""
        self._event_handlers['listening_start'].append(callback)

    def on_listening_end(self, callback: Callable):
        """Subscribe to listening end event."""
        self._event_handlers['listening_end'].append(callback)

    # ========== Internal Event Handlers ==========

    def _on_transcription_ready(self, event):
        """Handle transcription ready from facet."""
        self._state.last_transcription = event.data.get('text', '')
        self._state.partial_transcription = ""
        self._pending_events.append({
            'type': 'transcription_ready',
            'text': self._state.last_transcription
        })

    def _on_speech_start(self, event):
        """Handle speech start from facet."""
        self._state.is_speaking = True
        self._pending_events.append({
            'type': 'speech_start',
            'text': event.data.get('text', '')
        })

    def _on_speech_end(self, event):
        """Handle speech end from facet."""
        self._state.is_speaking = False
        self._pending_events.append({
            'type': 'speech_end',
            'text': event.data.get('text', '')
        })

    def _on_listening_start(self, event):
        """Handle listening start from facet."""
        self._state.is_listening = True
        self._pending_events.append({'type': 'listening_start'})

    def _on_listening_end(self, event):
        """Handle listening end from facet."""
        self._state.is_listening = False
        self._pending_events.append({'type': 'listening_end'})

    # ========== Sync with Facet Cycle ==========

    def get_pending_commands(self) -> Dict[str, Any]:
        """
        Get pending commands for AudioStreamFacet.

        Called at sync point to pass commands to facet.

        Returns:
            Dict of pending commands
        """
        commands = {}

        if self._state.pending_speak:
            commands['speak'] = self._state.pending_speak
            self._state.pending_speak = None

        if self._state.pending_listen:
            commands['start_listening'] = True
            self._state.pending_listen = False

        if self._state.pending_stop_listening:
            commands['stop_listening'] = True
            self._state.pending_stop_listening = False

        if self._state.pending_interrupt:
            commands['interrupt'] = True
            self._state.pending_interrupt = False

        return commands

    def update_from_facet(self, facet_data: Dict[str, Any]):
        """
        Update state from AudioStreamFacet sync data.

        Args:
            facet_data: Data returned from facet.sync()
        """
        self._state.state = facet_data.get('state', 'IDLE')
        self._state.is_speaking = facet_data.get('is_speaking', False)
        self._state.is_listening = facet_data.get('is_listening', False)

        if 'transcription' in facet_data:
            self._state.last_transcription = facet_data['transcription']

    def get_pending_events(self) -> List[Dict]:
        """
        Get and clear pending events for JavaScript callbacks.

        Returns:
            List of event dicts
        """
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def get_pending_emitter_updates(self) -> List[Dict[str, Any]]:
        """
        Get and clear pending emitter updates for audio system.

        Called at sync point to pass emitter commands to spatial audio.

        Returns:
            List of emitter update commands
        """
        updates = self._state.pending_emitter_updates.copy()
        self._state.pending_emitter_updates.clear()
        return updates

    # ========== JavaScript Interface ==========

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with state and method placeholders
        """
        return {
            # State (polling) - actual values
            'isListening': self._state.is_listening,
            'isSpeaking': self._state.is_speaking,
            'isTranscribing': self._state.is_transcribing,
            'lastTranscription': self._state.last_transcription,
            'partialTranscription': self._state.partial_transcription,
            'state': self._state.state,
            'voice': self._state.voice,
            'sensitivity': self._state.sensitivity,

            # Emitter state
            'emitters': {k: v.to_dict() for k, v in self._state.emitters.items()},

            # TTS/STT Methods (placeholders for JS binding)
            'speak': '__audio_speak__',
            'listen': '__audio_listen__',
            'stopListening': '__audio_stop_listening__',
            'interrupt': '__audio_interrupt__',
            'setVoice': '__audio_set_voice__',
            'setSensitivity': '__audio_set_sensitivity__',

            # Spatial Audio Methods (placeholders for JS binding)
            'attachSource': '__audio_attach_source__',
            'removeSource': '__audio_remove_source__',
            'play': '__audio_play__',
            'playAt': '__audio_play_at__',
            'stop': '__audio_stop__',
            'pause': '__audio_pause__',
            'resume': '__audio_resume__',
            'setVolume': '__audio_set_volume__',
            'setPitch': '__audio_set_pitch__',
            'isPlaying': '__audio_is_playing__',
            'getEmitter': '__audio_get_emitter__',
            'listEmitters': '__audio_list_emitters__',

            # Events (placeholders for JS binding)
            'onTranscriptionReady': '__audio_on_transcription_ready__',
            'onSpeechStart': '__audio_on_speech_start__',
            'onSpeechEnd': '__audio_on_speech_end__',
            'onListeningStart': '__audio_on_listening_start__',
            'onListeningEnd': '__audio_on_listening_end__',
            'on': '__audio_on__'  # Generic event handler for emitter events
        }


# Global singleton instance
_audio_api_instance = None


def get_audio_api() -> AudioAPI:
    """Get global AudioAPI singleton."""
    global _audio_api_instance
    if _audio_api_instance is None:
        _audio_api_instance = AudioAPI()
    return _audio_api_instance
