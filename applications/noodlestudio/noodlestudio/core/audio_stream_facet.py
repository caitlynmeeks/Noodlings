"""
Audio Stream Facet - Real-time audio processing for speech I/O.

Implements Option C architecture: Runs parallel to main facet cycle.

Audio Flow:
    Mic Input → AudioBuffer → Whisper chunks → Text (sync→ facet cycle)
    Text (sync←) → TTS Stream → Speaker Output

Features:
    - Voice Activity Detection (VAD) for smart chunking
    - Whisper transcription (local or API)
    - TTS synthesis (ElevenLabs, local, or system)
    - Interrupt handling (stop TTS when user speaks)
    - WebSocket streaming support (~250ms chunks)

Events:
    - transcription_ready: New transcription available
    - transcription_partial: Partial transcription (real-time)
    - speech_start: TTS started speaking
    - speech_end: TTS finished speaking
    - listening_start: Started listening
    - listening_end: Stopped listening

Scripting API (context.noodle.audio):
    - Events: onTranscriptionReady, onSpeechStart, onSpeechEnd
    - Polling: lastTranscription, isSpeaking, isListening
    - Control: speak(), listen(), stopListening(), interrupt()
    - Config: setSensitivity(), setVoice(), setModel()

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
"""

import asyncio
import time
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from collections import deque
from enum import Enum, auto

from .multimodal_facet import (
    MultimodalFacet, Modality, ModalityDirection,
    MultimodalBuffer, MultimodalEvent
)

logger = logging.getLogger(__name__)


class AudioState(Enum):
    """Current state of audio facet."""
    IDLE = auto()           # Not actively processing
    LISTENING = auto()      # Capturing audio input
    TRANSCRIBING = auto()   # Processing audio to text
    SPEAKING = auto()       # TTS output active
    INTERRUPTED = auto()    # TTS was interrupted


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata."""
    data: bytes             # Raw audio bytes
    sample_rate: int        # Sample rate (e.g., 16000)
    channels: int           # Number of channels
    timestamp: float        # When captured
    duration_ms: float      # Chunk duration
    is_speech: bool = True  # VAD result


@dataclass
class Transcription:
    """Result from speech-to-text."""
    text: str                   # Transcribed text
    is_final: bool              # Final vs partial
    confidence: float           # 0-1 confidence score
    language: str               # Detected language
    timestamp: float            # When transcribed
    audio_duration_ms: float    # Source audio duration
    segments: List[Dict] = field(default_factory=list)  # Word-level timing


@dataclass
class TTSRequest:
    """Request to synthesize speech."""
    text: str                   # Text to speak
    voice: str                  # Voice ID or name
    speed: float = 1.0          # Playback speed
    priority: int = 0           # Higher = skip queue
    interrupt: bool = False     # Interrupt current speech


class AudioStreamFacet(MultimodalFacet):
    """
    Real-time audio streaming facet.

    Runs parallel to main facet cycle, handling:
    - Microphone capture and buffering
    - Voice activity detection
    - Speech-to-text transcription
    - Text-to-speech synthesis
    - Audio playback

    Unity-like API for scripting:
        context.noodle.audio.speak("Hello!")
        context.noodle.audio.onTranscriptionReady((text) => { ... })
    """

    def __init__(
        self,
        facet_id: str,
        process_interval_ms: int = 250,
        # Audio settings
        sample_rate: int = 16000,
        chunk_duration_ms: int = 250,
        # VAD settings
        vad_enabled: bool = True,
        vad_threshold: float = 0.5,
        silence_duration_ms: int = 500,
        # Model settings
        transcription_model: str = "AUDIO_IN",  # Model label
        tts_model: str = "AUDIO_OUT",           # Model label
        tts_voice: str = "default"
    ):
        """
        Initialize audio stream facet.

        Args:
            facet_id: Unique identifier
            process_interval_ms: Processing loop interval
            sample_rate: Audio sample rate (Hz)
            chunk_duration_ms: Size of audio chunks
            vad_enabled: Enable voice activity detection
            vad_threshold: VAD sensitivity (0-1)
            silence_duration_ms: Silence before end of speech
            transcription_model: Model label for STT
            tts_model: Model label for TTS
            tts_voice: Voice to use for TTS
        """
        super().__init__(
            facet_id=facet_id,
            modality=Modality.AUDIO,
            direction=ModalityDirection.BOTH,
            process_interval_ms=process_interval_ms,
            model_label=transcription_model
        )

        # Audio settings
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms

        # VAD settings
        self.vad_enabled = vad_enabled
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms

        # Model settings
        self.transcription_model = transcription_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice

        # State
        self._state = AudioState.IDLE
        self._last_speech_time = 0.0

        # Audio buffers
        self._audio_chunks: deque = deque(maxlen=100)  # Raw audio chunks
        self._transcription_buffer: List[AudioChunk] = []  # Chunks being transcribed

        # Transcription state
        self._last_transcription: Optional[Transcription] = None
        self._partial_transcription: str = ""
        self._transcription_history: deque = deque(maxlen=50)

        # TTS state
        self._tts_queue: deque = deque(maxlen=10)
        self._current_tts: Optional[TTSRequest] = None
        self._is_speaking = False

        # Client references (set externally)
        self._transcription_client = None  # Whisper client
        self._tts_client = None            # TTS client
        self._audio_output = None          # Speaker output

        # Accumulated audio for transcription
        self._accumulated_audio: bytes = b''
        self._accumulated_duration_ms: float = 0.0

        logger.info(f"[AudioStreamFacet] Initialized (sample_rate={sample_rate}, "
                   f"vad={vad_enabled}, chunk={chunk_duration_ms}ms)")

    # ========== State Properties ==========

    @property
    def state(self) -> AudioState:
        """Get current audio state."""
        return self._state

    @property
    def is_listening(self) -> bool:
        """Check if actively listening."""
        return self._state == AudioState.LISTENING

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is playing."""
        return self._is_speaking

    @property
    def is_transcribing(self) -> bool:
        """Check if transcription is in progress."""
        return self._state == AudioState.TRANSCRIBING

    @property
    def last_transcription(self) -> Optional[str]:
        """Get last transcription text."""
        if self._last_transcription:
            return self._last_transcription.text
        return None

    @property
    def partial_transcription(self) -> str:
        """Get current partial transcription."""
        return self._partial_transcription

    # ========== Control Methods ==========

    async def listen(self):
        """
        Start listening for audio input.

        Activates microphone capture and VAD.
        """
        if self._state != AudioState.IDLE:
            logger.warning(f"[AudioStreamFacet] Cannot listen in state {self._state}")
            return

        self._state = AudioState.LISTENING
        self._accumulated_audio = b''
        self._accumulated_duration_ms = 0.0

        await self.emit("listening_start", {
            'timestamp': time.time()
        })

        logger.info("[AudioStreamFacet] Started listening")

    async def stop_listening(self):
        """
        Stop listening and process any accumulated audio.
        """
        if self._state != AudioState.LISTENING:
            return

        self._state = AudioState.IDLE

        # Trigger transcription of accumulated audio
        if self._accumulated_audio:
            await self._transcribe_accumulated()

        await self.emit("listening_end", {
            'timestamp': time.time()
        })

        logger.info("[AudioStreamFacet] Stopped listening")

    async def speak(self, text: str, voice: Optional[str] = None,
                   speed: float = 1.0, interrupt: bool = False) -> bool:
        """
        Synthesize and speak text.

        Args:
            text: Text to speak
            voice: Voice ID (uses default if None)
            speed: Playback speed multiplier
            interrupt: If True, interrupt current speech

        Returns:
            True if speech started, False if queued
        """
        request = TTSRequest(
            text=text,
            voice=voice or self.tts_voice,
            speed=speed,
            interrupt=interrupt
        )

        if interrupt:
            await self.interrupt()
            return await self._start_tts(request)

        if self._is_speaking:
            self._tts_queue.append(request)
            logger.info(f"[AudioStreamFacet] TTS queued: {text[:50]}...")
            return False

        return await self._start_tts(request)

    async def interrupt(self):
        """
        Interrupt current TTS playback.
        """
        if self._is_speaking:
            self._state = AudioState.INTERRUPTED
            self._is_speaking = False
            self._current_tts = None

            # Stop audio output
            if self._audio_output:
                await self._audio_output.stop()

            await self.emit("speech_interrupted", {
                'timestamp': time.time()
            })

            logger.info("[AudioStreamFacet] Speech interrupted")

    def clear_tts_queue(self):
        """Clear pending TTS requests."""
        self._tts_queue.clear()

    # ========== Configuration ==========

    def set_voice(self, voice: str):
        """Set TTS voice."""
        self.tts_voice = voice
        logger.info(f"[AudioStreamFacet] Voice set to: {voice}")

    def set_sensitivity(self, threshold: float):
        """Set VAD sensitivity (0-1)."""
        self.vad_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"[AudioStreamFacet] VAD threshold set to: {self.vad_threshold}")

    def set_transcription_model(self, model_label: str):
        """Set transcription model label."""
        self.transcription_model = model_label
        self.model_label = model_label

    def set_tts_model(self, model_label: str):
        """Set TTS model label."""
        self.tts_model = model_label

    # ========== Client Setup ==========

    def set_transcription_client(self, client):
        """
        Set transcription client (Whisper API or local).

        Client must implement:
            async transcribe(audio_bytes, sample_rate) -> Transcription
        """
        self._transcription_client = client

    def set_tts_client(self, client):
        """
        Set TTS client (ElevenLabs, local, etc.).

        Client must implement:
            async synthesize(text, voice, speed) -> bytes
        """
        self._tts_client = client

    def set_audio_output(self, output):
        """
        Set audio output (speaker).

        Output must implement:
            async play(audio_bytes, sample_rate)
            async stop()
        """
        self._audio_output = output

    # ========== Processing Loop ==========

    async def _process_loop(self):
        """
        Main processing loop (called at interval).

        Handles:
        1. VAD on incoming audio chunks
        2. Triggering transcription when speech ends
        3. Processing TTS queue
        """
        # Process any queued audio chunks
        while self._audio_chunks:
            chunk = self._audio_chunks.popleft()
            await self._process_audio_chunk(chunk)

        # Check if we should transcribe accumulated audio
        if self._state == AudioState.LISTENING:
            time_since_speech = (time.time() - self._last_speech_time) * 1000
            if (self._accumulated_audio and
                time_since_speech > self.silence_duration_ms):
                # Silence detected, transcribe
                await self._transcribe_accumulated()

        # Process TTS queue
        if not self._is_speaking and self._tts_queue:
            request = self._tts_queue.popleft()
            await self._start_tts(request)

    async def _process_audio_chunk(self, chunk: AudioChunk):
        """
        Process single audio chunk.

        Applies VAD and accumulates speech audio.
        """
        if self._state != AudioState.LISTENING:
            return

        # Voice Activity Detection
        is_speech = chunk.is_speech
        if self.vad_enabled and self._vad_model:
            is_speech = await self._detect_voice_activity(chunk)

        if is_speech:
            # Accumulate speech audio
            self._accumulated_audio += chunk.data
            self._accumulated_duration_ms += chunk.duration_ms
            self._last_speech_time = time.time()

            # Emit partial transcription periodically
            if self._accumulated_duration_ms > 1000:  # Every ~1 second
                await self._emit_partial_transcription()

    async def _detect_voice_activity(self, chunk: AudioChunk) -> bool:
        """
        Run VAD on audio chunk.

        Returns True if speech detected.
        """
        # TODO: Implement Silero VAD or WebRTC VAD
        # For now, simple energy-based detection
        if not chunk.data:
            return False

        # Calculate RMS energy
        import struct
        samples = struct.unpack(f'{len(chunk.data)//2}h', chunk.data)
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
        normalized = rms / 32768.0

        return normalized > self.vad_threshold

    async def _transcribe_accumulated(self):
        """
        Transcribe accumulated audio buffer.
        """
        if not self._accumulated_audio:
            return

        if not self._transcription_client:
            logger.warning("[AudioStreamFacet] No transcription client set")
            self._accumulated_audio = b''
            self._accumulated_duration_ms = 0.0
            return

        self._state = AudioState.TRANSCRIBING
        audio_data = self._accumulated_audio
        duration_ms = self._accumulated_duration_ms

        # Clear buffer
        self._accumulated_audio = b''
        self._accumulated_duration_ms = 0.0

        try:
            # Call transcription API
            result = await self._transcription_client.transcribe(
                audio_data,
                self.sample_rate
            )

            # Create transcription object
            transcription = Transcription(
                text=result.get('text', ''),
                is_final=True,
                confidence=result.get('confidence', 0.9),
                language=result.get('language', 'en'),
                timestamp=time.time(),
                audio_duration_ms=duration_ms,
                segments=result.get('segments', [])
            )

            # Store result
            self._last_transcription = transcription
            self._transcription_history.append(transcription)
            self._partial_transcription = ""

            # Emit event
            await self.emit("transcription_ready", {
                'text': transcription.text,
                'confidence': transcription.confidence,
                'language': transcription.language,
                'duration_ms': transcription.audio_duration_ms
            })

            # Push to sync data for main cycle
            await self.push_sync_data('transcription', transcription.text)

            logger.info(f"[AudioStreamFacet] Transcribed: {transcription.text}")

        except Exception as e:
            logger.error(f"[AudioStreamFacet] Transcription error: {e}")

        finally:
            self._state = AudioState.IDLE

    async def _emit_partial_transcription(self):
        """
        Emit partial transcription for real-time display.
        """
        # TODO: Implement streaming transcription
        pass

    # ========== TTS ==========

    async def _start_tts(self, request: TTSRequest) -> bool:
        """
        Start TTS synthesis and playback.

        Returns True if started successfully.
        """
        if not self._tts_client:
            logger.warning("[AudioStreamFacet] No TTS client set")
            return False

        self._current_tts = request
        self._is_speaking = True
        self._state = AudioState.SPEAKING

        await self.emit("speech_start", {
            'text': request.text,
            'voice': request.voice,
            'timestamp': time.time()
        })

        try:
            # Synthesize audio
            audio_data = await self._tts_client.synthesize(
                request.text,
                request.voice,
                request.speed
            )

            # Play audio
            if self._audio_output and audio_data:
                await self._audio_output.play(audio_data, self.sample_rate)

            await self.emit("speech_end", {
                'text': request.text,
                'timestamp': time.time()
            })

            logger.info(f"[AudioStreamFacet] Spoke: {request.text[:50]}...")
            return True

        except Exception as e:
            logger.error(f"[AudioStreamFacet] TTS error: {e}")
            return False

        finally:
            self._is_speaking = False
            self._current_tts = None
            self._state = AudioState.IDLE

    # ========== Sync Points ==========

    async def _sync_with_cycle(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync with main facet cycle.

        Exchanges:
        - Transcriptions from audio → cycle
        - TTS requests from cycle → audio

        Args:
            cycle_data: Data from main cycle

        Returns:
            Audio data for cycle (transcriptions, state)
        """
        result = {
            'state': self._state.name,
            'is_speaking': self._is_speaking,
            'is_listening': self.is_listening
        }

        # Include last transcription if available
        if self._last_transcription:
            result['transcription'] = self._last_transcription.text
            result['transcription_confidence'] = self._last_transcription.confidence

        # Check for TTS requests from cycle
        tts_text = cycle_data.get('speak')
        if tts_text:
            await self.speak(tts_text)

        # Check for control commands
        if cycle_data.get('start_listening'):
            await self.listen()
        elif cycle_data.get('stop_listening'):
            await self.stop_listening()
        elif cycle_data.get('interrupt'):
            await self.interrupt()

        return result

    # ========== External Audio Input ==========

    async def push_audio_chunk(self, data: bytes, sample_rate: int = None,
                               duration_ms: float = None):
        """
        Push audio chunk from external source (WebSocket, mic).

        Args:
            data: Raw audio bytes (PCM 16-bit)
            sample_rate: Sample rate (uses default if None)
            duration_ms: Chunk duration (calculated if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        if duration_ms is None:
            # Calculate from data size (16-bit PCM)
            samples = len(data) // 2
            duration_ms = (samples / sample_rate) * 1000

        chunk = AudioChunk(
            data=data,
            sample_rate=sample_rate,
            channels=1,
            timestamp=time.time(),
            duration_ms=duration_ms
        )

        self._audio_chunks.append(chunk)

    # ========== Serialization ==========

    def to_dict(self) -> Dict[str, Any]:
        """Serialize facet state."""
        base = super().to_dict()
        base.update({
            'sample_rate': self.sample_rate,
            'chunk_duration_ms': self.chunk_duration_ms,
            'vad_enabled': self.vad_enabled,
            'vad_threshold': self.vad_threshold,
            'transcription_model': self.transcription_model,
            'tts_model': self.tts_model,
            'tts_voice': self.tts_voice,
            'state': self._state.name,
            'is_speaking': self._is_speaking,
            'last_transcription': self.last_transcription
        })
        return base

    # ========== JavaScript API (for scripting) ==========

    def get_js_api(self) -> Dict[str, Any]:
        """
        Get JavaScript-compatible API object.

        Used by NoodleAPI to expose audio functionality.
        """
        return {
            # State (polling)
            'isListening': self.is_listening,
            'isSpeaking': self.is_speaking,
            'lastTranscription': self.last_transcription,
            'partialTranscription': self.partial_transcription,
            'state': self._state.name,

            # Methods (placeholders for JS binding)
            'speak': '__audio_speak__',
            'listen': '__audio_listen__',
            'stopListening': '__audio_stop_listening__',
            'interrupt': '__audio_interrupt__',
            'setVoice': '__audio_set_voice__',
            'setSensitivity': '__audio_set_sensitivity__',
            'clearQueue': '__audio_clear_queue__',

            # Events (placeholders)
            'onTranscriptionReady': '__audio_on_transcription_ready__',
            'onSpeechStart': '__audio_on_speech_start__',
            'onSpeechEnd': '__audio_on_speech_end__',
            'onListeningStart': '__audio_on_listening_start__',
            'onListeningEnd': '__audio_on_listening_end__'
        }


# ========== Client Factory ==========

def create_default_transcription_client():
    """Create default transcription client based on available backends."""
    from .transcription_clients import create_transcription_client
    return create_transcription_client(backend="auto")


def create_default_tts_client():
    """Create default TTS client based on available backends."""
    from .tts_clients import create_tts_client
    return create_tts_client(backend="auto")


# ========== Convenience Functions ==========

def create_audio_facet_with_clients(
    facet_id: str,
    transcription_backend: str = "auto",
    tts_backend: str = "auto",
    **kwargs
) -> AudioStreamFacet:
    """
    Create AudioStreamFacet with configured clients.

    Args:
        facet_id: Unique identifier
        transcription_backend: "groq", "local", "openai", or "auto"
        tts_backend: "elevenlabs", "openai", "piper", or "auto"
        **kwargs: Additional AudioStreamFacet arguments

    Returns:
        Configured AudioStreamFacet
    """
    from .transcription_clients import create_transcription_client
    from .tts_clients import create_tts_client

    facet = AudioStreamFacet(facet_id=facet_id, **kwargs)
    facet.set_transcription_client(create_transcription_client(transcription_backend))
    facet.set_tts_client(create_tts_client(tts_backend))

    return facet
