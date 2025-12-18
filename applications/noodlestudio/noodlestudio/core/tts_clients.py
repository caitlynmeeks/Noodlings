"""
TTS Clients - Text-to-speech implementations.

Supports multiple backends:
- ElevenLabs (high quality, expressive)
- OpenAI TTS (good quality, fast)
- Local Piper (offline, very fast)

All clients implement the same interface:
    async synthesize(text, voice, speed) -> bytes

Returns raw audio bytes (PCM 16-bit or format specified).

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import io

logger = logging.getLogger(__name__)


@dataclass
class TTSVoice:
    """Voice configuration."""
    id: str
    name: str
    language: str = "en"
    gender: str = "neutral"
    description: str = ""


class TTSClient(ABC):
    """Abstract base for TTS clients."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str = "default",
                        speed: float = 1.0) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak
            voice: Voice ID or name
            speed: Playback speed multiplier (1.0 = normal)

        Returns:
            Audio bytes (PCM 16-bit mono by default)
        """
        pass

    @abstractmethod
    def list_voices(self) -> List[TTSVoice]:
        """List available voices."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get output sample rate."""
        pass


class ElevenLabsClient(TTSClient):
    """
    ElevenLabs TTS client.

    High-quality, expressive text-to-speech.
    Requires ELEVENLABS_API_KEY environment variable.

    Features:
    - Natural sounding voices
    - Voice cloning support
    - Streaming audio
    - Multiple languages
    """

    # Default voices
    VOICES = {
        'default': 'EXAVITQu4vr4xnSDxMaL',  # Sarah
        'rachel': '21m00Tcm4TlvDq8ikWAM',
        'domi': 'AZnzlk1XvdvUeBnXmlld',
        'bella': 'EXAVITQu4vr4xnSDxMaL',
        'antoni': 'ErXwobaYiN019PkySvjV',
        'josh': 'TxGEqnHWrfWFTfGW9XjX',
        'arnold': 'VR6AewLTigWG4xSOukaG',
        'sam': 'yoZ06aMxZJJ28mfd3POQ',
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "eleven_turbo_v2_5"):
        """
        Initialize ElevenLabs client.

        Args:
            api_key: ElevenLabs API key (uses ELEVENLABS_API_KEY env var if not provided)
            model: Model to use (eleven_turbo_v2_5 for speed, eleven_multilingual_v2 for quality)
        """
        self.api_key = api_key or os.environ.get('ELEVENLABS_API_KEY')
        self.model = model
        self._sample_rate = 22050  # ElevenLabs default

        if not self.api_key:
            logger.warning("[ElevenLabs] No API key found. Set ELEVENLABS_API_KEY environment variable.")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def list_voices(self) -> List[TTSVoice]:
        """List available voices."""
        return [
            TTSVoice(id=vid, name=name, language="en")
            for name, vid in self.VOICES.items()
        ]

    async def synthesize(self, text: str, voice: str = "default",
                        speed: float = 1.0) -> bytes:
        """Synthesize speech using ElevenLabs API."""
        if not self.api_key:
            logger.warning("[ElevenLabs] No API key, returning empty audio")
            return b''

        # Resolve voice name to ID
        voice_id = self.VOICES.get(voice.lower(), voice)

        try:
            import httpx

            # Build request
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "model_id": self.model,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }

            # Add speed adjustment if not 1.0
            # ElevenLabs doesn't have direct speed control, but we can note it for post-processing

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                audio_bytes = response.content

            logger.info(f"[ElevenLabs] Synthesized {len(audio_bytes)} bytes for: {text[:30]}...")

            # If speed != 1.0, we'd need to resample (not implemented yet)
            return audio_bytes

        except Exception as e:
            logger.error(f"[ElevenLabs] Synthesis error: {e}")
            return b''


class OpenAITTSClient(TTSClient):
    """
    OpenAI TTS client.

    Good quality, fast text-to-speech.
    Requires OPENAI_API_KEY environment variable.

    Features:
    - Multiple voices
    - HD mode available
    - Good pronunciation
    """

    VOICES = {
        'default': 'nova',
        'alloy': 'alloy',
        'echo': 'echo',
        'fable': 'fable',
        'onyx': 'onyx',
        'nova': 'nova',
        'shimmer': 'shimmer'
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "tts-1",
                 response_format: str = "pcm"):
        """
        Initialize OpenAI TTS client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use (tts-1 for speed, tts-1-hd for quality)
            response_format: Output format (pcm, mp3, opus, aac, flac)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        self.response_format = response_format
        self._sample_rate = 24000  # OpenAI TTS sample rate

        if not self.api_key:
            logger.warning("[OpenAITTS] No API key found. Set OPENAI_API_KEY environment variable.")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def list_voices(self) -> List[TTSVoice]:
        """List available voices."""
        return [
            TTSVoice(id=vid, name=name, language="en")
            for name, vid in self.VOICES.items()
        ]

    async def synthesize(self, text: str, voice: str = "default",
                        speed: float = 1.0) -> bytes:
        """Synthesize speech using OpenAI TTS API."""
        if not self.api_key:
            logger.warning("[OpenAITTS] No API key, returning empty audio")
            return b''

        # Resolve voice name
        voice_id = self.VOICES.get(voice.lower(), voice)

        try:
            import httpx

            url = "https://api.openai.com/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "input": text,
                "voice": voice_id,
                "response_format": self.response_format,
                "speed": max(0.25, min(4.0, speed))  # OpenAI supports 0.25 to 4.0
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                audio_bytes = response.content

            logger.info(f"[OpenAITTS] Synthesized {len(audio_bytes)} bytes for: {text[:30]}...")

            return audio_bytes

        except Exception as e:
            logger.error(f"[OpenAITTS] Synthesis error: {e}")
            return b''


class PiperTTSClient(TTSClient):
    """
    Piper TTS client (local).

    Fast, offline text-to-speech using Piper.
    Requires piper-tts to be installed.

    Features:
    - Runs locally (no API keys)
    - Very fast (~10x realtime on CPU)
    - Multiple voices available
    - Low latency

    Install:
        pip install piper-tts
        # Or download binary from https://github.com/rhasspy/piper
    """

    def __init__(self, model_path: Optional[str] = None, voice: str = "en_US-lessac-medium"):
        """
        Initialize Piper TTS client.

        Args:
            model_path: Path to Piper model directory (auto-downloads if None)
            voice: Default voice name
        """
        self.model_path = model_path
        self.default_voice = voice
        self._sample_rate = 22050  # Piper default
        self._piper_path = None
        self._model_cache: Dict[str, str] = {}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def list_voices(self) -> List[TTSVoice]:
        """List available Piper voices."""
        # Common Piper voices
        return [
            TTSVoice(id="en_US-lessac-medium", name="Lessac (US)", language="en-US", gender="male"),
            TTSVoice(id="en_US-amy-medium", name="Amy (US)", language="en-US", gender="female"),
            TTSVoice(id="en_US-ryan-medium", name="Ryan (US)", language="en-US", gender="male"),
            TTSVoice(id="en_GB-alba-medium", name="Alba (UK)", language="en-GB", gender="female"),
            TTSVoice(id="en_GB-cori-medium", name="Cori (UK)", language="en-GB", gender="female"),
        ]

    async def _ensure_piper(self):
        """Ensure Piper is available."""
        if self._piper_path:
            return

        # Try to find piper in PATH
        import shutil
        piper_path = shutil.which('piper')

        if piper_path:
            self._piper_path = piper_path
            logger.info(f"[PiperTTS] Found piper at: {piper_path}")
            return

        # Try Python piper-tts package
        try:
            import piper
            # Use the package directly
            self._piper_path = "piper-python"
            logger.info("[PiperTTS] Using piper-tts Python package")
            return
        except ImportError:
            pass

        logger.warning("[PiperTTS] Piper not found. Install with: pip install piper-tts")

    async def _get_model_path(self, voice: str) -> Optional[str]:
        """Get or download model for voice."""
        if voice in self._model_cache:
            return self._model_cache[voice]

        # Check if model exists in standard locations
        model_dirs = [
            os.path.expanduser("~/.local/share/piper/voices"),
            "/usr/share/piper/voices",
            self.model_path
        ]

        for model_dir in model_dirs:
            if model_dir and os.path.exists(model_dir):
                model_file = os.path.join(model_dir, voice, f"{voice}.onnx")
                if os.path.exists(model_file):
                    self._model_cache[voice] = model_file
                    return model_file

        # Model not found - would need to download
        logger.warning(f"[PiperTTS] Model '{voice}' not found. Download from https://github.com/rhasspy/piper/releases")
        return None

    async def synthesize(self, text: str, voice: str = "default",
                        speed: float = 1.0) -> bytes:
        """Synthesize speech using Piper."""
        await self._ensure_piper()

        if not self._piper_path:
            logger.warning("[PiperTTS] Piper not available")
            return b''

        # Resolve voice
        if voice == "default":
            voice = self.default_voice

        try:
            # Use Python piper-tts if available
            if self._piper_path == "piper-python":
                return await self._synthesize_python(text, voice, speed)
            else:
                return await self._synthesize_cli(text, voice, speed)

        except Exception as e:
            logger.error(f"[PiperTTS] Synthesis error: {e}")
            return b''

    async def _synthesize_python(self, text: str, voice: str, speed: float) -> bytes:
        """Synthesize using piper-tts Python package."""
        try:
            from piper import PiperVoice

            # Load voice (cached)
            model_path = await self._get_model_path(voice)
            if not model_path:
                return b''

            loop = asyncio.get_event_loop()

            def synthesize():
                piper_voice = PiperVoice.load(model_path)
                wav_bytes = io.BytesIO()

                # Synthesize to WAV
                import wave
                with wave.open(wav_bytes, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self._sample_rate)

                    for audio_bytes in piper_voice.synthesize_stream_raw(text):
                        wav_file.writeframes(audio_bytes)

                wav_bytes.seek(0)
                return wav_bytes.read()

            audio_bytes = await loop.run_in_executor(None, synthesize)
            logger.info(f"[PiperTTS] Synthesized {len(audio_bytes)} bytes for: {text[:30]}...")
            return audio_bytes

        except Exception as e:
            logger.error(f"[PiperTTS] Python synthesis error: {e}")
            return b''

    async def _synthesize_cli(self, text: str, voice: str, speed: float) -> bytes:
        """Synthesize using Piper CLI."""
        model_path = await self._get_model_path(voice)
        if not model_path:
            return b''

        try:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                output_path = f.name

            # Build command
            cmd = [
                self._piper_path,
                "--model", model_path,
                "--output_file", output_path
            ]

            if speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / speed)])

            # Run Piper
            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    input=text.encode(),
                    capture_output=True,
                    timeout=30
                )
            )

            if process.returncode != 0:
                logger.error(f"[PiperTTS] CLI error: {process.stderr.decode()}")
                return b''

            # Read output file
            with open(output_path, 'rb') as f:
                audio_bytes = f.read()

            os.unlink(output_path)

            logger.info(f"[PiperTTS] Synthesized {len(audio_bytes)} bytes for: {text[:30]}...")
            return audio_bytes

        except Exception as e:
            logger.error(f"[PiperTTS] CLI synthesis error: {e}")
            return b''


# ========== Factory ==========

def create_tts_client(
    backend: str = "auto",
    **kwargs
) -> TTSClient:
    """
    Factory function to create TTS client.

    Args:
        backend: Backend to use ("elevenlabs", "openai", "piper", "auto")
        **kwargs: Backend-specific arguments

    Returns:
        TTSClient instance

    Auto selection priority:
    1. ElevenLabs (if ELEVENLABS_API_KEY set)
    2. OpenAI (if OPENAI_API_KEY set)
    3. Piper (if installed)
    """
    if backend == "elevenlabs":
        return ElevenLabsClient(**kwargs)

    elif backend == "openai":
        return OpenAITTSClient(**kwargs)

    elif backend == "piper":
        return PiperTTSClient(**kwargs)

    elif backend == "auto":
        # Try ElevenLabs first (best quality)
        if os.environ.get('ELEVENLABS_API_KEY'):
            logger.info("[TTS] Auto-selected: ElevenLabs")
            return ElevenLabsClient(**kwargs)

        # Try OpenAI
        if os.environ.get('OPENAI_API_KEY'):
            logger.info("[TTS] Auto-selected: OpenAI TTS")
            return OpenAITTSClient(**kwargs)

        # Try Piper (local)
        try:
            import shutil
            if shutil.which('piper'):
                logger.info("[TTS] Auto-selected: Piper (local)")
                return PiperTTSClient(**kwargs)
        except:
            pass

        try:
            import piper
            logger.info("[TTS] Auto-selected: Piper (Python package)")
            return PiperTTSClient(**kwargs)
        except ImportError:
            pass

        # No backend available
        logger.warning("[TTS] No TTS backend available. "
                      "Set ELEVENLABS_API_KEY, OPENAI_API_KEY, or install piper-tts.")
        return ElevenLabsClient()  # Will return empty audio

    else:
        raise ValueError(f"Unknown TTS backend: {backend}")
