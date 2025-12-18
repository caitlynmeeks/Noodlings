"""
Transcription Clients - Speech-to-text implementations.

Supports multiple backends:
- Groq Whisper (fast cloud API, recommended)
- Local faster-whisper (offline, requires model download)
- OpenAI Whisper API (fallback)

All clients implement the same interface:
    async transcribe(audio_bytes, sample_rate) -> Dict

Returns:
    {
        'text': str,           # Transcribed text
        'confidence': float,   # 0-1 confidence score
        'language': str,       # Detected language code
        'segments': List[Dict] # Word-level timing (optional)
    }

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
"""

import asyncio
import logging
import os
import tempfile
import wave
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text."""
    text: str
    confidence: float
    language: str
    segments: List[Dict]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'segments': self.segments
        }


class TranscriptionClient(ABC):
    """Abstract base for transcription clients."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: PCM 16-bit mono audio
            sample_rate: Audio sample rate (default 16000)

        Returns:
            Dict with 'text', 'confidence', 'language', 'segments'
        """
        pass

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
        """
        Convert raw PCM to WAV format.

        Args:
            pcm_bytes: Raw PCM 16-bit audio
            sample_rate: Sample rate
            channels: Number of channels (1 for mono)

        Returns:
            WAV file bytes
        """
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        wav_buffer.seek(0)
        return wav_buffer.read()


class GroqWhisperClient(TranscriptionClient):
    """
    Groq Whisper API client.

    Fast cloud-based transcription using Groq's optimized Whisper.
    Requires GROQ_API_KEY environment variable.

    Features:
    - Very fast (~10x realtime)
    - High accuracy
    - Word-level timestamps
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3"):
        """
        Initialize Groq Whisper client.

        Args:
            api_key: Groq API key (uses GROQ_API_KEY env var if not provided)
            model: Whisper model to use (whisper-large-v3 recommended)
        """
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        self.model = model
        self._client = None

        if not self.api_key:
            logger.warning("[GroqWhisper] No API key found. Set GROQ_API_KEY environment variable.")

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio using Groq Whisper API."""
        if not self.api_key:
            return {
                'text': '[Groq API key not configured]',
                'confidence': 0.0,
                'language': 'en',
                'segments': []
            }

        try:
            # Convert PCM to WAV
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate)

            # Use httpx for async HTTP
            import httpx

            # Create multipart form data
            files = {
                'file': ('audio.wav', wav_bytes, 'audio/wav'),
            }
            data = {
                'model': self.model,
                'response_format': 'verbose_json',
                'language': 'en'  # Can be made configurable
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://api.groq.com/openai/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    files=files,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

            # Parse response
            text = result.get('text', '').strip()
            language = result.get('language', 'en')

            # Extract word-level segments if available
            segments = []
            if 'segments' in result:
                for seg in result['segments']:
                    segments.append({
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', '')
                    })

            logger.info(f"[GroqWhisper] Transcribed: {text[:50]}...")

            return {
                'text': text,
                'confidence': 0.95,  # Groq doesn't return confidence
                'language': language,
                'segments': segments
            }

        except Exception as e:
            logger.error(f"[GroqWhisper] Transcription error: {e}")
            return {
                'text': f'[Transcription error: {e}]',
                'confidence': 0.0,
                'language': 'en',
                'segments': []
            }


class LocalWhisperClient(TranscriptionClient):
    """
    Local faster-whisper client.

    Runs Whisper locally using faster-whisper (CTranslate2 backend).
    Requires: pip install faster-whisper

    Features:
    - Offline operation
    - GPU acceleration (CUDA/Metal)
    - Multiple model sizes
    """

    def __init__(self, model_size: str = "base", device: str = "auto",
                 compute_type: str = "auto"):
        """
        Initialize local Whisper client.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (auto, cpu, cuda, mps)
            compute_type: Compute precision (auto, float16, int8)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        async with self._lock:
            if self._model is not None:
                return

            try:
                from faster_whisper import WhisperModel

                # Determine device
                device = self.device
                if device == "auto":
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = "cpu"  # faster-whisper doesn't support MPS directly
                    else:
                        device = "cpu"

                # Determine compute type
                compute_type = self.compute_type
                if compute_type == "auto":
                    compute_type = "float16" if device == "cuda" else "int8"

                logger.info(f"[LocalWhisper] Loading model '{self.model_size}' on {device} ({compute_type})")

                # Load model in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(self.model_size, device=device, compute_type=compute_type)
                )

                logger.info(f"[LocalWhisper] Model loaded successfully")

            except ImportError:
                logger.error("[LocalWhisper] faster-whisper not installed. Run: pip install faster-whisper")
                raise
            except Exception as e:
                logger.error(f"[LocalWhisper] Failed to load model: {e}")
                raise

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio using local faster-whisper."""
        try:
            await self._ensure_model()

            # Convert PCM to WAV and save to temp file
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(wav_bytes)
                temp_path = f.name

            try:
                # Run transcription in thread pool
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(temp_path, beam_size=5)
                )

                # Collect segments
                segment_list = []
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                    segment_list.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    })

                text = ''.join(text_parts).strip()
                language = info.language

                logger.info(f"[LocalWhisper] Transcribed: {text[:50]}...")

                return {
                    'text': text,
                    'confidence': info.language_probability,
                    'language': language,
                    'segments': segment_list
                }

            finally:
                # Clean up temp file
                os.unlink(temp_path)

        except ImportError:
            return {
                'text': '[faster-whisper not installed]',
                'confidence': 0.0,
                'language': 'en',
                'segments': []
            }
        except Exception as e:
            logger.error(f"[LocalWhisper] Transcription error: {e}")
            return {
                'text': f'[Transcription error: {e}]',
                'confidence': 0.0,
                'language': 'en',
                'segments': []
            }


class OpenAIWhisperClient(TranscriptionClient):
    """
    OpenAI Whisper API client.

    Uses OpenAI's hosted Whisper API.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize OpenAI Whisper client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use (whisper-1)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model

        if not self.api_key:
            logger.warning("[OpenAIWhisper] No API key found. Set OPENAI_API_KEY environment variable.")

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper API."""
        if not self.api_key:
            return {
                'text': '[OpenAI API key not configured]',
                'confidence': 0.0,
                'language': 'en',
                'segments': []
            }

        try:
            # Convert PCM to WAV
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate)

            import httpx

            files = {
                'file': ('audio.wav', wav_bytes, 'audio/wav'),
            }
            data = {
                'model': self.model,
                'response_format': 'verbose_json'
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://api.openai.com/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    files=files,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

            text = result.get('text', '').strip()
            language = result.get('language', 'en')

            segments = []
            if 'segments' in result:
                for seg in result['segments']:
                    segments.append({
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', '')
                    })

            logger.info(f"[OpenAIWhisper] Transcribed: {text[:50]}...")

            return {
                'text': text,
                'confidence': 0.95,
                'language': language,
                'segments': segments
            }

        except Exception as e:
            logger.error(f"[OpenAIWhisper] Transcription error: {e}")
            return {
                'text': f'[Transcription error: {e}]',
                'confidence': 0.0,
                'language': 'en',
                'segments': []
            }


# ========== Factory ==========

def create_transcription_client(
    backend: str = "auto",
    **kwargs
) -> TranscriptionClient:
    """
    Factory function to create transcription client.

    Args:
        backend: Backend to use ("groq", "local", "openai", "auto")
        **kwargs: Backend-specific arguments

    Returns:
        TranscriptionClient instance

    Auto selection priority:
    1. Groq (if GROQ_API_KEY set)
    2. Local (if faster-whisper installed)
    3. OpenAI (if OPENAI_API_KEY set)
    """
    if backend == "groq":
        return GroqWhisperClient(**kwargs)

    elif backend == "local":
        return LocalWhisperClient(**kwargs)

    elif backend == "openai":
        return OpenAIWhisperClient(**kwargs)

    elif backend == "auto":
        # Try Groq first (fastest)
        if os.environ.get('GROQ_API_KEY'):
            logger.info("[Transcription] Auto-selected: Groq Whisper")
            return GroqWhisperClient(**kwargs)

        # Try local faster-whisper
        try:
            import faster_whisper
            logger.info("[Transcription] Auto-selected: Local faster-whisper")
            return LocalWhisperClient(**kwargs)
        except ImportError:
            pass

        # Fall back to OpenAI
        if os.environ.get('OPENAI_API_KEY'):
            logger.info("[Transcription] Auto-selected: OpenAI Whisper")
            return OpenAIWhisperClient(**kwargs)

        # No backend available
        logger.warning("[Transcription] No transcription backend available. "
                      "Set GROQ_API_KEY, OPENAI_API_KEY, or install faster-whisper.")
        return GroqWhisperClient()  # Will return error messages

    else:
        raise ValueError(f"Unknown transcription backend: {backend}")
