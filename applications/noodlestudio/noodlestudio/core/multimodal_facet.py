# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Multimodal Facet - Base class for parallel audio/vision/image processing.
#
#   Implements Option C architecture: Parallel subsystem with...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.multimodal_facet
# PURPOSE:  multimodal facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Modality, ModalityDirection, MultimodalEvent, MultimodalBuffer, MultimodalFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Union
from collections import deque
import uuid

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Supported input/output modalities."""
    TEXT = auto()       # Standard text (default)
    AUDIO = auto()      # Audio stream (mic/speaker)
    IMAGE = auto()      # Static images
    VIDEO = auto()      # Video stream

    @staticmethod
    def detect(data: Any) -> 'Modality':
        """
        Auto-detect modality from data.

        Args:
            data: Input data to analyze

        Returns:
            Detected Modality enum value
        """
        if data is None:
            return Modality.TEXT

        # Check for explicit modality marker
        if isinstance(data, dict):
            if '_modality' in data:
                return Modality[data['_modality'].upper()]
            if 'audio_data' in data or 'audio_buffer' in data:
                return Modality.AUDIO
            if 'image_data' in data or 'image_path' in data:
                return Modality.IMAGE
            if 'video_data' in data or 'video_path' in data:
                return Modality.VIDEO

        # Check for binary audio data
        if isinstance(data, bytes):
            # WAV header check
            if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
                return Modality.AUDIO
            # PNG header check
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                return Modality.IMAGE
            # JPEG header check
            if data[:2] == b'\xff\xd8':
                return Modality.IMAGE

        # Check for file path strings
        if isinstance(data, str):
            lower = data.lower()
            if any(lower.endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']):
                return Modality.AUDIO
            if any(lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                return Modality.IMAGE
            if any(lower.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.webm']):
                return Modality.VIDEO

        return Modality.TEXT


class ModalityDirection(Enum):
    """Direction of modality flow."""
    INPUT = "input"    # Data coming IN (mic, camera)
    OUTPUT = "output"  # Data going OUT (speaker, display)
    BOTH = "both"      # Bidirectional


@dataclass
class MultimodalEvent:
    """Event emitted by multimodal facets."""
    event_type: str           # "transcription_ready", "speech_start", etc.
    facet_id: str            # Source facet UUID
    timestamp: float         # Event timestamp
    data: Dict[str, Any]     # Event payload
    modality: Modality       # Which modality this event relates to


@dataclass
class MultimodalBuffer:
    """
    Thread-safe buffer for multimodal data.

    Supports both continuous streaming (audio) and discrete items (images).
    """
    modality: Modality
    max_size: int = 100

    # Internal storage
    _items: deque = field(default_factory=lambda: deque(maxlen=100))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def push(self, item: Any, metadata: Optional[Dict] = None):
        """Add item to buffer."""
        async with self._lock:
            self._items.append({
                'data': item,
                'metadata': metadata or {},
                'timestamp': time.time()
            })

    async def pop(self) -> Optional[Dict]:
        """Remove and return oldest item."""
        async with self._lock:
            if self._items:
                return self._items.popleft()
            return None

    async def peek(self) -> Optional[Dict]:
        """Return oldest item without removing."""
        async with self._lock:
            if self._items:
                return self._items[0]
            return None

    async def get_all(self) -> List[Dict]:
        """Get all items (non-destructive)."""
        async with self._lock:
            return list(self._items)

    async def clear(self):
        """Clear all items."""
        async with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class MultimodalFacet(ABC):
    """
    Base class for multimodal facets (audio, vision, image generation).

    Implements Option C: Parallel subsystem with sync points.

    Subclasses must implement:
        - _process_loop(): Main processing loop
        - _sync_with_cycle(): Called when main facet cycle syncs

    Lifecycle:
        1. start() - Begin parallel processing
        2. _process_loop() runs continuously in background
        3. sync() - Called by FacetExecutor at cycle boundaries
        4. stop() - Shutdown gracefully

    Event System:
        - on(event_type, callback) - Subscribe to events
        - emit(event) - Emit event to subscribers

    Unity-like API:
        - Events: onTranscriptionReady, onSpeechStart, etc.
        - Polling: lastTranscription, isSpeaking, etc.
        - Control: speak(), listen(), interrupt()
    """

    def __init__(
        self,
        facet_id: str,
        modality: Modality,
        direction: ModalityDirection = ModalityDirection.BOTH,
        process_interval_ms: int = 250,  # ~4Hz default
        model_label: Optional[str] = None
    ):
        """
        Initialize multimodal facet.

        Args:
            facet_id: Unique identifier
            modality: Primary modality (AUDIO, IMAGE, VIDEO)
            direction: INPUT, OUTPUT, or BOTH
            process_interval_ms: How often to run process loop
            model_label: Model label to use (AUDIO_IN, VISION, etc.)
        """
        self.facet_id = facet_id
        self.modality = modality
        self.direction = direction
        self.process_interval_ms = process_interval_ms
        self.model_label = model_label

        # State
        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None

        # Buffers
        self._input_buffer = MultimodalBuffer(modality)
        self._output_buffer = MultimodalBuffer(modality)

        # Sync point data (exchanged with main cycle)
        self._sync_data: Dict[str, Any] = {}
        self._sync_lock = asyncio.Lock()

        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Statistics
        self._process_count = 0
        self._total_process_time = 0.0
        self._last_process_time = 0.0
        self._sync_count = 0

        logger.info(f"[MultimodalFacet] Created {self.__class__.__name__} "
                   f"(id={facet_id[:8]}, modality={modality.name}, interval={process_interval_ms}ms)")

    # ========== Lifecycle ==========

    async def start(self):
        """Start parallel processing loop."""
        if self._running:
            logger.warning(f"[MultimodalFacet] {self.facet_id[:8]} already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[MultimodalFacet] Started {self.__class__.__name__} (id={self.facet_id[:8]})")

    async def stop(self):
        """Stop parallel processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(f"[MultimodalFacet] Stopped {self.__class__.__name__} (id={self.facet_id[:8]})")

    def pause(self):
        """Pause processing (keeps loop running but skips work)."""
        self._paused = True

    def resume(self):
        """Resume processing after pause."""
        self._paused = False

    @property
    def is_running(self) -> bool:
        """Check if processing loop is running."""
        return self._running and self._task is not None

    @property
    def is_paused(self) -> bool:
        """Check if processing is paused."""
        return self._paused

    # ========== Main Loop ==========

    async def _run_loop(self):
        """
        Main processing loop (runs in background).

        Calls _process_loop() at configured interval.
        """
        interval = self.process_interval_ms / 1000.0

        while self._running:
            loop_start = time.time()

            if not self._paused:
                try:
                    start = time.time()
                    await self._process_loop()
                    elapsed = time.time() - start

                    self._process_count += 1
                    self._total_process_time += elapsed
                    self._last_process_time = elapsed

                except Exception as e:
                    logger.error(f"[MultimodalFacet] Process loop error: {e}")

            # Sleep remaining interval time
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    @abstractmethod
    async def _process_loop(self):
        """
        Override in subclass: Process one iteration.

        Called at configured interval (e.g., every 250ms).
        Should:
            1. Check input buffer for new data
            2. Process data (transcribe, generate, etc.)
            3. Update output buffer or emit events
        """
        pass

    # ========== Sync Points ==========

    async def sync(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize with main facet execution cycle.

        Called by FacetExecutor at cycle boundaries.

        Args:
            cycle_data: Data from main cycle (context, other facet outputs)

        Returns:
            Data to pass back to main cycle
        """
        self._sync_count += 1

        async with self._sync_lock:
            # Let subclass handle sync
            result = await self._sync_with_cycle(cycle_data)

            # Clear sync data after exchange
            self._sync_data = {}

            return result

    @abstractmethod
    async def _sync_with_cycle(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override in subclass: Handle sync with main cycle.

        Args:
            cycle_data: Data from main facet cycle

        Returns:
            Data to pass back (transcriptions, generated images, etc.)
        """
        pass

    async def push_sync_data(self, key: str, value: Any):
        """
        Push data to be exchanged at next sync point.

        Args:
            key: Data key
            value: Data value
        """
        async with self._sync_lock:
            self._sync_data[key] = value

    async def get_sync_data(self, key: str) -> Optional[Any]:
        """
        Get data from last sync.

        Args:
            key: Data key

        Returns:
            Data value or None
        """
        async with self._sync_lock:
            return self._sync_data.get(key)

    # ========== Event System ==========

    def on(self, event_type: str, callback: Callable[[MultimodalEvent], None]):
        """
        Subscribe to events.

        Args:
            event_type: Event type (e.g., "transcription_ready")
            callback: Function to call when event occurs

        Example:
            facet.on("transcription_ready", lambda e: print(e.data['text']))
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(callback)

    def off(self, event_type: str, callback: Optional[Callable] = None):
        """
        Unsubscribe from events.

        Args:
            event_type: Event type
            callback: Specific callback to remove (or all if None)
        """
        if event_type in self._event_handlers:
            if callback:
                self._event_handlers[event_type] = [
                    h for h in self._event_handlers[event_type] if h != callback
                ]
            else:
                self._event_handlers[event_type] = []

    async def emit(self, event_type: str, data: Dict[str, Any]):
        """
        Emit event to subscribers.

        Args:
            event_type: Event type
            data: Event payload
        """
        event = MultimodalEvent(
            event_type=event_type,
            facet_id=self.facet_id,
            timestamp=time.time(),
            data=data,
            modality=self.modality
        )

        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[MultimodalFacet] Event handler error: {e}")

    # ========== Input/Output ==========

    async def push_input(self, data: Any, metadata: Optional[Dict] = None):
        """
        Push data to input buffer.

        Args:
            data: Input data (audio chunk, image, etc.)
            metadata: Optional metadata
        """
        await self._input_buffer.push(data, metadata)

    async def get_output(self) -> Optional[Dict]:
        """
        Get next output from output buffer.

        Returns:
            Output data dict or None
        """
        return await self._output_buffer.pop()

    async def peek_output(self) -> Optional[Dict]:
        """
        Peek at next output without removing.

        Returns:
            Output data dict or None
        """
        return await self._output_buffer.peek()

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'process_count': self._process_count,
            'sync_count': self._sync_count,
            'total_process_time': self._total_process_time,
            'last_process_time': self._last_process_time,
            'avg_process_time': (
                self._total_process_time / self._process_count
                if self._process_count > 0 else 0
            ),
            'input_buffer_size': len(self._input_buffer),
            'output_buffer_size': len(self._output_buffer),
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }

    # ========== Serialization ==========

    def to_dict(self) -> Dict[str, Any]:
        """Serialize facet state for YAML/JSON."""
        return {
            'id': self.facet_id,
            'type': self.__class__.__name__,
            'modality': self.modality.name,
            'direction': self.direction.value,
            'process_interval_ms': self.process_interval_ms,
            'model_label': self.model_label,
            'stats': self.get_stats()
        }


# ========== Factory ==========

def create_multimodal_facet(
    facet_type: str,
    facet_id: Optional[str] = None,
    **kwargs
) -> MultimodalFacet:
    """
    Factory function to create multimodal facets.

    Args:
        facet_type: Type name ("AudioStreamFacet", "VisionFacet", etc.)
        facet_id: Optional UUID (generated if not provided)
        **kwargs: Additional arguments for facet constructor

    Returns:
        MultimodalFacet instance
    """
    if facet_id is None:
        facet_id = str(uuid.uuid4())

    # Import concrete implementations
    if facet_type == "AudioStreamFacet":
        from .audio_stream_facet import AudioStreamFacet
        return AudioStreamFacet(facet_id=facet_id, **kwargs)
    elif facet_type == "VisionFacet":
        # Future implementation
        raise NotImplementedError("VisionFacet not yet implemented")
    elif facet_type == "ImageGenFacet":
        # Future implementation
        raise NotImplementedError("ImageGenFacet not yet implemented")
    else:
        raise ValueError(f"Unknown multimodal facet type: {facet_type}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
