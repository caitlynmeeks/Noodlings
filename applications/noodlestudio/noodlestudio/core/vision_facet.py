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
#   Vision Facet - Image understanding for cognitive architectures.
#
#   Processes images through vision models (Claude Vision, GP...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.vision_facet
# PURPOSE:  vision facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ImageMemory, VisionFacet, create_vision_facet_with_client()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import base64
import hashlib
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .multimodal_facet import (
    MultimodalFacet, Modality, ModalityDirection,
    MultimodalBuffer, MultimodalEvent
)

logger = logging.getLogger(__name__)


@dataclass
class ImageMemory:
    """
    An image in the memory system.

    Tracks both the image data and its semantic representation.
    """
    id: str                      # UUID
    timestamp: float             # When captured/analyzed
    source: str                  # "file", "screenshot", "camera", "url"
    path: Optional[str]          # Path to stored image (cold storage)

    # Analysis results
    description: str             # Natural language description
    objects: List[str]           # Detected objects
    text: str                    # OCR text
    emotions: Dict[str, float]   # Detected emotions
    colors: List[str]            # Dominant colors

    # Memory tier
    tier: str = "hot"           # "hot", "warm", "cold"

    # For hot tier - actual image data (tokens)
    image_data: Optional[bytes] = None
    media_type: str = "image/png"

    # For warm/cold tier - embedding for semantic search
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/storage."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'source': self.source,
            'path': self.path,
            'description': self.description,
            'objects': self.objects,
            'text': self.text,
            'emotions': self.emotions,
            'colors': self.colors,
            'tier': self.tier,
            'media_type': self.media_type,
            'has_data': self.image_data is not None,
            'has_embedding': self.embedding is not None
        }

    def to_warm(self) -> 'ImageMemory':
        """Demote to warm tier (drop image data, keep description)."""
        return ImageMemory(
            id=self.id,
            timestamp=self.timestamp,
            source=self.source,
            path=self.path,
            description=self.description,
            objects=self.objects,
            text=self.text,
            emotions=self.emotions,
            colors=self.colors,
            tier="warm",
            image_data=None,  # Drop image data
            media_type=self.media_type,
            embedding=self.embedding
        )

    def to_cold(self) -> 'ImageMemory':
        """Demote to cold tier (reference only)."""
        return ImageMemory(
            id=self.id,
            timestamp=self.timestamp,
            source=self.source,
            path=self.path,
            description=self.description[:200],  # Truncate description
            objects=self.objects[:5],  # Keep top 5 objects
            text="",  # Drop OCR
            emotions={},  # Drop emotions
            colors=self.colors[:3],  # Keep top 3 colors
            tier="cold",
            image_data=None,
            media_type=self.media_type,
            embedding=self.embedding
        )


class VisionFacet(MultimodalFacet):
    """
    Vision facet for image understanding.

    Processes images and maintains a hybrid memory system
    for efficient context management.
    """

    def __init__(
        self,
        facet_id: str,
        process_interval_ms: int = 1000,  # Vision doesn't need high frequency
        model_label: str = "VISION",
        # Memory settings
        hot_limit: int = 3,      # Max images in hot storage
        warm_limit: int = 10,    # Max images in warm storage
        cold_storage_path: Optional[str] = None
    ):
        """
        Initialize vision facet.

        Args:
            facet_id: Unique identifier
            process_interval_ms: Processing loop interval
            model_label: Model label for vision
            hot_limit: Max images to keep as full tokens
            warm_limit: Max images in warm storage
            cold_storage_path: Path for cold image storage
        """
        super().__init__(
            facet_id=facet_id,
            modality=Modality.IMAGE,
            direction=ModalityDirection.INPUT,
            process_interval_ms=process_interval_ms,
            model_label=model_label
        )

        # Memory settings
        self.hot_limit = hot_limit
        self.warm_limit = warm_limit
        self.cold_storage_path = cold_storage_path or "/tmp/noodlings_vision"

        # Ensure cold storage exists
        Path(self.cold_storage_path).mkdir(parents=True, exist_ok=True)

        # Memory tiers
        self._hot_images: deque = deque(maxlen=hot_limit)
        self._warm_images: deque = deque(maxlen=warm_limit)
        self._cold_index: Dict[str, ImageMemory] = {}  # id -> metadata only

        # Vision client
        self._vision_client = None

        # Current analysis
        self._last_analysis: Optional[ImageMemory] = None
        self._pending_images: deque = deque(maxlen=10)

        # Statistics
        self._images_analyzed = 0
        self._total_analysis_time = 0.0

        logger.info(f"[VisionFacet] Initialized (hot={hot_limit}, warm={warm_limit})")

    # ========== Client Setup ==========

    def set_vision_client(self, client):
        """
        Set vision client.

        Client must implement:
            async analyze(image, prompt, max_tokens) -> Dict
        """
        self._vision_client = client
        logger.info("[VisionFacet] Vision client set")

    # ========== Image Analysis ==========

    async def analyze_image(
        self,
        image: Union[bytes, str, Path],
        prompt: str = "Describe this image in detail.",
        source: str = "file"
    ) -> ImageMemory:
        """
        Analyze an image and add to memory.

        Args:
            image: Image data, path, or URL
            prompt: Analysis prompt
            source: Source type ("file", "screenshot", "camera", "url")

        Returns:
            ImageMemory with analysis results
        """
        if not self._vision_client:
            logger.warning("[VisionFacet] No vision client set")
            return ImageMemory(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                source=source,
                path=None,
                description="[Vision client not configured]",
                objects=[],
                text="",
                emotions={},
                colors=[],
                tier="hot"
            )

        start_time = time.time()

        # Analyze image
        result = await self._vision_client.analyze(image, prompt)

        elapsed = time.time() - start_time
        self._images_analyzed += 1
        self._total_analysis_time += elapsed

        # Prepare image data for storage
        image_bytes = None
        media_type = "image/png"

        if isinstance(image, bytes):
            image_bytes = image
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                with open(path, 'rb') as f:
                    image_bytes = f.read()

        # Create memory entry
        memory = ImageMemory(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            source=source,
            path=str(image) if isinstance(image, (str, Path)) else None,
            description=result.get('description', ''),
            objects=result.get('objects', []),
            text=result.get('text', ''),
            emotions=result.get('emotions', {}),
            colors=result.get('colors', []),
            tier="hot",
            image_data=image_bytes,
            media_type=media_type
        )

        # Add to hot storage (will demote older images)
        await self._add_to_memory(memory)

        # Update last analysis
        self._last_analysis = memory

        # Emit event
        await self.emit("image_analyzed", {
            'id': memory.id,
            'description': memory.description,
            'objects': memory.objects,
            'source': source
        })

        logger.info(f"[VisionFacet] Analyzed image in {elapsed:.2f}s: {memory.description[:50]}...")

        return memory

    async def _add_to_memory(self, memory: ImageMemory):
        """Add image to memory with tier management."""
        # If hot is full, demote oldest to warm
        if len(self._hot_images) >= self.hot_limit:
            oldest_hot = self._hot_images.popleft()

            # Save to cold storage before demoting
            if oldest_hot.image_data:
                cold_path = Path(self.cold_storage_path) / f"{oldest_hot.id}.png"
                with open(cold_path, 'wb') as f:
                    f.write(oldest_hot.image_data)
                oldest_hot.path = str(cold_path)

            warm_memory = oldest_hot.to_warm()
            self._warm_images.append(warm_memory)

            logger.debug(f"[VisionFacet] Demoted {oldest_hot.id[:8]} to warm")

        # If warm is full, demote oldest to cold
        if len(self._warm_images) >= self.warm_limit:
            oldest_warm = self._warm_images.popleft()
            cold_memory = oldest_warm.to_cold()
            self._cold_index[cold_memory.id] = cold_memory

            logger.debug(f"[VisionFacet] Demoted {oldest_warm.id[:8]} to cold")

        # Add new memory to hot
        self._hot_images.append(memory)

    # ========== Screenshot Capture ==========

    async def capture_screenshot(
        self,
        region: Optional[tuple] = None,
        prompt: str = "Describe what you see on screen."
    ) -> ImageMemory:
        """
        Capture and analyze a screenshot.

        Args:
            region: Optional (x, y, width, height) to capture specific region
            prompt: Analysis prompt

        Returns:
            ImageMemory with analysis
        """
        try:
            # Try to use PIL for screenshot
            from PIL import ImageGrab
            import io

            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()

            # Convert to bytes
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            # Analyze
            memory = await self.analyze_image(image_bytes, prompt, source="screenshot")

            await self.emit("screenshot_captured", {
                'id': memory.id,
                'description': memory.description
            })

            return memory

        except ImportError:
            logger.error("[VisionFacet] PIL not available for screenshots")
            return ImageMemory(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                source="screenshot",
                path=None,
                description="[Screenshot capture requires PIL]",
                objects=[],
                text="",
                emotions={},
                colors=[],
                tier="hot"
            )
        except Exception as e:
            logger.error(f"[VisionFacet] Screenshot error: {e}")
            raise

    # ========== Memory Access ==========

    def get_last_image(self) -> Optional[ImageMemory]:
        """Get most recently analyzed image."""
        return self._last_analysis

    def get_hot_images(self) -> List[ImageMemory]:
        """Get all images in hot storage (full tokens)."""
        return list(self._hot_images)

    def get_warm_images(self) -> List[ImageMemory]:
        """Get all images in warm storage (descriptions only)."""
        return list(self._warm_images)

    def get_image_by_id(self, image_id: str) -> Optional[ImageMemory]:
        """
        Get image by ID from any tier.

        For cold images, loads data from disk if needed.
        """
        # Check hot
        for img in self._hot_images:
            if img.id == image_id:
                return img

        # Check warm
        for img in self._warm_images:
            if img.id == image_id:
                return img

        # Check cold
        if image_id in self._cold_index:
            cold = self._cold_index[image_id]
            # Load image data from disk if path exists
            if cold.path and Path(cold.path).exists():
                with open(cold.path, 'rb') as f:
                    cold.image_data = f.read()
            return cold

        return None

    def search_images(self, query: str, limit: int = 5) -> List[ImageMemory]:
        """
        Search images by text query.

        Simple keyword matching for now.
        TODO: Implement semantic search with embeddings.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching ImageMemory
        """
        query_lower = query.lower()
        results = []

        # Search all tiers
        all_images = (
            list(self._hot_images) +
            list(self._warm_images) +
            list(self._cold_index.values())
        )

        for img in all_images:
            score = 0
            # Check description
            if query_lower in img.description.lower():
                score += 3
            # Check objects
            for obj in img.objects:
                if query_lower in obj.lower():
                    score += 2
            # Check text (OCR)
            if query_lower in img.text.lower():
                score += 1

            if score > 0:
                results.append((score, img))

        # Sort by score and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [img for _, img in results[:limit]]

    # ========== Processing Loop ==========

    async def _process_loop(self):
        """Process pending images."""
        while self._pending_images:
            image_data, prompt, source = self._pending_images.popleft()
            await self.analyze_image(image_data, prompt, source)

    async def _sync_with_cycle(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync with main facet cycle.

        Args:
            cycle_data: Data from main cycle

        Returns:
            Vision data for cycle
        """
        result = {
            'last_description': self._last_analysis.description if self._last_analysis else '',
            'hot_count': len(self._hot_images),
            'warm_count': len(self._warm_images),
            'cold_count': len(self._cold_index)
        }

        # Check for image analysis request from cycle
        image_path = cycle_data.get('analyze_image')
        if image_path:
            prompt = cycle_data.get('prompt', 'Describe this image.')
            await self.analyze_image(image_path, prompt, source='cycle')

        # Check for screenshot request
        if cycle_data.get('screenshot'):
            await self.capture_screenshot()

        # Include hot images for context (if requested)
        if cycle_data.get('include_images'):
            result['hot_images'] = [img.to_dict() for img in self._hot_images]

        return result

    # ========== Queue for Analysis ==========

    def queue_image(self, image: Union[bytes, str, Path],
                    prompt: str = "Describe this image.",
                    source: str = "queue"):
        """
        Queue an image for analysis.

        Args:
            image: Image data, path, or URL
            prompt: Analysis prompt
            source: Source type
        """
        self._pending_images.append((image, prompt, source))

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get vision statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            'images_analyzed': self._images_analyzed,
            'total_analysis_time': self._total_analysis_time,
            'avg_analysis_time': (
                self._total_analysis_time / self._images_analyzed
                if self._images_analyzed > 0 else 0
            ),
            'hot_images': len(self._hot_images),
            'warm_images': len(self._warm_images),
            'cold_images': len(self._cold_index)
        })
        return base_stats

    # ========== Serialization ==========

    def to_dict(self) -> Dict[str, Any]:
        """Serialize facet state."""
        base = super().to_dict()
        base.update({
            'hot_limit': self.hot_limit,
            'warm_limit': self.warm_limit,
            'images_analyzed': self._images_analyzed,
            'last_description': self._last_analysis.description if self._last_analysis else None
        })
        return base


# ========== Factory ==========

def create_vision_facet_with_client(
    facet_id: str,
    vision_backend: str = "auto",
    **kwargs
) -> VisionFacet:
    """
    Create VisionFacet with configured client.

    Args:
        facet_id: Unique identifier
        vision_backend: "claude", "openai", "llava", or "auto"
        **kwargs: Additional VisionFacet arguments

    Returns:
        Configured VisionFacet
    """
    from .vision_clients import create_vision_client

    facet = VisionFacet(facet_id=facet_id, **kwargs)
    facet.set_vision_client(create_vision_client(vision_backend))

    return facet

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
