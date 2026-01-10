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
#   Image Generation Facet - Text-to-image output for cognitive architectures.
#
#   Generates images from text prompts using DALL-E, Flux, or...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.image_gen_facet
# PURPOSE:  image gen facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   StylePreset, GenerationRequest, ImageGenFacet, create_image_gen_facet_with_client()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

from .multimodal_facet import (
    MultimodalFacet, Modality, ModalityDirection
)
from .image_gen_clients import GeneratedImage

logger = logging.getLogger(__name__)


@dataclass
class StylePreset:
    """Preset style for image generation."""
    name: str
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 7.5
    num_steps: int = 30


# Built-in style presets
STYLE_PRESETS = {
    "photorealistic": StylePreset(
        name="photorealistic",
        prompt_suffix=", photorealistic, highly detailed, 8k, professional photography",
        negative_prompt="cartoon, anime, illustration, painting, drawing, blurry, low quality"
    ),
    "artistic": StylePreset(
        name="artistic",
        prompt_suffix=", artistic, painterly, expressive brushstrokes, fine art",
        negative_prompt="photo, photorealistic, 3d render"
    ),
    "anime": StylePreset(
        name="anime",
        prompt_suffix=", anime style, manga, japanese animation, cel shaded",
        negative_prompt="photo, photorealistic, western cartoon"
    ),
    "cinematic": StylePreset(
        name="cinematic",
        prompt_suffix=", cinematic lighting, dramatic, film still, movie scene, anamorphic",
        negative_prompt="cartoon, anime, low quality"
    ),
    "concept_art": StylePreset(
        name="concept_art",
        prompt_suffix=", concept art, digital painting, artstation, trending",
        negative_prompt="photo, photorealistic, blurry"
    ),
    "fantasy": StylePreset(
        name="fantasy",
        prompt_suffix=", fantasy art, magical, ethereal, detailed illustration",
        negative_prompt="photo, modern, urban"
    ),
    "scifi": StylePreset(
        name="scifi",
        prompt_suffix=", science fiction, futuristic, cyberpunk, neon, technological",
        negative_prompt="medieval, fantasy, natural"
    ),
    "portrait": StylePreset(
        name="portrait",
        prompt_suffix=", portrait photography, studio lighting, professional headshot",
        negative_prompt="full body, landscape, multiple people"
    ),
    "none": StylePreset(
        name="none",
        prompt_prefix="",
        prompt_suffix="",
        negative_prompt=""
    )
}


@dataclass
class GenerationRequest:
    """Queued generation request."""
    id: str
    prompt: str
    style: str
    width: int
    height: int
    seed: Optional[int]
    requested_at: float
    callback: Optional[callable] = None


class ImageGenFacet(MultimodalFacet):
    """
    Image generation facet for text-to-image output.

    Generates images from prompts and integrates with facet cycle.
    """

    def __init__(
        self,
        facet_id: str,
        process_interval_ms: int = 1000,
        model_label: str = "IMAGE_GEN",
        # Generation settings
        default_width: int = 1024,
        default_height: int = 1024,
        default_style: str = "none",
        # Output settings
        output_path: Optional[str] = None,
        cache_limit: int = 20
    ):
        """
        Initialize image generation facet.

        Args:
            facet_id: Unique identifier
            process_interval_ms: Processing loop interval
            model_label: Model label for generation
            default_width: Default image width
            default_height: Default image height
            default_style: Default style preset
            output_path: Path to save generated images
            cache_limit: Max images to keep in cache
        """
        super().__init__(
            facet_id=facet_id,
            modality=Modality.IMAGE,
            direction=ModalityDirection.OUTPUT,
            process_interval_ms=process_interval_ms,
            model_label=model_label
        )

        # Settings
        self.default_width = default_width
        self.default_height = default_height
        self.default_style = default_style
        self.output_path = output_path or "/tmp/noodlings_generated"

        # Ensure output path exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Client
        self._gen_client = None

        # Generation queue
        self._queue: deque = deque(maxlen=50)
        self._is_generating = False

        # Cache
        self._cache: deque = deque(maxlen=cache_limit)
        self._last_generated: Optional[GeneratedImage] = None

        # Statistics
        self._images_generated = 0
        self._total_generation_time = 0.0
        self._failed_generations = 0

        logger.info(f"[ImageGenFacet] Initialized (style={default_style}, "
                   f"size={default_width}x{default_height})")

    # ========== Client Setup ==========

    def set_generation_client(self, client):
        """
        Set image generation client.

        Client must implement:
            async generate(prompt, **kwargs) -> List[GeneratedImage]
        """
        self._gen_client = client
        logger.info("[ImageGenFacet] Generation client set")

    # ========== Style Management ==========

    def get_style(self, name: str) -> StylePreset:
        """Get style preset by name."""
        return STYLE_PRESETS.get(name, STYLE_PRESETS["none"])

    def set_default_style(self, style: str):
        """Set default style preset."""
        if style in STYLE_PRESETS:
            self.default_style = style
            logger.info(f"[ImageGenFacet] Default style set to: {style}")
        else:
            logger.warning(f"[ImageGenFacet] Unknown style: {style}")

    def list_styles(self) -> List[str]:
        """List available style presets."""
        return list(STYLE_PRESETS.keys())

    # ========== Image Generation ==========

    async def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Optional[GeneratedImage]:
        """
        Generate an image from text prompt.

        Args:
            prompt: Text description
            style: Style preset name (uses default if None)
            width: Image width (uses default if None)
            height: Image height (uses default if None)
            seed: Random seed for reproducibility
            **kwargs: Additional generation options

        Returns:
            GeneratedImage or None if failed
        """
        if not self._gen_client:
            logger.warning("[ImageGenFacet] No generation client set")
            return None

        start_time = time.time()

        # Apply style
        style_name = style or self.default_style
        style_preset = self.get_style(style_name)

        # Build full prompt with style
        full_prompt = style_preset.prompt_prefix + prompt + style_preset.prompt_suffix
        negative = style_preset.negative_prompt

        # Emit start event
        await self.emit("generation_started", {
            'prompt': prompt,
            'style': style_name,
            'full_prompt': full_prompt
        })

        try:
            self._is_generating = True

            # Generate
            results = await self._gen_client.generate(
                prompt=full_prompt,
                negative_prompt=negative,
                width=width or self.default_width,
                height=height or self.default_height,
                seed=seed,
                num_inference_steps=style_preset.num_steps,
                guidance_scale=style_preset.guidance_scale,
                **kwargs
            )

            if not results or not results[0].image_data:
                raise Exception("No image generated")

            generated = results[0]
            elapsed = time.time() - start_time

            # Save to disk
            if generated.image_data:
                filename = f"{uuid.uuid4().hex[:12]}.png"
                filepath = Path(self.output_path) / filename
                with open(filepath, 'wb') as f:
                    f.write(generated.image_data)
                logger.info(f"[ImageGenFacet] Saved: {filepath}")

            # Update stats
            self._images_generated += 1
            self._total_generation_time += elapsed

            # Cache
            self._cache.append(generated)
            self._last_generated = generated

            # Emit complete event
            await self.emit("image_generated", {
                'prompt': prompt,
                'style': style_name,
                'revised_prompt': generated.revised_prompt,
                'width': generated.width,
                'height': generated.height,
                'generation_time': elapsed
            })

            logger.info(f"[ImageGenFacet] Generated in {elapsed:.2f}s: {prompt[:50]}...")

            return generated

        except Exception as e:
            self._failed_generations += 1
            logger.error(f"[ImageGenFacet] Generation error: {e}")

            await self.emit("generation_failed", {
                'prompt': prompt,
                'error': str(e)
            })

            return None

        finally:
            self._is_generating = False

    # ========== Queue Management ==========

    def queue_generation(
        self,
        prompt: str,
        style: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> str:
        """
        Queue an image for generation.

        Args:
            prompt: Text description
            style: Style preset
            width: Image width
            height: Image height
            seed: Random seed
            callback: Called when generation completes

        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())

        request = GenerationRequest(
            id=request_id,
            prompt=prompt,
            style=style or self.default_style,
            width=width or self.default_width,
            height=height or self.default_height,
            seed=seed,
            requested_at=time.time(),
            callback=callback
        )

        self._queue.append(request)
        logger.info(f"[ImageGenFacet] Queued: {prompt[:30]}... (id={request_id[:8]})")

        return request_id

    # ========== Cache Access ==========

    def get_last_image(self) -> Optional[GeneratedImage]:
        """Get most recently generated image."""
        return self._last_generated

    def get_cached_images(self) -> List[GeneratedImage]:
        """Get all cached images."""
        return list(self._cache)

    # ========== Processing Loop ==========

    async def _process_loop(self):
        """Process queued generation requests."""
        if self._is_generating:
            return  # Already generating

        if not self._queue:
            return  # Nothing to process

        request = self._queue.popleft()

        result = await self.generate(
            prompt=request.prompt,
            style=request.style,
            width=request.width,
            height=request.height,
            seed=request.seed
        )

        # Call callback if provided
        if request.callback and result:
            try:
                request.callback(result)
            except Exception as e:
                logger.error(f"[ImageGenFacet] Callback error: {e}")

    async def _sync_with_cycle(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync with main facet cycle.

        Args:
            cycle_data: Data from main cycle

        Returns:
            Generation data for cycle
        """
        result = {
            'is_generating': self._is_generating,
            'queue_size': len(self._queue),
            'images_generated': self._images_generated
        }

        # Include last generated if available
        if self._last_generated:
            result['last_image'] = {
                'prompt': self._last_generated.revised_prompt,
                'width': self._last_generated.width,
                'height': self._last_generated.height,
                'has_data': self._last_generated.image_data is not None
            }

        # Check for generation request from cycle
        gen_prompt = cycle_data.get('generate')
        if gen_prompt:
            style = cycle_data.get('style', self.default_style)
            self.queue_generation(gen_prompt, style=style)

        return result

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            'images_generated': self._images_generated,
            'failed_generations': self._failed_generations,
            'total_generation_time': self._total_generation_time,
            'avg_generation_time': (
                self._total_generation_time / self._images_generated
                if self._images_generated > 0 else 0
            ),
            'queue_size': len(self._queue),
            'cache_size': len(self._cache),
            'is_generating': self._is_generating
        })
        return base_stats

    # ========== Serialization ==========

    def to_dict(self) -> Dict[str, Any]:
        """Serialize facet state."""
        base = super().to_dict()
        base.update({
            'default_width': self.default_width,
            'default_height': self.default_height,
            'default_style': self.default_style,
            'images_generated': self._images_generated,
            'is_generating': self._is_generating
        })
        return base


# ========== Factory ==========

def create_image_gen_facet_with_client(
    facet_id: str,
    gen_backend: str = "auto",
    **kwargs
) -> ImageGenFacet:
    """
    Create ImageGenFacet with configured client.

    Args:
        facet_id: Unique identifier
        gen_backend: "dalle", "flux", "sd", or "auto"
        **kwargs: Additional ImageGenFacet arguments

    Returns:
        Configured ImageGenFacet
    """
    from .image_gen_clients import create_image_gen_client

    facet = ImageGenFacet(facet_id=facet_id, **kwargs)
    facet.set_generation_client(create_image_gen_client(gen_backend))

    return facet

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
