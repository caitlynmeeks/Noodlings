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
#   Vision API - Scripting interface for image understanding and generation.
#
#   Provides context.noodle.vision in ScriptedFacets with Uni...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.vision_api
# PURPOSE:  Vision Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   VisionAPIState, VisionAPI, get_vision_api()
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
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VisionAPIState:
    """
    Snapshot of vision state for JavaScript access.
    """
    # Vision (understanding)
    last_description: str = ""
    last_objects: List[str] = field(default_factory=list)
    last_text: str = ""
    hot_image_count: int = 0
    warm_image_count: int = 0

    # Generation
    is_generating: bool = False
    last_generated_prompt: str = ""
    generation_queue_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JavaScript-compatible dict."""
        return {
            'lastDescription': self.last_description,
            'lastObjects': self.last_objects,
            'lastText': self.last_text,
            'hotImageCount': self.hot_image_count,
            'warmImageCount': self.warm_image_count,
            'isGenerating': self.is_generating,
            'lastGeneratedPrompt': self.last_generated_prompt,
            'generationQueueSize': self.generation_queue_size
        }


class VisionAPI:
    """
    Vision scripting API for context.noodle.vision.

    Combines image understanding (VisionFacet) and image generation (ImageGenFacet).

    Example (JavaScript in ScriptedFacet):
        function process(inputs, context) {
            // Analyze an image
            var analysis = context.noodle.vision.analyze("photo.jpg");
            if (analysis.objects.includes("cat")) {
                context.noodle.audio.speak("I see a cat!");
            }

            // Generate based on analysis
            context.noodle.vision.generate(
                "A painting of " + analysis.description,
                "artistic"
            );

            return {saw: analysis.description};
        }
    """

    def __init__(self):
        """Initialize Vision API."""
        self._state = VisionAPIState()
        self._vision_facet = None
        self._image_gen_facet = None

        # Pending commands
        self._pending_analyze: Optional[tuple] = None  # (path, prompt)
        self._pending_generate: Optional[tuple] = None  # (prompt, style)
        self._pending_screenshot: bool = False

    # ========== Facet Connection ==========

    def set_vision_facet(self, facet):
        """Connect to VisionFacet for image understanding."""
        self._vision_facet = facet
        logger.info("[VisionAPI] Connected to VisionFacet")

    def set_image_gen_facet(self, facet):
        """Connect to ImageGenFacet for image generation."""
        self._image_gen_facet = facet
        logger.info("[VisionAPI] Connected to ImageGenFacet")

    # ========== Image Understanding ==========

    def analyze(self, image_path: str, prompt: str = "Describe this image.") -> Dict[str, Any]:
        """
        Queue image for analysis.

        Args:
            image_path: Path to image file
            prompt: Analysis prompt

        Returns:
            Last analysis result (new analysis queued)
        """
        self._pending_analyze = (image_path, prompt)
        logger.info(f"[VisionAPI] Queued analysis: {image_path}")

        # Return last known result
        return {
            'description': self._state.last_description,
            'objects': self._state.last_objects,
            'text': self._state.last_text,
            'queued': True
        }

    def screenshot(self, prompt: str = "Describe what you see.") -> Dict[str, Any]:
        """
        Queue screenshot capture and analysis.

        Args:
            prompt: Analysis prompt

        Returns:
            Last analysis result (screenshot queued)
        """
        self._pending_screenshot = True
        logger.info("[VisionAPI] Queued screenshot")

        return {
            'description': self._state.last_description,
            'objects': self._state.last_objects,
            'queued': True
        }

    def get_last_image(self) -> Dict[str, Any]:
        """Get last analyzed image info."""
        return {
            'description': self._state.last_description,
            'objects': self._state.last_objects,
            'text': self._state.last_text
        }

    def search_images(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search image memory.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching images
        """
        if not self._vision_facet:
            return []

        results = self._vision_facet.search_images(query, limit)
        return [img.to_dict() for img in results]

    # ========== Image Generation ==========

    def generate(self, prompt: str, style: str = "none") -> Dict[str, Any]:
        """
        Queue image generation.

        Args:
            prompt: Text description
            style: Style preset ("photorealistic", "artistic", "anime", etc.)

        Returns:
            Generation status
        """
        self._pending_generate = (prompt, style)
        logger.info(f"[VisionAPI] Queued generation: {prompt[:50]}...")

        return {
            'queued': True,
            'prompt': prompt,
            'style': style,
            'isGenerating': self._state.is_generating
        }

    def get_last_generated(self) -> Dict[str, Any]:
        """Get last generated image info."""
        if not self._image_gen_facet:
            return {'error': 'No generation facet'}

        last = self._image_gen_facet.get_last_image()
        if not last:
            return {'hasImage': False}

        return {
            'hasImage': True,
            'prompt': last.revised_prompt,
            'width': last.width,
            'height': last.height,
            'hasData': last.image_data is not None
        }

    def list_styles(self) -> List[str]:
        """List available generation styles."""
        if self._image_gen_facet:
            return self._image_gen_facet.list_styles()
        return ["photorealistic", "artistic", "anime", "cinematic", "concept_art",
                "fantasy", "scifi", "portrait", "none"]

    def set_style(self, style: str):
        """Set default generation style."""
        if self._image_gen_facet:
            self._image_gen_facet.set_default_style(style)

    # ========== State Properties ==========

    @property
    def last_description(self) -> str:
        """Get last image description."""
        return self._state.last_description

    @property
    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        return self._state.is_generating

    # ========== Sync with Facet Cycle ==========

    def get_pending_commands(self) -> Dict[str, Any]:
        """
        Get pending commands for facets.

        Called at sync point.
        """
        commands = {}

        if self._pending_analyze:
            path, prompt = self._pending_analyze
            commands['analyze_image'] = path
            commands['prompt'] = prompt
            self._pending_analyze = None

        if self._pending_screenshot:
            commands['screenshot'] = True
            self._pending_screenshot = False

        if self._pending_generate:
            prompt, style = self._pending_generate
            commands['generate'] = prompt
            commands['style'] = style
            self._pending_generate = None

        return commands

    def update_from_vision_facet(self, facet_data: Dict[str, Any]):
        """Update state from VisionFacet sync data."""
        if 'last_description' in facet_data:
            self._state.last_description = facet_data['last_description']
        self._state.hot_image_count = facet_data.get('hot_count', 0)
        self._state.warm_image_count = facet_data.get('warm_count', 0)

        # Update from hot images
        hot_images = facet_data.get('hot_images', [])
        if hot_images:
            last = hot_images[-1]
            self._state.last_objects = last.get('objects', [])
            self._state.last_text = last.get('text', '')

    def update_from_gen_facet(self, facet_data: Dict[str, Any]):
        """Update state from ImageGenFacet sync data."""
        self._state.is_generating = facet_data.get('is_generating', False)
        self._state.generation_queue_size = facet_data.get('queue_size', 0)

        last_image = facet_data.get('last_image')
        if last_image:
            self._state.last_generated_prompt = last_image.get('prompt', '')

    # ========== JavaScript Interface ==========

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JavaScript-compatible dict for context injection.
        """
        return {
            # State (polling) - actual values
            'lastDescription': self._state.last_description,
            'lastObjects': self._state.last_objects,
            'lastText': self._state.last_text,
            'isGenerating': self._state.is_generating,
            'hotImageCount': self._state.hot_image_count,
            'warmImageCount': self._state.warm_image_count,

            # Methods (placeholders for JS binding)
            'analyze': '__vision_analyze__',
            'screenshot': '__vision_screenshot__',
            'getLastImage': '__vision_get_last_image__',
            'searchImages': '__vision_search_images__',
            'generate': '__vision_generate__',
            'getLastGenerated': '__vision_get_last_generated__',
            'listStyles': '__vision_list_styles__',
            'setStyle': '__vision_set_style__',

            # Events (placeholders)
            'onImageAnalyzed': '__vision_on_image_analyzed__',
            'onImageGenerated': '__vision_on_image_generated__'
        }


# Global singleton instance
_vision_api_instance = None


def get_vision_api() -> VisionAPI:
    """Get global VisionAPI singleton."""
    global _vision_api_instance
    if _vision_api_instance is None:
        _vision_api_instance = VisionAPI()
    return _vision_api_instance

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
