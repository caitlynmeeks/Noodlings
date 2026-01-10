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
#   Vision Clients - Image understanding implementations.
#
#   Supports multiple backends: - Claude Vision (Anthropic) -...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.vision_clients
# PURPOSE:  Vision Clients
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   VisionResult, VisionClient, ClaudeVisionClient, GPT4VisionClient, LLaVAClient
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
import logging
import os
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from image analysis."""
    description: str
    objects: List[str]
    text: str  # OCR
    emotions: Dict[str, float]  # face -> emotion scores
    colors: List[str]  # dominant colors
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'objects': self.objects,
            'text': self.text,
            'emotions': self.emotions,
            'colors': self.colors,
            'metadata': self.metadata
        }


class VisionClient(ABC):
    """Abstract base for vision clients."""

    @abstractmethod
    async def analyze(
        self,
        image: Union[bytes, str, Path],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Analyze an image.

        Args:
            image: Image data (bytes), base64 string, file path, or URL
            prompt: Analysis prompt
            max_tokens: Max response tokens

        Returns:
            Dict with 'description', 'objects', 'text', 'emotions', 'metadata'
        """
        pass

    def _prepare_image(self, image: Union[bytes, str, Path]) -> tuple[str, str]:
        """
        Prepare image for API submission.

        Returns:
            (base64_data, media_type)
        """
        # Handle file path
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                with open(path, 'rb') as f:
                    image_bytes = f.read()
                # Detect media type from extension
                ext = path.suffix.lower()
                media_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                media_type = media_types.get(ext, 'image/png')
                return base64.b64encode(image_bytes).decode('utf-8'), media_type
            elif str(image).startswith('http'):
                # URL - return as-is
                return str(image), 'url'
            else:
                # Assume base64
                return str(image), 'image/png'

        # Handle bytes
        if isinstance(image, bytes):
            # Detect format from magic bytes
            if image[:8] == b'\x89PNG\r\n\x1a\n':
                media_type = 'image/png'
            elif image[:2] == b'\xff\xd8':
                media_type = 'image/jpeg'
            elif image[:6] in (b'GIF87a', b'GIF89a'):
                media_type = 'image/gif'
            elif image[:4] == b'RIFF' and image[8:12] == b'WEBP':
                media_type = 'image/webp'
            else:
                media_type = 'image/png'  # default

            return base64.b64encode(image).decode('utf-8'), media_type

        raise ValueError(f"Unsupported image type: {type(image)}")


class ClaudeVisionClient(VisionClient):
    """
    Claude Vision client (Anthropic).

    Uses Claude's multimodal capabilities for image understanding.
    Best quality, excellent reasoning about images.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude Vision client.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model to use (claude-sonnet-4-20250514, claude-opus-4-20250514)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model

        if not self.api_key:
            logger.warning("[ClaudeVision] No API key found. Set ANTHROPIC_API_KEY environment variable.")

    async def analyze(
        self,
        image: Union[bytes, str, Path],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Analyze image using Claude Vision."""
        if not self.api_key:
            return {
                'description': '[Anthropic API key not configured]',
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'error': 'API key missing'}
            }

        try:
            import httpx

            image_data, media_type = self._prepare_image(image)

            # Build message content
            if media_type == 'url':
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_data
                    }
                }
            else:
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }

            # Enhanced prompt for structured output
            analysis_prompt = f"""{prompt}

Please also identify:
1. Main objects visible
2. Any text in the image
3. Emotions shown (if faces present)
4. Dominant colors"""

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": analysis_prompt}
                        ]
                    }
                ]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

            # Extract response
            description = ""
            if result.get("content"):
                for block in result["content"]:
                    if block.get("type") == "text":
                        description = block.get("text", "")

            logger.info(f"[ClaudeVision] Analyzed image: {description[:50]}...")

            # Parse structured elements from description
            return self._parse_response(description)

        except Exception as e:
            logger.error(f"[ClaudeVision] Analysis error: {e}")
            return {
                'description': f'[Vision error: {e}]',
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'error': str(e)}
            }

    def _parse_response(self, description: str) -> Dict[str, Any]:
        """Parse structured elements from Claude's response."""
        # Simple parsing - Claude usually structures its responses well
        objects = []
        text_found = ""
        emotions = {}
        colors = []

        lines = description.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()

            if 'object' in line_lower and ':' in line_lower:
                current_section = 'objects'
            elif 'text' in line_lower and ':' in line_lower:
                current_section = 'text'
            elif 'emotion' in line_lower and ':' in line_lower:
                current_section = 'emotions'
            elif 'color' in line_lower and ':' in line_lower:
                current_section = 'colors'
            elif line.startswith('-') or line.startswith('*'):
                item = line.lstrip('-* ').strip()
                if current_section == 'objects':
                    objects.append(item)
                elif current_section == 'colors':
                    colors.append(item)

        return {
            'description': description,
            'objects': objects,
            'text': text_found,
            'emotions': emotions,
            'colors': colors,
            'metadata': {'model': self.model}
        }


class GPT4VisionClient(VisionClient):
    """
    GPT-4 Vision client (OpenAI).

    Uses GPT-4's vision capabilities.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize GPT-4 Vision client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use (gpt-4o, gpt-4-turbo)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model

        if not self.api_key:
            logger.warning("[GPT4Vision] No API key found. Set OPENAI_API_KEY environment variable.")

    async def analyze(
        self,
        image: Union[bytes, str, Path],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Analyze image using GPT-4 Vision."""
        if not self.api_key:
            return {
                'description': '[OpenAI API key not configured]',
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'error': 'API key missing'}
            }

        try:
            import httpx

            image_data, media_type = self._prepare_image(image)

            # Build image content
            if media_type == 'url':
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": image_data}
                }
            else:
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                }

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            image_content
                        ]
                    }
                ]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

            description = result["choices"][0]["message"]["content"]
            logger.info(f"[GPT4Vision] Analyzed image: {description[:50]}...")

            return {
                'description': description,
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'model': self.model}
            }

        except Exception as e:
            logger.error(f"[GPT4Vision] Analysis error: {e}")
            return {
                'description': f'[Vision error: {e}]',
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'error': str(e)}
            }


class LLaVAClient(VisionClient):
    """
    LLaVA client (via Ollama).

    Local vision model - runs offline.
    Requires Ollama with LLaVA model installed.

    Install:
        ollama pull llava
    """

    def __init__(self, model: str = "llava", host: str = "http://localhost:11434"):
        """
        Initialize LLaVA client.

        Args:
            model: Ollama model name (llava, llava:13b, bakllava)
            host: Ollama server URL
        """
        self.model = model
        self.host = host

    async def analyze(
        self,
        image: Union[bytes, str, Path],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Analyze image using LLaVA via Ollama."""
        try:
            import httpx

            image_data, media_type = self._prepare_image(image)

            # Ollama expects base64 images in the 'images' array
            if media_type == 'url':
                # Need to download the image first
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(image_data)
                    img_bytes = img_response.content
                    image_data = base64.b64encode(img_bytes).decode('utf-8')

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=120.0  # Local models can be slower
                )
                response.raise_for_status()
                result = response.json()

            description = result.get("response", "")
            logger.info(f"[LLaVA] Analyzed image: {description[:50]}...")

            return {
                'description': description,
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'model': self.model, 'local': True}
            }

        except Exception as e:
            logger.error(f"[LLaVA] Analysis error: {e}")
            return {
                'description': f'[Vision error: {e}]',
                'objects': [],
                'text': '',
                'emotions': {},
                'colors': [],
                'metadata': {'error': str(e)}
            }


# ========== Factory ==========

def create_vision_client(
    backend: str = "auto",
    **kwargs
) -> VisionClient:
    """
    Factory function to create vision client.

    Args:
        backend: Backend to use ("claude", "openai", "llava", "auto")
        **kwargs: Backend-specific arguments

    Returns:
        VisionClient instance

    Auto selection priority:
    1. Claude (if ANTHROPIC_API_KEY set) - best quality
    2. OpenAI (if OPENAI_API_KEY set)
    3. LLaVA (if Ollama running)
    """
    if backend == "claude":
        return ClaudeVisionClient(**kwargs)

    elif backend == "openai":
        return GPT4VisionClient(**kwargs)

    elif backend == "llava":
        return LLaVAClient(**kwargs)

    elif backend == "auto":
        # Try Claude first (best quality)
        if os.environ.get('ANTHROPIC_API_KEY'):
            logger.info("[Vision] Auto-selected: Claude Vision")
            return ClaudeVisionClient(**kwargs)

        # Try OpenAI
        if os.environ.get('OPENAI_API_KEY'):
            logger.info("[Vision] Auto-selected: GPT-4 Vision")
            return GPT4VisionClient(**kwargs)

        # Try local LLaVA
        try:
            import httpx
            # Quick check if Ollama is running
            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if any('llava' in m.get('name', '').lower() for m in models):
                    logger.info("[Vision] Auto-selected: LLaVA (local)")
                    return LLaVAClient(**kwargs)
        except:
            pass

        # No backend available
        logger.warning("[Vision] No vision backend available. "
                      "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or install LLaVA via Ollama.")
        return ClaudeVisionClient()  # Will return error messages

    else:
        raise ValueError(f"Unknown vision backend: {backend}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
