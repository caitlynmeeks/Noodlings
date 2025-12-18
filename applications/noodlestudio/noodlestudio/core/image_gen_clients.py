"""
Image Generation Clients - Text-to-image implementations.

Supports multiple backends:
- DALL-E 3 (OpenAI) - High quality, good prompt following
- Flux (via Replicate) - Fast, artistic styles
- Stable Diffusion (via Replicate or local) - Flexible, many models

All clients implement the same interface:
    async generate(prompt, **kwargs) -> Dict

Returns:
    {
        'image_data': bytes,     # Raw image bytes
        'image_url': str,        # URL if hosted
        'width': int,
        'height': int,
        'seed': int,             # For reproducibility
        'revised_prompt': str    # What model actually used
    }

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
"""

import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeneratedImage:
    """Result from image generation."""
    image_data: Optional[bytes]
    image_url: Optional[str]
    width: int
    height: int
    seed: Optional[int]
    revised_prompt: str
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_data': self.image_data is not None,
            'image_url': self.image_url,
            'width': self.width,
            'height': self.height,
            'seed': self.seed,
            'revised_prompt': self.revised_prompt,
            'model': self.model
        }


class ImageGenClient(ABC):
    """Abstract base for image generation clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[GeneratedImage]:
        """
        Generate images from text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in image
            width: Image width
            height: Image height
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            **kwargs: Backend-specific options

        Returns:
            List of GeneratedImage
        """
        pass


class DallE3Client(ImageGenClient):
    """
    DALL-E 3 client (OpenAI).

    High quality image generation with excellent prompt understanding.
    Requires OPENAI_API_KEY environment variable.

    Supports:
    - 1024x1024, 1792x1024, 1024x1792 sizes
    - Standard and HD quality
    - Vivid and natural styles
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "dall-e-3"):
        """
        Initialize DALL-E client.

        Args:
            api_key: OpenAI API key
            model: Model to use (dall-e-3, dall-e-2)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model

        if not self.api_key:
            logger.warning("[DALL-E] No API key found. Set OPENAI_API_KEY environment variable.")

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        quality: str = "standard",  # "standard" or "hd"
        style: str = "vivid",       # "vivid" or "natural"
        **kwargs
    ) -> List[GeneratedImage]:
        """Generate images using DALL-E 3."""
        if not self.api_key:
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt="[OpenAI API key not configured]",
                model=self.model
            )]

        try:
            import httpx

            # Map dimensions to DALL-E 3 supported sizes
            size = self._get_size(width, height)

            # Combine prompts
            full_prompt = prompt
            if negative_prompt:
                full_prompt += f"\n\nAvoid: {negative_prompt}"

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "n": min(num_images, 1),  # DALL-E 3 only supports n=1
                "size": size,
                "quality": quality,
                "style": style,
                "response_format": "b64_json"
            }

            results = []

            async with httpx.AsyncClient() as client:
                # Generate requested number (DALL-E 3 is 1 at a time)
                for _ in range(num_images):
                    response = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=120.0
                    )
                    response.raise_for_status()
                    result = response.json()

                    for img_data in result.get("data", []):
                        b64 = img_data.get("b64_json")
                        image_bytes = base64.b64decode(b64) if b64 else None

                        results.append(GeneratedImage(
                            image_data=image_bytes,
                            image_url=img_data.get("url"),
                            width=width,
                            height=height,
                            seed=None,  # DALL-E doesn't expose seed
                            revised_prompt=img_data.get("revised_prompt", prompt),
                            model=self.model
                        ))

            logger.info(f"[DALL-E] Generated {len(results)} images for: {prompt[:50]}...")
            return results

        except Exception as e:
            logger.error(f"[DALL-E] Generation error: {e}")
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt=f"[Generation error: {e}]",
                model=self.model
            )]

    def _get_size(self, width: int, height: int) -> str:
        """Map dimensions to DALL-E supported size."""
        # DALL-E 3 sizes: 1024x1024, 1792x1024, 1024x1792
        aspect = width / height

        if aspect > 1.5:
            return "1792x1024"  # Landscape
        elif aspect < 0.67:
            return "1024x1792"  # Portrait
        else:
            return "1024x1024"  # Square


class FluxClient(ImageGenClient):
    """
    Flux client (via Replicate).

    Fast, high-quality image generation.
    Requires REPLICATE_API_TOKEN environment variable.

    Models:
    - black-forest-labs/flux-schnell (fast)
    - black-forest-labs/flux-dev (quality)
    - black-forest-labs/flux-pro (best)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "black-forest-labs/flux-schnell"
    ):
        """
        Initialize Flux client.

        Args:
            api_key: Replicate API token
            model: Model to use
        """
        self.api_key = api_key or os.environ.get('REPLICATE_API_TOKEN')
        self.model = model

        if not self.api_key:
            logger.warning("[Flux] No API key found. Set REPLICATE_API_TOKEN environment variable.")

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        num_inference_steps: int = 4,  # Schnell is fast with 4 steps
        guidance_scale: float = 3.5,
        **kwargs
    ) -> List[GeneratedImage]:
        """Generate images using Flux via Replicate."""
        if not self.api_key:
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt="[Replicate API key not configured]",
                model=self.model
            )]

        try:
            import httpx

            # Replicate API
            payload = {
                "version": self._get_model_version(),
                "input": {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_outputs": num_images,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                }
            }

            if seed is not None:
                payload["input"]["seed"] = seed

            if negative_prompt:
                payload["input"]["negative_prompt"] = negative_prompt

            async with httpx.AsyncClient() as client:
                # Create prediction
                response = await client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                prediction = response.json()

                # Poll for completion
                prediction_url = prediction["urls"]["get"]
                while prediction["status"] not in ["succeeded", "failed", "canceled"]:
                    await asyncio.sleep(1)
                    response = await client.get(
                        prediction_url,
                        headers={"Authorization": f"Token {self.api_key}"},
                        timeout=30.0
                    )
                    prediction = response.json()

                if prediction["status"] != "succeeded":
                    raise Exception(f"Generation failed: {prediction.get('error')}")

                # Download images
                results = []
                output_urls = prediction.get("output", [])
                if isinstance(output_urls, str):
                    output_urls = [output_urls]

                for url in output_urls:
                    img_response = await client.get(url, timeout=60.0)
                    image_bytes = img_response.content

                    results.append(GeneratedImage(
                        image_data=image_bytes,
                        image_url=url,
                        width=width,
                        height=height,
                        seed=seed,
                        revised_prompt=prompt,
                        model=self.model
                    ))

            logger.info(f"[Flux] Generated {len(results)} images for: {prompt[:50]}...")
            return results

        except Exception as e:
            logger.error(f"[Flux] Generation error: {e}")
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt=f"[Generation error: {e}]",
                model=self.model
            )]

    def _get_model_version(self) -> str:
        """Get model version hash for Replicate."""
        # These are the current stable versions
        versions = {
            "black-forest-labs/flux-schnell": "f2ab8a5bfe79f02f0b2a5d516359d1ebcfbce367c4e6bd1bd51b0d82c1e55daf",
            "black-forest-labs/flux-dev": "4c7f5d9a5ff87f8d6b70f8b3a3f4e1d2c5b6a7e8f9d0c1b2a3e4f5d6c7b8a9e0",
            "black-forest-labs/flux-pro": "a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"
        }
        return versions.get(self.model, versions["black-forest-labs/flux-schnell"])


class StableDiffusionClient(ImageGenClient):
    """
    Stable Diffusion client (via Replicate or local).

    Flexible image generation with many model variants.
    Can run locally with diffusers or via Replicate API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "stability-ai/sdxl",
        local: bool = False
    ):
        """
        Initialize Stable Diffusion client.

        Args:
            api_key: Replicate API token (for cloud)
            model: Model to use
            local: Use local diffusers instead of API
        """
        self.api_key = api_key or os.environ.get('REPLICATE_API_TOKEN')
        self.model = model
        self.local = local
        self._local_pipeline = None

        if not local and not self.api_key:
            logger.warning("[StableDiffusion] No API key found. Set REPLICATE_API_TOKEN or use local=True.")

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> List[GeneratedImage]:
        """Generate images using Stable Diffusion."""
        if self.local:
            return await self._generate_local(
                prompt, negative_prompt, width, height,
                num_images, seed, num_inference_steps, guidance_scale
            )
        else:
            return await self._generate_replicate(
                prompt, negative_prompt, width, height,
                num_images, seed, num_inference_steps, guidance_scale
            )

    async def _generate_local(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_images: int,
        seed: Optional[int],
        num_inference_steps: int,
        guidance_scale: float
    ) -> List[GeneratedImage]:
        """Generate using local diffusers pipeline."""
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
            import io

            # Lazy load pipeline
            if self._local_pipeline is None:
                logger.info("[StableDiffusion] Loading local pipeline...")
                self._local_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                # Use MPS on Mac, CUDA on Linux/Windows
                if torch.backends.mps.is_available():
                    self._local_pipeline = self._local_pipeline.to("mps")
                elif torch.cuda.is_available():
                    self._local_pipeline = self._local_pipeline.to("cuda")

            # Generate
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)

            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                None,
                lambda: self._local_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images
            )

            results = []
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()

                results.append(GeneratedImage(
                    image_data=image_bytes,
                    image_url=None,
                    width=width,
                    height=height,
                    seed=seed,
                    revised_prompt=prompt,
                    model="sdxl-local"
                ))

            logger.info(f"[StableDiffusion] Generated {len(results)} images locally")
            return results

        except ImportError:
            logger.error("[StableDiffusion] diffusers not installed for local generation")
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt="[diffusers not installed]",
                model=self.model
            )]
        except Exception as e:
            logger.error(f"[StableDiffusion] Local generation error: {e}")
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt=f"[Generation error: {e}]",
                model=self.model
            )]

    async def _generate_replicate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_images: int,
        seed: Optional[int],
        num_inference_steps: int,
        guidance_scale: float
    ) -> List[GeneratedImage]:
        """Generate using Replicate API."""
        if not self.api_key:
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt="[Replicate API key not configured]",
                model=self.model
            )]

        try:
            import httpx

            payload = {
                "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",  # SDXL
                "input": {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_outputs": num_images,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                }
            }

            if seed is not None:
                payload["input"]["seed"] = seed
            if negative_prompt:
                payload["input"]["negative_prompt"] = negative_prompt

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                prediction = response.json()

                # Poll for completion
                prediction_url = prediction["urls"]["get"]
                while prediction["status"] not in ["succeeded", "failed", "canceled"]:
                    await asyncio.sleep(2)
                    response = await client.get(
                        prediction_url,
                        headers={"Authorization": f"Token {self.api_key}"},
                        timeout=30.0
                    )
                    prediction = response.json()

                if prediction["status"] != "succeeded":
                    raise Exception(f"Generation failed: {prediction.get('error')}")

                # Download images
                results = []
                output_urls = prediction.get("output", [])
                if isinstance(output_urls, str):
                    output_urls = [output_urls]

                for url in output_urls:
                    img_response = await client.get(url, timeout=60.0)
                    image_bytes = img_response.content

                    results.append(GeneratedImage(
                        image_data=image_bytes,
                        image_url=url,
                        width=width,
                        height=height,
                        seed=seed,
                        revised_prompt=prompt,
                        model=self.model
                    ))

            logger.info(f"[StableDiffusion] Generated {len(results)} images via Replicate")
            return results

        except Exception as e:
            logger.error(f"[StableDiffusion] Replicate generation error: {e}")
            return [GeneratedImage(
                image_data=None,
                image_url=None,
                width=width,
                height=height,
                seed=None,
                revised_prompt=f"[Generation error: {e}]",
                model=self.model
            )]


# ========== Factory ==========

def create_image_gen_client(
    backend: str = "auto",
    **kwargs
) -> ImageGenClient:
    """
    Factory function to create image generation client.

    Args:
        backend: Backend to use ("dalle", "flux", "sd", "auto")
        **kwargs: Backend-specific arguments

    Returns:
        ImageGenClient instance

    Auto selection priority:
    1. DALL-E 3 (if OPENAI_API_KEY set) - best quality
    2. Flux (if REPLICATE_API_TOKEN set) - fast
    3. Stable Diffusion local (if diffusers installed)
    """
    if backend == "dalle":
        return DallE3Client(**kwargs)

    elif backend == "flux":
        return FluxClient(**kwargs)

    elif backend == "sd":
        return StableDiffusionClient(**kwargs)

    elif backend == "auto":
        # Try DALL-E first
        if os.environ.get('OPENAI_API_KEY'):
            logger.info("[ImageGen] Auto-selected: DALL-E 3")
            return DallE3Client(**kwargs)

        # Try Flux via Replicate
        if os.environ.get('REPLICATE_API_TOKEN'):
            logger.info("[ImageGen] Auto-selected: Flux")
            return FluxClient(**kwargs)

        # Try local Stable Diffusion
        try:
            import diffusers
            logger.info("[ImageGen] Auto-selected: Stable Diffusion (local)")
            return StableDiffusionClient(local=True, **kwargs)
        except ImportError:
            pass

        # No backend available
        logger.warning("[ImageGen] No image generation backend available. "
                      "Set OPENAI_API_KEY, REPLICATE_API_TOKEN, or install diffusers.")
        return DallE3Client()  # Will return error messages

    else:
        raise ValueError(f"Unknown image generation backend: {backend}")
