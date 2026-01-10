# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   OpenRouter Client - Multi-Model Aggregator
#
#   OpenRouter.ai is like a universal translator for AI models.
#   One API key gives you access to 200+ models from Anthropic,
#   Google, Meta, Mistral, and more. This client handles their
#   OpenAI-compatible format with proper attribution headers.
#   Great for comparing models or when you need fallback options.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.providers.openrouter_client
# PURPOSE:  Access 200+ models via OpenRouter aggregation
# LAYER:    Backend / LLM Interface
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   OpenRouterClient    LLMClient implementation for OpenRouter
#   OpenRouterAPIError  Exception for API errors
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
OpenRouter Client

Provides access to 200+ models via OpenRouter.ai aggregation service.
OpenAI-compatible API with special headers for attribution.
"""

import aiohttp
import time
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client_router import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterClient(LLMClient):
    """
    OpenRouter API client.

    Aggregates 200+ models from multiple providers:
    - anthropic/claude-3.5-sonnet
    - google/gemini-pro-1.5
    - meta-llama/llama-3.1-70b-instruct
    - mistralai/mistral-large
    - And many more!

    API Format: OpenAI-compatible with special headers
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenRouter client.

        Args:
            config: Provider configuration from ProviderManager
                {
                    'api_key': 'sk-or-v1-...',
                    'base_url': 'https://openrouter.ai/api/v1',
                    'site_url': 'https://noodlings.ai',
                    'site_name': 'Noodlings'
                }
        """
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://openrouter.ai/api/v1')
        self.site_url = config.get('site_url', 'https://noodlings.ai')
        self.site_name = config.get('site_name', 'Noodlings Multi-Timescale Affective Agents')

        # Default model if none specified
        self.default_model = config.get('default_model', 'anthropic/claude-3.5-sonnet')

        # Timeout settings
        self.timeout = aiohttp.ClientTimeout(total=120)  # 2 minutes for large models

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion via OpenRouter.

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}, ...]
            model: Model name (e.g., "anthropic/claude-3.5-sonnet")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            **kwargs: Additional OpenRouter parameters (top_p, frequency_penalty, etc.)

        Returns:
            LLMResponse with standardized format

        Raises:
            OpenRouterAPIError: If API call fails
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")

        # Headers (OpenRouter-specific)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': self.site_url,  # Attribution (optional but nice!)
            'X-Title': self.site_name,  # Shows in OpenRouter dashboard
        }

        # Request payload (OpenAI-compatible format)
        payload = {
            'model': model or self.default_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }

        # Make API call
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:

                    # Check for errors
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"OpenRouter API error ({resp.status}): {error_text}")
                        raise OpenRouterAPIError(
                            f"API returned {resp.status}: {error_text}"
                        )

                    data = await resp.json()

        except aiohttp.ClientError as e:
            logger.error(f"OpenRouter network error: {e}")
            raise OpenRouterAPIError(f"Network error: {str(e)}")

        latency_ms = (time.time() - start_time) * 1000

        # Parse response (OpenAI-compatible format)
        try:
            choice = data['choices'][0]
            usage = data['usage']

            return LLMResponse(
                content=choice['message']['content'],
                model=data.get('model', model),  # Actual model used
                provider='openrouter',
                input_tokens=usage['prompt_tokens'],
                output_tokens=usage['completion_tokens'],
                total_tokens=usage['total_tokens'],
                finish_reason=choice['finish_reason'],
                latency_ms=latency_ms,
                cached=False
            )

        except (KeyError, IndexError) as e:
            logger.error(f"OpenRouter response parsing error: {e}")
            logger.error(f"Response data: {data}")
            raise OpenRouterAPIError(f"Failed to parse response: {str(e)}")

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str = None,
        **kwargs
    ) -> str:
        """
        Simple completion interface.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            model: Model name (optional, uses default)

        Returns:
            Generated text
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        # Add user prompt
        messages.append({'role': 'user', 'content': user_prompt})

        # Generate
        response = await self.generate(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            **kwargs
        )

        return response.content

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.

        Useful for populating model browser in UI.

        Returns:
            List of model dicts with id, name, pricing, etc.
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key required for model listing")

        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://openrouter.ai/api/v1/models',
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise OpenRouterAPIError(
                            f"Failed to fetch models: {error_text}"
                        )

                    data = await resp.json()
                    return data.get('data', [])

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching OpenRouter models: {e}")
            raise OpenRouterAPIError(f"Network error: {str(e)}")


class OpenRouterAPIError(Exception):
    """OpenRouter API error"""
    pass


# Convenience function for testing
async def test_openrouter_client(api_key: str):
    """Test OpenRouter client with simple prompt"""

    client = OpenRouterClient({
        'api_key': api_key,
        'site_url': 'https://noodlings.ai',
        'site_name': 'Noodlings Test'
    })

    print("Testing OpenRouter client...")

    # Test with Claude via OpenRouter
    response = await client.complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Say 'Honque!' if you can hear me",
        model="anthropic/claude-3.5-sonnet"
    )

    print(f"Response: {response}")
    print("OpenRouter client working!")


if __name__ == '__main__':
    import asyncio

    # Test with API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        asyncio.run(test_openrouter_client(api_key))
    else:
        print("Set OPENROUTER_API_KEY environment variable to test")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
