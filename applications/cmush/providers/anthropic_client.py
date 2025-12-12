"""
Anthropic Client

Provides access to Claude models (Opus, Sonnet, Haiku) via Anthropic API.
Uses official 'anthropic' Python SDK.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client_router import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Import anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not installed - install with: pip install anthropic")


class AnthropicClient(LLMClient):
    """
    Anthropic API client for Claude models.

    Supports:
    - claude-opus-4.5
    - claude-sonnet-4.5
    - claude-haiku-4
    - And older versions

    API Format: Different from OpenAI!
    - System prompt is separate parameter (not in messages)
    - Uses 'anthropic' SDK (not direct HTTP)
    - Prompt caching available (reduces costs)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic client.

        Args:
            config: Provider configuration
                {
                    'api_key': 'sk-ant-...',
                    'default_model': 'claude-sonnet-4.5'
                }
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        self.api_key = config.get('api_key', '')
        # Anthropic API uses dated model IDs (not friendly names!)
        self.default_model = config.get('default_model', 'claude-sonnet-4-20250514')

        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        logger.info(f"Anthropic client initialized with model: {self.default_model}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion via Claude API.

        Args:
            messages: Chat messages
            model: Claude model name (e.g., "claude-sonnet-4.5")
            temperature: Sampling temperature (0.0-1.0 for Claude)
            max_tokens: Maximum output tokens

        Returns:
            LLMResponse with standardized format

        Raises:
            AnthropicAPIError: If API call fails
        """
        # Extract system prompt (Claude handles it separately!)
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                # Keep user/assistant messages
                user_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Ensure at least one user message
        if not user_messages:
            raise ValueError("At least one user message required")

        # Make API call
        start_time = time.time()

        try:
            # Use asyncio.to_thread for sync SDK
            import asyncio

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model or self.default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",  # System is separate!
                messages=user_messages,
                **kwargs
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            content = response.content[0].text

            return LLMResponse(
                content=content,
                model=response.model,
                provider='anthropic',
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                latency_ms=latency_ms,
                cached=False  # TODO: Check for cache_read_input_tokens
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise AnthropicClientError(f"Claude API failed: {str(e)}")

        except Exception as e:
            logger.error(f"Anthropic client error: {e}")
            raise AnthropicClientError(f"Unexpected error: {str(e)}")

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
            model: Model name (optional)

        Returns:
            Generated text
        """
        messages = [
            {'role': 'user', 'content': user_prompt}
        ]

        # System prompt handled separately in generate()
        if system_prompt:
            messages.insert(0, {'role': 'system', 'content': system_prompt})

        response = await self.generate(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            **kwargs
        )

        return response.content


class AnthropicClientError(Exception):
    """Anthropic client error"""
    pass


# Convenience function for testing
async def test_anthropic_client(api_key: str):
    """Test Anthropic client with simple prompt"""

    client = AnthropicClient({
        'api_key': api_key,
        'default_model': 'claude-sonnet-4-20250514'
    })

    print("Testing Anthropic client...")

    response = await client.complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Say 'Honque!' if you can hear me"
    )

    print(f"Response: {response}")
    print("Anthropic client working!")


if __name__ == '__main__':
    import asyncio

    # Test with API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        asyncio.run(test_anthropic_client(api_key))
    else:
        print("Set ANTHROPIC_API_KEY environment variable to test")
        print("Get your key at: https://console.anthropic.com/settings/keys")
