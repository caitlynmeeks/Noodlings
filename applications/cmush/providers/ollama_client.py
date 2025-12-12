"""
Ollama Client

Wraps existing OpenAICompatibleLLM for consistency with multi-provider system.
Ollama provides OpenAI-compatible API at localhost:11434.
"""

import time
import logging
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client_router import LLMClient, LLMResponse
from llm_interface import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    Ollama client (wraps existing OpenAICompatibleLLM implementation).

    Ollama is local inference server with OpenAI-compatible API.
    No API key needed, runs on localhost:11434.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama client.

        Args:
            config: Provider configuration
                {
                    'base_url': 'http://localhost:11434',
                    'default_model': 'deepseek-r1:14b'
                }
        """
        base_url = config.get('base_url', 'http://localhost:11434')

        # Ollama uses /v1 suffix for OpenAI compatibility
        api_base = f"{base_url}/v1" if not base_url.endswith('/v1') else base_url

        self.default_model = config.get('default_model', 'deepseek-r1:14b')

        # Use existing OpenAICompatibleLLM implementation
        self.impl = OpenAICompatibleLLM(
            api_base=api_base,
            api_key='not-needed',  # Ollama doesn't need API key
            model=self.default_model,
            use_model_instances=True
        )

        logger.info(f"Ollama client initialized: {api_base}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion via Ollama.

        Args:
            messages: Chat messages
            model: Ollama model name (e.g., "deepseek-r1:14b")
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            LLMResponse with standardized format
        """
        # Extract system prompt if present (for existing API compatibility)
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                # Keep user/assistant messages
                user_messages.append(msg)

        # For now, use the simple completion interface
        # (Existing OpenAICompatibleLLM uses system + user format)
        if len(user_messages) == 1 and user_messages[0]['role'] == 'user':
            user_prompt = user_messages[0]['content']
        else:
            # Multiple messages - concatenate (simplification for now)
            user_prompt = '\n'.join(msg['content'] for msg in user_messages)

        # Call existing implementation
        start_time = time.time()

        try:
            # Use existing _complete method
            response_text, actual_model, finish_reason = await self.impl._complete(
                system_prompt=system_prompt or "",
                user_prompt=user_prompt,
                temperature=temperature,
                model=model or self.default_model
            )

            latency_ms = (time.time() - start_time) * 1000

            # Estimate token counts (Ollama doesn't always return usage)
            # Rough estimate: ~4 chars per token
            input_tokens = (len(system_prompt or "") + len(user_prompt)) // 4
            output_tokens = len(response_text) // 4

            return LLMResponse(
                content=response_text,
                model=actual_model or model,
                provider='ollama',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                finish_reason=finish_reason or 'stop',
                latency_ms=latency_ms,
                cached=False
            )

        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            raise OllamaClientError(f"Ollama generation failed: {str(e)}")

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
        messages = []

        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': user_prompt})

        response = await self.generate(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            **kwargs
        )

        return response.content


class OllamaClientError(Exception):
    """Ollama client error"""
    pass


# Convenience function for testing
async def test_ollama_client():
    """Test Ollama client with simple prompt"""

    client = OllamaClient({
        'base_url': 'http://localhost:11434',
        'default_model': 'deepseek-r1:7b'
    })

    print("Testing Ollama client...")

    response = await client.complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Say 'Honque!' if you can hear me"
    )

    print(f"Response: {response}")
    print("Ollama client working!")


if __name__ == '__main__':
    import asyncio
    asyncio.run(test_ollama_client())
