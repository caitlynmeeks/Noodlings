"""
Headless LLM Client - Multi-provider LLM interface without Qt dependencies

For use in the NoodleStudio runtime (standalone applications).
Provides the interface expected by FacetExecutor.

Supported providers:
- noodlings: Noodlings cloud routing service (api.noodlings.ai)
- ollama: Local Ollama server
- anthropic: Anthropic Claude API (direct, own key)
- openai: OpenAI API (direct, own key)
- openrouter: OpenRouter aggregated API

The 'noodlings' provider is the recommended option for built applications.
It routes through our cloud service, billing the user's Noodlings account.

Author: Caitlyn + Claude
Date: January 3, 2026
"""

import aiohttp
import asyncio
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for headless LLM client."""

    provider: str = "ollama"  # noodlings, ollama, anthropic, openai, openrouter
    model: str = ""  # Model name (provider-specific)
    api_key: str = ""  # API key (from env or config)
    base_url: str = ""  # Custom base URL
    timeout: int = 120  # Request timeout in seconds
    max_concurrent: int = 5  # Max parallel requests

    # Model label mapping (SMALL/MEDIUM/LARGE -> actual model)
    model_labels: Dict[str, str] = None

    def __post_init__(self):
        if self.model_labels is None:
            self.model_labels = {}

        # Set defaults based on provider
        if not self.base_url:
            if self.provider == "noodlings":
                self.base_url = "https://api.noodlings.ai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "anthropic":
                self.base_url = "https://api.anthropic.com/v1"
            elif self.provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"

        # Get API key from environment if not provided
        if not self.api_key:
            env_map = {
                "noodlings": "NOODLINGS_API_KEY",  # User's Noodlings account token
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            if self.provider in env_map:
                self.api_key = os.environ.get(env_map[self.provider], "")


class HeadlessLLMClient:
    """
    Headless LLM client for runtime execution.

    Provides the interface expected by FacetExecutor:
    - generate_with_tokens(prompt, system_prompt, model, temperature, max_tokens)

    No Qt dependencies - pure async/aiohttp.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize headless LLM client.

        Args:
            config: LLM configuration. Defaults to Ollama.
        """
        self.config = config or LLMConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _resolve_model(self, model_label: Optional[str]) -> str:
        """
        Resolve model label to actual model name.

        Args:
            model_label: Model label (SMALL/MEDIUM/LARGE) or actual name

        Returns:
            Actual model name for the provider
        """
        if not model_label:
            return self.config.model or self._default_model()

        # Check if it's a label
        label_upper = model_label.upper()
        if label_upper in self.config.model_labels:
            return self.config.model_labels[label_upper]

        # It's an actual model name
        return model_label

    def _default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "noodlings": "anthropic/claude-3.5-sonnet",  # Our routing uses provider/model format
            "ollama": "llama3.2",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o-mini",
            "openrouter": "anthropic/claude-3.5-sonnet",
        }
        return defaults.get(self.config.provider, "gpt-4o-mini")

    async def generate_with_tokens(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400
    ) -> Tuple[str, int]:
        """
        Generate text with token count tracking.

        This is the interface expected by FacetExecutor.

        Args:
            prompt: User prompt
            system_prompt: System message
            model: Model label or name (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, token_count)
        """
        await self._ensure_session()

        resolved_model = self._resolve_model(model)

        # Route to provider-specific implementation
        if self.config.provider == "anthropic":
            return await self._generate_anthropic(
                prompt, system_prompt, resolved_model, temperature, max_tokens
            )
        elif self.config.provider == "noodlings":
            # Noodlings cloud uses OpenAI-compatible format
            return await self._generate_noodlings(
                prompt, system_prompt, resolved_model, temperature, max_tokens
            )
        else:
            # OpenAI-compatible (Ollama, OpenAI, OpenRouter)
            return await self._generate_openai_compatible(
                prompt, system_prompt, resolved_model, temperature, max_tokens
            )

    async def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, int]:
        """Generate using OpenAI-compatible API (Ollama, OpenAI, OpenRouter)."""
        async with self._semaphore:
            url = f"{self.config.base_url}/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            # OpenRouter requires additional headers
            if self.config.provider == "openrouter":
                headers["HTTP-Referer"] = "https://noodlings.ai"
                headers["X-Title"] = "NoodleStudio Runtime"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            logger.debug(f"LLM request to {model}: {prompt[:100]}...")

            try:
                async with self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"LLM API error {resp.status}: {text}")
                        raise Exception(f"LLM API error {resp.status}: {text}")

                    data = await resp.json()

                    # Extract response text
                    response_text = data['choices'][0]['message']['content']

                    # Extract token count
                    token_count = data.get('usage', {}).get('total_tokens', 0)

                    # Strip thinking tags if present
                    response_text = self._strip_thinking_tags(response_text)

                    logger.debug(f"LLM response ({token_count} tokens): {response_text[:100]}...")

                    return response_text, token_count

            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {e}")
                raise

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, int]:
        """Generate using Anthropic's native API."""
        async with self._semaphore:
            url = f"{self.config.base_url}/messages"

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01"
            }

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }

            logger.debug(f"Anthropic request to {model}: {prompt[:100]}...")

            try:
                async with self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Anthropic API error {resp.status}: {text}")
                        raise Exception(f"Anthropic API error {resp.status}: {text}")

                    data = await resp.json()

                    # Extract response text from Anthropic format
                    content = data.get('content', [])
                    response_text = ""
                    for block in content:
                        if block.get('type') == 'text':
                            response_text += block.get('text', '')

                    # Extract token count
                    usage = data.get('usage', {})
                    token_count = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)

                    logger.debug(f"Anthropic response ({token_count} tokens): {response_text[:100]}...")

                    return response_text, token_count

            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {e}")
                raise

    async def _generate_noodlings(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, int]:
        """
        Generate using Noodlings cloud routing service.

        Uses OpenAI-compatible format at api.noodlings.ai/v1/chat/completions.
        Bills the user's Noodlings account via credits.
        """
        async with self._semaphore:
            url = f"{self.config.base_url}/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            }

            # Use model format: provider/model-name (e.g., anthropic/claude-3.5-sonnet)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,  # Streaming not yet supported
            }

            logger.debug(f"Noodlings request to {model}: {prompt[:100]}...")

            try:
                async with self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status == 402:
                        # Insufficient credits
                        data = await resp.json()
                        error_msg = data.get('error', {}).get('message', 'Insufficient credits')
                        logger.error(f"Noodlings billing error: {error_msg}")
                        raise Exception(f"Billing error: {error_msg}")

                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Noodlings API error {resp.status}: {text}")
                        raise Exception(f"Noodlings API error {resp.status}: {text}")

                    data = await resp.json()

                    # OpenAI-compatible response format
                    response_text = data['choices'][0]['message']['content']
                    token_count = data.get('usage', {}).get('total_tokens', 0)

                    # Strip thinking tags if present
                    response_text = self._strip_thinking_tags(response_text)

                    logger.debug(f"Noodlings response ({token_count} tokens): {response_text[:100]}...")

                    return response_text, token_count

            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {e}")
                raise

    def _strip_thinking_tags(self, text: str) -> str:
        """Strip <thinking> tags from LLM response."""
        import re

        patterns = [
            r'<thinking>.*?</thinking>',
            r'<think>.*?</think>',
        ]

        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        return result.strip()

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400
    ) -> str:
        """
        Simple generation interface (discards token count).

        Args:
            prompt: User prompt
            system_prompt: System message
            model: Model label or name
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        text, _ = await self.generate_with_tokens(
            prompt, system_prompt, model, temperature, max_tokens
        )
        return text


def create_llm_client_from_env() -> HeadlessLLMClient:
    """
    Create LLM client from environment variables.

    Environment variables:
    - NOODLE_LLM_PROVIDER: ollama, anthropic, openai, openrouter
    - NOODLE_LLM_MODEL: default model name
    - NOODLE_LLM_BASE_URL: custom base URL
    - ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY: API keys

    Returns:
        Configured HeadlessLLMClient
    """
    provider = os.environ.get("NOODLE_LLM_PROVIDER", "ollama")
    model = os.environ.get("NOODLE_LLM_MODEL", "")
    base_url = os.environ.get("NOODLE_LLM_BASE_URL", "")

    # Model label mapping from environment
    model_labels = {}
    for label in ["SMALL", "MEDIUM", "LARGE"]:
        env_key = f"NOODLE_LLM_MODEL_{label}"
        if env_key in os.environ:
            model_labels[label] = os.environ[env_key]

    config = LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        model_labels=model_labels
    )

    return HeadlessLLMClient(config)
