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
#   LLM Client Router
#
#   A switchboard that routes AI requests to the right provider.
#   When code asks for "SMALL" model, the router figures out which
#   service to use (Ollama running locally? Anthropic Claude in
#   the cloud? OpenRouter?) and creates the appropriate connection.
#   This lets the rest of the code not worry about which provider
#   is configured - it just asks for what it needs.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.llm_client_router
# PURPOSE:  Route model labels to appropriate LLM provider clients
# LAYER:    Backend / LLM Infrastructure
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LLMClientRouter   Routes SMALL/MEDIUM/LARGE to provider clients
#   LLMClient         Abstract base class for provider implementations
#   LLMResponse       Standardized response format across providers
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
LLM Client Router - Multi-Provider Execution Infrastructure

Routes model labels (SMALL/MEDIUM/LARGE) to appropriate provider clients.
Unified interface for Ollama, Anthropic, OpenAI, OpenRouter, LM Studio.

Author: Caitlyn + Claude
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Standardized LLM response format across all providers.

    Providers have different response structures - this normalizes them.
    """
    content: str  # Generated text
    model: str  # Actual model used (may differ from requested)
    provider: str  # Provider ID (ollama, anthropic, openrouter, etc.)

    # Usage statistics
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Metadata
    finish_reason: str  # stop, length, error, etc.
    latency_ms: float  # API call duration

    # Optional
    cached: bool = False  # For providers with prompt caching
    error: Optional[str] = None


class LLMClient(ABC):
    """
    Abstract base class for LLM provider clients.

    Each provider (Anthropic, OpenAI, OpenRouter, Ollama) implements this interface.
    Ensures consistent calling patterns regardless of provider.
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from message list.

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}, ...]
            model: Model name (provider-specific)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with standardized format
        """
        pass

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str = None,
        **kwargs
    ) -> str:
        """
        Simple completion interface (system + user prompt).

        Convenience wrapper around generate() for common use case.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            model: Model name (optional, uses client default)

        Returns:
            Generated text (just the string, not full response)
        """
        pass

    async def complete_with_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str = None,
        **kwargs
    ) -> LLMResponse:
        """
        Complete with full response metadata.

        Like complete() but returns full LLMResponse for token tracking.
        """
        messages = []

        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': user_prompt})

        return await self.generate(messages, model, temperature, **kwargs)


class LLMClientRouter:
    """
    Routes LLM requests to appropriate provider based on model labels.

    Usage:
        router = LLMClientRouter(provider_manager, model_label_manager)

        # Get client for model label
        client = router.get_client("LARGE")  # → AnthropicClient

        # Generate
        response = await client.complete(
            system_prompt="You are Red",
            user_prompt="Say hi",
            model="claude-sonnet-4.5"
        )
    """

    def __init__(self, provider_manager=None, model_label_manager=None):
        """
        Initialize router.

        Args:
            provider_manager: ProviderManager instance (for configs)
            model_label_manager: ModelLabelManager instance (for label→provider mapping)
        """
        self.provider_manager = provider_manager
        self.label_manager = model_label_manager
        self.clients: Dict[str, LLMClient] = {}

        # Usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'by_provider': {},
            'total_tokens': 0,
        }

    def get_client(self, model_label: str) -> LLMClient:
        """
        Get LLM client for model label.

        Args:
            model_label: Label like "SMALL", "MEDIUM", "LARGE"

        Returns:
            LLMClient instance for the configured provider

        Raises:
            ValueError: If label not configured or provider unknown
        """
        # Look up provider + model for this label
        provider_id, model_name = self.label_manager.get_label(model_label)

        if not provider_id:
            raise ValueError(f"Model label '{model_label}' not configured")

        # Get or create client for this provider
        if provider_id not in self.clients:
            self.clients[provider_id] = self._create_client(provider_id)

        return self.clients[provider_id]

    def get_model_for_label(self, model_label: str) -> tuple[str, str]:
        """
        Get (provider, model) tuple for label.

        Returns:
            (provider_id, model_name) tuple
        """
        return self.label_manager.get_label(model_label)

    def _create_client(self, provider_id: str) -> LLMClient:
        """
        Factory method: Create client for provider.

        Args:
            provider_id: Provider identifier (ollama, anthropic, openrouter, etc.)

        Returns:
            LLMClient subclass instance
        """
        # Get provider configuration
        config = self.provider_manager.get_provider_config(provider_id)

        # Create appropriate client
        if provider_id == 'ollama':
            from .providers.ollama_client import OllamaClient
            return OllamaClient(config)

        elif provider_id == 'anthropic':
            from .providers.anthropic_client import AnthropicClient
            return AnthropicClient(config)

        elif provider_id == 'openai':
            from .providers.openai_client import OpenAIClient
            return OpenAIClient(config)

        elif provider_id == 'openrouter':
            from .providers.openrouter_client import OpenRouterClient
            return OpenRouterClient(config)

        elif provider_id == 'lmstudio':
            from .providers.lmstudio_client import LMStudioClient
            return LMStudioClient(config)

        else:
            raise ValueError(f"Unknown provider: {provider_id}")

    async def generate_with_fallback(
        self,
        model_label: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        fallback_to_ollama: bool = True
    ) -> LLMResponse:
        """
        Generate with automatic fallback to Ollama if provider fails.

        Useful for handling API errors gracefully without breaking agents.
        """
        try:
            # Try primary provider
            client = self.get_client(model_label)
            provider_id, model_name = self.get_model_for_label(model_label)

            response = await client.generate(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Track usage
            self._track_usage(response)

            return response

        except Exception as e:
            logger.error(f"LLM call failed for {model_label}: {e}")

            if fallback_to_ollama and provider_id != 'ollama':
                logger.warning(f"Falling back to Ollama for {model_label}")

                # Get Ollama client
                ollama_client = self.clients.get('ollama')
                if not ollama_client:
                    ollama_client = self._create_client('ollama')
                    self.clients['ollama'] = ollama_client

                # Get Ollama model for this tier
                ollama_model = self._get_ollama_fallback_model(model_label)

                response = await ollama_client.generate(
                    messages=messages,
                    model=ollama_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                response.error = f"Primary provider failed: {str(e)}"
                return response

            raise

    def _get_ollama_fallback_model(self, model_label: str) -> str:
        """Get Ollama model for fallback based on tier"""
        fallbacks = {
            'SMALL': 'deepseek-r1:7b',
            'MEDIUM': 'deepseek-r1:14b',
            'LARGE': 'deepseek-r1:70b',
        }
        return fallbacks.get(model_label, 'deepseek-r1:14b')

    def _track_usage(self, response: LLMResponse):
        """Track usage statistics"""
        self.usage_stats['total_requests'] += 1
        self.usage_stats['total_tokens'] += response.total_tokens

        provider = response.provider
        if provider not in self.usage_stats['by_provider']:
            self.usage_stats['by_provider'][provider] = {
                'requests': 0,
                'tokens': 0,
                'cost_estimate': 0.0
            }

        self.usage_stats['by_provider'][provider]['requests'] += 1
        self.usage_stats['by_provider'][provider]['tokens'] += response.total_tokens

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics (for monitoring/cost tracking)"""
        return self.usage_stats.copy()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
