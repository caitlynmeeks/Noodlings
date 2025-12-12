"""
LLM Provider Clients

Multi-provider execution layer for Noodlings.
Each provider implements the LLMClient interface.
"""

from .openrouter_client import OpenRouterClient
from .ollama_client import OllamaClient
from .anthropic_client import AnthropicClient

__all__ = ['OpenRouterClient', 'OllamaClient', 'AnthropicClient']
