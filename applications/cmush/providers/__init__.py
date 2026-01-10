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
#   LLM Provider Clients - Multi-Provider AI Interface
#
#   Different AI providers have different APIs. This package
#   wraps them all behind a consistent LLMClient interface.
#   Whether you're using Anthropic's Claude, OpenRouter's 200+
#   models, or local Ollama - same method calls, same response
#   format. Swap providers without changing application code.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.providers
# PURPOSE:  Unified LLM client interface for multiple providers
# LAYER:    Backend / LLM Interface
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   OpenRouterClient   Access 200+ models via OpenRouter.ai
#   OllamaClient       Local inference via Ollama (no API key)
#   AnthropicClient    Direct access to Claude models
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
LLM Provider Clients

Multi-provider execution layer for Noodlings.
Each provider implements the LLMClient interface.
"""

from .openrouter_client import OpenRouterClient
from .ollama_client import OllamaClient
from .anthropic_client import AnthropicClient

__all__ = ['OpenRouterClient', 'OllamaClient', 'AnthropicClient']

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
