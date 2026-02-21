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
#   Headless LLM Client - Multi-provider LLM interface without Qt dependencies
#
#   For use in the NoodleStudio runtime (standalone applicati...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.llm_client
# PURPOSE:  Llm Client
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LLMConfig, HeadlessLLMClient, create_llm_client_from_env()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import aiohttp
import asyncio
import json
import os
import logging
import re
from typing import Callable, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for headless LLM client."""

    provider: str = "ollama"  # noodlerouter, ollama, anthropic, openai, openrouter
    model: str = ""  # Model name (provider-specific)
    api_key: str = ""  # API key (from env or config)
    base_url: str = ""  # Custom base URL
    timeout: int = 120  # Request timeout in seconds
    max_concurrent: int = 5  # Max parallel requests

    # Model label mapping (SMALL/MEDIUM/LARGE -> actual model)
    model_labels: Dict[str, str] = None

    # Thinking mode per label (label -> bool). None = all enabled (default).
    thinking_modes: Dict[str, bool] = None

    def __post_init__(self):
        if self.model_labels is None:
            self.model_labels = {}
        if self.thinking_modes is None:
            self.thinking_modes = {}

        # Set defaults based on provider
        if not self.base_url:
            if self.provider == "noodlerouter":
                self.base_url = "https://api.noodlings.ai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "anthropic":
                self.base_url = "https://api.anthropic.com/v1"
            elif self.provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"
            elif self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"
            elif self.provider == "custom":
                self.base_url = "http://localhost:8080/v1"

        # Get API key from environment if not provided
        if not self.api_key:
            env_map = {
                "noodlerouter": "NOODLEROUTER_API_KEY",  # User's Noodlings account token
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
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None  # Track which loop owns session
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def _ensure_session(self):
        """Ensure aiohttp session is created and valid for current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Check if session exists and is still valid
        needs_new_session = False

        if self.session is None:
            needs_new_session = True
        elif self.session.closed:
            needs_new_session = True
            self.session = None
        elif self._session_loop is not current_loop:
            # Session was created in a different event loop - it's now invalid
            # Don't try to close it (that would fail in the wrong loop)
            # Just abandon it and create a new one
            logger.debug(f"Session loop mismatch: creating new session for current loop")
            self.session = None
            needs_new_session = True

        if needs_new_session:
            self.session = aiohttp.ClientSession()
            self._session_loop = current_loop
            logger.debug(f"Created new aiohttp session for loop {id(current_loop)}")

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
            "noodlerouter": "anthropic/claude-3.5-sonnet",  # NoodleROUTER uses provider/model format
            "ollama": "llama3.2",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o-mini",
            "openrouter": "anthropic/claude-3.5-sonnet",
        }
        return defaults.get(self.config.provider, "gpt-4o-mini")

    def _is_thinking_enabled(self, label: Optional[str]) -> bool:
        """Check if thinking mode is enabled for a given model label.

        Returns True (default) when no label is provided or when thinking_modes
        has no entry for the label.
        """
        if not label or not self.config.thinking_modes:
            return True
        # Normalize label for lookup (check both as-is and uppercase)
        if label in self.config.thinking_modes:
            return self.config.thinking_modes[label]
        if label.upper() in self.config.thinking_modes:
            return self.config.thinking_modes[label.upper()]
        return True

    async def generate_with_tokens(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
        label: Optional[str] = None
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
            label: Model label for thinking mode lookup (optional)

        Returns:
            Tuple of (generated_text, token_count)
        """
        await self._ensure_session()

        resolved_model = self._resolve_model(model)

        # When thinking is OFF: hint in system prompt + will strip tags after
        thinking_enabled = self._is_thinking_enabled(label or model)
        if not thinking_enabled:
            system_prompt = system_prompt + "\n\nRespond directly without chain-of-thought reasoning."

        # Route to provider-specific implementation
        if self.config.provider == "anthropic":
            text, tokens = await self._generate_anthropic(
                prompt, system_prompt, resolved_model, temperature, max_tokens
            )
        elif self.config.provider == "noodlerouter":
            # NoodleROUTER uses OpenAI-compatible format
            text, tokens = await self._generate_noodlerouter(
                prompt, system_prompt, resolved_model, temperature, max_tokens
            )
        else:
            # OpenAI-compatible (Ollama, OpenAI, OpenRouter)
            text, tokens = await self._generate_openai_compatible(
                prompt, system_prompt, resolved_model, temperature, max_tokens
            )

        # Strip thinking tags unless thinking mode is explicitly enabled
        # for a known label. Default behavior (no label) always strips
        # for backward compatibility.
        if not thinking_enabled or not label:
            text = self._strip_thinking_tags(text)

        return text, tokens

    async def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, int]:
        """Generate using OpenAI-compatible API (Ollama, OpenAI, OpenRouter, LMStudio)."""
        async with self._semaphore:
            base = self.config.base_url.rstrip("/")
            if base.endswith("/v1"):
                url = f"{base}/chat/completions"
            else:
                url = f"{base}/v1/chat/completions"

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

                    # Extract response text, handling thinking models
                    message = data['choices'][0]['message']
                    response_text = message.get('content') or ''

                    # Discard reasoning_content (thinking models like Kimi 2.5)
                    # Some models return content=None with only reasoning_content
                    if not response_text and 'reasoning_content' in message:
                        logger.debug("Content was None/empty, reasoning_content discarded")

                    # Extract token count
                    token_count = data.get('usage', {}).get('total_tokens', 0)

                    # Tag stripping is handled by generate_with_tokens()
                    # based on thinking mode (ON preserves, OFF strips)

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

    async def _generate_noodlerouter(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, int]:
        """
        Generate using NoodleROUTER cloud routing service.

        Uses OpenAI-compatible format at api.noodlings.ai/v1/chat/completions.
        Bills the user's Noodlings account via credits.
        """
        async with self._semaphore:
            base = self.config.base_url.rstrip("/")
            if base.endswith("/v1"):
                url = f"{base}/chat/completions"
            else:
                url = f"{base}/v1/chat/completions"

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

            logger.debug(f"NoodleROUTER request to {model}: {prompt[:100]}...")

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
                        logger.error(f"NoodleROUTER billing error: {error_msg}")
                        raise Exception(f"Billing error: {error_msg}")

                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"NoodleROUTER API error {resp.status}: {text}")
                        raise Exception(f"NoodleROUTER API error {resp.status}: {text}")

                    data = await resp.json()

                    # OpenAI-compatible response format (handle thinking models)
                    message = data['choices'][0]['message']
                    response_text = message.get('content') or ''
                    token_count = data.get('usage', {}).get('total_tokens', 0)

                    # Discard reasoning_content from thinking models
                    if not response_text and 'reasoning_content' in message:
                        logger.debug("NoodleROUTER: content was None/empty, reasoning_content discarded")

                    # Tag stripping handled by generate_with_tokens() based on thinking mode

                    logger.debug(f"NoodleROUTER response ({token_count} tokens): {response_text[:100]}...")

                    return response_text, token_count

            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {e}")
                raise

    def _strip_thinking_tags(self, text: str) -> str:
        """Strip thinking/reasoning tags from LLM response.

        Handles multiple tag variants used by different thinking models:
        - <thinking>/<think>: DeepSeek R1, Qwen QwQ
        - <reasoning>: Kimi 2.5 via LMStudio
        - <reflection>: Some reasoning models
        - <inner_thoughts>/<analysis>: Other CoT wrappers
        """
        import re

        patterns = [
            r'<thinking>.*?</thinking>',
            r'<think>.*?</think>',
            r'<reasoning>.*?</reasoning>',
            r'<reflection>.*?</reflection>',
            r'<inner_thoughts>.*?</inner_thoughts>',
            r'<analysis>.*?</analysis>',
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

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
        on_token: Optional[Callable[[str], None]] = None,
        label: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Generate text via streaming, calling on_token for each chunk.

        Args:
            prompt: User prompt
            system_prompt: System message
            model: Model label or name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            on_token: Callback for each text chunk
            label: Model label for thinking mode lookup

        Returns:
            Tuple of (full_text, token_count)
        """
        await self._ensure_session()

        resolved_model = self._resolve_model(model)
        thinking_enabled = self._is_thinking_enabled(label or model)

        if not thinking_enabled:
            system_prompt = system_prompt + "\n\nRespond directly without chain-of-thought reasoning."

        # Build the tag filter for streaming
        tag_filter = ThinkingTagFilter(suppress=not thinking_enabled)

        base = self.config.base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://noodlings.ai"
            headers["X-Title"] = "NoodleStudio Runtime"

        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        full_text = ""
        token_count = 0

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Stream API error {resp.status}: {text}")
                    # Fall back to buffered
                    return await self.generate_with_tokens(
                        prompt, system_prompt, model, temperature, max_tokens, label=label
                    )

                buffer = ""
                async for raw_line in resp.content:
                    line = raw_line.decode('utf-8', errors='replace').strip()
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = event.get("choices", [])
                    if not choices:
                        # Check for usage in final event
                        if "usage" in event:
                            token_count = event["usage"].get("total_tokens", 0)
                        continue

                    delta = choices[0].get("delta", {})

                    # Extract text from content or reasoning_content
                    chunk = ""
                    if "content" in delta and delta["content"]:
                        chunk = delta["content"]
                    elif "reasoning_content" in delta and delta["reasoning_content"]:
                        if thinking_enabled:
                            chunk = delta["reasoning_content"]

                    if chunk:
                        # Run through tag filter
                        filtered = tag_filter.feed(chunk)
                        if filtered and on_token:
                            on_token(filtered)
                        full_text += filtered

                # Flush any remaining buffered content
                remaining = tag_filter.flush()
                if remaining and on_token:
                    on_token(remaining)
                full_text += remaining

                # Get token count from usage if available
                if token_count == 0:
                    # Estimate from text length
                    token_count = len(full_text) // 4

        except Exception as e:
            logger.error(f"Streaming failed, falling back to buffered: {e}")
            return await self.generate_with_tokens(
                prompt, system_prompt, model, temperature, max_tokens, label=label
            )

        return full_text, token_count


class ThinkingTagFilter:
    """
    State machine that filters thinking tags from a stream of text chunks.

    Handles tags split across chunk boundaries (e.g., '<thi' + 'nk>').
    When suppress=True, content between opening and closing tags is discarded.
    When suppress=False, all text passes through unchanged.
    """

    # Tag names to filter
    TAG_NAMES = ['think', 'thinking', 'reasoning', 'reflection', 'inner_thoughts', 'analysis']

    def __init__(self, suppress: bool = True):
        self._suppress = suppress
        self._buffer = ""         # Partial tag buffer
        self._inside_tag = False  # Whether we're inside a thinking tag
        self._tag_name = ""       # Current tag we're matching

    def feed(self, text: str) -> str:
        """
        Feed a text chunk through the filter.

        Returns text that should be output (may be empty if inside a tag).
        """
        if not self._suppress:
            return text

        output = []
        self._buffer += text

        while self._buffer:
            if self._inside_tag:
                # Look for closing tag
                close_tag = f"</{self._tag_name}>"
                close_idx = self._buffer.lower().find(close_tag.lower())
                if close_idx >= 0:
                    # Found closing tag -- discard everything up to and including it
                    self._buffer = self._buffer[close_idx + len(close_tag):]
                    self._inside_tag = False
                    self._tag_name = ""
                elif self._could_contain_partial_close():
                    # Buffer might contain start of closing tag -- wait for more data
                    break
                else:
                    # No closing tag anywhere -- discard entire buffer
                    self._buffer = ""
                    break
            else:
                # Look for opening tag
                best_idx = -1
                best_name = ""
                for name in self.TAG_NAMES:
                    open_tag = f"<{name}>"
                    idx = self._buffer.lower().find(open_tag.lower())
                    if idx >= 0 and (best_idx < 0 or idx < best_idx):
                        best_idx = idx
                        best_name = name

                if best_idx >= 0:
                    # Found opening tag -- output text before it
                    output.append(self._buffer[:best_idx])
                    open_tag = f"<{best_name}>"
                    self._buffer = self._buffer[best_idx + len(open_tag):]
                    self._inside_tag = True
                    self._tag_name = best_name
                elif self._could_contain_partial_open():
                    # Buffer might contain start of opening tag -- output safe prefix
                    # Keep last N chars that could be start of a tag
                    safe_end = self._find_safe_output_end()
                    if safe_end > 0:
                        output.append(self._buffer[:safe_end])
                        self._buffer = self._buffer[safe_end:]
                    break
                else:
                    # No tag found -- output everything
                    output.append(self._buffer)
                    self._buffer = ""

        return "".join(output)

    def flush(self) -> str:
        """Flush remaining buffer. Call when stream is done."""
        if not self._suppress:
            result = self._buffer
            self._buffer = ""
            return result

        if self._inside_tag:
            # Unclosed tag -- discard remaining buffer
            self._buffer = ""
            return ""

        result = self._buffer
        self._buffer = ""
        return result

    def _could_contain_partial_open(self) -> bool:
        """Check if buffer ends with a partial opening tag."""
        lower = self._buffer.lower()
        # Check if buffer ends with '<' or '<t' or '<th' etc.
        for i in range(1, min(len(lower), 20)):
            suffix = lower[-i:]
            if suffix.startswith('<'):
                for name in self.TAG_NAMES:
                    tag = f"<{name}>"
                    if tag.startswith(suffix):
                        return True
        return False

    def _could_contain_partial_close(self) -> bool:
        """Check if buffer might contain start of a closing tag."""
        if not self._tag_name:
            return False
        close_tag = f"</{self._tag_name}>"
        lower = self._buffer.lower()
        for i in range(1, min(len(lower) + 1, len(close_tag))):
            suffix = lower[-i:]
            if close_tag.lower().startswith(suffix):
                return True
        return False

    def _find_safe_output_end(self) -> int:
        """Find the last position in buffer that's safe to output (before any partial tag)."""
        lower = self._buffer.lower()
        # Find last '<' that could be start of a tag
        for i in range(len(lower) - 1, -1, -1):
            if lower[i] == '<':
                return i
        return len(lower)


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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
