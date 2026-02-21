# ------------------------------------------------------------------
#   LMStudio / Custom Provider Tests
#
#   Verifies that LMStudio and custom OpenAI-compatible providers
#   work correctly in NoodleCode dispatch, HeadlessLLMClient URL
#   construction, and Settings UI.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_lmstudio_provider
# PURPOSE:  LMStudio Provider Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# =====================================================================
# LLMConfig Defaults
# =====================================================================

class TestLLMConfigDefaults:
    """Verify LLMConfig sets correct default base_url per provider."""

    def test_lmstudio_default_base_url(self):
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig(provider="lmstudio")
        assert config.base_url == "http://localhost:1234/v1"

    def test_custom_default_base_url(self):
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig(provider="custom")
        assert config.base_url == "http://localhost:8080/v1"

    def test_lmstudio_explicit_base_url_not_overridden(self):
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig(
            provider="lmstudio",
            base_url="http://100.85.191.79:1234/v1"
        )
        assert config.base_url == "http://100.85.191.79:1234/v1"

    def test_ollama_default_unchanged(self):
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig(provider="ollama")
        assert config.base_url == "http://localhost:11434/v1"

    def test_openai_default_unchanged(self):
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig(provider="openai")
        assert config.base_url == "https://api.openai.com/v1"


# =====================================================================
# URL Normalization
# =====================================================================

class TestURLNormalization:
    """Verify URL construction appends /v1 only when needed."""

    def _build_url(self, base_url: str) -> str:
        """Replicate the URL normalization logic from llm_client."""
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        else:
            return f"{base}/v1/chat/completions"

    def test_url_with_v1_suffix(self):
        url = self._build_url("http://localhost:1234/v1")
        assert url == "http://localhost:1234/v1/chat/completions"

    def test_url_without_v1_suffix(self):
        url = self._build_url("http://localhost:1234")
        assert url == "http://localhost:1234/v1/chat/completions"

    def test_url_with_trailing_slash(self):
        url = self._build_url("http://localhost:1234/v1/")
        assert url == "http://localhost:1234/v1/chat/completions"

    def test_url_remote_with_v1(self):
        url = self._build_url("http://100.85.191.79:1234/v1")
        assert url == "http://100.85.191.79:1234/v1/chat/completions"

    def test_url_remote_without_v1(self):
        url = self._build_url("http://100.85.191.79:1234")
        assert url == "http://100.85.191.79:1234/v1/chat/completions"

    def test_noodlerouter_url_preserved(self):
        url = self._build_url("https://api.noodlings.ai/v1")
        assert url == "https://api.noodlings.ai/v1/chat/completions"

    def test_openrouter_url_preserved(self):
        url = self._build_url("https://openrouter.ai/api/v1")
        assert url == "https://openrouter.ai/api/v1/chat/completions"


# =====================================================================
# NoodleCode Provider Dispatch
# =====================================================================

class TestNoodleCodeProviderDispatch:
    """Verify _call_llm doesn't reject lmstudio/custom providers."""

    def _make_engine(self):
        """Create a real NoodleCodeEngine with no external dependencies."""
        from noodlestudio.core.noodle_code_engine import NoodleCodeEngine
        return NoodleCodeEngine()

    def test_lmstudio_not_unsupported(self):
        """lmstudio provider should route to openai-compatible, not error."""
        engine = self._make_engine()

        async def check():
            chunks = []
            try:
                async for chunk in engine._call_llm(
                    provider_type="lmstudio",
                    model_id="test-model",
                    base_url="http://localhost:1234/v1",
                    api_key="",
                    system_prompt_override="test",
                ):
                    chunks.append(chunk)
                    if len(chunks) > 2:
                        break
            except Exception:
                # Connection errors are expected (no LMStudio server running)
                pass
            return chunks

        chunks = asyncio.run(check())
        for chunk in chunks:
            assert "Unsupported provider type" not in (chunk.content or "")

    def test_custom_not_unsupported(self):
        """custom provider should route to openai-compatible, not error."""
        engine = self._make_engine()

        async def check():
            chunks = []
            try:
                async for chunk in engine._call_llm(
                    provider_type="custom",
                    model_id="test-model",
                    base_url="http://localhost:8080/v1",
                    api_key="",
                    system_prompt_override="test",
                ):
                    chunks.append(chunk)
                    if len(chunks) > 2:
                        break
            except Exception:
                pass
            return chunks

        chunks = asyncio.run(check())
        for chunk in chunks:
            assert "Unsupported provider type" not in (chunk.content or "")

    def test_bogus_provider_yields_error(self):
        """An unrecognized provider should yield an error chunk."""
        engine = self._make_engine()

        async def check():
            chunks = []
            async for chunk in engine._call_llm(
                provider_type="nonexistent_provider",
                model_id="test-model",
                base_url="http://localhost:9999",
                api_key="",
                system_prompt_override="test",
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(check())
        assert len(chunks) == 1
        assert chunks[0].type == "error"
        assert "Unsupported provider type" in chunks[0].content


# =====================================================================
# Settings UI: Port Field Removed
# =====================================================================

class TestSettingsUIPortField:
    """Verify LMStudio config dialog no longer has a Port field."""

    def test_lmstudio_dialog_has_no_port_input(self, qapp):
        from noodlestudio.core.provider_manager import ProviderConfig
        from noodlestudio.panels.model_manager_panel_v2 import ProviderConfigDialog

        config = ProviderConfig(
            id="lmstudio",
            name="LM Studio",
            type="lmstudio",
            base_url="http://localhost:1234",
        )
        dialog = ProviderConfigDialog(config)
        assert not hasattr(dialog, 'port_input'), \
            "Port field should be removed from LMStudio config dialog"
        dialog.close()
        dialog.deleteLater()

    def test_lmstudio_dialog_has_base_url_input(self, qapp):
        from noodlestudio.core.provider_manager import ProviderConfig
        from noodlestudio.panels.model_manager_panel_v2 import ProviderConfigDialog

        config = ProviderConfig(
            id="lmstudio",
            name="LM Studio",
            type="lmstudio",
            base_url="http://localhost:1234",
        )
        dialog = ProviderConfigDialog(config)
        assert hasattr(dialog, 'base_url_input'), \
            "Base URL field should still exist"
        dialog.close()
        dialog.deleteLater()
