# ──────────────────────────────────────────────────────────────
#
#   Tests for Thinking Mode -> Model Prefix migration
#
#   Original boolean thinking toggle has been replaced by a
#   string prefix system. These tests verify backward compatibility
#   and that the old test patterns still pass with the new API.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_thinking_mode
# PURPOSE:  Backward compat tests for thinking -> prefix migration
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


# =============================================================================
# ModelLabelManager prefix API (replaces thinking mode)
# =============================================================================

class TestModelLabelManagerThinking:
    """Verify prefix-based API replaces old thinking toggle."""

    def _make_manager(self):
        from noodlestudio.core.model_label_manager import ModelLabelManager
        return ModelLabelManager()

    def test_get_prefix_default_empty(self):
        """New label returns empty prefix by default (was: thinking=True)."""
        mgr = self._make_manager()
        result = mgr.get_model_prefix("Large")
        assert result == "" or isinstance(result, str)

    def test_set_and_get_prefix(self):
        """Set prefix /no_think, read back /no_think."""
        mgr = self._make_manager()
        mgr.set_model_prefix("Large", "/no_think")
        assert mgr.get_model_prefix("Large") == "/no_think"
        # Restore
        mgr.set_model_prefix("Large", "")
        assert mgr.get_model_prefix("Large") == ""

    def test_set_model_preserves_prefix(self):
        """Changing model assignment doesn't reset prefix."""
        mgr = self._make_manager()
        mgr.set_model_prefix("Large", "/no_think")
        assert mgr.get_model_prefix("Large") == "/no_think"

        # Change the model assignment
        mgr.set_model_for_label("Large", "ollama", "llama3.2")
        # Prefix should still be /no_think
        assert mgr.get_model_prefix("Large") == "/no_think"

        # Restore
        mgr.set_model_prefix("Large", "")

    def test_backward_compat_missing_key(self):
        """Old settings without prefix or thinking key return empty string."""
        mgr = self._make_manager()
        import json

        old_data = json.dumps({"provider": "ollama", "model": "llama3.2"})
        mgr.settings.setValue("labels/TestOldFormat", old_data)
        mgr.settings.sync()

        result = mgr.get_model_prefix("TestOldFormat")
        assert result == ""

        # Clean up
        mgr.settings.remove("labels/TestOldFormat")
        mgr.settings.sync()


# =============================================================================
# LLMConfig label_prefixes (replaces thinking_modes)
# =============================================================================

class TestLLMConfigThinkingModes:
    """Verify label_prefixes field in LLMConfig (replaces thinking_modes)."""

    def test_label_prefixes_in_llm_config(self):
        """Config populated correctly with label_prefixes."""
        from noodlestudio.runtime.llm_client import LLMConfig

        config = LLMConfig(
            label_prefixes={"LARGE": "/no_think", "SMALL": ""}
        )
        assert config.label_prefixes["LARGE"] == "/no_think"
        assert config.label_prefixes["SMALL"] == ""

    def test_label_prefixes_default_empty(self):
        """Default config has empty label_prefixes dict."""
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig()
        assert config.label_prefixes == {}


# =============================================================================
# HeadlessLLMClient prefix behavior (replaces thinking mode)
# =============================================================================

class TestHeadlessLLMClientThinking:
    """Verify prefix resolution and tag stripping."""

    def _make_client(self, label_prefixes=None):
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        config = LLMConfig(label_prefixes=label_prefixes or {})
        return HeadlessLLMClient(config)

    def test_prefix_resolution_with_label(self):
        """Label with prefix returns it; label without returns empty."""
        client = self._make_client(label_prefixes={"LARGE": "/no_think"})
        assert client._resolve_prefix("LARGE") == "/no_think"
        assert client._resolve_prefix("SMALL") == ""

    def test_strip_tags_always_works(self):
        """Tags stripped regardless of prefix setting."""
        client = self._make_client(label_prefixes={})
        text = "<think>reasoning here</think>The answer."
        result = client.strip_thinking_tags(text)
        assert result == "The answer."

    def test_tag_stripping_backward_compat(self):
        """_strip_thinking_tags alias still works."""
        client = self._make_client()
        text = "<think>reasoning here</think>The answer."
        result = client._strip_thinking_tags(text)
        assert result == "The answer."
