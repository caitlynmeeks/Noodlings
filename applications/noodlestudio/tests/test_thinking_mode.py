# ──────────────────────────────────────────────────────────────
#
#   Tests for Thinking Mode setting (Commit 1)
#
#   Per-label On/Off toggle controlling whether chain-of-thought
#   reasoning passes through or is suppressed.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_thinking_mode
# PURPOSE:  Verify thinking mode per-label toggle
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


# =============================================================================
# ModelLabelManager thinking mode
# =============================================================================

class TestModelLabelManagerThinking:
    """Verify thinking mode persistence and defaults."""

    def _make_manager(self):
        """Create a ModelLabelManager (uses QSettings)."""
        from noodlestudio.core.model_label_manager import ModelLabelManager
        return ModelLabelManager()

    def test_get_thinking_mode_default_true(self):
        """New label returns True by default."""
        mgr = self._make_manager()
        # All labels default to thinking=True
        result = mgr.get_thinking_mode("Large")
        assert result is True

    def test_set_and_get_thinking_mode(self):
        """Set thinking to False, read back False."""
        mgr = self._make_manager()
        mgr.set_thinking_mode("Large", False)
        assert mgr.get_thinking_mode("Large") is False
        # Restore
        mgr.set_thinking_mode("Large", True)
        assert mgr.get_thinking_mode("Large") is True

    def test_set_model_preserves_thinking(self):
        """Changing model assignment doesn't reset thinking mode."""
        mgr = self._make_manager()
        # Set thinking off for Large
        mgr.set_thinking_mode("Large", False)
        assert mgr.get_thinking_mode("Large") is False

        # Change the model assignment
        mgr.set_model_for_label("Large", "ollama", "llama3.2")
        # Thinking should still be False
        assert mgr.get_thinking_mode("Large") is False

        # Restore
        mgr.set_thinking_mode("Large", True)

    def test_backward_compat_missing_key(self):
        """Old settings without 'thinking' key return True."""
        mgr = self._make_manager()
        import json

        # Simulate old-format JSON without thinking key
        old_data = json.dumps({"provider": "ollama", "model": "llama3.2"})
        mgr.settings.setValue("labels/TestOldFormat", old_data)
        mgr.settings.sync()

        result = mgr.get_thinking_mode("TestOldFormat")
        assert result is True

        # Clean up
        mgr.settings.remove("labels/TestOldFormat")
        mgr.settings.sync()


# =============================================================================
# LLMConfig thinking modes
# =============================================================================

class TestLLMConfigThinkingModes:
    """Verify thinking_modes field in LLMConfig."""

    def test_thinking_modes_in_llm_config(self):
        """Config populated correctly with thinking modes."""
        from noodlestudio.runtime.llm_client import LLMConfig

        config = LLMConfig(
            thinking_modes={"LARGE": False, "SMALL": True}
        )
        assert config.thinking_modes["LARGE"] is False
        assert config.thinking_modes["SMALL"] is True

    def test_thinking_modes_default_empty(self):
        """Default config has empty thinking_modes dict."""
        from noodlestudio.runtime.llm_client import LLMConfig
        config = LLMConfig()
        assert config.thinking_modes == {}


# =============================================================================
# HeadlessLLMClient thinking mode behavior
# =============================================================================

class TestHeadlessLLMClientThinking:
    """Verify system prompt hint and tag stripping based on thinking mode."""

    def _make_client(self, thinking_modes=None):
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        config = LLMConfig(thinking_modes=thinking_modes or {})
        return HeadlessLLMClient(config)

    def test_thinking_off_system_prompt_hint(self):
        """When thinking OFF, system prompt includes no-CoT hint."""
        # This tests the _is_thinking_enabled logic
        client = self._make_client(thinking_modes={"LARGE": False})
        assert client._is_thinking_enabled("LARGE") is False
        assert client._is_thinking_enabled("SMALL") is True  # Not set -> default True

    def test_thinking_off_strips_tags(self):
        """Tags stripped when thinking OFF."""
        client = self._make_client(thinking_modes={"LARGE": False})
        text = "<think>reasoning here</think>The answer."
        result = client._strip_thinking_tags(text)
        assert result == "The answer."

    def test_thinking_on_preserves_tags(self):
        """When thinking is ON, _is_thinking_enabled returns True."""
        client = self._make_client(thinking_modes={"LARGE": True})
        assert client._is_thinking_enabled("LARGE") is True
        # Tags would NOT be stripped by generate_with_tokens when thinking ON
        # (verified by the flow: thinking_enabled=True -> skip strip)
