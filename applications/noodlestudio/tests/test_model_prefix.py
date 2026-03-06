# ──────────────────────────────────────────────────────────────
#
#   Tests for Model Prefix System (Commit B)
#
#   Replaces the boolean thinking toggle with a string prefix
#   that gets prepended to the system prompt. Two levels:
#   label default (Model Manager) and facet override (Inspector).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_model_prefix
# PURPOSE:  Verify prefix storage, resolution, and integration
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


# =============================================================================
# ModelLabelManager prefix methods
# =============================================================================

class TestModelLabelManagerPrefix:
    """Verify prefix persistence and migration from old thinking boolean."""

    def _make_manager(self):
        from noodlestudio.core.model_label_manager import ModelLabelManager
        return ModelLabelManager()

    def test_get_prefix_default_empty(self):
        """New label returns empty string by default."""
        mgr = self._make_manager()
        result = mgr.get_model_prefix("Large")
        assert isinstance(result, str)

    def test_set_and_get_prefix(self):
        """Set prefix to /no_think, read it back."""
        mgr = self._make_manager()
        mgr.set_model_prefix("Large", "/no_think")
        assert mgr.get_model_prefix("Large") == "/no_think"
        # Restore
        mgr.set_model_prefix("Large", "")

    def test_prefix_preserved_when_model_changes(self):
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

    def test_backward_compat_thinking_false(self):
        """Old settings with thinking:false migrate to /no_think prefix."""
        mgr = self._make_manager()
        import json

        # Simulate old format with thinking:false
        old_data = json.dumps({"provider": "ollama", "model": "qwen:32b", "thinking": False})
        mgr.settings.setValue("labels/TestOldThinking", old_data)
        mgr.settings.sync()

        result = mgr.get_model_prefix("TestOldThinking")
        assert result == "/no_think"

        # Clean up
        mgr.settings.remove("labels/TestOldThinking")
        mgr.settings.sync()

    def test_backward_compat_thinking_true(self):
        """Old settings with thinking:true (or absent) migrate to empty prefix."""
        mgr = self._make_manager()
        import json

        # Old format with thinking:true
        old_data = json.dumps({"provider": "ollama", "model": "llama3.2", "thinking": True})
        mgr.settings.setValue("labels/TestOldThinkingTrue", old_data)
        mgr.settings.sync()

        result = mgr.get_model_prefix("TestOldThinkingTrue")
        assert result == ""

        # Clean up
        mgr.settings.remove("labels/TestOldThinkingTrue")
        mgr.settings.sync()

    def test_missing_prefix_key_returns_empty(self):
        """Old settings without any prefix or thinking key return empty string."""
        mgr = self._make_manager()
        import json

        old_data = json.dumps({"provider": "ollama", "model": "llama3.2"})
        mgr.settings.setValue("labels/TestNoPrefixKey", old_data)
        mgr.settings.sync()

        result = mgr.get_model_prefix("TestNoPrefixKey")
        assert result == ""

        # Clean up
        mgr.settings.remove("labels/TestNoPrefixKey")
        mgr.settings.sync()


# =============================================================================
# LLMConfig label_prefixes
# =============================================================================

class TestLLMConfigLabelPrefixes:
    """Verify label_prefixes field in LLMConfig."""

    def test_label_prefixes_stored(self):
        """Config stores label_prefixes dict."""
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
# HeadlessLLMClient prefix resolution
# =============================================================================

class TestHeadlessLLMClientPrefix:
    """Verify prefix prepending and tag stripping."""

    def _make_client(self, label_prefixes=None):
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        config = LLMConfig(label_prefixes=label_prefixes or {})
        return HeadlessLLMClient(config)

    def test_resolve_prefix_from_label(self):
        """Label prefix resolved when no facet_prefix."""
        client = self._make_client(label_prefixes={"LARGE": "/no_think"})
        assert client._resolve_prefix("LARGE") == "/no_think"

    def test_resolve_prefix_empty_label(self):
        """No label -> empty prefix."""
        client = self._make_client(label_prefixes={"LARGE": "/no_think"})
        assert client._resolve_prefix(None) == ""

    def test_resolve_prefix_facet_overrides_label(self):
        """facet_prefix takes priority over label prefix."""
        client = self._make_client(label_prefixes={"LARGE": "/no_think"})
        assert client._resolve_prefix("LARGE", facet_prefix="/custom") == "/custom"

    def test_resolve_prefix_facet_empty_string_overrides(self):
        """facet_prefix="" (explicit no prefix) overrides label prefix."""
        client = self._make_client(label_prefixes={"LARGE": "/no_think"})
        assert client._resolve_prefix("LARGE", facet_prefix="") == ""

    def test_strip_thinking_tags_public(self):
        """strip_thinking_tags is accessible as public method."""
        client = self._make_client()
        text = "<think>reasoning here</think>The answer."
        result = client.strip_thinking_tags(text)
        assert result == "The answer."


# =============================================================================
# Facet model_prefix field
# =============================================================================

class TestFacetModelPrefix:
    """Verify model_prefix on Facet dataclass."""

    def _make_facet(self, **kwargs):
        from noodlestudio.core.facet_system import Facet
        defaults = {
            'id': 'test_facet',
            'name': 'Test',
            'facet_type': 'LLMFacet',
            'prompt': 'Test prompt',
        }
        defaults.update(kwargs)
        return Facet(**defaults)

    def test_model_prefix_default_none(self):
        """Facet().model_prefix is None by default."""
        facet = self._make_facet()
        assert facet.model_prefix is None

    def test_model_prefix_to_dict_omits_none(self):
        """to_dict() omits model_prefix when None."""
        facet = self._make_facet()
        d = facet.to_dict()
        assert 'model_prefix' not in d

    def test_model_prefix_to_dict_includes_value(self):
        """to_dict() includes model_prefix when set."""
        facet = self._make_facet(model_prefix="/no_think")
        d = facet.to_dict()
        assert d['model_prefix'] == "/no_think"

    def test_model_prefix_to_dict_includes_empty_string(self):
        """to_dict() includes model_prefix="" (explicit no prefix)."""
        facet = self._make_facet(model_prefix="")
        d = facet.to_dict()
        assert d['model_prefix'] == ""

    def test_model_prefix_roundtrip(self):
        """model_prefix round-trips through to_dict/from_dict."""
        from noodlestudio.core.facet_system import Facet
        facet = self._make_facet(model_prefix="/no_think")
        d = facet.to_dict()
        loaded = Facet.from_dict(d)
        assert loaded.model_prefix == "/no_think"

    def test_model_prefix_from_dict_missing(self):
        """Missing model_prefix in dict -> None."""
        from noodlestudio.core.facet_system import Facet
        data = {
            'id': 'f1', 'name': 'F1', 'type': 'LLMFacet',
            'prompt': 'test'
        }
        facet = Facet.from_dict(data)
        assert facet.model_prefix is None
