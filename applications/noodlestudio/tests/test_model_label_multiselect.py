# ──────────────────────────────────────────────────────────────
#   Tests for Multi-Label Model Assignment
#
#   Verifies that a single model can be assigned to multiple
#   labels simultaneously (many-to-one), and that the QPushButton
#   + QMenu UI reflects the correct state.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtCore import Qt, QSettings


# =============================================================================
# Data Model Tests -- ModelLabelManager.get_labels_for_model()
# =============================================================================

class TestGetLabelsForModel:
    """Tests for the plural reverse-lookup method."""

    @pytest.fixture(autouse=True)
    def label_manager(self):
        """Fresh ModelLabelManager with isolated QSettings."""
        # Patch QSettings to use a temp scope so tests don't pollute user prefs
        with patch("noodlestudio.core.model_label_manager.QSettings") as MockSettings:
            mock_settings = MagicMock(spec=QSettings)
            MockSettings.return_value = mock_settings

            # Storage backing for the mock
            self._storage = {}

            def mock_value(key, default=None):
                return self._storage.get(key, default)

            def mock_set_value(key, val):
                self._storage[key] = val

            def mock_remove(key):
                self._storage.pop(key, None)

            def mock_child_keys():
                prefix = "labels/"
                return [k[len(prefix):] for k in self._storage if k.startswith(prefix)]

            def mock_begin_group(group):
                pass

            def mock_end_group():
                pass

            mock_settings.value = mock_value
            mock_settings.setValue = mock_set_value
            mock_settings.remove = mock_remove
            mock_settings.childKeys = mock_child_keys
            mock_settings.beginGroup = mock_begin_group
            mock_settings.endGroup = mock_end_group
            mock_settings.sync = MagicMock()

            from noodlestudio.core.model_label_manager import ModelLabelManager
            mgr = ModelLabelManager()
            self.mgr = mgr
            yield mgr

    def test_get_labels_for_model_empty(self):
        """No labels assigned returns empty list."""
        result = self.mgr.get_labels_for_model("anthropic", "claude-opus-4-5-20250220")
        assert result == []

    def test_get_labels_for_model_single(self):
        """Single label returns list with one element."""
        self.mgr.set_model_for_label("Large", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        result = self.mgr.get_labels_for_model("anthropic", "claude-opus-4-5-20250220")
        assert result == ["Large"]

    def test_get_labels_for_model_multiple(self):
        """Two labels on same model returns both."""
        self.mgr.set_model_for_label("Large", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        self.mgr.set_model_for_label("Noodle Code", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        result = self.mgr.get_labels_for_model("anthropic", "claude-opus-4-5-20250220")
        assert set(result) == {"Large", "Noodle Code"}

    def test_get_labels_for_model_excludes_other_models(self):
        """Labels assigned to different models are excluded."""
        self.mgr.set_model_for_label("Large", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        self.mgr.set_model_for_label("Small", "ollama", "deepseek-r1:7b", emit_signal=False)
        self.mgr.set_model_for_label("Noodle Code", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)

        result = self.mgr.get_labels_for_model("anthropic", "claude-opus-4-5-20250220")
        assert set(result) == {"Large", "Noodle Code"}
        assert "Small" not in result

    def test_get_label_for_model_backward_compat(self):
        """Singular get_label_for_model() still returns first match."""
        self.mgr.set_model_for_label("Large", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        self.mgr.set_model_for_label("Noodle Code", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        result = self.mgr.get_label_for_model("anthropic", "claude-opus-4-5-20250220")
        # Should return one of the assigned labels (first found)
        assert result in ("Large", "Noodle Code")

    def test_unassign_one_label_keeps_others(self):
        """Clearing one label leaves the other intact."""
        self.mgr.set_model_for_label("Large", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)
        self.mgr.set_model_for_label("Noodle Code", "anthropic", "claude-opus-4-5-20250220", emit_signal=False)

        # Clear "Large"
        self.mgr.set_model_for_label("Large", None, None, emit_signal=False)

        result = self.mgr.get_labels_for_model("anthropic", "claude-opus-4-5-20250220")
        assert result == ["Noodle Code"]

        # Verify "Large" is now unassigned
        p, m = self.mgr.get_model_for_label("Large")
        assert p is None
        assert m is None


# =============================================================================
# UI Tests -- ModelRow button display
# =============================================================================

class TestModelRowButtonDisplay:
    """Tests for the QPushButton text in ModelRow."""

    @pytest.fixture
    def model_row(self, qapp):
        """Create a ModelRow for testing."""
        from noodlestudio.panels.model_manager_panel_v2 import ModelRow
        row = ModelRow("anthropic", {"id": "claude-opus-4-5-20250220", "name": "Claude Opus 4.5"})
        yield row

    def test_button_shows_none_when_empty(self, model_row):
        """Button shows '(None)' with no labels assigned."""
        model_row.update_labels(["Small", "Large", "Noodle Code"], [])
        assert model_row.label_button.text() == "(None)"

    def test_button_shows_single_label(self, model_row):
        """Button shows the label name when one label is assigned."""
        model_row.update_labels(["Small", "Large", "Noodle Code"], ["Large"])
        assert model_row.label_button.text() == "Large"

    def test_button_shows_two_labels_comma_separated(self, model_row):
        """Button shows comma-separated labels for two assignments."""
        model_row.update_labels(["Small", "Large", "Noodle Code"], ["Large", "Noodle Code"])
        text = model_row.label_button.text()
        assert "Large" in text
        assert "Noodle Code" in text
        assert ", " in text

    def test_button_shows_count_for_three_plus(self, model_row):
        """Button shows count for three or more labels."""
        model_row.update_labels(
            ["Small", "Medium", "Large", "Noodle Code"],
            ["Small", "Large", "Noodle Code"]
        )
        assert model_row.label_button.text() == "3 labels"
        # Tooltip should show full list
        tooltip = model_row.label_button.toolTip()
        assert "Large" in tooltip
        assert "Noodle Code" in tooltip
        assert "Small" in tooltip


# =============================================================================
# UI Tests -- Signal emission
# =============================================================================

class TestModelRowSignals:
    """Tests for labelChanged signal emission on check/uncheck."""

    @pytest.fixture
    def model_row(self, qapp):
        """Create a ModelRow with labels configured."""
        from noodlestudio.panels.model_manager_panel_v2 import ModelRow
        row = ModelRow("anthropic", {"id": "claude-opus-4-5-20250220", "name": "Claude Opus 4.5"})
        row.update_labels(["Small", "Large", "Noodle Code"], ["Large"])
        yield row

    def test_label_toggled_signal_on_check(self, model_row, qtbot):
        """Checking a label emits the label name."""
        with qtbot.waitSignal(model_row.labelChanged, timeout=1000) as blocker:
            model_row._on_label_toggled("Noodle Code", True)
        assert blocker.args == ["anthropic", "claude-opus-4-5-20250220", "Noodle Code"]

    def test_label_toggled_signal_on_uncheck(self, model_row, qtbot):
        """Unchecking a label emits __UNCHECK__ prefix."""
        with qtbot.waitSignal(model_row.labelChanged, timeout=1000) as blocker:
            model_row._on_label_toggled("Large", False)
        assert blocker.args == ["anthropic", "claude-opus-4-5-20250220", "__UNCHECK__Large"]
