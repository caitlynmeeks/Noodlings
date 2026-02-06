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
#   Model Label Manager for dynamic LLM tier configuration.
#
#   Provides persistent mapping between label names (SMALL, M...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.model_label_manager
# PURPOSE:  Model Label Manager
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ModelLabelManager, get_model_label_manager()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Optional, Tuple
from PyQt6.QtCore import QSettings, QObject, pyqtSignal
import json


class ModelLabelManager(QObject):
    """
    Manages dynamic label→(provider, model) mappings with persistence.

    Features:
    - Arbitrary number of custom labels
    - Multi-backend support (Ollama, Anthropic, OpenAI, OpenRouter, etc.)
    - Radio-button behavior (one model per label per provider)
    - QSettings persistence
    - Default labels: SMALL, MEDIUM, LARGE

    Usage:
        manager = ModelLabelManager()
        manager.set_model_for_label("SMALL", "ollama", "deepseek-r1:7b")
        provider, model = manager.get_model_for_label("MEDIUM")
        all_labels = manager.get_all_labels()
    """

    # Signal emitted when label mappings change
    mappingsChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.settings = QSettings("Noodlings", "ModelLabelManager")
        self._ensure_defaults()

    # System labels that cannot be deleted
    PROTECTED_LABELS = ["Small", "Medium", "Large", "Noodle Code", "Computer Use"]

    def _ensure_defaults(self):
        """Ensure default labels exist with DeepSeek R1 models on Ollama."""
        defaults = {
            # Text generation tiers
            "Small": ("ollama", "deepseek-r1:7b"),
            "Medium": ("ollama", "deepseek-r1:14b"),
            "Large": ("ollama", "deepseek-r1:70b"),

            # Multimodal labels (unassigned by default - user configures)
            # VISION: Image understanding (Claude Vision, GPT-4V, LLaVA)
            # AUDIO_IN: Speech-to-text (Whisper, Groq Whisper)
            # AUDIO_OUT: Text-to-speech (ElevenLabs, OpenAI TTS, local)
            # IMAGE_GEN: Image generation (Flux, DALL-E, Stable Diffusion)
            # VIDEO_IN: Video understanding (future)
        }

        # Multimodal labels - created unassigned so user can configure
        multimodal_labels = ["VISION", "AUDIO_IN", "AUDIO_OUT", "IMAGE_GEN", "VIDEO_IN"]

        # System labels for NoodleStudio features (unassigned by default)
        # Noodle Code: AI assistant in the Noodle Code panel (needs smart model)
        # Computer Use: Vision+action loop for UI automation (needs computer_use capability)
        system_labels = ["Noodle Code", "Computer Use"]

        # Only set if not already configured
        for label, (provider, model) in defaults.items():
            existing = self.get_model_for_label(label)
            if not existing or not existing[0]:  # No provider set
                self.set_model_for_label(label, provider, model, emit_signal=False)

        # Create multimodal labels (unassigned - user configures in Model Manager)
        for label in multimodal_labels:
            if label not in self.get_all_labels():
                self.create_label(label)

        # Create system labels (unassigned - user configures in Model Manager)
        for label in system_labels:
            if label not in self.get_all_labels():
                self.create_label(label)

    def get_model_for_label(self, label: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get (provider, model) assigned to a label.

        Args:
            label: Label name (e.g., "SMALL", "CUSTOM_REASONING")

        Returns:
            Tuple of (provider_id, model_name) or (None, None) if unset
        """
        value = self.settings.value(f"labels/{label}", None)

        if value is None:
            return (None, None)

        # Try to parse as JSON first (new format)
        try:
            if isinstance(value, str):
                data = json.loads(value)
            else:
                data = value

            provider = data.get("provider")
            model = data.get("model")

            # Treat empty strings as unassigned
            if not provider or not model:
                return (None, None)

            return (provider, model)
        except (json.JSONDecodeError, AttributeError):
            # Not valid JSON - must be legacy format (just model name string)
            if isinstance(value, str):
                return ("ollama", value)
            return (None, None)

    def set_model_for_label(self, label: str, provider_id: Optional[str], model_name: Optional[str], emit_signal: bool = True):
        """
        Set (provider, model) for a label.

        Args:
            label: Label name (e.g., "SMALL")
            provider_id: Provider ID (e.g., "ollama", "anthropic")
            model_name: Model name (e.g., "deepseek-r1:7b", "claude-sonnet-4.5")
            emit_signal: Whether to emit mappingsChanged signal
        """
        # print(f"DEBUG set_model_for_label: label='{label}', provider='{provider_id}', model='{model_name}'")

        # Store unassigned labels with special marker (so they appear in get_all_labels)
        if provider_id is None or model_name is None:
            data = {"provider": "", "model": ""}
            self.settings.setValue(f"labels/{label}", json.dumps(data))
        else:
            # Store as JSON
            data = {"provider": provider_id, "model": model_name}
            json_str = json.dumps(data)
            # print(f"DEBUG: Storing JSON: {json_str}")
            self.settings.setValue(f"labels/{label}", json_str)

        self.settings.sync()

        # Verify what was actually saved
        saved = self.settings.value(f"labels/{label}")
        # print(f"DEBUG: Read back from settings: {saved}")

        if emit_signal:
            self.mappingsChanged.emit()

    def get_label_for_model(self, provider_id: str, model_name: str) -> Optional[str]:
        """
        Get label assigned to a (provider, model) pair (reverse lookup).

        Returns the first label found. For multi-label lookups, use
        get_labels_for_model() instead.

        Args:
            provider_id: Provider ID (e.g., "ollama")
            model_name: Model name (e.g., "deepseek-r1:7b")

        Returns:
            Label name or None if (provider, model) has no label
        """
        for label in self.get_all_labels():
            p, m = self.get_model_for_label(label)
            if p == provider_id and m == model_name:
                return label

        return None

    def get_labels_for_model(self, provider_id: str, model_name: str) -> List[str]:
        """
        Get ALL labels assigned to a (provider, model) pair.

        Multiple labels can point to the same model (many-to-one).
        For example, a single model can be assigned to both "Large"
        and "Noodle Code" simultaneously.

        Args:
            provider_id: Provider ID (e.g., "anthropic")
            model_name: Model name (e.g., "claude-opus-4-5-20250220")

        Returns:
            List of label names (empty if no labels assigned)
        """
        labels = []
        for label in self.get_all_labels():
            p, m = self.get_model_for_label(label)
            if p == provider_id and m == model_name:
                labels.append(label)
        return labels

    def get_all_labels(self) -> List[str]:
        """
        Get all defined label names.

        Returns:
            List of label names (e.g., ["SMALL", "MEDIUM", "LARGE", "REASONING"])
        """
        self.settings.beginGroup("labels")
        labels = self.settings.childKeys()
        self.settings.endGroup()
        return labels

    def create_label(self, label: str) -> bool:
        """
        Create a new custom label (unassigned).

        Args:
            label: Label name (must be uppercase with underscores)

        Returns:
            True if created, False if already exists
        """
        if label in self.get_all_labels():
            return False

        # Create with None values to reserve the label (unassigned)
        self.set_model_for_label(label, None, None)
        return True

    def delete_label(self, label: str) -> bool:
        """
        Delete a custom label. Cannot delete protected/system labels.

        Args:
            label: Label name to delete

        Returns:
            True if deleted, False if protected or doesn't exist
        """
        # Protect system labels
        if label in self.PROTECTED_LABELS:
            return False

        if label not in self.get_all_labels():
            return False

        self.settings.remove(f"labels/{label}")
        self.settings.sync()
        self.mappingsChanged.emit()
        return True

    def is_protected_label(self, label: str) -> bool:
        """Check if a label is protected (cannot be deleted)."""
        return label in self.PROTECTED_LABELS

    def get_noodle_code_model(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the model configured for Noodle Code (AI assistant).

        Returns:
            Tuple of (provider_id, model_name) or (None, None) if not configured.
        """
        return self.get_model_for_label("Noodle Code")

    def get_computer_use_model(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the model configured for Computer Use (vision + action).

        Returns:
            Tuple of (provider_id, model_name) or (None, None) if not configured.
            Should be a model with 'computer_use' capability.
        """
        return self.get_model_for_label("Computer Use")

    def get_all_mappings(self) -> Dict[str, Tuple[str, str]]:
        """
        Get all label→(provider, model) mappings.

        Returns:
            Dict of {label: (provider_id, model_name)} (excludes unassigned labels)
        """
        mappings = {}
        for label in self.get_all_labels():
            provider, model = self.get_model_for_label(label)
            if provider and model:
                mappings[label] = (provider, model)
        return mappings

    def clear_all_custom_labels(self):
        """Clear all custom labels, keeping only Small/Medium/Large."""
        for label in self.get_all_labels():
            if label not in ["Small", "Medium", "Large"]:
                self.delete_label(label)


# Global singleton instance
_manager_instance = None


def get_model_label_manager() -> ModelLabelManager:
    """Get global ModelLabelManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelLabelManager()
    return _manager_instance

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
