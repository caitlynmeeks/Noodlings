"""
Provider Manager - Multi-backend LLM provider system.

Supports:
- Internal (Ollama): Local models, download-based
- Anthropic: Claude API
- OpenAI: GPT API
- OpenRouter: Aggregated API marketplace
- LM Studio: Local OpenAI-compatible API
- Custom: User-defined endpoints

Each provider has:
- Configuration (API keys, endpoints)
- Model discovery (list available models)
- Client abstraction (unified generate interface)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from PyQt6.QtCore import QSettings, QObject, pyqtSignal
import json
import requests


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    id: str  # Unique identifier (e.g., "ollama", "anthropic", "custom_groq")
    name: str  # Display name (e.g., "Internal (Ollama)", "Anthropic")
    type: str  # Provider type: ollama, anthropic, openai, openrouter, lmstudio, custom

    # Optional configuration (depends on type)
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For custom/lmstudio
    port: Optional[int] = None  # For lmstudio

    # Cached data
    available_models: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'ProviderConfig':
        """Deserialize from dict."""
        return ProviderConfig(**data)


class ProviderManager(QObject):
    """
    Manages multiple LLM provider configurations.

    Features:
    - Provider CRUD (add, remove, configure)
    - Model discovery per provider
    - Unified client interface
    - QSettings persistence
    """

    providersChanged = pyqtSignal()  # Emitted when provider list changes

    def __init__(self):
        super().__init__()
        self.settings = QSettings("Noodlings", "ProviderManager")
        self._ensure_default_providers()

    def _ensure_default_providers(self):
        """Ensure default providers exist."""
        defaults = [
            ProviderConfig(
                id="ollama",
                name="Internal (Ollama)",
                type="ollama",
                base_url="http://localhost:11434"
            ),
            ProviderConfig(
                id="anthropic",
                name="Anthropic",
                type="anthropic"
            ),
            ProviderConfig(
                id="openai",
                name="OpenAI",
                type="openai"
            ),
            ProviderConfig(
                id="openrouter",
                name="OpenRouter",
                type="openrouter",
                base_url="https://openrouter.ai/api/v1"
            ),
        ]

        existing_ids = set(self.get_all_provider_ids())

        for provider in defaults:
            if provider.id not in existing_ids:
                self._save_provider(provider)

    def get_all_provider_ids(self) -> List[str]:
        """Get list of all provider IDs."""
        self.settings.beginGroup("providers")
        ids = self.settings.childGroups()
        self.settings.endGroup()
        return ids

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration by ID."""
        self.settings.beginGroup(f"providers/{provider_id}")

        if not self.settings.contains("type"):
            self.settings.endGroup()
            return None

        data = {
            "id": provider_id,
            "name": self.settings.value("name", ""),
            "type": self.settings.value("type", ""),
            "api_key": self.settings.value("api_key", None),
            "base_url": self.settings.value("base_url", None),
            "port": self.settings.value("port", None, type=int),
            "available_models": json.loads(self.settings.value("available_models", "[]"))
        }

        self.settings.endGroup()
        return ProviderConfig.from_dict(data)

    def _save_provider(self, provider: ProviderConfig):
        """Save provider configuration."""
        self.settings.beginGroup(f"providers/{provider.id}")

        self.settings.setValue("name", provider.name)
        self.settings.setValue("type", provider.type)

        if provider.api_key:
            self.settings.setValue("api_key", provider.api_key)
        if provider.base_url:
            self.settings.setValue("base_url", provider.base_url)
        if provider.port:
            self.settings.setValue("port", provider.port)

        self.settings.setValue("available_models", json.dumps(provider.available_models))

        self.settings.endGroup()
        self.settings.sync()

    def add_provider(self, provider: ProviderConfig) -> bool:
        """Add a new provider."""
        if provider.id in self.get_all_provider_ids():
            return False

        self._save_provider(provider)
        self.providersChanged.emit()
        return True

    def update_provider(self, provider: ProviderConfig):
        """Update existing provider configuration."""
        self._save_provider(provider)
        self.providersChanged.emit()

    def delete_provider(self, provider_id: str) -> bool:
        """Delete a provider. Cannot delete default providers."""
        if provider_id in ["ollama", "anthropic", "openai", "openrouter"]:
            return False

        self.settings.beginGroup("providers")
        self.settings.remove(provider_id)
        self.settings.endGroup()
        self.settings.sync()

        self.providersChanged.emit()
        return True

    def fetch_available_models(self, provider_id: str) -> List[str]:
        """
        Fetch available models from provider API.

        Returns list of model names (e.g., ["claude-sonnet-4.5", "gpt-4"]).
        Updates provider's cached available_models.
        """
        provider = self.get_provider(provider_id)
        if not provider:
            return []

        try:
            models = self._fetch_models_by_type(provider)

            # Update cached models
            provider.available_models = models
            self._save_provider(provider)

            return models

        except Exception as e:
            print(f"Error fetching models from {provider_id}: {e}")
            return provider.available_models  # Return cached

    def _fetch_models_by_type(self, provider: ProviderConfig) -> List[str]:
        """Fetch models based on provider type."""

        if provider.type == "ollama":
            return self._fetch_ollama_models(provider)

        elif provider.type == "anthropic":
            return self._fetch_anthropic_models(provider)

        elif provider.type == "openai":
            return self._fetch_openai_models(provider)

        elif provider.type == "openrouter":
            return self._fetch_openrouter_models(provider)

        elif provider.type == "lmstudio":
            return self._fetch_lmstudio_models(provider)

        elif provider.type == "custom":
            return self._fetch_custom_models(provider)

        return []

    def _fetch_ollama_models(self, provider: ProviderConfig) -> List[str]:
        """Fetch models from Ollama."""
        import subprocess
        import os

        try:
            env = os.environ.copy()
            env["OLLAMA_MODELS"] = "/Volumes/DOUBLETROUBLE/models"

            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env
            )

            if result.returncode != 0:
                return []

            models = []
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 1:
                    models.append(parts[0])

            return models

        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    def _fetch_anthropic_models(self, provider: ProviderConfig) -> List[str]:
        """Return known Anthropic models (no discovery API)."""
        return [
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            "claude-sonnet-4",
            "claude-sonnet-3.7",
            "claude-sonnet-3.5",
            "claude-haiku-3.5",
            "claude-opus-3",
            "claude-sonnet-3",
            "claude-haiku-3",
        ]

    def _fetch_openai_models(self, provider: ProviderConfig) -> List[str]:
        """Fetch models from OpenAI API."""
        if not provider.api_key:
            # Return known models if no API key
            return [
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "o1",
                "o1-mini",
                "o3-mini",
            ]

        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {provider.api_key}"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]

            return []

        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            return []

    def _fetch_openrouter_models(self, provider: ProviderConfig) -> List[str]:
        """Fetch models from OpenRouter API."""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]

            return []

        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
            return []

    def _fetch_lmstudio_models(self, provider: ProviderConfig) -> List[str]:
        """Fetch models from LM Studio (OpenAI-compatible)."""
        if not provider.base_url:
            return []

        try:
            url = f"{provider.base_url}/v1/models"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]

            return []

        except Exception as e:
            print(f"Error fetching LM Studio models: {e}")
            return []

    def _fetch_custom_models(self, provider: ProviderConfig) -> List[str]:
        """Fetch models from custom OpenAI-compatible endpoint."""
        return self._fetch_lmstudio_models(provider)


# Global singleton
_provider_manager_instance = None


def get_provider_manager() -> ProviderManager:
    """Get global ProviderManager singleton."""
    global _provider_manager_instance
    if _provider_manager_instance is None:
        _provider_manager_instance = ProviderManager()
    return _provider_manager_instance
