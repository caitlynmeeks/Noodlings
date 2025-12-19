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

    # Ollama-specific concurrency settings
    num_parallel: Optional[int] = None  # OLLAMA_NUM_PARALLEL (requests per model)
    max_loaded_models: Optional[int] = None  # OLLAMA_MAX_LOADED_MODELS
    max_queue: Optional[int] = None  # OLLAMA_MAX_QUEUE

    # Cached data (full model objects with metadata)
    available_models: List[Dict[str, Any]] = field(default_factory=list)

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
            ProviderConfig(
                id="lmstudio",
                name="LM Studio",
                type="lmstudio",
                base_url="http://localhost:1234",
                port=1234
            ),
            ProviderConfig(
                id="groq",
                name="Groq",
                type="openai",  # OpenAI-compatible API
                base_url="https://api.groq.com/openai/v1"
            ),
            ProviderConfig(
                id="together",
                name="Together AI",
                type="openai",  # OpenAI-compatible API
                base_url="https://api.together.xyz/v1"
            ),
            ProviderConfig(
                id="mistral",
                name="Mistral AI",
                type="openai",  # OpenAI-compatible API
                base_url="https://api.mistral.ai/v1"
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
            # Ollama concurrency settings
            "num_parallel": self.settings.value("num_parallel", None, type=int),
            "max_loaded_models": self.settings.value("max_loaded_models", None, type=int),
            "max_queue": self.settings.value("max_queue", None, type=int),
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

        # Ollama concurrency settings
        if provider.num_parallel is not None:
            self.settings.setValue("num_parallel", provider.num_parallel)
        if provider.max_loaded_models is not None:
            self.settings.setValue("max_loaded_models", provider.max_loaded_models)
        if provider.max_queue is not None:
            self.settings.setValue("max_queue", provider.max_queue)

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
        if provider_id in ["ollama", "anthropic", "openai", "openrouter", "lmstudio", "groq", "together", "mistral"]:
            return False

        self.settings.beginGroup("providers")
        self.settings.remove(provider_id)
        self.settings.endGroup()
        self.settings.sync()

        self.providersChanged.emit()
        return True

    def fetch_available_models(self, provider_id: str) -> List[Dict]:
        """
        Fetch available models from provider API.

        Returns list of model dicts with metadata (id, name, description, context_length, etc.).
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
            # Return cached models, handling old string format
            cached = provider.available_models
            if cached and isinstance(cached[0], str):
                # Convert old string format to dict format
                return [{'id': m, 'name': m, 'description': ''} for m in cached]
            return cached

    def _fetch_models_by_type(self, provider: ProviderConfig) -> List[Dict]:
        """Fetch models based on provider type (returns model dicts with metadata)."""

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

    def _fetch_ollama_models(self, provider: ProviderConfig) -> List[Dict]:
        """Fetch models from Ollama with size and modified date."""
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
            # Ollama output: NAME ID SIZE MODIFIED
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 1:
                    model_dict = {
                        'id': parts[0],
                        'name': parts[0],
                        'size': parts[2] if len(parts) >= 3 else 'Unknown',
                        'modified': ' '.join(parts[3:]) if len(parts) >= 4 else '',
                        'description': f'Local Ollama model',
                        'context_length': 0,  # Ollama doesn't provide this
                        'architecture': {'modality': 'text->text'},
                        'supported_parameters': ['temperature', 'top_p', 'max_tokens'],
                    }
                    models.append(model_dict)

            return models

        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    def _fetch_anthropic_models(self, provider: ProviderConfig) -> List[Dict]:
        """Return known Anthropic models with metadata (no discovery API)."""
        # Known Anthropic models with their specs
        models_data = [
            {"id": "claude-opus-4.5", "name": "Claude Opus 4.5", "context": 200000, "desc": "Most capable model, best for complex tasks"},
            {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "context": 200000, "desc": "Balanced intelligence and speed"},
            {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "context": 200000, "desc": "Strong intelligence, fast responses"},
            {"id": "claude-sonnet-3.7", "name": "Claude Sonnet 3.7", "context": 200000, "desc": "Enhanced version of 3.5"},
            {"id": "claude-sonnet-3.5", "name": "Claude Sonnet 3.5", "context": 200000, "desc": "Excellent balance of capability"},
            {"id": "claude-haiku-3.5", "name": "Claude Haiku 3.5", "context": 200000, "desc": "Fastest model, near-instant responses"},
            {"id": "claude-opus-3", "name": "Claude Opus 3", "context": 200000, "desc": "Previous flagship model"},
            {"id": "claude-sonnet-3", "name": "Claude Sonnet 3", "context": 200000, "desc": "Previous balanced model"},
            {"id": "claude-haiku-3", "name": "Claude Haiku 3", "context": 200000, "desc": "Previous fast model"},
        ]

        return [{
            'id': m['id'],
            'name': m['name'],
            'description': m['desc'],
            'context_length': m['context'],
            'architecture': {'modality': 'text+image->text'},
            'supported_parameters': ['temperature', 'max_tokens', 'top_p', 'top_k', 'tools'],
        } for m in models_data]

    def _fetch_openai_models(self, provider: ProviderConfig) -> List[Dict]:
        """Fetch models from OpenAI API with metadata."""
        # Known OpenAI models with specs (fallback if no API key)
        known_models = [
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "context": 128000, "desc": "Most capable GPT-4 model"},
            {"id": "gpt-4", "name": "GPT-4", "context": 8192, "desc": "Original GPT-4 model"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context": 16385, "desc": "Fast, cost-effective model"},
            {"id": "o1", "name": "O1", "context": 200000, "desc": "Reasoning model with extended thinking"},
            {"id": "o1-mini", "name": "O1 Mini", "context": 128000, "desc": "Faster reasoning model"},
            {"id": "o3-mini", "name": "O3 Mini", "context": 200000, "desc": "Latest reasoning model"},
        ]

        if not provider.api_key:
            # Return known models as dicts
            return [{
                'id': m['id'],
                'name': m['name'],
                'description': m['desc'],
                'context_length': m['context'],
                'architecture': {'modality': 'text->text'},
                'supported_parameters': ['temperature', 'max_tokens', 'top_p', 'tools'],
            } for m in known_models]

        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {provider.api_key}"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Find known metadata for this model
                    known = next((m for m in known_models if m['id'] == model_id), None)
                    model_dict = {
                        'id': model_id,
                        'name': known['name'] if known else model_id,
                        'description': known['desc'] if known else 'OpenAI model',
                        'context_length': known['context'] if known else 0,
                        'architecture': {'modality': 'text->text'},
                        'supported_parameters': ['temperature', 'max_tokens', 'top_p', 'tools'],
                        'created': model.get('created', 0),
                    }
                    models.append(model_dict)
                return models

            return []

        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            return []

    def _fetch_openrouter_models(self, provider: ProviderConfig) -> List[Dict]:
        """Fetch models from OpenRouter API with full metadata."""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("data", []):
                    # Extract and normalize metadata
                    model_dict = {
                        'id': model.get('id', ''),
                        'name': model.get('name', model.get('id', '')),
                        'description': model.get('description', ''),
                        'context_length': model.get('context_length', 0),
                        'pricing': model.get('pricing', {}),
                        'architecture': model.get('architecture', {}),
                        'supported_parameters': model.get('supported_parameters', []),
                        'top_provider': model.get('top_provider', {}),
                        'created': model.get('created', 0),
                    }
                    models.append(model_dict)
                return models

            return []

        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
            return []

    def _fetch_lmstudio_models(self, provider: ProviderConfig) -> List[Dict]:
        """Fetch models from LM Studio (OpenAI-compatible) with metadata."""
        if not provider.base_url:
            return []

        try:
            url = f"{provider.base_url}/v1/models"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("data", []):
                    model_dict = {
                        'id': model.get("id", ""),
                        'name': model.get("id", ""),
                        'description': 'Local LM Studio model',
                        'context_length': 0,  # LM Studio doesn't provide this
                        'architecture': {'modality': 'text->text'},
                        'supported_parameters': ['temperature', 'max_tokens', 'top_p'],
                    }
                    models.append(model_dict)
                return models

            return []

        except Exception as e:
            print(f"Error fetching LM Studio models: {e}")
            return []

    def _fetch_custom_models(self, provider: ProviderConfig) -> List[Dict]:
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
