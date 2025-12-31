"""
Models API - Scriptable interface to model/provider system.

Provides JavaScript-accessible methods for:
- Getting/setting label assignments (SMALL, MEDIUM, LARGE)
- Listing available models from providers
- Configuring providers

Part of the unified Noodlings scripting API (context.noodle.models).

Author: Commander Spock + Cadet Caity
Date: December 10, 2025
"""

from typing import Dict, List, Tuple, Optional, Any


class ModelsAPI:
    """
    Scriptable interface to model/provider configuration.

    Available to JavaScript via context.noodle.models
    """

    def __init__(self, model_label_manager, provider_manager):
        """
        Initialize Models API.

        Args:
            model_label_manager: ModelLabelManager instance
            provider_manager: ProviderManager instance
        """
        self._label_manager = model_label_manager
        self._provider_manager = provider_manager

    def get_label(self, label: str) -> Dict[str, Optional[str]]:
        """
        Get (provider, model) assigned to a label.

        Args:
            label: Label name (e.g., "SMALL", "LARGE")

        Returns:
            Dict with {provider, model} keys (null if unset)

        Example (JavaScript):
            var assignment = context.noodle.models.get_label("SMALL");
            console.log(assignment.provider);  // "ollama"
            console.log(assignment.model);     // "deepseek-r1:7b"
        """
        provider, model = self._label_manager.get_model_for_label(label)
        return {
            'provider': provider,
            'model': model
        }

    def set_label(self, label: str, provider: str, model: str) -> bool:
        """
        Set (provider, model) for a label.

        Args:
            label: Label name (e.g., "MEDIUM")
            provider: Provider ID (e.g., "anthropic", "ollama")
            model: Model name (e.g., "claude-sonnet-4.5")

        Returns:
            True if set successfully

        Example (JavaScript):
            context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        """
        try:
            self._label_manager.set_model_for_label(label, provider, model)
            return True
        except Exception as e:
            return False

    def get_all_labels(self) -> Dict[str, Dict[str, str]]:
        """
        Get all label assignments.

        Returns:
            Dict of {label: {provider, model}} (excludes unassigned)

        Example (JavaScript):
            var labels = context.noodle.models.get_all_labels();
            // {
            //   "SMALL": {provider: "ollama", model: "deepseek-r1:7b"},
            //   "LARGE": {provider: "anthropic", model: "claude-opus-4.5"}
            // }
        """
        result = {}
        mappings = self._label_manager.get_all_mappings()
        for label, (provider, model) in mappings.items():
            result[label] = {
                'provider': provider,
                'model': model
            }
        return result

    def list_available(self, provider: str) -> List[str]:
        """
        List available models from a provider.

        Args:
            provider: Provider ID (e.g., "openrouter", "ollama")

        Returns:
            List of model names

        Example (JavaScript):
            var models = context.noodle.models.list_available("anthropic");
            // ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.0"]
        """
        try:
            config = self._provider_manager.get_provider(provider)
            if config:
                # Refresh models if needed
                self._provider_manager.refresh_models(provider)
                return config.available_models or []
            return []
        except Exception:
            return []

    def list_providers(self) -> List[Dict[str, str]]:
        """
        List all configured providers.

        Returns:
            List of {id, name, type} dicts

        Example (JavaScript):
            var providers = context.noodle.models.list_providers();
            // [
            //   {id: "ollama", name: "Internal (Ollama)", type: "ollama"},
            //   {id: "anthropic", name: "Anthropic", type: "anthropic"}
            // ]
        """
        provider_ids = self._provider_manager.get_all_provider_ids()
        result = []
        for provider_id in provider_ids:
            config = self._provider_manager.get_provider(provider_id)
            if config:
                result.append({
                    'id': config.id,
                    'name': config.name,
                    'type': config.type
                })
        return result

    def configure_provider(self, provider: str, **kwargs) -> bool:
        """
        Configure provider settings.

        Args:
            provider: Provider ID (e.g., "anthropic")
            **kwargs: Configuration options (api_key, base_url, port)

        Returns:
            True if configured successfully

        Example (JavaScript):
            context.noodle.models.configure_provider("anthropic", {
                api_key: "sk-ant-..."
            });
        """
        try:
            config = self._provider_manager.get_provider(provider)
            if not config:
                return False

            # Update config
            if 'api_key' in kwargs:
                config.api_key = kwargs['api_key']
            if 'base_url' in kwargs:
                config.base_url = kwargs['base_url']
            if 'port' in kwargs:
                config.port = kwargs['port']

            self._provider_manager._save_provider(config)
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JavaScript-compatible dict for context injection.

        Returns:
            Dict with method names as keys, values as placeholder strings
        """
        return {
            'get_label': '__models_get_label__',
            'set_label': '__models_set_label__',
            'get_all_labels': '__models_get_all_labels__',
            'list_available': '__models_list_available__',
            'list_providers': '__models_list_providers__',
            'configure_provider': '__models_configure_provider__'
        }
