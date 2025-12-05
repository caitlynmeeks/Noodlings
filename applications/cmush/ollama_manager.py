"""
Ollama LLM Manager with full observability and model lifecycle control.

Replaces LM Studio black box with instrumented Ollama client:
- Auto-loads models on first use
- Tracks per-model statistics (calls, tokens, timing, errors)
- Provides real-time status dashboard
- Graceful error handling and reconnection

Architecture:
    OllamaManager
        ├→ Model tier system (SMALL, MEDIUM, LARGE)
        ├→ Automatic model loading/pulling
        ├→ Per-model usage statistics
        └→ Real-time status API
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import ollama
from ollama import AsyncClient

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Per-model usage statistics."""

    model_name: str
    total_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_duration_seconds: float = 0.0
    last_call_time: Optional[datetime] = None
    errors: int = 0
    last_error: Optional[str] = None
    is_loaded: bool = False
    load_time: Optional[datetime] = None


@dataclass
class OllamaConfig:
    """Configuration for Ollama models and preferences."""

    # Model tiers - can be configured in advanced settings
    small_model: str = "qwen2.5:3b"
    medium_model: str = "qwen2.5:14b"
    large_model: str = "qwen2.5:32b"

    # Ollama server settings
    host: str = "http://localhost:11434"
    models_directory: str = "/Volumes/DOUBLETROUBLE/models"

    # Timeouts
    default_timeout: int = 120  # 2 minutes for large models
    load_timeout: int = 300  # 5 minutes to load/pull

    def get_model_for_tier(self, tier: str) -> str:
        """Get model name for a tier (SMALL, MEDIUM, LARGE)."""
        tier_map = {
            "SMALL": self.small_model,
            "MEDIUM": self.medium_model,
            "LARGE": self.large_model,
            "$$": self.large_model,  # Alias for LARGE
        }
        return tier_map.get(tier.upper(), self.medium_model)


class OllamaManager:
    """
    Manages Ollama LLM lifecycle with full observability.

    Features:
    - Auto-loads models on first use
    - Tracks detailed per-model statistics
    - Provides real-time status API
    - Graceful error handling

    Usage:
        manager = OllamaManager(config)
        await manager.initialize()

        response = await manager.generate(
            prompt="Hello",
            model_tier="MEDIUM"
        )

        status = await manager.get_status()
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client: Optional[AsyncClient] = None
        self.stats: Dict[str, ModelStats] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        self._ollama_process: Optional[subprocess.Popen] = None

    async def _ensure_ollama_server_running(self) -> bool:
        """
        Ensure Ollama server is running, starting it if necessary.

        Returns:
            True if server is running or was successfully started
        """
        try:
            # First, try to connect to see if it's already running
            self.client = AsyncClient(host=self.config.host)
            await self.client.list()
            logger.info("Ollama server already running")
            return True

        except Exception:
            # Server not running, try to start it
            logger.info("Ollama server not detected, starting...")

            try:
                # Set OLLAMA_MODELS environment variable to use local models directory
                env = os.environ.copy()
                env['OLLAMA_MODELS'] = self.config.models_directory

                # Start ollama serve in background with custom models directory
                self._ollama_process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    start_new_session=True  # Detach from parent process
                )

                logger.info(f"Started Ollama server (PID: {self._ollama_process.pid})")
                logger.info(f"Using models directory: {self.config.models_directory}")

                # Wait for server to be ready (up to 10 seconds)
                for attempt in range(20):
                    await asyncio.sleep(0.5)
                    try:
                        self.client = AsyncClient(host=self.config.host)
                        await self.client.list()
                        logger.info("Ollama server ready")
                        return True
                    except:
                        continue

                logger.error("Ollama server failed to become ready after 10 seconds")
                return False

            except FileNotFoundError:
                logger.error("ollama command not found. Please install Ollama: https://ollama.com/download")
                return False
            except Exception as e:
                logger.error(f"Failed to start Ollama server: {e}")
                return False

    async def initialize(self) -> bool:
        """
        Initialize Ollama client and verify connection.
        Automatically starts Ollama server if not running.

        Returns:
            True if initialization successful
        """
        # Ensure server is running
        if not await self._ensure_ollama_server_running():
            return False

        try:
            # Re-test connection to get model count
            self.client = AsyncClient(host=self.config.host)
            models = await self.client.list()
            logger.info(f"Ollama connected: {len(models.get('models', []))} models available")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            return False

    def _get_stats(self, model_name: str) -> ModelStats:
        """Get or create stats for a model."""
        if model_name not in self.stats:
            self.stats[model_name] = ModelStats(model_name=model_name)
        return self.stats[model_name]

    async def ensure_model_loaded(self, model_name: str) -> bool:
        """
        Ensure model is loaded, pulling from Ollama registry if necessary.

        Args:
            model_name: Name of model to load (e.g., "qwen2.5:3b")

        Returns:
            True if model is ready to use
        """
        async with self._lock:
            stats = self._get_stats(model_name)

            try:
                # Check if model exists locally
                models_response = await self.client.list()

                # Extract model names - handle different response formats
                models_list = models_response.get('models', [])
                available_models = []
                for m in models_list:
                    if isinstance(m, dict):
                        # Try different possible keys
                        name = m.get('name') or m.get('model') or m.get('id')
                        if name:
                            available_models.append(name)
                    elif isinstance(m, str):
                        available_models.append(m)

                logger.info(f"Checking for model {model_name} in: {available_models}")

                if model_name not in available_models:
                    logger.info(f"Model {model_name} not found locally, pulling from Ollama registry...")
                    logger.info(f"This may take several minutes depending on model size.")

                    # Pull model with progress tracking
                    stream = await self.client.pull(model_name, stream=True)
                    last_status = None
                    async for chunk in stream:
                        status = chunk.get('status', '')
                        # Only log when status changes to avoid spam
                        if status != last_status:
                            logger.info(f"  {status}")
                            last_status = status

                    logger.info(f"Model {model_name} downloaded successfully")

                # Mark as loaded
                stats.is_loaded = True
                stats.load_time = datetime.now()
                logger.info(f"Model {model_name} ready")
                return True

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                stats.errors += 1
                stats.last_error = str(e)
                return False

    async def generate(
        self,
        prompt: str,
        model_tier: str = "MEDIUM",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using specified model tier or model name.

        Args:
            prompt: Input prompt
            model_tier: Model tier (SMALL, MEDIUM, LARGE)
            model_name: Override with specific model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt (optional)
            **kwargs: Additional ollama.generate parameters

        Returns:
            {
                'text': str,  # Generated text
                'model': str,  # Model used
                'tokens': int,  # Total tokens
                'duration_seconds': float,  # Generation time
                'error': str | None,  # Error message if failed
            }
        """
        if not self._initialized:
            raise RuntimeError("OllamaManager not initialized. Call await manager.initialize()")

        # Resolve model name from tier
        if model_name is None:
            model_name = self.config.get_model_for_tier(model_tier)

        stats = self._get_stats(model_name)
        start_time = time.time()

        try:
            # Ensure model is loaded
            if not stats.is_loaded:
                loaded = await self.ensure_model_loaded(model_name)
                if not loaded:
                    return {
                        'text': '',
                        'model': model_name,
                        'tokens': 0,
                        'duration_seconds': 0.0,
                        'error': f"Failed to load model {model_name}",
                    }

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Generate
            logger.info(f"🧠 Generating with {model_name} (tier={model_tier}, prompt_len={len(prompt)})")

            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            # Filter out 'model' from kwargs to avoid duplicate argument
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'model'}

            response = await self.client.chat(
                model=model_name,
                messages=messages,
                options=options,
                **filtered_kwargs
            )

            # Extract response
            generated_text = response['message']['content']
            duration = time.time() - start_time

            # Update statistics
            stats.total_calls += 1
            stats.last_call_time = datetime.now()
            stats.total_duration_seconds += duration

            # Extract token counts if available
            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('eval_count', 0)
            total_tokens = prompt_tokens + completion_tokens

            stats.total_prompt_tokens += prompt_tokens
            stats.total_completion_tokens += completion_tokens
            stats.total_tokens += total_tokens

            logger.info(
                f"✓ Generated {completion_tokens} tokens in {duration:.2f}s "
                f"({completion_tokens/duration:.1f} tok/s)"
            )

            return {
                'text': generated_text,
                'model': model_name,
                'tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'duration_seconds': duration,
                'error': None,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ Generation failed with {model_name}: {e}")

            stats.errors += 1
            stats.last_error = str(e)

            return {
                'text': '',
                'model': model_name,
                'tokens': 0,
                'duration_seconds': duration,
                'error': str(e),
            }

    async def get_status(self) -> Dict[str, Any]:
        """
        Get real-time status of all models.

        Returns:
            {
                'connected': bool,
                'host': str,
                'models': [
                    {
                        'name': str,
                        'tier': str | None,  # SMALL/MEDIUM/LARGE if configured
                        'is_loaded': bool,
                        'total_calls': int,
                        'total_tokens': int,
                        'avg_duration_seconds': float,
                        'errors': int,
                        'last_call_time': str | None,
                    },
                    ...
                ],
                'tiers': {
                    'SMALL': str,
                    'MEDIUM': str,
                    'LARGE': str,
                }
            }
        """
        status = {
            'connected': self._initialized,
            'host': self.config.host,
            'models': [],
            'tiers': {
                'SMALL': self.config.small_model,
                'MEDIUM': self.config.medium_model,
                'LARGE': self.config.large_model,
            }
        }

        # Add per-model statistics
        for model_name, stats in self.stats.items():
            # Determine tier if this model is configured
            tier = None
            if model_name == self.config.small_model:
                tier = 'SMALL'
            elif model_name == self.config.medium_model:
                tier = 'MEDIUM'
            elif model_name == self.config.large_model:
                tier = 'LARGE'

            avg_duration = (
                stats.total_duration_seconds / stats.total_calls
                if stats.total_calls > 0 else 0.0
            )

            status['models'].append({
                'name': model_name,
                'tier': tier,
                'is_loaded': stats.is_loaded,
                'total_calls': stats.total_calls,
                'total_tokens': stats.total_tokens,
                'total_prompt_tokens': stats.total_prompt_tokens,
                'total_completion_tokens': stats.total_completion_tokens,
                'avg_duration_seconds': avg_duration,
                'errors': stats.errors,
                'last_error': stats.last_error,
                'last_call_time': stats.last_call_time.isoformat() if stats.last_call_time else None,
            })

        return status

    async def shutdown(self):
        """Clean shutdown. Stops Ollama server if we started it."""
        logger.info("Shutting down OllamaManager")
        self._initialized = False
        self.client = None

        # Stop Ollama server if we started it
        if self._ollama_process:
            try:
                logger.info(f"Stopping Ollama server (PID: {self._ollama_process.pid})")
                self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
                logger.info("Ollama server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Ollama server did not stop gracefully, killing...")
                self._ollama_process.kill()
            except Exception as e:
                logger.error(f"Error stopping Ollama server: {e}")
            finally:
                self._ollama_process = None

    async def close(self):
        """Alias for shutdown (OpenAICompatibleLLM interface compatibility)."""
        await self.shutdown()

    # ===== OpenAICompatibleLLM Interface Compatibility =====
    # These methods provide drop-in compatibility with the existing LLM interface

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.shutdown()

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Complete a prompt (compatibility method for OpenAICompatibleLLM interface).

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            model: Model name (falls back to MEDIUM tier if not specified)
            **kwargs: Additional parameters

        Returns:
            (response_text, usage_dict, model_used)
        """
        # If no model specified, use MEDIUM tier
        if model is None:
            model = self.config.medium_model

        result = await self.generate(
            prompt=user_prompt,
            model_name=model,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs
        )

        if result['error']:
            logger.error(f"Ollama generation failed: {result['error']}")
            return ("", {}, model)

        usage = {
            'prompt_tokens': result.get('prompt_tokens', 0),
            'completion_tokens': result.get('completion_tokens', 0),
            'total_tokens': result.get('tokens', 0)
        }

        return (result['text'], usage, model)

    async def text_to_affect(
        self,
        text: str,
        context: Optional[list] = None,
        agent_id: Optional[str] = None
    ) -> list:
        """
        Convert text to 5-D affect vector (compatibility method).

        Returns:
            [valence, arousal, dominance, sorrow, boredom]
        """
        system_prompt = """You are an emotion analysis expert. Analyze the emotional affect of text and return ONLY a JSON object with these exact keys:
{
  "valence": <number from -1.0 to 1.0>,
  "arousal": <number from 0.0 to 1.0>,
  "dominance": <number from 0.0 to 1.0>,
  "sorrow": <number from 0.0 to 1.0>,
  "boredom": <number from 0.0 to 1.0>
}

Where:
- valence: negative (-1) to positive (+1)
- arousal: calm (0) to excited (1)
- dominance: submissive (0) to dominant/confident (1)
- sorrow: content (0) to sad (1)
- boredom: engaged (0) to bored (1)

Return ONLY the JSON, no other text."""

        user_prompt = f'Text: "{text}"'
        if context:
            user_prompt += f"\n\nRecent context:\n" + "\n".join(context[-3:])

        result = await self.generate(
            prompt=user_prompt,
            model_tier="SMALL",  # Use fast model for affect
            system_prompt=system_prompt,
            temperature=0.3  # Lower temp for more consistent affect extraction
        )

        if result['error']:
            logger.warning(f"Affect extraction failed, using neutral: {result['error']}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            # Parse JSON from response
            text_response = result['text'].strip()

            # Remove markdown code blocks if present
            if '```json' in text_response:
                text_response = text_response.split('```json')[1].split('```')[0]
            elif '```' in text_response:
                text_response = text_response.split('```')[1].split('```')[0]

            # Find JSON object
            start_idx = text_response.find('{')
            end_idx = text_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                text_response = text_response[start_idx:end_idx]

            affect_dict = json.loads(text_response)

            return [
                float(affect_dict.get('valence', 0.0)),
                float(affect_dict.get('arousal', 0.0)),
                float(affect_dict.get('dominance', 0.0)),
                float(affect_dict.get('sorrow', 0.0)),
                float(affect_dict.get('boredom', 0.0))
            ]

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse affect JSON, using neutral: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    async def generate_response(
        self,
        phenomenal_state: dict,
        target_user: str,
        conversation_context: list,
        agent_name: str,
        species: str = "unknown",
        model: Optional[str] = None,
        max_tokens: int = 180,
        **kwargs
    ) -> dict:
        """
        Generate agent response from phenomenal state (compatibility method).

        Returns:
            {
                'response': str,
                'model_used': str,
                'usage': dict
            }
        """
        # Build prompt from phenomenal state
        affect = phenomenal_state.get('affect', [0]*5)
        valence, arousal, dominance, sorrow, boredom = affect[:5]

        # Format conversation context
        context_str = "\n".join([
            f"{c.get('speaker', 'Unknown')}: {c.get('text', '')}"
            for c in conversation_context[-5:]
        ])

        system_prompt = f"""You are {agent_name}, a {species}.
Respond naturally and in-character based on your current emotional state.

Current affect:
- Valence: {valence:.2f} (-1=negative, +1=positive)
- Arousal: {arousal:.2f} (0=calm, 1=excited)
- Dominance: {dominance:.2f} (0=submissive, 1=confident)
- Sorrow: {sorrow:.2f}
- Boredom: {boredom:.2f}

Keep responses brief and natural. Speak as {agent_name} would."""

        user_prompt = f"""Recent conversation:
{context_str}

Respond to {target_user} based on your current emotional state."""

        result = await self.generate(
            prompt=user_prompt,
            model_name=model,
            model_tier="MEDIUM",  # Use medium model for responses
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.8,
            **kwargs
        )

        return {
            'response': result['text'],
            'model_used': result['model'],
            'usage': {
                'prompt_tokens': result.get('prompt_tokens', 0),
                'completion_tokens': result.get('completion_tokens', 0),
                'total_tokens': result.get('tokens', 0)
            }
        }

    async def generate_rumination(
        self,
        phenomenal_state: dict,
        conversation_context: list,
        agent_name: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate internal thought/rumination (compatibility method)."""
        affect = phenomenal_state.get('affect', [0]*5)
        valence, arousal = affect[0], affect[1]

        context_str = "\n".join([
            f"{c.get('speaker', 'Unknown')}: {c.get('text', '')}"
            for c in conversation_context[-3:]
        ])

        system_prompt = f"""You are {agent_name}'s internal voice. Generate a brief internal thought.
Current mood: valence={valence:.2f}, arousal={arousal:.2f}
Be concise and introspective."""

        user_prompt = f"Recent events:\n{context_str}\n\nWhat are you thinking?"

        result = await self.generate(
            prompt=user_prompt,
            model_name=model,
            model_tier="SMALL",
            system_prompt=system_prompt,
            max_tokens=60,
            temperature=0.7,
            **kwargs
        )

        return result['text']

    async def self_reflection(
        self,
        phenomenal_state: dict,
        conversation_context: list,
        agent_name: str,
        model: Optional[str] = None,
        **kwargs
    ) -> dict:
        """Self-reflection for withdrawal decision (compatibility method)."""
        system_prompt = f"You are {agent_name}. Briefly evaluate if you should withdraw from this conversation."

        context_str = "\n".join([
            f"{c.get('speaker', 'Unknown')}: {c.get('text', '')}"
            for c in conversation_context[-3:]
        ])

        user_prompt = f"Recent conversation:\n{context_str}\n\nShould you withdraw? Answer YES or NO and explain briefly."

        result = await self.generate(
            prompt=user_prompt,
            model_name=model,
            model_tier="SMALL",
            system_prompt=system_prompt,
            max_tokens=50,
            temperature=0.5,
            **kwargs
        )

        # Parse YES/NO
        response_text = result['text'].upper()
        should_withdraw = 'YES' in response_text[:20]

        return {
            'withdraw': should_withdraw,
            'reason': result['text']
        }

    async def detect_toxicity(self, text: str, **kwargs) -> dict:
        """Detect toxicity in text (compatibility method)."""
        system_prompt = "You are a content safety analyzer. Rate toxicity from 0.0 (safe) to 1.0 (toxic). Return only a number."

        user_prompt = f'Rate toxicity (0.0-1.0): "{text}"'

        result = await self.generate(
            prompt=user_prompt,
            model_tier="SMALL",
            system_prompt=system_prompt,
            max_tokens=10,
            temperature=0.1,
            **kwargs
        )

        try:
            # Extract number from response
            import re
            numbers = re.findall(r'0?\.\d+|[01]\.0', result['text'])
            score = float(numbers[0]) if numbers else 0.0
        except:
            score = 0.0

        return {
            'score': score,
            'toxic': score > 0.7
        }

    async def generate_with_tokens(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400
    ) -> tuple:
        """
        Generate with token count tracking (for facet execution).

        Model parameter can be either:
        - Tier name: "SMALL", "MEDIUM", "LARGE"
        - Actual model name: "qwen2.5:3b"

        Returns:
            Tuple of (generated_text, token_count)
        """
        # Check if model is a tier name or actual model
        if model and model.upper() in ['SMALL', 'MEDIUM', 'LARGE', '$$']:
            # It's a tier name
            result = await self.generate(
                prompt=prompt,
                model_tier=model.upper(),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # It's an actual model name (or None)
            result = await self.generate(
                prompt=prompt,
                model_name=model,
                model_tier="MEDIUM" if model is None else None,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        return (result['text'], result.get('tokens', 0))
