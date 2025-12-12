# Multi-Provider LLM Execution Plan

**Status:** Discovery/UI complete ✅ | Execution layer missing ❌
**Priority:** HIGH - Users can configure but not use external providers
**Effort:** 2-3 days

---

## Current State Analysis

### What Works ✅

**1. Model Discovery:**
- `ProviderManager` can list models from all providers
- Anthropic: claude-opus-4.5, claude-sonnet-4.5, etc.
- OpenAI: gpt-4, gpt-3.5-turbo, etc.
- OpenRouter: 200+ models
- LM Studio: Local discovery
- Ollama: Local discovery

**2. Label Configuration:**
- `ModelLabelManager` stores (provider, model) tuples
- Settings UI shows provider info
- Inspector shows "Using: claude-sonnet-4.5 (Anthropic)"

**3. Ollama Execution:**
- `OpenAICompatibleLLM` class in llm_interface.py
- Works with Ollama's OpenAI-compatible API
- Used by all facets currently

### What's Missing ❌

**Execution layer for:**
- ❌ Anthropic API (different format than OpenAI)
- ❌ OpenAI API (needs real API key handling)
- ❌ OpenRouter API (different auth)
- ❌ Provider routing (which client to use?)

---

## Architecture Design

### Current Call Flow (Ollama Only)

```python
LLMFacet.execute()
    ↓
cognitive_components.py: _call_llm_tracked()
    ↓
llm_interface.py: OpenAICompatibleLLM.generate()
    ↓
POST http://localhost:11434/v1/chat/completions
    ↓
Ollama returns response
```

### Desired Call Flow (Multi-Provider)

```python
LLMFacet.execute(model_label="LARGE")
    ↓
ModelLabelManager.get_label("LARGE")
    → (provider="anthropic", model="claude-sonnet-4.5")
    ↓
LLMClientRouter.get_client(provider)
    → AnthropicClient | OpenAIClient | OpenRouterClient | OllamaClient
    ↓
Client.generate(prompt, model)
    ↓
Provider-specific API call
    ↓
Standardized response format
```

---

## Implementation Plan

### Phase 1: Unified LLM Client Interface (~200 lines)

```python
# applications/cmush/llm_client_router.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMClient(ABC):
    """Abstract base for all LLM providers"""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate completion from messages"""
        pass

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str = None
    ) -> str:
        """Simple completion (system + user prompt)"""
        pass

@dataclass
class LLMResponse:
    """Standardized response format"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]  # input_tokens, output_tokens
    finish_reason: str
    latency_ms: float


class LLMClientRouter:
    """Routes requests to appropriate provider client"""

    def __init__(self, provider_manager, model_label_manager):
        self.provider_manager = provider_manager
        self.label_manager = model_label_manager
        self.clients = {}  # provider_id → LLMClient

    def get_client(self, model_label: str) -> LLMClient:
        """Get client for model label (e.g., 'LARGE')"""

        # Look up provider + model
        provider_id, model_name = self.label_manager.get_label(model_label)

        # Get or create client for this provider
        if provider_id not in self.clients:
            self.clients[provider_id] = self._create_client(provider_id)

        return self.clients[provider_id]

    def _create_client(self, provider_id: str) -> LLMClient:
        """Factory method for provider clients"""

        config = self.provider_manager.get_provider_config(provider_id)

        if provider_id == 'ollama':
            return OllamaClient(config)
        elif provider_id == 'anthropic':
            return AnthropicClient(config)
        elif provider_id == 'openai':
            return OpenAIClient(config)
        elif provider_id == 'openrouter':
            return OpenRouterClient(config)
        elif provider_id == 'lmstudio':
            return LMStudioClient(config)
        else:
            raise ValueError(f"Unknown provider: {provider_id}")
```

---

### Phase 2: Provider-Specific Clients

#### A. Anthropic Client (~300 lines)

```python
# applications/cmush/providers/anthropic_client.py

import anthropic
from llm_client_router import LLMClient, LLMResponse

class AnthropicClient(LLMClient):
    """Claude API client"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Call Claude API

        Format differences from OpenAI:
        - Uses 'anthropic' package (not requests)
        - Messages format: [{"role": "user", "content": "..."}]
        - System prompt is SEPARATE parameter
        - Returns different JSON structure
        """

        # Extract system message if present
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                user_messages.append(msg)

        # Call Claude API
        start_time = time.time()

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=user_messages
        )

        latency_ms = (time.time() - start_time) * 1000

        # Convert to standardized format
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider='anthropic',
            usage={
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            latency_ms=latency_ms
        )

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.7, model: str = None) -> str:
        """Simple completion wrapper"""

        messages = [
            {'role': 'user', 'content': user_prompt}
        ]

        response = await self.generate(
            messages=messages,
            model=model or 'claude-sonnet-4.5',
            temperature=temperature,
            max_tokens=1000
        )

        return response.content
```

#### B. OpenAI Client (~250 lines)

```python
# applications/cmush/providers/openai_client.py

from openai import AsyncOpenAI
from llm_client_router import LLMClient, LLMResponse

class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Call OpenAI API

        Format similar to our current OpenAICompatibleLLM but with:
        - Real API key authentication
        - Different base URL (api.openai.com)
        - Billing/rate limits to handle
        """

        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider='openai',
            usage={
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency_ms
        )
```

#### C. OpenRouter Client (~200 lines)

```python
# applications/cmush/providers/openrouter_client.py

import aiohttp
from llm_client_router import LLMClient, LLMResponse

class OpenRouterClient(LLMClient):
    """OpenRouter API client (aggregates 200+ models)"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.base_url = "https://openrouter.ai/api/v1"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Call OpenRouter API

        Format: OpenAI-compatible but with:
        - Different auth header (Authorization: Bearer)
        - Model names include provider prefix (anthropic/claude-3.5-sonnet)
        - HTTP-Referer header recommended
        """

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://noodlings.ai',
            'X-Title': 'Noodlings Multi-Timescale Affective Agents'
        }

        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                data = await resp.json()

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=data['choices'][0]['message']['content'],
            model=data['model'],
            provider='openrouter',
            usage={
                'input_tokens': data['usage']['prompt_tokens'],
                'output_tokens': data['usage']['completion_tokens'],
            },
            finish_reason=data['choices'][0]['finish_reason'],
            latency_ms=latency_ms
        )
```

#### D. Refactor Existing OllamaClient (~150 lines)

```python
# Wrap existing OpenAICompatibleLLM for consistency

class OllamaClient(LLMClient):
    """Ollama client (wraps existing OpenAICompatibleLLM)"""

    def __init__(self, config: Dict[str, Any]):
        # Use existing OpenAICompatibleLLM implementation
        self.impl = OpenAICompatibleLLM(
            api_base=config.get('base_url', 'http://localhost:11434/v1'),
            api_key='not-needed',
            use_model_instances=True
        )

    async def generate(self, messages, model, temperature=0.7, max_tokens=1000):
        """Delegate to existing implementation"""
        # Call existing OpenAICompatibleLLM.generate()
        # Convert response to LLMResponse format
        # ...
```

---

### Phase 3: Integration Points

#### 1. Update cognitive_components.py

```python
# BEFORE (line ~71):
async def _call_llm_tracked(self, llm_client, prompt: str, ...):
    # Assumes llm_client is OpenAICompatibleLLM
    response = await llm_client.generate(...)

# AFTER:
async def _call_llm_tracked(self, model_label: str, prompt: str, ...):
    # Get appropriate client via router
    client = self.llm_router.get_client(model_label)
    response = await client.generate(...)
```

#### 2. Update LLMFacet Execution

```python
# In facet executor or agent_bridge.py

# BEFORE:
async def execute_llm_facet(facet: LLMFacet):
    # Hardcoded to Ollama
    ollama_client = get_ollama_client()
    result = await ollama_client.complete(facet.prompt, ...)

# AFTER:
async def execute_llm_facet(facet: LLMFacet):
    # Use configured provider
    model_label = facet.model  # "LARGE"
    client = llm_router.get_client(model_label)
    result = await client.complete(facet.prompt, model=model_name, ...)
```

#### 3. Pass Provider Config from NoodleStudio → cmush

```python
# NoodleStudio sends provider configs via noodleScope API

# BEFORE:
# cmush only knows about Ollama config

# AFTER:
# POST /api/provider-config
{
    "anthropic": {
        "api_key": "sk-ant-...",
        "models": ["claude-opus-4.5", ...]
    },
    "openai": {
        "api_key": "sk-...",
        "models": ["gpt-4", ...]
    }
}

# cmush creates LLMClientRouter with all providers
```

---

## Implementation Checklist

### Day 1: Infrastructure (~4 hours)
- [ ] Create `llm_client_router.py` with abstract `LLMClient` class
- [ ] Create `LLMResponse` dataclass
- [ ] Create `LLMClientRouter` class
- [ ] Test with mock clients

### Day 2: Provider Clients (~6 hours)
- [ ] Implement `AnthropicClient` (install `anthropic` package)
- [ ] Implement `OpenAIClient` (install `openai` package)
- [ ] Implement `OpenRouterClient` (aiohttp)
- [ ] Wrap existing Ollama as `OllamaClient`
- [ ] Test each client independently

### Day 3: Integration (~6 hours)
- [ ] Update `cognitive_components.py` to use router
- [ ] Add provider config sync (NoodleStudio → cmush)
- [ ] Update facet executor
- [ ] Add error handling (API key missing, rate limits)
- [ ] Test end-to-end with each provider

### Day 4: Testing & Polish (~4 hours)
- [ ] Test Red using Claude Opus
- [ ] Test Context Intelligence using GPT-4
- [ ] Test Convergence using OpenRouter
- [ ] Add fallback logic (if provider fails, try Ollama)
- [ ] Add usage tracking (token counts per provider)

---

## Package Dependencies

Add to `requirements.txt`:
```
anthropic>=0.40.0
openai>=1.0.0
aiohttp>=3.9.0
```

---

## Cost Implications

**Testing costs (using external APIs):**
- Anthropic: ~$0.015 per agent response (Claude Sonnet)
- OpenAI: ~$0.03 per agent response (GPT-4)
- OpenRouter: Varies by model

**Mitigation:**
- Keep Ollama as default for development
- Only use external for production/demo
- Add token budget limits
- Cache responses for repeated prompts

---

## Error Handling Strategy

```python
class LLMClientRouter:

    async def generate_with_fallback(self, model_label: str, ...):
        """Try configured provider, fallback to Ollama if fails"""

        try:
            # Try primary provider
            client = self.get_client(model_label)
            return await client.generate(...)

        except AnthropicAPIError as e:
            logger.warning(f"Anthropic API failed: {e}, falling back to Ollama")
            fallback_client = self.clients['ollama']
            return await fallback_client.generate(...)

        except OpenAIRateLimitError:
            logger.warning("OpenAI rate limit hit, falling back to Ollama")
            # ...
```

---

## Testing Plan

### 1. Unit Tests (per client)
```python
# Test Anthropic client
async def test_anthropic_client():
    client = AnthropicClient({'api_key': 'sk-ant-...'})

    response = await client.complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Say hello",
        model="claude-sonnet-4.5"
    )

    assert "hello" in response.lower()
    assert response.provider == 'anthropic'
```

### 2. Integration Tests
```python
# Test router
async def test_router():
    # Configure LARGE → Anthropic
    label_manager.set_label("LARGE", "anthropic", "claude-sonnet-4.5")

    router = LLMClientRouter(provider_manager, label_manager)
    client = router.get_client("LARGE")

    assert isinstance(client, AnthropicClient)

    response = await client.complete("Test", "Hello")
    assert len(response) > 0
```

### 3. Live Agent Test
```bash
# In noodleMUSH:
1. Configure LARGE → Anthropic (claude-sonnet-4.5)
2. Set Red's Mind facet to use LARGE
3. Chat with Red: "hi red"
4. Verify response comes from Claude API (check logs)
5. Check token usage in Anthropic dashboard
```

---

## Quick Start Implementation

Want me to start vith **Phase 1** (infrastructure)?

**Steps:**
1. Create `llm_client_router.py` with abstract base
2. Create `LLMResponse` dataclass
3. Create router class
4. Show you the architecture before building all clients

**Zen you approve, ve build ze provider clients!**

*Schlag zu pour ze multi-provider execution?* 🦆
