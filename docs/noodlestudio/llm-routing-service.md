# LLM Routing Service

**Status**: Planning
**Last Updated**: January 3, 2026
**Authors**: Caitlyn + Claude
**Inspiration**: OpenRouter

---

## Overview

Instead of requiring users to configure their own API keys for built applications, Noodlings provides a unified LLM routing service. Built apps connect to `api.noodlings.ai`, we route to providers (Anthropic, OpenAI, etc.), and bill users with a small service fee.

### Why This Model?

1. **Simpler UX**: Users don't need API keys from multiple providers
2. **Revenue stream**: Small margin on each request sustains the service
3. **Centralized management**: One woman (Caity) can manage from admin dashboard
4. **Flexibility**: Route to cheapest/fastest/best provider transparently

### How OpenRouter Works

OpenRouter's model (which we're copying):
1. Single API endpoint (OpenAI-compatible format)
2. User has account with credits (prepaid or usage-based)
3. Request comes in with model preference
4. OpenRouter routes to provider using THEIR API keys
5. Charges user: `provider_cost + margin` (typically 0-20%)
6. Handles rate limits, fallbacks, retries

---

## Architecture

### Request Flow

```
Built App                    Noodlings API                 Providers
   |                              |                            |
   |  POST /v1/chat/completions   |                            |
   |  Authorization: Bearer xxx   |                            |
   |----------------------------->|                            |
   |                              |  1. Validate token         |
   |                              |  2. Check credits          |
   |                              |  3. Select provider        |
   |                              |                            |
   |                              |  POST to Anthropic/OpenAI  |
   |                              |--------------------------->|
   |                              |                            |
   |                              |<---------------------------|
   |                              |  4. Count tokens           |
   |                              |  5. Calculate cost         |
   |                              |  6. Deduct credits         |
   |                              |  7. Log usage              |
   |<-----------------------------|                            |
   |  Response (streamed)         |                            |
```

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    api.noodlings.ai                         │
│                  (Cloudflare Workers)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Auth      │  │   Router    │  │   Billing   │         │
│  │   Layer     │  │   Layer     │  │   Layer     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         v                v                v                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Cloudflare D1 Database              │       │
│  │  - users, credits, usage_logs, provider_config  │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           v                  v                  v
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Anthropic│       │  OpenAI  │       │  Google  │
    │   API    │       │   API    │       │   API    │
    └──────────┘       └──────────┘       └──────────┘
```

---

## API Design

### Endpoint: POST /v1/chat/completions

OpenAI-compatible format (same as OpenRouter):

```bash
curl https://api.noodlings.ai/v1/chat/completions \
  -H "Authorization: Bearer nood_xxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Model Naming Convention

Follow OpenRouter's pattern: `provider/model-name`

```
anthropic/claude-sonnet-4
anthropic/claude-opus-4
anthropic/claude-haiku-3.5
openai/gpt-4o
openai/gpt-4o-mini
google/gemini-2.0-flash
meta/llama-3.3-70b
```

### Response Format

Standard OpenAI format:

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1704307200,
  "model": "anthropic/claude-sonnet-4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

---

## Database Schema

### New Tables

```sql
-- Provider configurations (managed via admin dashboard)
CREATE TABLE provider_configs (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,           -- 'anthropic', 'openai', 'google'
  api_key_encrypted TEXT NOT NULL,  -- Encrypted API key
  enabled INTEGER DEFAULT 1,
  priority INTEGER DEFAULT 0,       -- For fallback ordering
  rate_limit_rpm INTEGER,           -- Requests per minute
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Model pricing
CREATE TABLE model_pricing (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,           -- 'anthropic/claude-sonnet-4'
  provider TEXT NOT NULL,
  input_cost_per_1m REAL NOT NULL,  -- Cost per 1M input tokens (USD)
  output_cost_per_1m REAL NOT NULL, -- Cost per 1M output tokens (USD)
  margin_percent REAL DEFAULT 20,   -- Our markup (20% default)
  enabled INTEGER DEFAULT 1,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Usage logs (for billing and analytics)
CREATE TABLE llm_usage_logs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  provider_cost_usd REAL NOT NULL,  -- What we paid
  user_cost_usd REAL NOT NULL,      -- What user paid (with margin)
  latency_ms INTEGER,
  success INTEGER DEFAULT 1,
  error_message TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- API keys for built apps (separate from user login tokens)
CREATE TABLE app_api_keys (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  key_hash TEXT NOT NULL,           -- SHA256 of the key
  name TEXT,                        -- "Red's World Production"
  last_used_at TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Existing Tables (Already Have)

- `users` - User accounts
- `credits` - User credit balances
- `credit_transactions` - Credit history

---

## Pricing Model

### Example Pricing (per 1M tokens)

| Model | Provider Cost | Our Price (20% margin) |
|-------|---------------|------------------------|
| claude-sonnet-4 | $3 / $15 | $3.60 / $18 |
| claude-opus-4 | $15 / $75 | $18 / $90 |
| claude-haiku-3.5 | $0.80 / $4 | $0.96 / $4.80 |
| gpt-4o | $2.50 / $10 | $3 / $12 |
| gpt-4o-mini | $0.15 / $0.60 | $0.18 / $0.72 |

### Credit System

1 credit = $0.01 USD

So a request costing $0.036 = 3.6 credits deducted.

### Billing Flow

```python
def process_request(user_id, model, input_tokens, output_tokens):
    pricing = get_pricing(model)

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    provider_cost = input_cost + output_cost

    # Add margin
    margin = 1 + (pricing.margin_percent / 100)
    user_cost = provider_cost * margin

    # Convert to credits (1 credit = $0.01)
    credits_to_deduct = user_cost * 100

    # Deduct
    deduct_credits(user_id, credits_to_deduct)

    # Log
    log_usage(user_id, model, input_tokens, output_tokens,
              provider_cost, user_cost)
```

---

## Admin Dashboard Features

### Model Management Page

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Routing > Models                              [+ Add]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ anthropic/claude-sonnet-4              [Enabled] ●  │   │
│  │ Input: $3.00/1M  Output: $15.00/1M  Margin: 20%     │   │
│  │ Our Price: $3.60/1M input, $18.00/1M output         │   │
│  │ Usage today: 2.3M tokens ($8.28)         [Edit]     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ anthropic/claude-haiku-3.5             [Enabled] ●  │   │
│  │ Input: $0.80/1M  Output: $4.00/1M  Margin: 20%      │   │
│  │ Our Price: $0.96/1M input, $4.80/1M output          │   │
│  │ Usage today: 15.1M tokens ($14.50)       [Edit]     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Provider Keys Page

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Routing > Provider Keys                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Anthropic                                    [Connected] ● │
│  API Key: sk-ant-...xxxxx (last 5 chars)                   │
│  Rate Limit: 4000 RPM                                       │
│  Monthly Spend: $1,234.56                    [Update Key]   │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  OpenAI                                    [Not Connected] ○│
│  [Add API Key]                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Usage Analytics Page

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Routing > Analytics                    [Last 7 days ▼]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Revenue         Provider Cost      Margin                  │
│  $2,456.78       $2,047.32          $409.46 (16.7%)        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  [Chart: Daily revenue/cost over time]              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Top Models                          Top Users              │
│  1. claude-sonnet-4    $1,234       1. user@x.com  $456    │
│  2. claude-haiku-3.5   $890         2. dev@y.org   $234    │
│  3. gpt-4o-mini        $332         3. test@z.io   $123    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### What We Already Have

- Cloudflare Workers infrastructure
- D1 database with users, credits
- Admin dashboard (SvelteKit)
- Authentication system
- Credit transaction system

### What We Need to Build

#### Phase 1: Core Routing (MVP)
- [ ] `/v1/chat/completions` endpoint
- [ ] Anthropic provider integration
- [ ] Token counting (tiktoken or similar)
- [ ] Basic billing (deduct credits)
- [ ] Usage logging

#### Phase 2: Admin Dashboard
- [ ] Provider keys management page
- [ ] Model pricing configuration
- [ ] Usage analytics dashboard
- [ ] Margin configuration

#### Phase 3: Multi-Provider
- [ ] OpenAI integration
- [ ] Google (Gemini) integration
- [ ] Fallback routing (if primary fails)
- [ ] Smart routing (cheapest/fastest)

#### Phase 4: Advanced
- [ ] Rate limiting per user
- [ ] Spending limits/alerts
- [ ] Streaming optimization
- [ ] Caching for identical requests

---

## Built App Integration

### How Built Apps Connect

In `build.yaml`:

```yaml
name: "Red's World"
version: "1.0.0"

# LLM routing - use Noodlings service
llm:
  provider: "noodlings"  # Use our routing service
  # OR
  provider: "local"      # Use local Ollama only
  # OR
  provider: "custom"     # User provides own keys
  custom_endpoint: "https://api.openai.com/v1"
```

### First-Run Experience (for "noodlings" provider)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    Welcome to Red's World                   │
│                                                             │
│  This app uses AI powered by Noodlings.                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Email: [_______________________________]           │   │
│  │  Password: [_______________________________]        │   │
│  │                                                     │   │
│  │  [Sign In]              [Create Account]            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Or continue with local AI only (requires Ollama)          │
│  [Use Local AI]                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Runtime Configuration

The built app stores:
- API key (in system keychain)
- Preferred model
- Local fallback preference

---

## Security Considerations

### API Key Storage
- Provider API keys encrypted at rest in D1
- Decrypted only in Worker memory during request
- Never logged or exposed in responses

### User API Keys
- Generated: `nood_` prefix + 32 random chars
- Stored as SHA256 hash in database
- Can be revoked from dashboard

### Rate Limiting
- Per-user rate limits (configurable)
- Per-provider rate limits (respect their limits)
- Automatic backoff on 429s

---

## Cost Analysis

### Break-Even

If we add 20% margin:
- $1000 in user spending = $833 provider cost + $167 margin
- Cloudflare Workers: ~$0.50/million requests (negligible)
- D1 storage: ~$0.75/GB/month (negligible)

### Scaling Considerations

At scale, we could negotiate volume discounts with providers, increasing effective margin.

---

## Questions to Resolve

### Q1: Prepaid vs Usage-Based?

| Option | Pros | Cons |
|--------|------|------|
| **Prepaid credits** | Simple, no billing complexity | Users must buy credits upfront |
| **Usage-based billing** | Pay-as-you-go | Need Stripe subscriptions |
| **Hybrid** | Flexibility | More complex |

**Current thinking**: Start with prepaid credits (already have this). Add usage-based later.

### Q2: Free Tier?

Should new users get free credits to try?

**Current thinking**: Yes. 1000 free credits ($10 worth) on signup. Good for demos.

### Q3: Model Access Tiers?

Should some models require higher account tier?

**Current thinking**: No tiers for v1. All models available to all users. Revisit if needed.

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-03 | Initial planning document |
