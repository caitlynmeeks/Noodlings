# LLM Routing Service

**Status**: Phase 2 COMPLETE - Deployed to Production
**Last Updated**: January 8, 2026
**Authors**: Caitlyn + Claude
**Inspiration**: OpenRouter

---

## Implementation Status

| Phase | Status | Date |
|-------|--------|------|
| Phase 1: Core Routing (Anthropic) | COMPLETE | Jan 3, 2026 |
| Phase 1.5: API Keys (`nood_xxxxx`) | COMPLETE | Jan 8, 2026 |
| Phase 2: Admin Dashboard | Not Started | - |
| Phase 3: Multi-Provider (OpenAI, Google) | Not Started | - |
| Phase 4: Advanced (Rate limiting, Caching) | Not Started | - |

### What's Live

**Endpoint**: `https://api.noodlings.ai/v1/chat/completions`

```bash
# Example request (using API key)
curl -X POST https://api.noodlings.ai/v1/chat/completions \
  -H "Authorization: Bearer nood_xxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3.5-haiku",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Tested**: Jan 8, 2026 (end-to-end verification)
- Authentication: Working (session tokens AND API keys)
- API Keys: Working (`nood_xxxxx` format, create/list/revoke)
- Credit checking: Working (402 when insufficient)
- Anthropic routing: Working
- Token counting: Working
- Billing: Working (credits deducted correctly)
- OpenAI-compatible response format: Working

**Authentication**: Supports both session tokens and API keys (`nood_xxxxx` format). See [API Keys](../backend/api-keys.md) for key management. For CLI/runtime use, set `NOODLINGS_API_KEY` to either a session token or an API key.

### Supported Models (Production)

| Model ID | Input/1M | Output/1M |
|----------|----------|-----------|
| `anthropic/claude-opus-4-5` | $18.00 | $90.00 |
| `anthropic/claude-sonnet-4` | $3.60 | $18.00 |
| `anthropic/claude-3.5-sonnet` | $3.60 | $18.00 |
| `anthropic/claude-3.5-haiku` | $0.96 | $4.80 |
| `anthropic/claude-3-opus` | $18.00 | $90.00 |
| `anthropic/claude-3-sonnet` | $3.60 | $18.00 |
| `anthropic/claude-3-haiku` | $0.30 | $1.50 |

All prices include 20% margin over provider cost.

---

## Overview

Instead of requiring users to configure their own API keys for built applications, Noodlings provides a unified LLM routing service. Built apps connect to `api.noodlings.ai`, we route to providers (Anthropic, OpenAI, etc.), and bill users with a small service fee.

---

## Architecture Decision: Parallel Systems

**CRITICAL**: The new `/v1/chat/completions` endpoint is **completely separate** from the existing `/llm/*` routes.

```
EXISTING (DO NOT TOUCH):
  /llm/generate         -> OpenRouter -> Providers
  /llm/generate/stream  -> OpenRouter -> Providers
  Used by: NoodleStudio internal operations (may be phased out later)

NEW (for built apps):
  /v1/chat/completions  -> Direct to Anthropic/OpenAI/Google
  Used by: Built applications, external consumers
```

The existing `/llm/*` routes continue working exactly as they do now. They use OpenRouter as a convenience layer. The new `/v1/chat/completions` endpoint routes **directly** to providers using our API keys, giving us:
- No OpenRouter middleman fees
- Direct control over provider keys
- Better error handling and rate limit management
- Foundation for smart multi-provider routing

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

## Runtime LLM Provider Architecture

### The Three Paths

Built applications (and the headless runtime) support three LLM provider paths:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Built Application                            │
│                                                                 │
│  LLM Provider Selection (user chooses one):                     │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  noodlings  │  │    local    │  │  own_keys   │             │
│  │   (cloud)   │  │  (ollama)   │  │  (byok)     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          v                v                v
   api.noodlings.ai   localhost:11434   api.anthropic.com
   /v1/chat/completions                 api.openai.com
   (we bill user)     (free, local)     (user's keys)
```

| Provider | Description | API Key Needed | Billing |
|----------|-------------|----------------|---------|
| `noodlings` | Our cloud routing service | User's Noodlings account token | Credits deducted |
| `ollama` / `lmstudio` | Local inference | None | Free |
| `anthropic` / `openai` / etc. | Direct to provider | User provides own | User pays provider |

### HeadlessLLMClient Configuration

The `noodlestudio/runtime/llm_client.py` supports these providers:

```python
# Environment variables for runtime configuration
NOODLE_LLM_PROVIDER=noodlings       # Use our cloud service
NOODLE_LLM_PROVIDER=ollama          # Use local Ollama
NOODLE_LLM_PROVIDER=anthropic       # Use own Anthropic key

# For noodlings provider, also need:
NOODLINGS_API_KEY=<session_token>   # Session ID from web login (API keys TBD)

# For own keys:
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
```

### First-Run Experience

Built apps using `noodlings` provider show a first-run dialog:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    Welcome to Red's World                       │
│                                                                 │
│  This app uses AI powered by Noodlings.                         │
│                                                                 │
│  Choose how to connect:                                         │
│                                                                 │
│  ○ Noodlings Cloud (Recommended)                                │
│    Sign in with your Noodlings account                          │
│    Uses your credit balance for AI requests                     │
│    [Sign In] [Create Account]                                   │
│                                                                 │
│  ○ Local AI (Ollama)                                            │
│    Free, runs on your computer                                  │
│    Requires Ollama installed                                    │
│    [Use Local AI]                                               │
│                                                                 │
│  ○ Own API Keys                                                 │
│    Use your own Anthropic/OpenAI keys                           │
│    [Configure Keys]                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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

## Institutional Accounts

### Overview

Organizations (schools, companies, labs) can manage billing for their members:
- Org admin invites members (by email)
- Org sets per-member monthly credit limits
- Usage bills to org, not individual
- Members toggle between personal/institutional billing
- Org admin audits usage by member

### User Experience

**For Members:**
```
┌─────────────────────────────────────────────────────────────┐
│  Account > Billing                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Billing Source:                                            │
│  ○ Personal Credits (142 credits remaining)                │
│  ● Stanford AI Lab (Institutional)                          │
│     Monthly limit: 5,000 credits                            │
│     Used this month: 1,234 credits                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**For Org Admins:**
```
┌─────────────────────────────────────────────────────────────┐
│  Stanford AI Lab > Members                         [Invite] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Member              Limit     This Month    Status         │
│  ─────────────────────────────────────────────────────────  │
│  alice@stanford.edu  5,000     1,234         Active         │
│  bob@stanford.edu    5,000     4,891         Near Limit     │
│  carol@stanford.edu  10,000    2,100         Active         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Total This Month: 8,225 credits ($82.25)                   │
│                                                             │
│  [Download Usage Report]                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema

```sql
-- Organizations
CREATE TABLE organizations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,                -- "Stanford AI Lab"
  billing_email TEXT NOT NULL,
  stripe_customer_id TEXT,           -- For invoicing
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Org membership
CREATE TABLE org_members (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT DEFAULT 'member',        -- 'admin' or 'member'
  monthly_limit INTEGER DEFAULT 5000, -- Credits per month
  invited_by TEXT,
  joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (org_id) REFERENCES organizations(id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  UNIQUE(org_id, user_id)
);

-- Track which billing source user is using
-- (stored in users table or session)
ALTER TABLE users ADD COLUMN active_billing TEXT DEFAULT 'personal';
-- 'personal' or org_id
```

### Billing Flow

```python
def get_billing_source(user_id):
    user = get_user(user_id)

    if user.active_billing == 'personal':
        return {'type': 'personal', 'credits': user.credits}

    # Institutional billing
    org_id = user.active_billing
    membership = get_org_membership(user_id, org_id)

    if not membership:
        # Org removed them, fall back to personal
        return {'type': 'personal', 'credits': user.credits}

    month_start = get_month_start()
    used_this_month = get_user_org_usage(user_id, org_id, since=month_start)
    remaining = membership.monthly_limit - used_this_month

    return {
        'type': 'institutional',
        'org_id': org_id,
        'org_name': membership.org.name,
        'monthly_limit': membership.monthly_limit,
        'remaining': remaining
    }

def process_request(user_id, model, tokens):
    billing = get_billing_source(user_id)
    cost = calculate_cost(model, tokens)

    if billing['type'] == 'personal':
        deduct_credits(user_id, cost)
    else:
        # Log to org, check limit
        if cost > billing['remaining']:
            raise InsufficientCreditsError("Monthly institutional limit reached")
        log_org_usage(user_id, billing['org_id'], cost)
```

### Admin Dashboard Pages

**Organizations List** (super-admin view):
- List all orgs
- Total members, monthly spend
- Create new org

**Org Detail** (org admin view):
- Member list with usage
- Invite members
- Set per-member limits
- Download usage CSV
- Billing settings (payment method, invoices)

### Invite Flow

1. Org admin enters email
2. System sends invite email with link
3. User clicks link:
   - If existing account: Add to org
   - If new: Create account, add to org
4. User can now switch billing source to org

### Audit/Reporting

Org admins can:
- View real-time usage by member
- Download CSV reports (date range, by member, by model)
- Set up usage alerts (email when member hits 80% of limit)

### Implementation Priority

| Feature | Priority | Notes |
|---------|----------|-------|
| Org creation | P1 | **Self-service** - users create their own orgs |
| Member management | P1 | Invite, remove, set limits |
| Billing source toggle | P1 | Users switch personal/org |
| Usage tracking by org | P1 | Required for billing |
| Usage reports/CSV | P2 | Audit requirement |
| Stripe invoicing | P2 | Start with prepaid credits |
| Usage alerts | P3 | Nice to have |

---

## Unified Credits Economy

### One Currency, Multiple Uses

The credits system powers everything:

| Use Case | How Credits Work |
|----------|------------------|
| **LLM Routing** | Deducted per request (token-based) |
| **Asset Store** | Purchase noodlings, stages, radiances |
| **Premium Features** | Unlock advanced capabilities |

### Asset Store Integration

Same `credits` and `credit_transactions` tables:

```sql
-- Asset Store purchases use same credits
INSERT INTO credit_transactions (user_id, amount, type, description)
VALUES ('user123', -500, 'asset_purchase', 'Red Fire Anklebiter by @caitlyn');

-- Creators earn credits (revenue share)
INSERT INTO credit_transactions (user_id, amount, type, description)
VALUES ('creator456', 350, 'asset_sale', 'Red Fire Anklebiter sold to @user123');
```

### Institutional + Asset Store

Orgs can control asset purchases too:
- "Members can purchase assets up to 100 credits each"
- "All purchases require admin approval"
- "Block marketplace entirely (internal assets only)"

**Org-Owned Assets**: Assets purchased with org billing belong to the org, shared with all members:

```sql
-- Assets can belong to user OR org
ALTER TABLE asset_purchases ADD COLUMN owner_type TEXT DEFAULT 'user';
-- 'user' or 'org'
ALTER TABLE asset_purchases ADD COLUMN owner_id TEXT NOT NULL;
-- user_id or org_id depending on owner_type

-- Check if user has access to asset
-- True if: they bought it personally OR their org bought it
```

**Institutional Licensing** (future consideration):
- Some assets may have "institutional license" option
- Higher price, but shared across all org members
- vs. personal license per user
- Creator sets whether institutional license is available

```sql
ALTER TABLE org_members ADD COLUMN asset_purchase_limit INTEGER DEFAULT 0;
-- 0 = no limit, >0 = max per purchase, -1 = blocked
```

### Revenue Flow

```
User buys asset (500 credits)
       │
       ├── Creator gets 70% (350 credits)
       ├── Noodlings gets 30% (150 credits)
       │
       └── If institutional billing:
           └── Charged to org's account
```

### Creator Payouts

Creators accumulate credits from sales:
- Can use credits for their own LLM usage
- Can request payout (credits → USD via Stripe)
- Min payout threshold: 1000 credits ($10)

---

## Questions to Resolve

### Q1: Prepaid vs Usage-Based?

| Option | Pros | Cons |
|--------|------|------|
| **Prepaid credits** | Simple, no billing complexity | Users must buy credits upfront |
| **Usage-based billing** | Pay-as-you-go | Need Stripe subscriptions |
| **Hybrid** | Flexibility | More complex |

**Decision**: **Prepaid with auto-topup** (like Anthropic Console). Users set a threshold, card charged automatically when credits run low.

### Q2: Free Tier?

Should new users get free credits to try?

**Decision**: Yes. **1000 free credits** ($10 worth) on signup. Good for demos.

### Q3: Model Access Tiers?

Should some models require higher account tier?

**Decision**: No tiers for v1. All models available to all users. Revisit if needed.

---

## Regression Testing

### Manual Test Checklist

Before deploying changes to `/v1/chat/completions`:

```bash
# 1. Models endpoint (no auth)
curl -s https://api.noodlings.ai/v1/models | jq '.data | length'
# Expected: 7 (or more as models are added)

# 2. Auth required
curl -s -X POST https://api.noodlings.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-3.5-haiku", "messages": []}' | jq '.error'
# Expected: "Unauthorized" or similar

# 3. Credit check (with auth, 0 credits)
# Expected: 402 with "Insufficient credits"

# 4. Successful completion (with auth, >0 credits)
curl -s -X POST https://api.noodlings.ai/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-3.5-haiku", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 10}' | jq '.choices[0].message.content'
# Expected: Non-empty string response

# 5. Credit deduction
# Check balance before and after - should decrease
```

### Automated Tests (TODO)

Need to add to `backend/noodlings-api/`:
- `tests/v1.test.ts` - Unit tests for `/v1/*` routes
- Mock Anthropic API responses
- Test credit deduction logic
- Test error handling (invalid model, provider errors)

### Integration Tests

For runtime `noodlings` provider:
- `applications/noodlestudio/tests/test_noodlings_provider.py`
- Test `HeadlessLLMClient` with `provider="noodlings"`
- Mock API responses for offline testing

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-03 | Initial planning document |
| 2026-01-03 | Added Architecture Decision (parallel systems), Runtime LLM Provider Architecture |
| 2026-01-03 | Phase 1 COMPLETE: Deployed `/v1/chat/completions` with Anthropic routing |
| 2026-01-08 | Phase 1.5 COMPLETE: API keys (`nood_xxxxx`) implemented and deployed |
