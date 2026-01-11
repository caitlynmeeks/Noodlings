# NoodleServices

**Status**: Vision Specification
**Date**: 2026-01-10
**Authors**: Caity + Claude
**Priority**: Platform infrastructure

---

## Overview

NoodleServices is a unified gateway for cloud services, delivered through MCP servers. Just as NoodleROUTER provides frictionless LLM access, NoodleServices extends this model to quantum computing, cloud storage, external APIs, and more.

Users get one account, one billing relationship, no API key juggling. We handle the complexity, take a margin, and make powerful services accessible to everyone - including 12-year-olds running quantum circuits on real IBM hardware.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NoodleServices                           │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │NoodleROUTER │  │NoodleQUANTUM│  │ NoodleCLOUD │            │
│   │   (LLMs)    │  │(IBM Qiskit) │  │  (AWS/GCP)  │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                    │
│   ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐            │
│   │NoodleDATA   │  │NoodleCOMPUTE│  │NoodleSTORE  │            │
│   │(Weather,etc)│  │  (Lambda)   │  │    (S3)     │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                    │
│          └────────────────┼────────────────┘                    │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │   MCP Gateway   │                            │
│                  │                 │                            │
│                  │  • Auth/Identity│                            │
│                  │  • Metering     │                            │
│                  │  • Quota mgmt   │                            │
│                  │  • Billing      │                            │
│                  │  • Rate limiting│                            │
│                  └────────┬────────┘                            │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Let's    │  │ User's   │  │ Creator's│
        │ Quantum! │  │ Project  │  │   App    │
        └──────────┘  └──────────┘  └──────────┘
```

---

## Services

### NoodleROUTER (LLMs)

**Provider**: Anthropic (primary), OpenAI, local Ollama

**Capabilities**:
- Chat completions
- Tool use
- Vision
- Long context

**Why through us**:
- No API key management
- Automatic model selection by capability
- Fallback routing
- Usage tracking
- Cost optimization

---

### NoodleQUANTUM

**Provider**: IBM Quantum

**Capabilities**:
- `run_circuit` - Execute quantum circuit on real hardware
- `get_backends` - List available quantum computers
- `queue_status` - Check job queue
- `get_results` - Retrieve completed job results
- `simulate` - Run on simulator (free tier)

**Available Backends** (via our IBM account):
| Backend | Qubits | Type | Queue |
|---------|--------|------|-------|
| ibm_brisbane | 127 | Eagle r3 | Medium |
| ibm_kyoto | 127 | Eagle r3 | Low |
| ibm_osaka | 127 | Eagle r3 | Low |
| ibm_sherbrooke | 127 | Eagle r3 | Medium |

**Why through us**:
- No IBM Quantum account needed
- No waiting for access approval
- Pre-paid credits (we buy in bulk)
- Simplified API (Qiskit complexity hidden)
- Educational tier for Let's! apps

---

### NoodleDATA

**Providers**: OpenWeather, NASA, USGS, etc.

**Services**:

| Service | Provider | Capabilities |
|---------|----------|--------------|
| Weather | OpenWeather | Current, forecast, historical |
| Space | NASA APIs | APOD, Mars rovers, asteroids |
| Earth | USGS | Earthquakes, water data |
| Maps | OpenStreetMap | Geocoding, routing |

**Why through us**:
- Unified interface (one MCP, many sources)
- No individual API keys
- Bundled in tiers (not metered per-call)
- Caching/optimization

---

### NoodleCLOUD

**Provider**: AWS (primary), GCP (backup)

**Services**:

| Service | AWS Backend | Capabilities |
|---------|-------------|--------------|
| Storage | S3 | Upload, download, list, presigned URLs |
| Compute | Lambda | Run Python functions, scheduled tasks |
| Database | DynamoDB | Key-value storage for app state |
| Media | MediaConvert | Video/audio transcoding |

**Why through us**:
- No AWS account complexity
- Sandboxed (can't accidentally spin up expensive resources)
- Simplified APIs
- Automatic cleanup of orphaned resources

---

## Pricing Tiers

### Free Tier

For learning and experimentation.

| Service | Quota | Notes |
|---------|-------|-------|
| LLM (NoodleROUTER) | 1,000 tokens/day | Sonnet-class |
| Quantum (NoodleQUANTUM) | 10 circuits/month | Simulator only |
| Data (NoodleDATA) | 100 calls/day | All data services |
| Storage (NoodleCLOUD) | 100 MB | 30-day retention |
| **Attribution** | **Required** | "Made with NoodleSTUDIO" |

---

### Creator Tier - $9/month

For hobbyists and indie creators.

| Service | Quota | Notes |
|---------|-------|-------|
| LLM | 50,000 tokens/day | Sonnet + Haiku |
| Quantum | 100 circuits/month | **Real hardware** |
| Data | Unlimited | All data services |
| Storage | 5 GB | Permanent |
| Compute | 10 Lambda hours | |
| **Attribution** | Optional | |

---

### Studio Tier - $29/month

For serious creators and small teams.

| Service | Quota | Notes |
|---------|-------|-------|
| LLM | 500,000 tokens/day | All models incl. Opus |
| Quantum | 1,000 circuits/month | Priority queue |
| Data | Unlimited | |
| Storage | 50 GB | |
| Compute | 100 Lambda hours | |
| Database | 1 GB | DynamoDB |
| **Code signing** | Included | NoodleStudio cert |
| **Attribution** | Optional | |

---

### Enterprise Tier - Custom

For organizations and institutions.

| Feature | Options |
|---------|---------|
| LLM | Unlimited / bring your own key |
| Quantum | Direct IBM account / volume pricing |
| All services | Direct provider accounts available |
| On-premises | Self-hosted option |
| Support | Dedicated contact |
| SLA | 99.9% uptime guarantee |
| Compliance | SOC2, HIPAA available |

---

## MCP Server Specifications

### NoodleQUANTUM MCP

```yaml
# mcp://noodle.services/quantum

name: noodle_quantum
version: 1.0.0

tools:
  - name: run_circuit
    description: Execute a quantum circuit on IBM hardware
    parameters:
      circuit:
        type: string
        description: OpenQASM 3.0 circuit definition
      backend:
        type: string
        enum: [ibm_brisbane, ibm_kyoto, ibm_osaka, simulator]
        default: simulator
      shots:
        type: integer
        default: 1024
        max: 8192
    returns:
      job_id: string
      status: enum[queued, running, completed, failed]

  - name: get_results
    description: Get results from a completed quantum job
    parameters:
      job_id:
        type: string
    returns:
      counts: object  # measurement outcomes
      metadata: object

  - name: list_backends
    description: List available quantum backends
    returns:
      backends:
        type: array
        items:
          name: string
          qubits: integer
          queue_length: integer
          status: enum[online, maintenance, offline]

  - name: estimate_cost
    description: Estimate credits for a circuit
    parameters:
      circuit: string
      backend: string
      shots: integer
    returns:
      credits: number
      queue_estimate_minutes: number
```

### NoodleDATA MCP (Weather)

```yaml
# mcp://noodle.services/weather

name: noodle_weather
version: 1.0.0

tools:
  - name: current
    description: Get current weather for a location
    parameters:
      location:
        type: string
        description: City name, zip code, or lat,lon
    returns:
      temperature: number
      feels_like: number
      humidity: number
      conditions: string
      wind_speed: number

  - name: forecast
    description: Get weather forecast
    parameters:
      location: string
      days:
        type: integer
        default: 5
        max: 14
    returns:
      daily:
        type: array
        items:
          date: string
          high: number
          low: number
          conditions: string
          precipitation_chance: number
```

---

## Integration in Let's! Apps

### Project Configuration

```yaml
# project.yaml

name: Let's Quantum!
version: 1.0.0

services:
  required:
    - noodle_router    # For Guide
    - noodle_quantum   # For quantum circuits

  optional:
    - noodle_weather   # For weather-based quantum randomness demo

service_config:
  noodle_quantum:
    default_backend: simulator  # Start with simulator
    allow_real_hardware: true   # Can upgrade to real
    max_shots: 4096
```

### Runtime Access

```python
# In a facet or script

async def run_quantum_demo(self):
    # Get the quantum service
    quantum = self.context.services.get('noodle_quantum')

    # Build a Bell state circuit
    circuit = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c = measure q;
    """

    # Run it
    result = await quantum.run_circuit(
        circuit=circuit,
        backend='ibm_brisbane',  # Real hardware!
        shots=1024
    )

    # Guide explains the results
    self.context.guide.say(
        f"Look at those results! See how we got roughly equal "
        f"counts of 00 and 11, but almost no 01 or 10? "
        f"That's entanglement - the qubits are correlated."
    )
```

---

## Billing & Metering

### Credit System

All metered services use a unified credit system:

| Service | Credit Cost |
|---------|-------------|
| LLM: 1K tokens (Haiku) | 1 credit |
| LLM: 1K tokens (Sonnet) | 5 credits |
| LLM: 1K tokens (Opus) | 25 credits |
| Quantum: 1 circuit (simulator) | 1 credit |
| Quantum: 1 circuit (real, 1K shots) | 50 credits |
| Storage: 1 GB/month | 10 credits |
| Compute: 1 Lambda hour | 20 credits |

### Tier Credit Allocations

| Tier | Monthly Credits | Overage Rate |
|------|-----------------|--------------|
| Free | 100 | N/A (hard cap) |
| Creator | 5,000 | $0.01/credit |
| Studio | 50,000 | $0.008/credit |
| Enterprise | Custom | Volume pricing |

### Usage Dashboard

Users can see:
- Current credit balance
- Usage by service
- Usage by project
- Projected monthly usage
- Overage warnings

---

## Business Model

### Revenue Streams

1. **Subscription tiers** - Monthly recurring
2. **Overage charges** - Pay-as-you-go beyond tier
3. **Enterprise contracts** - Annual commitments
4. **Education partnerships** - Institutional licenses

### Cost Structure

| Service | Our Cost | User Pays | Margin |
|---------|----------|-----------|--------|
| Anthropic API | ~$3/M tokens | ~$4/M tokens | ~25% |
| IBM Quantum | Volume discount | Per-circuit | ~20% |
| AWS S3 | $0.023/GB | $0.10/GB | ~77% |
| AWS Lambda | $0.20/M req | $0.50/M req | ~60% |
| Weather APIs | Flat fee | Bundled | N/A |

### Volume Strategy

- Buy IBM Quantum credits in bulk (education discount)
- AWS reserved capacity for predictable baseline
- Cache aggressively for data APIs
- Pass savings to users, keep margin for sustainability

---

## Security & Compliance

### Isolation

- Each user's resources in separate namespace
- No cross-user data access
- Automatic resource cleanup on account deletion

### Credentials

- User never sees provider API keys
- Our service accounts are heavily scoped
- Rotation on regular schedule

### Audit

- All service calls logged
- Billing audit trail
- Anomaly detection for abuse

### Compliance Path

- SOC2 Type II (in progress)
- HIPAA BAA available for Enterprise
- GDPR compliant (EU data residency option)

---

## Future Services

### Planned

| Service | Provider | Timeline |
|---------|----------|----------|
| NoodleAUDIO | ElevenLabs | Q2 2026 |
| NoodleVISION | Replicate | Q2 2026 |
| NoodleSEARCH | Brave/Perplexity | Q3 2026 |
| NoodleDB | PlanetScale | Q3 2026 |

### Community MCP

Eventually, third parties can register MCP servers:

```yaml
# Community service registration
name: astronomy_data
provider: community/astro_enthusiast
description: Access to various astronomy catalogs
pricing: free  # or metered
review_status: approved
```

---

## The Vision

A 12-year-old opens Let's Quantum!

She talks to Guide, learns about superposition.

Guide says "Want to try it for real?"

She builds a circuit. Clicks "Run on Real Hardware."

Her circuit runs on ibm_brisbane - a quantum computer colder than space, in a lab in New York.

Results come back. Guide explains what happened.

She goes to school and says: **"I wrote a quantum program and ran it on a real quantum computer."**

No IBM account. No university affiliation. No gatekeeping.

Just curiosity, conversation, and access.

That's NoodleServices.

---

*"The best technology disappears. You just do the thing you wanted to do."*
