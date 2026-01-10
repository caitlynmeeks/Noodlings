# NoodleStudio Secure Enterprise Tier

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Version**: 1.0 DRAFT
**Date**: 2025-01-05
**Contact**: Caitlyn Meeks, Principal - caitlyn@noodlings.ai

---

## Executive Summary

NoodleStudio is a cognitive architecture development platform enabling organizations to build AI-driven simulations, training environments, and intelligent agents. The Secure Enterprise Tier provides deployment options meeting the stringent security requirements of defense, intelligence, federal, and security-conscious commercial clients.

**Key Differentiators:**
- Air-gapped deployment capability (zero internet dependency)
- CAC/PIV authentication native support
- FedRAMP-ready architecture
- ITAR-compliant data handling
- Full audit trail with SIEM integration
- On-premise or GovCloud deployment options
- No vendor lock-in on LLM providers (local Ollama, Azure OpenAI Gov, etc.)

---

## Table of Contents

1. [Security Architecture Overview](#1-security-architecture-overview)
2. [Deployment Models](#2-deployment-models)
3. [Authentication & Access Control](#3-authentication--access-control)
4. [Data Protection](#4-data-protection)
5. [Audit & Compliance](#5-audit--compliance)
6. [Network Security](#6-network-security)
7. [Incident Response](#7-incident-response)
8. [Compliance Certifications](#8-compliance-certifications)
9. [LLM Security & Data Sovereignty](#9-llm-security--data-sovereignty)
10. [Implementation & Onboarding](#10-implementation--onboarding)
11. [Support & SLAs](#11-support--slas)
12. [Pricing Structure](#12-pricing-structure)
13. [Security Questionnaire Pre-Answers](#13-security-questionnaire-pre-answers)
14. [Reference Architectures](#14-reference-architectures)

---

## 1. Security Architecture Overview

### Defense-in-Depth Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERIMETER SECURITY                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 NETWORK SECURITY                          │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              APPLICATION SECURITY                   │  │  │
│  │  │  ┌───────────────────────────────────────────────┐  │  │  │
│  │  │  │              DATA SECURITY                    │  │  │  │
│  │  │  │  ┌─────────────────────────────────────────┐  │  │  │  │
│  │  │  │  │         IDENTITY & ACCESS               │  │  │  │  │
│  │  │  │  │  ┌───────────────────────────────────┐  │  │  │  │  │
│  │  │  │  │  │      COGNITIVE ASSETS             │  │  │  │  │  │
│  │  │  │  │  │   (Noodlings, Stages, Models)     │  │  │  │  │  │
│  │  │  │  │  └───────────────────────────────────┘  │  │  │  │  │
│  │  │  │  └─────────────────────────────────────────┘  │  │  │  │
│  │  │  └───────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Security Principles

| Principle | Implementation |
|-----------|----------------|
| **Zero Trust** | All requests authenticated and authorized, regardless of network location |
| **Least Privilege** | Role-based access with minimal default permissions |
| **Defense in Depth** | Multiple independent security layers |
| **Fail Secure** | System defaults to secure state on failure |
| **Audit Everything** | Complete audit trail of all operations |
| **Data Minimization** | Collect and retain only necessary data |

---

## 2. Deployment Models

### Model A: Dedicated Cloud Tenant

**Best for**: Organizations requiring isolation but not on-premise infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                 DEDICATED CLOUDFLARE ZONE                       │
│                 (US Data Centers Only)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Dedicated  │  │  Dedicated  │  │  Dedicated  │             │
│  │  Workers    │  │  D1 (SQL)   │  │  R2 (Blob)  │             │
│  │  (Compute)  │  │  (Database) │  │  (Storage)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                   [Customer VPN]                                │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                    Customer Network
```

**Specifications:**
- Separate Cloudflare account (not shared zone)
- Dedicated database instance (no multi-tenancy)
- US-only data residency guaranteed
- Customer-controlled encryption keys (BYOK)
- SOC2 Type II compliant (via Cloudflare)
- 99.9% uptime SLA

**Isolation Guarantees:**
- No shared compute resources
- No shared storage
- No shared database tables
- Separate TLS certificates
- Dedicated IP ranges (optional)

---

### Model B: AWS GovCloud / Azure Government

**Best for**: Organizations requiring FedRAMP High or IL4/IL5 compliance

```
┌─────────────────────────────────────────────────────────────────┐
│              AWS GOVCLOUD / AZURE GOVERNMENT                    │
│                   (FedRAMP High)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    VPC / VNET                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   EC2/VM    │  │  RDS/SQL    │  │   S3/Blob   │     │   │
│  │  │ NoodleMUSH  │  │  Database   │  │  Storage    │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │         │                │                │             │   │
│  │  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │   Lambda/   │  │   Azure     │                      │   │
│  │  │  Functions  │  │  OpenAI     │ (FedRAMP Auth)       │   │
│  │  │  (API)      │  │  (LLM)      │                      │   │
│  │  └─────────────┘  └─────────────┘                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│              [Private Link / ExpressRoute]                      │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                  Customer On-Premise
```

**Specifications:**
- FedRAMP High authorized infrastructure
- IL4/IL5 capable (with appropriate controls)
- ITAR compliant data handling
- Azure OpenAI (FedRAMP authorized) for LLM
- Customer-owned subscription (we manage deployment)
- HSM-backed key management (AWS CloudHSM / Azure Key Vault HSM)

**Compliance Inheritance:**
- Inherits cloud provider's FedRAMP authorization
- Reduces customer ATO burden
- Continuous monitoring via cloud provider

---

### Model C: On-Premise Deployment

**Best for**: Organizations requiring full infrastructure control

```
┌─────────────────────────────────────────────────────────────────┐
│                 CUSTOMER DATA CENTER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 NOODLESTUDIO SERVER                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ NoodleMUSH  │  │ PostgreSQL  │  │   MinIO     │     │   │
│  │  │  (Python)   │  │ (Database)  │  │  (Storage)  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │         │                │                │             │   │
│  │  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │  NoodleHub  │  │   Ollama    │                      │   │
│  │  │  (Web UI)   │  │   (LLM)     │ ← Local Models       │   │
│  │  └─────────────┘  └─────────────┘                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OPTIONAL: LICENSE SERVER                   │   │
│  │         (For air-gapped license validation)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Specifications:**
- Runs on customer hardware (bare metal or VM)
- Docker containers or direct installation
- PostgreSQL database (customer-managed)
- MinIO or S3-compatible storage
- Local LLM via Ollama (Llama, Mistral, etc.)
- No cloud dependency

**Requirements:**
- Linux server (Ubuntu 22.04 LTS recommended)
- 32GB+ RAM (64GB+ for LLM hosting)
- NVIDIA GPU recommended for local LLM (RTX 3090+)
- 500GB+ SSD storage

---

### Model D: Air-Gapped Deployment

**Best for**: SCIFs, classified environments, maximum isolation

```
┌─────────────────────────────────────────────────────────────────┐
│           SECURE COMPARTMENTED INFORMATION FACILITY             │
│                    (NO NETWORK CONNECTION)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              STANDALONE WORKSTATION                     │   │
│  │                                                         │   │
│  │  NoodleStudio ──► NoodleMUSH ──► Ollama (Local LLM)    │   │
│  │       │               │              │                  │   │
│  │       └───────────────┼──────────────┘                  │   │
│  │                       │                                 │   │
│  │                 Local PostgreSQL                        │   │
│  │                 Local File Storage                      │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   UPDATES    │    │   LICENSE    │    │   EXPORTS    │      │
│  │   via USB    │    │  via Dongle  │    │  via Diode   │      │
│  │  (Signed)    │    │  (Hardware)  │    │  (One-way)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         ▲                   ▲                   │               │
│         │                   │                   ▼               │
│    ─────┴───────────────────┴───────────────────┴─────         │
│                    PHYSICAL SECURITY BOUNDARY                   │
└─────────────────────────────────────────────────────────────────┘
```

**Specifications:**
- Zero network connectivity
- Offline license validation (hardware dongle + cryptographic)
- Updates via signed USB packages (manual review)
- Export via data diode (one-way transfer out)
- Local LLM only (no API calls)
- All data remains within physical boundary

**Update Process:**
1. Noodlings releases signed update package
2. Customer security team reviews package contents
3. Package transferred via approved media
4. Cryptographic signature verified before installation
5. Installation logged to local audit trail

**License Validation:**
- Hardware security dongle (YubiKey HSM or similar)
- Cryptographic challenge-response (no network needed)
- Time-limited licenses with grace period
- Tamper-evident logging

---

## 3. Authentication & Access Control

### Supported Authentication Methods

| Method | Use Case | Compliance |
|--------|----------|------------|
| **CAC/PIV** | DoD, Federal civilian | HSPD-12, FIPS 201 |
| **SAML 2.0** | Enterprise SSO | FedRAMP |
| **OIDC** | Modern IdP integration | FedRAMP |
| **LDAP/AD** | On-premise directory | Enterprise |
| **Local accounts** | Air-gapped, standalone | All |
| **MFA (TOTP)** | Additional factor | NIST 800-63B |
| **Hardware tokens** | High-assurance | FIPS 140-2 |

### CAC/PIV Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAC AUTHENTICATION FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   User   │───►│   Card   │───►│  Client  │───►│  Server  │  │
│  │ + CAC    │    │  Reader  │    │   TLS    │    │  Verify  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │         │
│       │  1. Insert CAC                               │         │
│       │  2. Enter PIN                                │         │
│       │  3. Client extracts certificate              │         │
│       │  4. Mutual TLS handshake                     │         │
│       │  5. Server validates against DoD PKI ────────┘         │
│       │  6. Extract EDIPI/UPN from certificate                 │
│       │  7. Map to local user record                           │
│       │  8. Establish session                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Certificate Validation:**
- Validates against DoD PKI trust chain
- Checks CRL/OCSP for revocation (when network available)
- Caches CRL for air-gapped operation
- Extracts EDIPI for user identification
- Supports both PIV-AUTH and CAC certificates

**NoodleStudio Desktop Integration:**
- Native smart card support via PKCS#11
- PIN prompt integrated into login flow
- Session timeout configurable by policy
- Re-authentication required for sensitive operations

### Role-Based Access Control (RBAC)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERMISSION HIERARCHY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ORGANIZATION ADMIN                                             │
│  ├── Manage organization settings                               │
│  ├── Manage all users and roles                                 │
│  ├── View all projects                                          │
│  ├── Access audit logs                                          │
│  └── Configure security policies                                │
│       │                                                         │
│  PROJECT ADMIN                                                  │
│  ├── Manage project settings                                    │
│  ├── Manage project members                                     │
│  ├── View project audit logs                                    │
│  └── Export project data                                        │
│       │                                                         │
│  DEVELOPER                                                      │
│  ├── Create/edit noodlings                                      │
│  ├── Create/edit stages                                         │
│  ├── Run simulations                                            │
│  └── View own audit trail                                       │
│       │                                                         │
│  VIEWER                                                         │
│  ├── View projects (read-only)                                  │
│  └── Run simulations (no edit)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Session Management

| Control | Setting |
|---------|---------|
| Session timeout (idle) | Configurable: 15-60 min (default: 30) |
| Session timeout (absolute) | Configurable: 4-12 hours (default: 8) |
| Concurrent sessions | Configurable: 1-5 per user |
| Session binding | IP address + User agent |
| Re-auth for exports | Required |
| Re-auth for admin actions | Required |

---

## 4. Data Protection

### Encryption Standards

| Data State | Method | Standard |
|------------|--------|----------|
| **At Rest (Database)** | AES-256-GCM | FIPS 140-2 |
| **At Rest (Files)** | AES-256-GCM | FIPS 140-2 |
| **At Rest (Backups)** | AES-256-GCM | FIPS 140-2 |
| **In Transit** | TLS 1.3 | FIPS 140-2 |
| **In Memory** | Secure enclaves (optional) | SGX/SEV |

### Key Management

**Cloud Deployments:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    KEY HIERARCHY                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CUSTOMER MASTER KEY (CMK)                  │   │
│  │         (Stored in HSM - Customer controlled)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│              ┌───────────┼───────────┐                         │
│              ▼           ▼           ▼                         │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │  Database     │ │   Storage     │ │   Backup      │        │
│  │  Encryption   │ │  Encryption   │ │  Encryption   │        │
│  │  Key (DEK)    │ │  Key (SEK)    │ │  Key (BEK)    │        │
│  └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Bring Your Own Key (BYOK):**
- Customer generates and controls master key
- Key never leaves customer's HSM boundary
- Noodlings cannot access plaintext data without customer key
- Key rotation supported without data re-encryption (envelope encryption)

**On-Premise:**
- Local HSM integration (Thales Luna, AWS CloudHSM, etc.)
- Software HSM option for lower security tiers (HashiCorp Vault)
- Key ceremonies documented and auditable

### Data Classification Support

NoodleStudio supports customer-defined data classification:

| Level | Label | Handling |
|-------|-------|----------|
| 1 | PUBLIC | Standard controls |
| 2 | INTERNAL | Encrypted, access logged |
| 3 | CONFIDENTIAL | Encrypted, MFA required, access logged |
| 4 | RESTRICTED | Encrypted, MFA required, approval workflow, full audit |

Projects and assets can be tagged with classification levels. System enforces appropriate controls automatically.

### Data Residency

| Deployment | Data Location | Guarantee |
|------------|---------------|-----------|
| Dedicated Cloud | US only | Contractual |
| GovCloud | US only (FedRAMP boundary) | Regulatory |
| On-Premise | Customer controlled | Physical |
| Air-Gapped | Customer controlled | Physical |

**No data ever leaves the designated boundary.** This includes:
- User data
- Project assets
- Cognitive models
- LLM prompts and responses
- Telemetry (disabled in secure tier)
- Crash reports (local only)

---

## 5. Audit & Compliance

### Comprehensive Audit Logging

Every operation generates an immutable audit record:

```json
{
  "event_id": "evt_a1b2c3d4e5f6",
  "timestamp": "2025-01-05T14:32:00.000Z",
  "event_type": "project.asset.create",
  "actor": {
    "user_id": "usr_abc123",
    "username": "john.smith",
    "edipi": "1234567890",
    "email": "john.smith@contractor.mil",
    "auth_method": "CAC",
    "session_id": "sess_xyz789"
  },
  "resource": {
    "type": "noodling",
    "id": "ndl_def456",
    "name": "Training_Agent_v2",
    "project_id": "prj_ghi789",
    "classification": "CONFIDENTIAL"
  },
  "action": {
    "operation": "CREATE",
    "details": {
      "source": "fork",
      "forked_from": "ndl_original123"
    }
  },
  "context": {
    "ip_address": "10.0.1.42",
    "user_agent": "NoodleStudio/1.2.3 (Windows NT 10.0)",
    "geo_location": "US-VA",
    "org_id": "org_securedefense"
  },
  "outcome": {
    "status": "SUCCESS",
    "response_time_ms": 234
  },
  "integrity": {
    "hash": "sha256:abc123...",
    "previous_hash": "sha256:xyz789...",
    "signature": "RS256:..."
  }
}
```

### Audit Event Categories

| Category | Events |
|----------|--------|
| **Authentication** | Login, logout, failed login, MFA challenge, session timeout, CAC insertion/removal |
| **Authorization** | Permission granted, permission denied, role change, privilege escalation |
| **Data Access** | Read, create, update, delete, export, import, share |
| **Administration** | User management, policy change, configuration change, key rotation |
| **Security** | Anomaly detected, threshold exceeded, integrity violation |
| **System** | Startup, shutdown, backup, restore, update |

### Log Integrity

- **Hash chaining**: Each log entry includes hash of previous entry
- **Digital signatures**: Entries signed with system key
- **Tamper detection**: Any modification breaks chain
- **Secure storage**: Logs stored separately from application data

### SIEM Integration

Native export formats:
- **CEF** (Common Event Format) - ArcSight
- **LEEF** (Log Event Extended Format) - QRadar
- **JSON** - Splunk, Elastic
- **Syslog** (RFC 5424) - Universal

Real-time streaming options:
- Syslog over TLS
- Kafka
- AWS Kinesis / Azure Event Hub
- Webhook (HTTPS POST)

### Retention & Archival

| Data Type | Active Retention | Archive | Total |
|-----------|------------------|---------|-------|
| Security events | 90 days | 7 years | 7 years |
| Access logs | 90 days | 3 years | 3 years |
| Admin actions | 90 days | 7 years | 7 years |
| Application logs | 30 days | 1 year | 1 year |

Archive format: Encrypted, compressed, integrity-verified bundles.

---

## 6. Network Security

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK ZONES                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INTERNET                                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      DMZ                                │   │
│  │  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │     WAF     │  │   Reverse   │                      │   │
│  │  │             │  │   Proxy     │                      │   │
│  │  └─────────────┘  └─────────────┘                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 APPLICATION TIER                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ NoodleHub   │  │ NoodleMUSH  │  │     API     │     │   │
│  │  │   (Web)     │  │  (Server)   │  │  Gateway    │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DATA TIER                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │  Database   │  │   Object    │  │     LLM     │     │   │
│  │  │             │  │   Storage   │  │   Service   │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Security Controls

| Control | Implementation |
|---------|----------------|
| **Firewall** | Stateful inspection, deny-by-default |
| **WAF** | OWASP Top 10 protection, custom rules |
| **DDoS** | Volumetric and application-layer protection |
| **IDS/IPS** | Signature and anomaly-based detection |
| **TLS** | 1.3 only, strong cipher suites |
| **mTLS** | Optional for API clients |
| **Network segmentation** | Separate VLANs per tier |
| **Micro-segmentation** | Service mesh (optional) |

### Allowed Cipher Suites (TLS 1.3)

```
TLS_AES_256_GCM_SHA384
TLS_CHACHA20_POLY1305_SHA256
TLS_AES_128_GCM_SHA256
```

Legacy TLS versions (1.0, 1.1, 1.2) disabled by default. TLS 1.2 available on request for compatibility.

### Certificate Management

- Certificates from trusted CA (DigiCert, Let's Encrypt, or customer CA)
- Automated renewal (ACME protocol)
- Certificate pinning supported for mobile/desktop clients
- Short-lived certificates option (24-hour)

---

## 7. Incident Response

### Severity Levels

| Level | Description | Response Time | Update Frequency |
|-------|-------------|---------------|------------------|
| **P1 - Critical** | Service down, data breach | 1 hour | Hourly |
| **P2 - High** | Major feature unavailable, security vulnerability | 4 hours | Every 4 hours |
| **P3 - Medium** | Minor feature issue, performance degradation | 24 hours | Daily |
| **P4 - Low** | Cosmetic, documentation | 72 hours | As needed |

### Incident Response Process

```
┌─────────────────────────────────────────────────────────────────┐
│                 INCIDENT RESPONSE WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ DETECT   │───►│ ANALYZE  │───►│ CONTAIN  │───►│ ERADICATE│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │         │
│       │                                               ▼         │
│       │              ┌──────────┐    ┌──────────┐              │
│       └──────────────│  REPORT  │◄───│ RECOVER  │              │
│                      └──────────┘    └──────────┘              │
│                           │                                     │
│                           ▼                                     │
│                      ┌──────────┐                               │
│                      │  REVIEW  │                               │
│                      │ (Lessons │                               │
│                      │ Learned) │                               │
│                      └──────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Security Incident Communication

**For Secure Tier customers:**
- Dedicated security contact (named individual)
- Encrypted communication channel (Signal, secure email)
- Phone escalation for P1 incidents
- Customer security team included in incident bridge
- Post-incident report within 72 hours

### Breach Notification

In the event of a confirmed data breach:
- Customer notified within 24 hours of confirmation
- Affected data scope identified
- Remediation plan provided
- Regulatory notification support (if required)
- Forensic report available upon request

---

## 8. Compliance Certifications

### Current Status

| Certification | Status | Evidence |
|---------------|--------|----------|
| **SOC 2 Type II** | Available (via Cloudflare) | Report on request |
| **ISO 27001** | Planned | - |
| **FedRAMP Moderate** | Architecture ready | Awaiting sponsor |
| **FedRAMP High** | Architecture ready (GovCloud) | Awaiting sponsor |
| **StateRAMP** | Architecture ready | Awaiting sponsor |

### Compliance Frameworks Supported

| Framework | Support Level |
|-----------|---------------|
| **NIST 800-53 Rev 5** | High baseline controls implemented |
| **NIST 800-171** | CUI handling controls implemented |
| **NIST CSF** | Aligned to all five functions |
| **CMMC 2.0** | Level 2 controls implemented |
| **ITAR** | Data segregation, access controls, audit |
| **HIPAA** | BAA available for healthcare customers |
| **GDPR** | Privacy controls, data subject rights |

### FedRAMP Readiness

**Control Implementation Status:**

| Control Family | Implemented | Planned | N/A |
|----------------|-------------|---------|-----|
| Access Control (AC) | 22 | 3 | 0 |
| Audit (AU) | 16 | 0 | 0 |
| Security Assessment (CA) | 9 | 0 | 0 |
| Configuration Mgmt (CM) | 11 | 0 | 0 |
| Contingency (CP) | 13 | 0 | 0 |
| Identification (IA) | 12 | 0 | 0 |
| Incident Response (IR) | 10 | 0 | 0 |
| Maintenance (MA) | 6 | 0 | 0 |
| Media Protection (MP) | 8 | 0 | 0 |
| Physical (PE) | 0 | 0 | 20 |
| Planning (PL) | 9 | 0 | 0 |
| Personnel (PS) | 8 | 0 | 0 |
| Risk Assessment (RA) | 6 | 0 | 0 |
| System Acquisition (SA) | 22 | 0 | 0 |
| System Protection (SC) | 44 | 0 | 0 |
| System Integrity (SI) | 17 | 0 | 0 |

Physical security controls (PE) inherited from cloud provider or customer facility.

---

## 9. LLM Security & Data Sovereignty

### The LLM Challenge

Cognitive simulations require Large Language Model inference. This creates unique security challenges:

1. **Prompt data exposure** - What goes into the LLM
2. **Response data exposure** - What comes out
3. **Model training** - Is your data used to train?
4. **Third-party transmission** - Where does data go?

### Noodlings LLM Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM PROVIDER OPTIONS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   OPTION 1:     │   Best for: Air-gapped, maximum control   │
│  │   LOCAL OLLAMA  │   Models: Llama 3, Mistral, Phi, etc.     │
│  │                 │   Data: Never leaves your network         │
│  │   [ON-PREMISE]  │   Latency: Lowest (local)                 │
│  └─────────────────┘   Cost: Hardware + power only             │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   OPTION 2:     │   Best for: FedRAMP High, IL4+            │
│  │   AZURE OPENAI  │   Models: GPT-4, GPT-4 Turbo              │
│  │   (GOVERNMENT)  │   Data: FedRAMP boundary only             │
│  │                 │   Compliance: FedRAMP High authorized     │
│  │   [GOVCLOUD]    │   Data use: NOT used for training         │
│  └─────────────────┘                                           │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   OPTION 3:     │   Best for: Performance + privacy         │
│  │   ANTHROPIC     │   Models: Claude 3.5 Sonnet, Claude 3     │
│  │   (API)         │   Data: NOT used for training             │
│  │                 │   SOC 2 Type II certified                 │
│  │   [CLOUD]       │   Zero data retention option              │
│  └─────────────────┘                                           │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   OPTION 4:     │   Best for: Cost + flexibility            │
│  │   CUSTOMER      │   Customer provides API keys              │
│  │   BYOK (Keys)   │   Customer controls provider relationship │
│  │                 │   Noodlings never sees keys (HSM)         │
│  │   [ANY]         │                                           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Local LLM Recommendations (Air-Gapped)

| Model | Parameters | VRAM Required | Quality | Use Case |
|-------|------------|---------------|---------|----------|
| Llama 3.1 70B | 70B | 48GB | Excellent | Full capability |
| Llama 3.1 8B | 8B | 8GB | Good | Resource-constrained |
| Mistral 7B | 7B | 8GB | Good | Fast inference |
| Phi-3 Medium | 14B | 12GB | Good | Balanced |
| Mixtral 8x7B | 46.7B | 32GB | Excellent | MoE efficiency |

**Hardware Recommendation for Local LLM:**
- NVIDIA A100 (40GB or 80GB) - Best performance
- NVIDIA RTX 4090 (24GB) - Good cost/performance
- Apple M2 Ultra (192GB unified) - Mac environments

### Data Flow Guarantees

**For Local/On-Premise:**
- Zero data transmitted externally
- All inference local to your network
- Model weights stored locally
- No telemetry or usage reporting

**For GovCloud (Azure OpenAI):**
- Data stays within FedRAMP boundary
- Microsoft contractually prohibited from training on your data
- Data encrypted in transit and at rest
- Audit logs available

---

## 10. Implementation & Onboarding

### Deployment Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Discovery** | 1-2 weeks | Requirements gathering, security questionnaire review, architecture planning |
| **Design** | 1-2 weeks | Detailed architecture, network design, integration planning |
| **Build** | 2-4 weeks | Infrastructure provisioning, configuration, deployment |
| **Test** | 1-2 weeks | Security testing, penetration testing (optional), UAT |
| **Go-Live** | 1 week | Cutover, training, documentation handoff |
| **Hypercare** | 2-4 weeks | Elevated support, issue resolution, optimization |

**Total: 8-15 weeks** depending on deployment model and customer requirements.

### Customer Responsibilities

| Area | Customer | Noodlings |
|------|----------|-----------|
| Infrastructure (On-Prem) | Provision, maintain | Specify requirements |
| Infrastructure (Cloud) | Subscription ownership | Deployment, management |
| Network connectivity | Provide | Specify requirements |
| Identity provider | Manage | Integrate |
| Security policies | Define | Implement |
| User management | Manage users | Provide tools |
| LLM provider (BYOK) | Contract, keys | Integrate |
| Training | Attend, adopt | Deliver |

### Training Included

| Course | Duration | Audience |
|--------|----------|----------|
| NoodleStudio Fundamentals | 4 hours | All users |
| NoodleStudio Advanced | 8 hours | Power users |
| Administration | 4 hours | IT admins |
| Security Configuration | 2 hours | Security team |

Training delivered virtually or on-site (on-site incurs travel costs).

---

## 11. Support & SLAs

### Support Tiers

| Tier | Availability | Response (P1) | Response (P2) | Included |
|------|--------------|---------------|---------------|----------|
| **Standard** | Business hours | 4 hours | 8 hours | With license |
| **Premium** | 12x5 | 2 hours | 4 hours | Add-on |
| **Enterprise** | 24x7 | 1 hour | 2 hours | Secure tier |

### Uptime SLA (Cloud Deployments)

| Tier | Uptime | Credits |
|------|--------|---------|
| Dedicated Cloud | 99.9% | 10% per 0.1% under |
| GovCloud | 99.95% | 10% per 0.05% under |

On-premise SLA depends on customer infrastructure.

### Support Channels

| Channel | Standard | Premium | Enterprise |
|---------|----------|---------|------------|
| Documentation | ✓ | ✓ | ✓ |
| Email | ✓ | ✓ | ✓ |
| Portal/Ticketing | ✓ | ✓ | ✓ |
| Phone | - | ✓ | ✓ |
| Dedicated Slack/Teams | - | - | ✓ |
| Named support engineer | - | - | ✓ |
| Quarterly business reviews | - | - | ✓ |

---

## 12. Pricing Structure

### Deployment Pricing

| Model | Setup Fee | Monthly | Notes |
|-------|-----------|---------|-------|
| **Dedicated Cloud** | $10,000 | $2,500 - $5,000 | Based on usage |
| **GovCloud Managed** | $25,000 | $5,000 - $10,000 | + cloud costs |
| **On-Premise** | $50,000 | Support contract | Perpetual license |
| **Air-Gapped** | $75,000 | Support contract | Perpetual + HSM |

### Credit Pricing (LLM Usage)

Credits used for LLM inference when using Noodlings-provided LLM routing.

| Volume | Price per 1M Credits | Effective Rate |
|--------|---------------------|----------------|
| < 1M | $10,000 | $0.0100 |
| 1M - 5M | $8,500 / M | $0.0085 |
| 5M - 10M | $8,000 / M | $0.0080 |
| 10M+ | Custom | Negotiated |

Local LLM (Ollama) deployments do not consume credits.

### Support Pricing

| Tier | Annual Cost |
|------|-------------|
| Standard | Included |
| Premium | $12,000 |
| Enterprise | $36,000 |

### Volume & Multi-Year Discounts

| Commitment | Discount |
|------------|----------|
| 1-year prepay | 10% |
| 3-year prepay | 20% |
| 5+ deployments | 15% additional |

---

## 13. Security Questionnaire Pre-Answers

Common security questionnaire items with responses:

### Data Security

**Q: Where is data stored?**
A: Configurable by deployment model. Options include US-only cloud (Cloudflare), AWS GovCloud, Azure Government, customer on-premise, or air-gapped. Customer has full control over data residency.

**Q: Is data encrypted at rest?**
A: Yes. AES-256-GCM encryption for all data at rest. Customer-managed keys (BYOK) supported and recommended for secure tier.

**Q: Is data encrypted in transit?**
A: Yes. TLS 1.3 for all data in transit. mTLS optional for API integrations.

**Q: Is customer data used to train models?**
A: No. Customer data is never used to train any models. For local LLM deployments, no data leaves the customer environment. For cloud LLM providers (Azure OpenAI, Anthropic), we use only providers that contractually guarantee no training on customer data.

**Q: How long is data retained?**
A: Customer-configurable. Default retention policies documented. Customers can request data deletion at any time.

### Access Control

**Q: What authentication methods are supported?**
A: SAML 2.0, OIDC, LDAP/AD, CAC/PIV (HSPD-12 compliant), local accounts with MFA.

**Q: Is MFA supported?**
A: Yes. TOTP, hardware tokens (YubiKey, RSA), and CAC/PIV.

**Q: How are access rights managed?**
A: Role-based access control (RBAC) with organization, project, and resource-level permissions. Principle of least privilege enforced by default.

### Audit & Compliance

**Q: What audit logging is available?**
A: Comprehensive audit logging of all authentication, authorization, data access, and administrative events. Logs are immutable with hash chaining and digital signatures.

**Q: Can logs be exported to our SIEM?**
A: Yes. Native support for CEF, LEEF, JSON, and Syslog formats. Real-time streaming via syslog/TLS, Kafka, or webhook.

**Q: What compliance certifications do you have?**
A: SOC 2 Type II (via infrastructure provider). FedRAMP-ready architecture. NIST 800-53 High baseline controls implemented. See Section 8 for full details.

### Incident Response

**Q: What is your incident response process?**
A: Documented IR process aligned with NIST 800-61. P1 incidents responded to within 1 hour for enterprise customers. Breach notification within 24 hours. See Section 7 for details.

**Q: Do you conduct penetration testing?**
A: Yes. Annual third-party penetration testing. Customers may conduct their own testing with coordination.

### Vendor Security

**Q: Do you have a security team?**
A: Yes. Dedicated security function responsible for security architecture, vulnerability management, and incident response.

**Q: How do you vet employees?**
A: Background checks for all employees with access to customer data or systems. Annual security awareness training.

**Q: How do you manage vulnerabilities?**
A: Automated vulnerability scanning (weekly). Critical vulnerabilities patched within 72 hours. Dependency management with automated alerts.

---

## 14. Reference Architectures

### DoD Training Simulation Environment

```
┌─────────────────────────────────────────────────────────────────┐
│           DOD TRAINING SIMULATION - REFERENCE ARCHITECTURE      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    NIPRNET ENCLAVE                      │   │
│  │                                                         │   │
│  │  ┌───────────────┐    ┌───────────────┐                │   │
│  │  │ Instructor    │    │   Student     │                │   │
│  │  │ Workstations  │    │ Workstations  │                │   │
│  │  │ (NoodleStudio)│    │ (Viewers)     │                │   │
│  │  └───────────────┘    └───────────────┘                │   │
│  │         │                    │                          │   │
│  │         └─────────┬──────────┘                          │   │
│  │                   │                                     │   │
│  │         ┌─────────┴─────────┐                          │   │
│  │         │                   │                          │   │
│  │  ┌──────┴──────┐    ┌──────┴──────┐                   │   │
│  │  │ NoodleMUSH  │    │   Ollama    │                   │   │
│  │  │  (Server)   │    │ (Llama 3.1) │                   │   │
│  │  └─────────────┘    └─────────────┘                   │   │
│  │         │                   │                          │   │
│  │         └─────────┬─────────┘                          │   │
│  │                   │                                     │   │
│  │         ┌─────────┴─────────┐                          │   │
│  │         │    PostgreSQL    │                           │   │
│  │         │    (Encrypted)   │                           │   │
│  │         └───────────────────┘                          │   │
│  │                                                         │   │
│  │  Authentication: CAC/PIV via DoD PKI                   │   │
│  │  Data Classification: CUI                              │   │
│  │  Network: NIPRNET                                      │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Defense Contractor R&D Lab

```
┌─────────────────────────────────────────────────────────────────┐
│      DEFENSE CONTRACTOR R&D - REFERENCE ARCHITECTURE            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────┐  ┌───────────────────────────┐  │
│  │     CORPORATE NETWORK     │  │     ITAR-CONTROLLED       │  │
│  │                           │  │         ENCLAVE           │  │
│  │  ┌───────────────────┐   │  │                           │  │
│  │  │  NoodleStudio     │   │  │  ┌───────────────────┐   │  │
│  │  │  (Unclass work)   │   │  │  │  NoodleStudio     │   │  │
│  │  └───────────────────┘   │  │  │  (ITAR projects)  │   │  │
│  │           │              │  │  └───────────────────┘   │  │
│  │           ▼              │  │           │              │  │
│  │  ┌───────────────────┐   │  │           ▼              │  │
│  │  │  Azure OpenAI     │   │  │  ┌───────────────────┐   │  │
│  │  │  (Commercial)     │   │  │  │   Local Ollama    │   │  │
│  │  └───────────────────┘   │  │  │  (Air-gapped LLM) │   │  │
│  │                           │  │  └───────────────────┘   │  │
│  └───────────────────────────┘  │                           │  │
│                                  │  No internet connection  │  │
│                                  │  No data exfiltration    │  │
│                                  │  US persons only access  │  │
│                                  │                           │  │
│                                  └───────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Federal Agency (FedRAMP High)

```
┌─────────────────────────────────────────────────────────────────┐
│         FEDERAL AGENCY - FEDRAMP HIGH ARCHITECTURE              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    AWS GOVCLOUD (US)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │     ALB     │  │   WAF       │  │   Shield    │     │   │
│  │  │             │  │             │  │   Advanced  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │                    VPC                          │   │   │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │   │   │
│  │  │  │   ECS     │  │   RDS     │  │    S3     │   │   │   │
│  │  │  │ (Fargate) │  │ (Aurora)  │  │(Encrypted)│   │   │   │
│  │  │  └───────────┘  └───────────┘  └───────────┘   │   │   │
│  │  │        │                                        │   │   │
│  │  │        ▼                                        │   │   │
│  │  │  ┌───────────────────────────────────────┐     │   │   │
│  │  │  │         Azure OpenAI                  │     │   │   │
│  │  │  │     (FedRAMP High - East US Gov)      │     │   │   │
│  │  │  └───────────────────────────────────────┘     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  CloudHSM    CloudTrail    GuardDuty    Config  │   │   │
│  │  │  (Keys)      (Audit)       (Threat)     (Compliance)│   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Agency Network ◄──── AWS Direct Connect ────► AWS GovCloud    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Contact

**Caitlyn Meeks**
Principal, Noodlings Technologies, LLC
Email: caitlyn@noodlings.ai
Web: https://noodlings.ai

For secure communications: Signal available upon request.

---

*Document Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY*
*This document contains proprietary information of Noodlings Technologies, LLC.*
*Distribution limited to potential customers under NDA.*

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-05 | C. Meeks / Claude | Initial draft |
