# Version Control, Collaboration & NoodleHub

**Status**: Specification (Draft)
**Date**: 2025-01-05
**Authors**: Caity Meeks, Claude

---

## Overview

This spec covers three interconnected systems:

1. **Git Behind Glass** - Invisible version control managed by Claude
2. **Collaboration** - Team projects with institutional support
3. **NoodleHub** - Public sharing and discovery platform

Design principles:
- **Local-first**: Everything works offline, cloud is optional sync
- **Claude-managed**: Users never see git commands
- **Conversational UI**: "Go back to yesterday's version" not "git reset"
- **Production-grade**: No shortcuts, real git under the hood

---

## Part 1: Git Behind Glass

### Architecture

```
NoodleStudio (PyQt6)
│
├── dulwich (pure Python git)
│   └── Local .git repo in every project
│
├── NoodleCode (Claude agent)
│   └── Manages all git operations conversationally
│
└── Optional remotes
    ├── Noodlings Backend (R2 bundles)
    └── GitHub (direct integration)
```

### Why dulwich

| Factor | Decision |
|--------|----------|
| Pure Python | No native dependencies, easy distribution |
| Cross-platform | Works everywhere Python works |
| Performance | Adequate for YAML-based projects |
| Distribution | `pip install dulwich` - done |

### Project Structure

```
MyProject/
├── .git/                  # Hidden, managed by Claude
├── .noodleignore          # Like .gitignore
├── project.yaml
├── Noodlings/
├── Stages/
├── Prims/
└── Assets/
```

### How Claude Manages Versions

**Automatic commits on meaningful changes:**
- Save project → Claude evaluates what changed
- Generates semantic commit message
- Commits silently in background

```
# What Claude sees internally:
git commit -m "Added memory facet to Bartender noodling"
git commit -m "Tuned attention weights in Player perception"
git commit -m "Created new stage: Tavern"

# What user sees:
Nothing. It just works.
```

**User-initiated version operations:**

| User says | Claude does |
|-----------|-------------|
| "We broke the project" | `git log`, finds last good state, offers restore options |
| "Go back to yesterday" | `git log --since`, shows options, `git checkout` |
| "What changed today?" | `git diff`, summarizes in plain English |
| "Save this as a checkpoint" | `git tag -a "checkpoint-name"` |
| "Show me the history" | `git log`, presents as timeline |
| "Undo the last few changes" | `git revert` or `git reset` as appropriate |

### Version Browser UI (Optional)

For users who want visual history:

```
┌─────────────────────────────────────────────────┐
│ Project History                            [x]  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ● Today, 3:42 PM                              │
│  │ Added emotion system to Bartender           │
│  │                                             │
│  ● Today, 2:15 PM                              │
│  │ Created Tavern stage with 3 zones           │
│  │                                             │
│  ● Yesterday, 6:30 PM                          │
│  │ Initial project setup                       │
│  │                                             │
│  ○ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│    Project created                             │
│                                                 │
│         [Restore Selected]  [Compare]          │
└─────────────────────────────────────────────────┘
```

Purely optional - conversational interface is primary.

### .noodleignore Defaults

```gitignore
# Runtime state (not versioned)
*.pyc
__pycache__/
.noodle_cache/
*.log

# Large binary assets (handled separately)
*.radiance.blob
*.mp4
*.wav

# OS junk
.DS_Store
Thumbs.db

# Secrets
.env
credentials.yaml
```

### Binary Asset Strategy

Large assets (gaussians, video, audio) are problematic in git.

**Options:**
1. **Git LFS** - Store pointers in git, blobs elsewhere
2. **Separate asset sync** - Assets in R2, git tracks manifest
3. **Hybrid** - Small assets in git, large assets separate

**Recommendation**: Start simple (everything in git), add LFS/separation later if projects get huge.

---

## Part 2: Collaboration

### Sharing Model

Every project has a **visibility** setting:

```
┌─────────────────────────────────────────────────┐
│ Sharing                                         │
├─────────────────────────────────────────────────┤
│                                                 │
│  Who can access this project?                   │
│                                                 │
│  ○ Nobody (private)                            │
│      Only you can see and edit                  │
│                                                 │
│  ○ My Organization                             │
│      Members of "SFU Cognitive Arts"            │
│                                                 │
│  ○ Anyone with the link                        │
│      Unlisted on NoodleHub, but shareable       │
│                                                 │
│  ○ Public on NoodleHub                         │
│      Discoverable by anyone                     │
│                                                 │
│  ─────────────────────────────────────────────  │
│                                                 │
│  Collaborators (can edit):                      │
│  ┌─────────────────────────────────────────┐   │
│  │ + Add people...                         │   │
│  │                                         │   │
│  │ Sarah Chen (owner)              [Admin] │   │
│  │ Marcus Webb                      [Edit] │   │
│  │ Prof. DiPaola                    [View] │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Collaborator Roles

| Role | Can View | Can Edit | Can Share | Can Delete | Can Transfer |
|------|----------|----------|-----------|------------|--------------|
| Viewer | Yes | No | No | No | No |
| Editor | Yes | Yes | No | No | No |
| Admin | Yes | Yes | Yes | No | No |
| Owner | Yes | Yes | Yes | Yes | Yes |

### Fork Model

When visibility allows, users can **fork** a project:

```
User: "I want to build on Sarah's Bartender project"

Claude: "Sarah's 'Emotional Bartender' is shared with your
        organization. I can:

        1. Fork it - Create your own copy to modify freely
        2. Request collaboration - Ask Sarah to add you as editor

        Which would you prefer?"
```

**Fork creates:**
- Full copy of project at current state
- New git history starting from fork point
- Original attribution preserved in project.yaml
- No ongoing sync with original (it's your copy now)

### Sync Protocol

**For shared projects (not forks):**

```
┌──────────────┐     push bundle      ┌──────────────┐
│   Client A   │ ──────────────────>  │   R2 Storage │
│  (dulwich)   │                      │  (git bundles)│
└──────────────┘                      └──────────────┘
                                             │
                                             │ pull bundle
                                             v
                                      ┌──────────────┐
                                      │   Client B   │
                                      │  (dulwich)   │
                                      └──────────────┘
```

**Bundle-based sync:**
- No git server to maintain
- R2 is just dumb object storage ($0.015/GB/month)
- Each push creates incremental bundle of new commits
- Pull downloads and applies bundles
- D1 tracks sync state per project

**Sync state table (D1):**

```sql
CREATE TABLE project_sync (
  project_id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  visibility TEXT NOT NULL,  -- private|org|link|public
  org_id TEXT,               -- if org-owned
  latest_bundle_hash TEXT,
  latest_bundle_url TEXT,    -- R2 URL
  updated_at TEXT,
  FOREIGN KEY (owner_id) REFERENCES users(id),
  FOREIGN KEY (org_id) REFERENCES organizations(id)
);

CREATE TABLE project_collaborators (
  project_id TEXT,
  user_id TEXT,
  role TEXT NOT NULL,  -- viewer|editor|admin
  added_at TEXT,
  added_by TEXT,
  PRIMARY KEY (project_id, user_id)
);
```

### Conflict Resolution

**Claude handles merges conversationally:**

```
Claude: "I synced with the team's changes. Marcus updated the
        Bartender while you were working on it too.

        His changes:
        - Added backstory about lost love
        - Changed greeting dialogue

        Your changes:
        - Made responses shorter and grumpier
        - Added new memory facet

        The memory facet merges cleanly. But the personality
        changes conflict. How should I handle it?

        1. Keep yours (discard Marcus's personality changes)
        2. Keep his (discard your personality changes)
        3. Try to blend both (grumpy + backstory)
        4. Let me look at both versions"
```

**For YAML files, semantic merge is often possible:**
- Different keys = auto-merge
- Same key, different values = conflict
- Claude can often suggest intelligent blends

### GitHub Integration (Optional)

Power users can connect GitHub:

```
User: "Use GitHub for this project"

Claude: "I'll connect to GitHub. Do you want to:

        1. Create new repo (noodlings-emotional-bartender)
        2. Use existing repo (pick from your repos)
        3. Connect to organization repo"
```

When GitHub is connected:
- Push/pull goes to GitHub directly
- Can use GitHub's collaboration features
- Still wrapped in Claude's conversational interface

---

## Part 3: Institutional Accounts

### Organization Model

```
Organization (e.g., "SFU Cognitive Arts")
│
├── Settings
│   ├── Name, logo, description
│   ├── Credit pool balance
│   ├── Default member credit limit
│   └── Auto top-up rules
│
├── Members
│   ├── Admins (full control)
│   ├── Managers (manage members, view usage)
│   └── Members (use credits, create projects)
│
├── Projects
│   ├── Org-owned (belong to org, not individual)
│   └── Member projects (shared with org)
│
└── Billing
    ├── Credit purchases
    ├── Usage by member
    └── Export reports
```

### Organization Roles

| Role | Manage Members | Manage Credits | View All Projects | Org Settings | Billing |
|------|----------------|----------------|-------------------|--------------|---------|
| Member | No | No | Shared only | No | No |
| Manager | Yes | View only | Yes | No | View |
| Admin | Yes | Yes | Yes | Yes | Yes |
| Owner | Yes | Yes | Yes | Yes | Yes |

### Member Credit System

```
┌─────────────────────────────────────────────────┐
│ Member: Sarah Chen                              │
├─────────────────────────────────────────────────┤
│                                                 │
│  Credit Balance: 3,240 / 5,000 monthly         │
│  ████████████████░░░░░░░ 64%                   │
│                                                 │
│  Settings:                                      │
│  ┌─────────────────────────────────────────┐   │
│  │ Monthly limit:        [5,000    ] ▼     │   │
│  │ Auto top-up:          [x] Enabled       │   │
│  │ Top-up threshold:     [1,000    ]       │   │
│  │ Top-up amount:        [2,000    ]       │   │
│  │ Hard cap:             [ ] No limit      │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  This month's usage:                           │
│  • LLM calls: 1,580 credits                    │
│  • Storage: 180 credits                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Credit flow:**
1. Org purchases credits in bulk (volume discounts?)
2. Admin allocates to members (monthly limits)
3. Members consume against their allocation
4. Auto top-up pulls from org pool if enabled
5. Hard cap prevents runaway usage

### Professor DiPaola's View

```
┌─────────────────────────────────────────────────┐
│ SFU Cognitive Arts                    [Admin]   │
├─────────────────────────────────────────────────┤
│                                                 │
│  Org Credits: 247,000 remaining                │
│  Members: 34 active                             │
│  Projects: 89 total (12 org-owned)             │
│                                                 │
│  ─────────────────────────────────────────────  │
│                                                 │
│  Quick Actions:                                 │
│  [Add Members]  [Purchase Credits]  [Reports]  │
│                                                 │
│  ─────────────────────────────────────────────  │
│                                                 │
│  Recent Activity:                              │
│  • Sarah C. created "Assignment 3 Submission"  │
│  • Marcus W. forked "Emotional NPC Starter"    │
│  • Wei L. used 450 credits on LLM calls        │
│                                                 │
│  ─────────────────────────────────────────────  │
│                                                 │
│  Member Usage (this month):                    │
│  ┌─────────────────────────────────────────┐   │
│  │ Name            Used    Limit   Status  │   │
│  │ Sarah Chen      3,240   5,000   OK      │   │
│  │ Marcus Webb     4,891   5,000   Warning │   │
│  │ Wei Liu         1,200   5,000   OK      │   │
│  │ ...                                     │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Database Schema (D1 additions)

```sql
-- Organizations
CREATE TABLE organizations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  owner_id TEXT NOT NULL,
  credit_balance INTEGER DEFAULT 0,
  default_member_limit INTEGER DEFAULT 5000,
  created_at TEXT,
  FOREIGN KEY (owner_id) REFERENCES users(id)
);

-- Organization membership
CREATE TABLE org_members (
  org_id TEXT,
  user_id TEXT,
  role TEXT NOT NULL,  -- member|manager|admin|owner
  credit_limit INTEGER,
  credit_used_this_month INTEGER DEFAULT 0,
  auto_topup_enabled INTEGER DEFAULT 0,
  auto_topup_threshold INTEGER,
  auto_topup_amount INTEGER,
  joined_at TEXT,
  PRIMARY KEY (org_id, user_id)
);

-- Organization credit transactions
CREATE TABLE org_credit_transactions (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL,
  user_id TEXT,           -- NULL for org-level purchases
  amount INTEGER NOT NULL, -- positive = add, negative = use
  type TEXT NOT NULL,      -- purchase|allocation|usage|refund
  description TEXT,
  created_at TEXT,
  FOREIGN KEY (org_id) REFERENCES organizations(id)
);

-- Organization-owned projects
CREATE TABLE org_projects (
  project_id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL,
  created_by TEXT NOT NULL,
  created_at TEXT,
  FOREIGN KEY (org_id) REFERENCES organizations(id)
);
```

---

## Part 4: NoodleHub

### What is NoodleHub?

A public platform for discovering and sharing NoodleStudio creations:
- Browse public noodlings, stages, facet assemblies
- Fork projects to start your own
- Follow creators
- Featured/trending content

### NoodleHub URL Structure

```
noodlehub.noodlings.ai/
├── /explore                    # Browse all public content
├── /explore/noodlings          # Just noodlings
├── /explore/stages             # Just stages
├── /explore/assemblies         # Just facet assemblies
├── /@username                  # Creator profile
├── /@username/project-slug     # Project page
├── /orgs/sfu-cognitive-arts    # Organization page
└── /join/abc123                # Join link (unlisted projects)
```

### Project Page

```
┌─────────────────────────────────────────────────────────────┐
│ noodlehub.noodlings.ai/@sarahchen/emotional-bartender       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Emotional Bartender                                        │
│  by Sarah Chen (@sarahchen)                                │
│                                                             │
│  A bartender NPC with dynamic emotional responses,          │
│  memory of regular customers, and context-aware dialogue.   │
│                                                             │
│  [Open in NoodleStudio]    [Fork]    [Download]            │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  Preview:                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │     [Interactive demo or video here]               │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  Includes:                                                  │
│  • 1 Noodling (Bartender)                                  │
│  • 1 Stage (Tavern with 3 zones)                           │
│  • 3 Facet Assemblies (memory, emotion, dialogue)          │
│  • 2 Gaussian avatars                                       │
│                                                             │
│  Stats: 234 forks • 1.2k views • Updated 2 days ago        │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  README                                                     │
│                                                             │
│  This bartender demonstrates how to use the continuous     │
│  affect system to create emotionally responsive NPCs...    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deep Links

**noodle:// protocol handler:**

```
noodle://join/abc123           → Open NoodleStudio, join project
noodle://fork/sarahchen/bartender  → Fork and open
noodle://open/local/path       → Open local project
```

Registered when NoodleStudio installs, so clicking links in browser opens the app.

### Discovery & Curation

**Browse categories:**
- Noodlings (AI characters)
- Stages (worlds/environments)
- Facet Assemblies (cognitive components)
- Full Projects (complete packages)
- Templates (starting points)

**Sorting:**
- Recent
- Popular (forks + views)
- Featured (staff picks)

**Filtering:**
- By creator
- By organization
- By tags
- By license

### Licensing

Creators choose a license when making public:

```
┌─────────────────────────────────────────────────┐
│ License                                         │
├─────────────────────────────────────────────────┤
│                                                 │
│  How can others use your project?               │
│                                                 │
│  ○ Open (CC0)                                  │
│      No restrictions, public domain             │
│                                                 │
│  ○ Attribution (CC-BY)                         │
│      Free to use with credit                    │
│                                                 │
│  ○ Non-Commercial (CC-BY-NC)                   │
│      Free for non-commercial use               │
│                                                 │
│  ○ Share-Alike (CC-BY-SA)                      │
│      Derivatives must use same license          │
│                                                 │
│  ○ Personal Use Only                           │
│      View and learn, no redistribution          │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Fork Attribution

When you fork, the lineage is preserved:

```yaml
# project.yaml in forked project
project:
  name: "My Improved Bartender"
  forked_from:
    creator: "sarahchen"
    project: "emotional-bartender"
    forked_at: "2025-01-05T14:32:00Z"
    commit: "a1b2c3d4"
    license: "CC-BY"
```

NoodleHub displays fork trees:

```
emotional-bartender (sarahchen) - Original
├── grumpy-bartender (marcuswebb) - Fork
│   └── angry-bartender (student42) - Fork of fork
├── friendly-bartender (weiliu) - Fork
└── robot-bartender (janesmith) - Fork
```

---

## Part 5: API Endpoints

### Version Control

```
# Project sync (authenticated)
POST   /v1/projects/:id/push     # Upload bundle
GET    /v1/projects/:id/pull     # Download latest bundle
GET    /v1/projects/:id/history  # Get commit history

# Request body for push:
{
  "bundle": "<base64 git bundle>",
  "from_commit": "abc123",
  "to_commit": "def456",
  "message": "Added memory facet"
}
```

### Collaboration

```
# Project sharing
GET    /v1/projects/:id/collaborators
POST   /v1/projects/:id/collaborators
DELETE /v1/projects/:id/collaborators/:user_id
PATCH  /v1/projects/:id/visibility

# Invitations
POST   /v1/projects/:id/invite    # Generate join link
GET    /v1/join/:code             # Resolve join link
POST   /v1/join/:code/accept      # Accept invitation
```

### Organizations

```
# Organization management
POST   /v1/orgs                   # Create org
GET    /v1/orgs/:slug             # Get org details
PATCH  /v1/orgs/:slug             # Update org
DELETE /v1/orgs/:slug             # Delete org

# Members
GET    /v1/orgs/:slug/members
POST   /v1/orgs/:slug/members
PATCH  /v1/orgs/:slug/members/:user_id
DELETE /v1/orgs/:slug/members/:user_id

# Credits
GET    /v1/orgs/:slug/credits
POST   /v1/orgs/:slug/credits/purchase
POST   /v1/orgs/:slug/credits/allocate
GET    /v1/orgs/:slug/credits/usage
```

### NoodleHub

```
# Discovery
GET    /v1/hub/explore           # Browse public projects
GET    /v1/hub/explore/noodlings
GET    /v1/hub/explore/stages
GET    /v1/hub/explore/assemblies
GET    /v1/hub/featured

# Project pages
GET    /v1/hub/@:username/:slug  # Get public project
POST   /v1/hub/@:username/:slug/fork
GET    /v1/hub/@:username/:slug/forks

# Profiles
GET    /v1/hub/@:username        # Creator profile
GET    /v1/hub/orgs/:slug        # Org public page
```

---

## Part 6: Implementation Phases

### Phase 1: Local Git (Foundation)

- [ ] Integrate dulwich into NoodleStudio
- [ ] Auto-init .git on project create
- [ ] Claude commit on save (semantic messages)
- [ ] Basic conversational commands (history, restore, diff)
- [ ] .noodleignore handling

### Phase 2: Cloud Sync (Personal)

- [ ] R2 bundle storage
- [ ] Push/pull endpoints
- [ ] Sync state in D1
- [ ] Offline queue (sync when reconnected)
- [ ] Conflict detection

### Phase 3: Collaboration

- [ ] Project visibility settings
- [ ] Collaborator management
- [ ] Join links
- [ ] Fork functionality
- [ ] Conflict resolution UI (Claude-mediated)

### Phase 4: Institutions

- [ ] Organization CRUD
- [ ] Member management
- [ ] Role-based permissions
- [ ] Credit allocation per member
- [ ] Auto top-up system
- [ ] Usage reports

### Phase 5: NoodleHub

- [ ] Public project listings
- [ ] Project pages
- [ ] Search/browse
- [ ] noodle:// protocol handler
- [ ] Fork trees
- [ ] Licensing system

### Phase 6: Community & Sales

- [ ] Star ratings
- [ ] Comments system
- [ ] Notifications
- [ ] Sales dashboard for institutions
- [ ] Volume discount management
- [ ] Invoice generation

### Phase 7: Private Hubs

- [ ] Custom subdomains
- [ ] Team management within orgs
- [ ] Team-scoped visibility
- [ ] Org admin content curation
- [ ] Private hub branding (logo, colors)

### Phase 8: Secure Tier

- [ ] Audit logging infrastructure
- [ ] Data residency controls
- [ ] CAC/PIV authentication
- [ ] On-premise deployment package
- [ ] Air-gapped licensing system
- [ ] GovCloud deployment scripts

### Phase 9: Polish

- [ ] GitHub integration
- [ ] Large asset handling (LFS or separate)
- [ ] Real-time collaboration (maybe)
- [ ] Mobile NoodleHub browser
- [ ] Verified creator badges

---

## Part 7: Sales & Billing Dashboard

### Institutional Sales Flow

When Caity closes a deal with an institution (SFU, game studio, etc.):

1. **Create organization** in admin dashboard
2. **Apply volume discount** (negotiated per deal)
3. **Charge and credit** the agreed amount
4. **Set up admin accounts** for their people

### Admin Dashboard: Sales View

```
┌─────────────────────────────────────────────────────────────┐
│ Admin > Sales > New Institutional Account                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Organization Details                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Name:             [Simon Fraser University      ]   │   │
│  │ Slug:             [sfu-cognitive-arts           ]   │   │
│  │ Contact Email:    [sdipaola@sfu.ca              ]   │   │
│  │ Contact Name:     [Prof. Steve DiPaola          ]   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Credit Package                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Base credits:     [1,000,000    ]                   │   │
│  │ Unit price:       $0.01 (1 credit = $0.01)          │   │
│  │ Base total:       $10,000.00                        │   │
│  │                                                     │   │
│  │ Volume discount:  [15        ] %                    │   │
│  │ Discount amount:  -$1,500.00                        │   │
│  │ ─────────────────────────────────────────────────── │   │
│  │ Final charge:     $8,500.00                         │   │
│  │ Credits granted:  1,000,000                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Notes (internal)                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Academic partnership. Annual renewal expected.      │   │
│  │ Contact via email, responds within 48h.             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [Preview Invoice]        [Create & Charge]                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Volume Discount Tiers (Suggested)

| Credits | Discount | Effective Rate |
|---------|----------|----------------|
| < 100k | 0% | $0.0100 |
| 100k - 500k | 5% | $0.0095 |
| 500k - 1M | 10% | $0.0090 |
| 1M - 5M | 15% | $0.0085 |
| 5M+ | 20% | $0.0080 |
| Custom | Negotiated | Per deal |

**Note**: Sales dashboard allows overriding with custom discount per deal.

### Credit Top-Up (Existing Orgs)

```
┌─────────────────────────────────────────────────────────────┐
│ Admin > SFU Cognitive Arts > Add Credits                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Current balance: 47,000 credits                           │
│  Discount rate: 15% (established)                          │
│                                                             │
│  Add credits:     [500,000       ]                         │
│  Charge amount:   $4,250.00                                │
│                                                             │
│  [Charge & Credit]                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Sales Database Schema

```sql
-- Institutional deals
CREATE TABLE institutional_deals (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL,
  credits_purchased INTEGER NOT NULL,
  discount_percent REAL NOT NULL,
  base_amount_usd REAL NOT NULL,
  discount_amount_usd REAL NOT NULL,
  final_amount_usd REAL NOT NULL,
  payment_method TEXT,           -- invoice|card|wire
  payment_status TEXT,           -- pending|paid|failed
  notes TEXT,
  created_by TEXT NOT NULL,      -- admin user id
  created_at TEXT,
  FOREIGN KEY (org_id) REFERENCES organizations(id)
);

-- Organization billing settings
ALTER TABLE organizations ADD COLUMN discount_percent REAL DEFAULT 0;
ALTER TABLE organizations ADD COLUMN billing_email TEXT;
ALTER TABLE organizations ADD COLUMN billing_contact TEXT;
ALTER TABLE organizations ADD COLUMN internal_notes TEXT;
```

---

## Part 8: Moderation

### Moderation Actions

Admins can take these actions on any public content:

| Action | Effect | Reversible? |
|--------|--------|-------------|
| **Hide** | Removed from browse/search, still accessible via direct link | Yes |
| **Delete** | Fully removed, R2 bundles deleted | No |
| **Change visibility** | Force to private/unlisted | Yes |
| **Warn creator** | Send warning email, log incident | N/A |
| **Suspend creator** | Disable account temporarily | Yes |

### Moderation Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│ Admin > Moderation > Reported Content                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Pending Reports (3)                                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ "Offensive NPC" by @badactor                        │   │
│  │ Reported: 2 times                                   │   │
│  │ Reason: Inappropriate content                       │   │
│  │                                                     │   │
│  │ [View Project]  [Hide]  [Delete]  [Dismiss]        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  Quick Actions (any project):                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Project URL or ID: [                            ]   │   │
│  │                                                     │   │
│  │ [Hide]  [Set Private]  [Delete]                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Report System (User-Facing)

On any public project page:

```
┌─────────────────────────────────────────────────┐
│ Report this project                             │
├─────────────────────────────────────────────────┤
│                                                 │
│  Why are you reporting this?                    │
│                                                 │
│  ○ Inappropriate or offensive content          │
│  ○ Copyright violation                         │
│  ○ Spam or misleading                          │
│  ○ Other: [                              ]     │
│                                                 │
│  Additional details (optional):                │
│  ┌─────────────────────────────────────────┐   │
│  │                                         │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  [Submit Report]                               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Moderation Log

All actions are logged for accountability:

```sql
CREATE TABLE moderation_log (
  id TEXT PRIMARY KEY,
  target_type TEXT NOT NULL,     -- project|user|comment
  target_id TEXT NOT NULL,
  action TEXT NOT NULL,          -- hide|delete|visibility|warn|suspend
  reason TEXT,
  admin_id TEXT NOT NULL,
  created_at TEXT
);

CREATE TABLE content_reports (
  id TEXT PRIMARY KEY,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  reporter_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  details TEXT,
  status TEXT DEFAULT 'pending', -- pending|reviewed|dismissed
  reviewed_by TEXT,
  reviewed_at TEXT,
  created_at TEXT
);
```

---

## Part 9: Community Features

### Star Ratings (GitHub-style)

Users can star projects they like:

```
┌─────────────────────────────────────────────────┐
│ Emotional Bartender              ★ 234  [Star] │
└─────────────────────────────────────────────────┘
```

- One star per user per project
- Stars contribute to "Popular" sorting
- Creators see who starred (maybe? or keep anonymous?)

### Comments

Threaded comments on project pages:

```
┌─────────────────────────────────────────────────────────────┐
│ Comments (12)                                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  @marcuswebb • 2 days ago                                  │
│  This is exactly what I needed for my game! The memory     │
│  system is really well designed. Quick question - how do   │
│  you handle the case where a customer returns after the    │
│  bartender's shift change?                                 │
│                                          [Reply] [Report]  │
│                                                             │
│    ↳ @sarahchen • 2 days ago                               │
│      Good question! The memory is stored in the stage      │
│      state, so it persists across noodling instances.      │
│      Check the memory facet config for the decay settings. │
│                                          [Reply] [Report]  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  @student42 • 1 day ago                                    │
│  I forked this and added a grumpy mode toggle!            │
│                                          [Reply] [Report]  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Add a comment...                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│  [Post Comment]                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Community Database Schema

```sql
-- Stars
CREATE TABLE project_stars (
  project_id TEXT,
  user_id TEXT,
  created_at TEXT,
  PRIMARY KEY (project_id, user_id)
);

-- Add star count to projects for fast queries
ALTER TABLE project_sync ADD COLUMN star_count INTEGER DEFAULT 0;

-- Comments
CREATE TABLE project_comments (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  parent_id TEXT,              -- NULL for top-level, comment_id for replies
  content TEXT NOT NULL,
  created_at TEXT,
  updated_at TEXT,
  deleted_at TEXT,             -- soft delete
  FOREIGN KEY (parent_id) REFERENCES project_comments(id)
);

-- Comment notifications
CREATE TABLE notifications (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  type TEXT NOT NULL,          -- comment|reply|star|fork
  source_user_id TEXT,
  target_type TEXT,
  target_id TEXT,
  read_at TEXT,
  created_at TEXT
);
```

---

## Part 10: Private NoodleHub (Enterprise)

### The Concept

Large organizations (game studios, universities, enterprises) want their own internal NoodleHub:
- Private to their organization
- Same features as public NoodleHub
- Isolated from public content
- Custom branding (maybe)
- Full admin control

### How It Works

**NOT self-hosted** (too complex). Instead: **tenant isolation** on our infrastructure.

```
┌─────────────────────────────────────────────────────────────┐
│                    Noodlings Infrastructure                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Public NoodleHub          Private Hubs                     │
│  ┌─────────────┐          ┌─────────────┐                  │
│  │ noodlehub.  │          │ sfu.noodle  │ ← SFU only       │
│  │ noodlings.ai│          │ hub.ai      │                  │
│  └─────────────┘          ├─────────────┤                  │
│         │                 │ ubisoft.    │ ← Ubisoft only   │
│         │                 │ noodlehub.ai│                  │
│         │                 └─────────────┘                  │
│         │                        │                          │
│         └────────────┬───────────┘                          │
│                      │                                      │
│              Same R2, D1, Workers                           │
│              (data isolated by org_id)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Private Hub Features

| Feature | Public Hub | Private Hub |
|---------|------------|-------------|
| Browse org projects | Via /orgs/slug | Full hub experience |
| Visibility options | Public/link/org/private | Org/team/private (no public) |
| Custom subdomain | No | Yes (sfu.noodlehub.ai) |
| Custom branding | No | Logo, colors (future) |
| Featured content | Staff picks | Org admin picks |
| User base | All users | Org members only |

### Access Control

```
User visits sfu.noodlehub.ai
        │
        ▼
   Authenticated?
        │
   ┌────┴────┐
   │         │
   No        Yes
   │         │
   ▼         ▼
Login     Member of SFU org?
page          │
          ┌───┴───┐
          │       │
          No      Yes
          │       │
          ▼       ▼
       "Access    Show
       denied"    hub
```

### Private Hub URL Structure

```
sfu.noodlehub.ai/
├── /explore                    # Browse org's shared projects
├── /explore/noodlings
├── /explore/stages
├── /explore/assemblies
├── /@username                  # Member profile (within org)
├── /@username/project-slug     # Project page
├── /teams/graphics             # Team page (sub-groups)
└── /featured                   # Org admin's picks
```

### Teams Within Orgs

For large orgs, projects can be shared with specific teams:

```
Organization: Ubisoft Montreal
├── Team: AI Characters
│   └── Shared projects for AI team
├── Team: Level Design
│   └── Shared projects for LD team
├── Team: Audio
│   └── Shared projects for audio team
└── All Members
    └── Org-wide shared projects
```

**Visibility options in Private Hub:**
- Private (just me)
- My Team (specific team)
- All Teams (whole org)

### Database Additions

```sql
-- Teams within orgs
CREATE TABLE org_teams (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL,
  name TEXT NOT NULL,
  slug TEXT NOT NULL,
  description TEXT,
  created_at TEXT,
  UNIQUE(org_id, slug),
  FOREIGN KEY (org_id) REFERENCES organizations(id)
);

-- Team membership
CREATE TABLE team_members (
  team_id TEXT,
  user_id TEXT,
  role TEXT DEFAULT 'member',  -- member|lead
  joined_at TEXT,
  PRIMARY KEY (team_id, user_id)
);

-- Project team sharing
CREATE TABLE project_team_access (
  project_id TEXT,
  team_id TEXT,
  PRIMARY KEY (project_id, team_id)
);

-- Private hub settings
ALTER TABLE organizations ADD COLUMN private_hub_enabled INTEGER DEFAULT 0;
ALTER TABLE organizations ADD COLUMN private_hub_subdomain TEXT UNIQUE;
ALTER TABLE organizations ADD COLUMN private_hub_logo_url TEXT;
ALTER TABLE organizations ADD COLUMN private_hub_accent_color TEXT;
```

### Pricing Model (Suggestion)

| Tier | Price | Includes |
|------|-------|----------|
| **Standard Org** | Credits only | Shared projects, basic collab |
| **Private Hub** | +$500/mo | Custom subdomain, full hub UI |
| **Enterprise** | Custom | SSO, custom branding, SLA |
| **Secure/MS&T** | Premium custom | See Part 11 |

---

## Part 11: Secure Tier (MS&T / Defense / Gov)

### The Market

Military, defense contractors, government agencies, and security-conscious enterprises need:
- Data sovereignty (know exactly where data lives)
- Air-gapped or isolated infrastructure
- Compliance certifications (FedRAMP, ITAR, SOC2, etc.)
- Audit trails for everything
- No data leaving their security perimeter

These clients have **deep pockets** and will pay significant premiums for verified security.

### Deployment Options

**Option A: Dedicated Tenant (Our Infrastructure)**
```
┌─────────────────────────────────────────────────────────────┐
│           Cloudflare (US-only data centers)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Isolated Workers    Isolated D1      Isolated R2          │
│  ┌─────────────┐    ┌───────────┐    ┌───────────┐        │
│  │ client.     │    │ client_   │    │ client_   │        │
│  │ noodlings   │    │ db        │    │ storage   │        │
│  └─────────────┘    └───────────┘    └───────────┘        │
│        │                  │                │               │
│        └──────────────────┼────────────────┘               │
│                           │                                │
│                    Zero shared resources                   │
│                    with other tenants                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- Separate Cloudflare account/zone
- Dedicated D1 database (not shared tables)
- Dedicated R2 bucket
- US-only data residency
- SOC2 Type II via Cloudflare

**Option B: On-Premise / Private Cloud**
```
┌─────────────────────────────────────────────────────────────┐
│              Client's Infrastructure                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NoodleMUSH Server     NoodleHub (self-hosted)             │
│  ┌─────────────┐      ┌─────────────────────┐             │
│  │ Docker or   │      │ Static site +       │             │
│  │ bare metal  │      │ internal API        │             │
│  └─────────────┘      └─────────────────────┘             │
│         │                      │                           │
│         └──────────┬───────────┘                           │
│                    │                                        │
│  ┌─────────────────┴─────────────────┐                     │
│  │        PostgreSQL / SQLite        │                     │
│  │        (client-managed DB)        │                     │
│  └───────────────────────────────────┘                     │
│                                                             │
│  LLM: Local Ollama / Azure Gov / AWS GovCloud             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- Runs entirely on client infrastructure
- Air-gapped option (no internet required)
- Client provides own LLM (Ollama, Azure OpenAI Gov, etc.)
- We provide deployment package + support contract

**Option C: GovCloud**
```
AWS GovCloud / Azure Government
├── NoodleMUSH on EC2/VM
├── NoodleHub on S3/Blob + Lambda/Functions
├── Database on RDS/Azure SQL
└── LLM via Azure OpenAI (FedRAMP authorized)
```

- FedRAMP High authorized infrastructure
- ITAR compliant
- We manage deployment, they own infrastructure

### Security Features (All Secure Tiers)

| Feature | Standard | Secure |
|---------|----------|--------|
| Data encryption at rest | Yes | Yes (customer-managed keys) |
| Data encryption in transit | TLS 1.3 | TLS 1.3 + mTLS option |
| SSO | SAML/OIDC | SAML/OIDC + CAC/PIV |
| Audit logging | Basic | Comprehensive (every action) |
| Data residency | Best effort | Guaranteed (US/EU/specific) |
| Backup location | Our choice | Customer-specified |
| Penetration testing | Annual | Customer can conduct own |
| Compliance reports | SOC2 via Cloudflare | SOC2 + custom attestations |
| Incident response SLA | Best effort | Contractual (4hr/24hr) |
| Dedicated support | No | Named account manager |

### Air-Gapped Mode

For truly isolated environments:

```
┌─────────────────────────────────────────────────────────────┐
│                   Secure Facility (No Internet)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NoodleStudio ──────> NoodleMUSH ──────> Local LLM         │
│  (desktop)            (server)           (Ollama/vLLM)     │
│       │                   │                   │             │
│       └───────────────────┼───────────────────┘             │
│                           │                                 │
│                    Local NoodleHub                          │
│                    (internal only)                          │
│                                                             │
│  Updates: Sneakernet (USB) with signed packages            │
│  Licensing: Offline activation with hardware key           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Air-gapped requirements:**
- Offline license validation (cryptographic, time-limited)
- Update packages signed and verifiable
- No telemetry, no phone-home
- Local-only everything

### Audit Trail

Every action logged with:
```json
{
  "timestamp": "2025-01-05T14:32:00.000Z",
  "user_id": "usr_abc123",
  "user_email": "jsmith@contractor.mil",
  "action": "project.create",
  "resource_type": "project",
  "resource_id": "prj_def456",
  "ip_address": "10.0.1.42",
  "user_agent": "NoodleStudio/1.2.3",
  "session_id": "sess_xyz789",
  "org_id": "org_securedefense",
  "details": {
    "project_name": "Mission Planning Prototype",
    "visibility": "private"
  }
}
```

Logs exportable in:
- JSON (for SIEM ingestion)
- CSV (for compliance review)
- Syslog format (for existing infrastructure)

### CAC/PIV Authentication

For DoD clients, support Common Access Card authentication:

```
┌─────────────────────────────────────────────────────────────┐
│ NoodleStudio Login                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Authentication Method:                                     │
│                                                             │
│  ○ Username / Password                                     │
│  ○ SSO (SAML)                                              │
│  ● CAC / PIV Card                                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │     Insert your CAC card and enter PIN             │   │
│  │                                                     │   │
│  │     [••••••]  [Authenticate]                       │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pricing (Ballpark)

| Option | Setup | Monthly | Notes |
|--------|-------|---------|-------|
| **Dedicated Tenant** | $10k | $2-5k | Isolated on our infra |
| **GovCloud Managed** | $25k | $5-10k | We deploy on their GovCloud |
| **On-Premise** | $50k+ | Support contract | They run everything |
| **Air-Gapped** | $75k+ | Support contract | Maximum isolation |

Plus credits at standard rates (or bundled).

**Value prop**: Compare to building cognitive AI tooling in-house. We're offering a turnkey platform that would cost $500k+ to develop internally.

### Compliance Roadmap

| Certification | Status | Timeline |
|---------------|--------|----------|
| SOC2 Type II | Via Cloudflare | Now |
| FedRAMP Moderate | Not started | If customer demand |
| FedRAMP High | Not started | Requires GovCloud |
| ITAR | Architecture supports | Per engagement |
| IL4/IL5 | Not started | Requires specific infra |

**Strategy**: Don't pursue certifications speculatively. Let customer demand and contracts fund the compliance work.

### Database Schema Additions

```sql
-- Audit log (secure tier)
CREATE TABLE audit_log (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  user_id TEXT,
  user_email TEXT,
  action TEXT NOT NULL,
  resource_type TEXT,
  resource_id TEXT,
  ip_address TEXT,
  user_agent TEXT,
  session_id TEXT,
  org_id TEXT,
  details TEXT,  -- JSON
  created_at TEXT
);

-- Create index for compliance queries
CREATE INDEX idx_audit_user ON audit_log(user_id, timestamp);
CREATE INDEX idx_audit_org ON audit_log(org_id, timestamp);
CREATE INDEX idx_audit_action ON audit_log(action, timestamp);

-- Secure org settings
ALTER TABLE organizations ADD COLUMN security_tier TEXT DEFAULT 'standard';
ALTER TABLE organizations ADD COLUMN data_residency TEXT;
ALTER TABLE organizations ADD COLUMN audit_log_enabled INTEGER DEFAULT 0;
ALTER TABLE organizations ADD COLUMN sso_required INTEGER DEFAULT 0;
ALTER TABLE organizations ADD COLUMN allowed_auth_methods TEXT;  -- JSON array
```

### NoodleStudio Integration

When user belongs to org with Private Hub:

```
┌─────────────────────────────────────────────────┐
│ Share Project                                   │
├─────────────────────────────────────────────────┤
│                                                 │
│  Share to:                                      │
│                                                 │
│  ○ Private (just me)                           │
│                                                 │
│  ○ My Team (AI Characters)                     │
│      Visible to 12 team members                 │
│                                                 │
│  ○ Ubisoft Montreal (Private Hub)              │
│      Visible to all 340 org members             │
│      Browsable at ubisoft.noodlehub.ai         │
│                                                 │
│  ○ Public NoodleHub                            │
│      ⚠️ Visible to everyone on the internet    │
│      Requires org admin approval               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Admin Controls for Private Hub

Org admins can:
- Enable/disable public sharing (force private-only)
- Set default visibility for new projects
- Feature projects on their hub's front page
- Manage teams
- View all org projects
- Moderate content within their hub

```
┌─────────────────────────────────────────────────────────────┐
│ Org Settings > Private Hub                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Private Hub: [x] Enabled                                  │
│  Subdomain:   ubisoft.noodlehub.ai                         │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  Policies:                                                  │
│  [x] Allow members to share publicly on NoodleHub          │
│      (requires admin approval)                              │
│  [ ] Require all projects to be private                    │
│  [x] Auto-add new members to "General" team                │
│                                                             │
│  Default visibility for new projects:                      │
│  ( ) Private  (•) Team  ( ) Org                            │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  Featured Projects (shown on hub front page):              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ + Add featured project...                           │   │
│  │                                                     │   │
│  │ 1. Emotional NPC Starter Kit (template)            │   │
│  │ 2. Q4 Demo - Tavern Scene                          │   │
│  │ 3. Best Practices: Memory Systems                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Resolved Decisions

| Question | Decision |
|----------|----------|
| Volume discounts | Yes - tiered discounts, sales dashboard for custom deals |
| Moderation | Hide/delete/change visibility + report system |
| Verified badges | Yes, but later implementation phase |
| Comments/ratings | Star ratings (GitHub-style) + threaded comments |
| Private NoodleHub | Tenant isolation on our infra, custom subdomains |
| MS&T / Secure tier | Yes - dedicated tenant, on-prem, GovCloud, air-gapped options |

---

## References

- [NoodleStudio Architecture](../architecture.md)
- [Backend Overview](../backend/overview.md)
- [LLM Routing Service](./llm-routing-service.md)
- [dulwich Documentation](https://www.dulwich.io/)
- [Git Bundle Format](https://git-scm.com/docs/git-bundle)
