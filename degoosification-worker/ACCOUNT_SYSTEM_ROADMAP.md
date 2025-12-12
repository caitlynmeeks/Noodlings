# Account System Roadmap

**Current Status:** Email collection via degoosification (Phase 1)
**Next:** Full account system for Asset Store (Phase 2)

---

## Phase 1: Email Collection (DONE ✅)

**What we have NOW:**
- Email collection via degoosification
- Cloudflare KV storage (90-day expiration)
- 1 email registered so far

**Export emails:**
```bash
cd degoosification-worker
node export-emails.js > emails.csv
```

**Use cases:**
- ✅ Import to Mailchimp / SendGrid for marketing
- ✅ Import to Zendesk for support ticketing
- ✅ Build initial user base before Asset Store launch
- ✅ Send announcements about Asset Store beta

---

## Phase 2: Full Account System (Future)

### When to Add Passwords?

**Option A: Add NOW (proactive)**
- Collect password during degoosification registration
- Users create account immediately
- Ready for Asset Store when it launches

**Option B: Add LATER (just-in-time)**
- Keep current email-only system
- Add password collection when Asset Store is ready
- Send "Complete your account" email to existing users

**Recommendation:** **Option B (later)** - Don't add friction to degoosification now. When Asset Store launches, send existing users an email: "Complete your Noodlings account to access the Asset Store!"

---

## Phase 2 Architecture

### User Database

**Option 1: Cloudflare D1 (SQLite)**
- Serverless SQL database
- Free tier: 5GB storage, 5M reads/day
- Perfect for user accounts

**Option 2: External (Supabase, Firebase Auth)**
- More features (OAuth, 2FA)
- More expensive
- Overkill for now

**Recommendation:** Cloudflare D1 (keep everything in Cloudflare ecosystem)

### Account Data Structure

```sql
CREATE TABLE users (
  id TEXT PRIMARY KEY,  -- UUID
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,  -- bcrypt
  display_name TEXT,
  created_at INTEGER NOT NULL,
  last_login INTEGER,
  email_verified INTEGER DEFAULT 0,

  -- Asset Store fields
  downloads_count INTEGER DEFAULT 0,
  purchases_total REAL DEFAULT 0.0,

  -- Support fields
  zendesk_id TEXT,

  -- Preferences
  newsletter_opt_in INTEGER DEFAULT 1,
  beta_tester INTEGER DEFAULT 0
);

CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_created_at ON users(created_at);
```

### New Worker Endpoints

```
POST /api/auth/register
  - Create account with email + password
  - Send verification email
  - Store in D1

POST /api/auth/login
  - Validate credentials
  - Return JWT token
  - Track last_login

POST /api/auth/verify-email
  - Verify email via token link

GET /api/user/profile
  - Get user data (requires auth)

PATCH /api/user/profile
  - Update display name, preferences

POST /api/user/reset-password
  - Send password reset email
```

---

## Integration Roadmap

### Marketing (Mailchimp / SendGrid)

**NOW:**
```bash
# Export emails to CSV
node export-emails.js > emails.csv

# Import to Mailchimp
# Mailchimp → Audience → Import → Upload CSV
```

**Use cases:**
- Newsletter about Noodlings development
- Asset Store launch announcement
- Beta testing invites

---

### Support (Zendesk)

**NOW:**
```bash
# Export emails
node export-emails.js > emails.csv

# Import to Zendesk
# Zendesk → Customers → Import users → Upload CSV
```

**Phase 2 (with accounts):**
- Auto-create Zendesk ticket when user reports bug in Asset Store
- Link support tickets to user accounts
- Show user's purchase history to support agents

---

### Asset Store (Phase 3)

**Requirements:**
- User login (email + password)
- Payment processing (Stripe)
- Download history
- Ratings/reviews
- Purchase receipts

**Architecture:**
```
Asset Store Worker
  ├─ Auth: JWT from account system
  ├─ Payments: Stripe integration
  ├─ Storage: Cloudflare R2 (assets)
  ├─ Database: D1 (purchases, ratings)
  └─ CDN: Cloudflare (fast downloads)
```

---

## Migration Plan: Email-Only → Full Accounts

When Asset Store launches:

**Step 1: Email existing users**
```
Subject: Complete Your Noodlings Account 🦆
From: Henri Bergamot <henri@noodlings.ai>

Bonjour, my friend!

You 'ave already degoosified your NoodleStudio (très bien!),
and now we 'ave something exciting: Ze Noodlings Asset Store!

To access ze Asset Store, please complete your account by
creating a password:

[Create Password Button] → https://noodlings.ai/complete-account?email=...

Honque honque,
Henri Bergamot
Product Specialist, Asset Store Services
```

**Step 2: Add password collection endpoint**
```javascript
POST /api/auth/complete-account
{
  "email": "user@example.com",
  "password": "secure_password",
  "token": "email_verification_token"
}
```

**Step 3: Migrate existing emails to D1**
```javascript
// Migration script
async function migrateEmailsToD1() {
  const emails = await fetchAllEmailsFromKV();

  for (const emailData of emails) {
    await db.prepare(`
      INSERT INTO users (id, email, created_at, email_verified)
      VALUES (?, ?, ?, 1)
    `).bind(
      crypto.randomUUID(),
      emailData.email,
      emailData.timestamp,
    ).run();
  }
}
```

---

## Cost Analysis

### Current (Email-only):
- **Cloudflare KV:** Free (< 1GB, < 100k reads/day)
- **Worker:** Free (< 100k requests/day)
- **Total:** $0/month

### With Accounts (D1):
- **D1 Database:** Free tier (5GB, 5M reads/day)
- **Still:** $0/month until very large scale

### With Asset Store:
- **R2 Storage:** $0.015/GB/month
- **Stripe:** 2.9% + $0.30 per transaction
- **Estimated at 10k users:** ~$50-100/month

---

## Security Considerations

### Password Storage
- **Use bcrypt** (not plain text!)
- **Min 12 characters**
- **Rate limiting** on login attempts

### Email Verification
- **Required before Asset Store access**
- **Token-based** (JWT with expiration)

### Session Management
- **JWT tokens** (short-lived, 1 hour)
- **Refresh tokens** (long-lived, 30 days)
- **Revocation** on logout

### GDPR Compliance
- **Data export:** Users can download their data
- **Right to deletion:** Users can delete accounts
- **Opt-out:** Marketing emails have unsubscribe

---

## Recommended Timeline

**Now → Asset Store Launch (Phase 1):**
- ✅ Keep email-only degoosification
- ✅ Export emails periodically for marketing
- ✅ Build user base (target: 1,000+ emails before launch)

**Asset Store Launch (Phase 2):**
- Add password collection
- Migrate existing emails to D1
- Implement auth endpoints
- Launch with "Complete your account" campaign

**Post-Launch (Phase 3):**
- Zendesk integration
- OAuth providers (Google, GitHub)
- Two-factor authentication
- Admin dashboard

---

## Quick Actions You Can Do NOW

### 1. Export Current Emails
```bash
cd degoosification-worker
node export-emails.js > emails.csv
```

### 2. Set Up Mailchimp Audience
- Create free Mailchimp account
- Import emails.csv
- Create "Noodlings Beta Users" audience
- Send welcome email

### 3. Monitor Growth
```bash
# Check total registrations
npx wrangler kv:key get "stats:total_registrations" --binding GOOSE_USERS
```

### 4. Periodic Exports
Add to cron:
```bash
# Export emails weekly
0 0 * * 0 cd ~/degoosification-worker && node export-emails.js > emails-$(date +%Y%m%d).csv
```

---

## Questions to Consider

1. **When do you want to launch Asset Store?**
   - If soon (< 3 months): Start building account system now
   - If later (6+ months): Keep email-only, build user base first

2. **What marketing do you want to do?**
   - Newsletter about Noodlings progress?
   - Beta testing program?
   - Early access to Asset Store?

3. **Support system preference?**
   - Zendesk (paid, feature-rich)
   - Discord (free, community-focused)
   - GitHub Issues (free, developer-focused)

---

**Ordnung muss sein!** But also: *Organic growth* (Christopher Alexander style)

Build what you need **when you need it**, not before. 🦆
