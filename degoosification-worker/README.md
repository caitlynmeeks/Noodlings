# Degoosification Worker

Serverless email collection backend for Noodlings Gooseware system.

**HonkCrypt™ powered** - QUANTUM ALGORITHMIC ENCRYPTION (it's XOR with a silly key)

## Overview

This Cloudflare Worker handles degoosification code registration and delivery:

1. User enters email in NoodleStudio Settings
2. Worker generates code using HonkCrypt™
3. Code sent via Resend email service
4. Email stored in Cloudflare KV (user base building!)
5. User enters code → goose defeated

## Architecture

```
NoodleStudio Client
    ↓ POST /api/degoosify/register
Cloudflare Worker (this project)
    ├→ Cloudflare KV (email storage)
    └→ Resend API (email delivery)
```

## Prerequisites

1. **Cloudflare Account** - Free tier sufficient
   - Sign up: https://dash.cloudflare.com/sign-up

2. **Resend Account** - Free tier (3,000 emails/month)
   - Sign up: https://resend.com/signup
   - Get API key: https://resend.com/api-keys

3. **Domain** (optional)
   - Can use `*.workers.dev` subdomain (free)
   - Or configure custom domain: `degoosify.noodlings.ai`

4. **Node.js** - For Wrangler CLI
   - Install: https://nodejs.org/

## Quick Start

### 1. Install Wrangler CLI

```bash
npm install -g wrangler
```

### 2. Login to Cloudflare

```bash
wrangler login
```

This will open a browser for authentication.

### 3. Create KV Namespace

```bash
# Create production KV namespace
wrangler kv:namespace create "GOOSE_USERS"

# Create preview KV namespace (for testing)
wrangler kv:namespace create "GOOSE_USERS" --preview
```

You'll see output like:
```
{ binding = "GOOSE_USERS", id = "abc123..." }
```

**Important:** Copy these IDs and update `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "GOOSE_USERS"
id = "YOUR_PRODUCTION_ID"        # ← Paste production ID here
preview_id = "YOUR_PREVIEW_ID"   # ← Paste preview ID here
```

### 4. Set Resend API Key

```bash
wrangler secret put RESEND_API_KEY
```

When prompted, paste your Resend API key (starts with `re_`).

### 5. Local Testing

```bash
# Start local dev server
wrangler dev

# In another terminal, test the endpoint
curl -X POST http://localhost:8787/api/degoosify/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

Expected response:
```json
{
  "success": true,
  "message": "Degoosification code sent to your email!",
  "email": "test@example.com"
}
```

Check your email for the degoosification code!

### 6. Deploy to Production

```bash
wrangler deploy
```

Your Worker will be live at:
```
https://degoosification-worker.<your-account>.workers.dev
```

### 7. Update NoodleStudio Client

Edit `noodlestudio/panels/settings_panel.py`, line 294:

```python
# Change from:
backend_url = "https://degoosify.noodlings.ai/api/degoosify/register"

# To your Worker URL:
backend_url = "https://degoosification-worker.<your-account>.workers.dev/api/degoosify/register"
```

## Custom Domain Setup (Optional)

### Using Cloudflare DNS

If you have a domain on Cloudflare (e.g., `noodlings.ai`):

1. **Add route to `wrangler.toml`:**

```toml
routes = [
  { pattern = "degoosify.noodlings.ai/*", zone_name = "noodlings.ai" }
]
```

2. **Deploy:**

```bash
wrangler deploy
```

3. **Configure DNS:**

Go to Cloudflare Dashboard → DNS → Add record:
- Type: `AAAA`
- Name: `degoosify`
- Content: `100::` (dummy IPv6, Worker handles routing)
- Proxy: ✅ Enabled (orange cloud)

Your Worker will now be accessible at:
```
https://degoosify.noodlings.ai
```

### Email Domain Configuration (Resend)

**Important:** Resend requires domain verification to send from custom domains.

1. **Add domain in Resend:**
   - Go to: https://resend.com/domains
   - Add `noodlings.ai`

2. **Verify DNS records:**
   - Resend will provide DNS records (SPF, DKIM, DMARC)
   - Add these to Cloudflare DNS
   - Wait for verification (usually < 1 hour)

3. **Update email sender in `src/email.js`:**

```javascript
from: 'The Goose <goose@noodlings.ai>'  // ← Your verified domain
```

**Free tier limitation:** Resend free tier only allows sending from ONE verified domain.

## API Endpoints

### POST /api/degoosify/register

Register email and send degoosification code.

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Degoosification code sent to your email!",
  "email": "user@example.com"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Invalid email address. The goose demands valid emails!"
}
```

### GET /api/degoosify/stats

Get registration statistics (admin endpoint).

**Response:**
```json
{
  "total_users": 42,
  "degoosified": 42,
  "message": "Gooseware statistics"
}
```

### GET /

Health check / status endpoint.

**Response:**
```json
{
  "service": "Degoosification Service",
  "status": "operational",
  "message": "🦆 Honk! The goose is ready.",
  "version": "1.0.0"
}
```

## Testing

### Manual Testing

```bash
# Test registration
curl -X POST http://localhost:8787/api/degoosify/register \
  -H "Content-Type: application/json" \
  -d '{"email": "your.email@example.com"}'

# Check stats
curl http://localhost:8787/api/degoosify/stats

# Health check
curl http://localhost:8787/
```

### Check KV Storage

```bash
# List all keys
wrangler kv:key list --binding GOOSE_USERS

# Get specific email
wrangler kv:key get "email:test@example.com" --binding GOOSE_USERS
```

### Test Email Delivery

Check that email was sent via Resend dashboard:
https://resend.com/emails

## Cost Analysis

### Free Tier (First 3,000 users/month)

| Service | Free Tier | Cost After |
|---------|-----------|------------|
| Cloudflare Workers | 100k requests/day | $5 / 10M requests |
| Cloudflare KV | 1GB storage, 100k reads/day | $0.50/GB/month |
| Resend | 3k emails/month | $20/month (50k) |

**Total: $0/month** for first ~3k users

### At Scale (100k users)

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Workers | ~10M requests | $5 |
| KV Storage | ~10GB | $5 |
| Resend | ~100k emails | $80 |

**Total: ~$90/month**

## Monitoring

### Cloudflare Dashboard

Monitor Worker performance:
- https://dash.cloudflare.com → Workers & Pages
- View metrics: requests, errors, CPU time
- Tail logs: `wrangler tail`

### Resend Dashboard

Monitor email delivery:
- https://resend.com/emails
- View sent emails, bounces, opens (if tracking enabled)

## Security Notes

**This is intentionally weak security theater!**

The HonkCrypt™ "QUANTUM ALGORITHMIC ENCRYPTION" is just XOR with a hardcoded key. Anyone reading the source code can:

1. Find bypass codes (ROT13 "HONK", "esoog", etc.)
2. Reverse the XOR encryption
3. Generate valid codes offline

**This is by design!** We're open source. The goal is user base building, not actual security. Curious tinkerers who find bypass codes will appreciate the humor.

### What We Store

```json
{
  "code": "GOOSE-abc123xyz==",
  "email": "user@example.com",
  "timestamp": 1702234567890,
  "goose_defeated": true,
  "version": "noodlestudio-1.0",
  "user_agent": "NoodleStudio/1.0",
  "ip": "1.2.3.4"
}
```

**Privacy:** Emails stored for 90 days, then auto-expired.

## Troubleshooting

### "Module not found" errors

Make sure you're using ES modules syntax:
```javascript
import { foo } from './bar.js';  // ✅ Include .js extension
export default { ... };
```

### KV writes not working

Check namespace binding in `wrangler.toml`:
```toml
[[kv_namespaces]]
binding = "GOOSE_USERS"  # ← Must match code
id = "your-namespace-id"
```

### Resend emails not sending

1. Check API key is set: `wrangler secret list`
2. Verify domain if using custom domain
3. Check Resend dashboard for errors
4. Free tier: Must use verified domain as sender

### CORS errors from NoodleStudio

Worker includes CORS headers by default. If still blocked:

```javascript
// In src/index.js, update CORS headers:
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',  // Or specific domain
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};
```

## Future Enhancements

### Phase 2: User Accounts
- Login/signup system
- User dashboard at noodlings.ai
- Download history

### Phase 3: Asset Store
- Extend Worker to serve asset metadata
- R2 storage for asset files (cheaper than S3!)
- Payment integration (Stripe)

### Phase 4: Analytics
- Track conversion rates (emails → codes → redeemed)
- Geographic distribution
- Integration with PostHog or Plausible

## License

MIT - Open source consciousness architecture for everyone!

## Contact

- Goose issues: File at https://github.com/noodlings/noodlings/issues
- Email: goose@noodlings.ai (once deployed!)

---

**Ordnung muss sein!** (But make it fun!)

🦆 The goose awaits its serverless destiny.
