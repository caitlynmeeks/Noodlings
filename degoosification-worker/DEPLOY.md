# Quick Deployment Guide

**YOUR API KEY IS SAFE:** Stored in `.dev.vars` (gitignored)

## Local Testing (Now!)

```bash
cd degoosification-worker

# Install dependencies
npm install

# Test locally (uses .dev.vars automatically)
npx wrangler dev

# In another terminal, test:
curl -X POST http://localhost:8787/api/degoosify/register \
  -H "Content-Type: application/json" \
  -d '{"email": "your.email@example.com"}'
```

Check your email! You should receive a degoosification code.

## Production Deployment

### 1. Login to Cloudflare

```bash
npx wrangler login
```

### 2. Create KV Namespace

```bash
# Production namespace
npx wrangler kv:namespace create "GOOSE_USERS"

# Preview namespace (for testing)
npx wrangler kv:namespace create "GOOSE_USERS" --preview
```

You'll see output like:
```
{ binding = "GOOSE_USERS", id = "abc123..." }
{ binding = "GOOSE_USERS", preview_id = "xyz789..." }
```

**IMPORTANT:** Update `wrangler.toml` with these IDs:

```toml
[[kv_namespaces]]
binding = "GOOSE_USERS"
id = "abc123..."        # ← Your production ID
preview_id = "xyz789..."  # ← Your preview ID
```

### 3. Set Production Secret

```bash
# Set Resend API key for production
npx wrangler secret put RESEND_API_KEY
```

When prompted, paste: `re_PxXYkjiP_7px1Ps9JtAdLqTmRjgqXXnrb`

**Security Note:** This stores the key encrypted in Cloudflare, NOT in git!

### 4. Deploy!

```bash
npx wrangler deploy
```

Your Worker will be live at:
```
https://degoosification-worker.YOUR_ACCOUNT.workers.dev
```

### 5. Update NoodleStudio Client

Edit `applications/noodlestudio/noodlestudio/panels/settings_panel.py`, line 294:

```python
# Change from:
backend_url = "https://degoosify.noodlings.ai/api/degoosify/register"

# To your deployed Worker URL:
backend_url = "https://degoosification-worker.YOUR_ACCOUNT.workers.dev/api/degoosify/register"
```

### 6. Test End-to-End

1. Run NoodleStudio
2. Open Settings (Cmd+,) → General tab
3. Enter your email
4. Click "Register & Turn off goose"
5. Goose appears! 🦆
6. Check email for code
7. Click "I already have a code"
8. Enter code → Goose defeated!

## Custom Domain (Optional)

If you have `noodlings.ai` on Cloudflare:

### 1. Configure Domain in Resend

https://resend.com/domains → Add `noodlings.ai`

Add DNS records (provided by Resend) to Cloudflare.

### 2. Update Worker Route

Edit `wrangler.toml`:

```toml
routes = [
  { pattern = "degoosify.noodlings.ai/*", zone_name = "noodlings.ai" }
]
```

### 3. Deploy

```bash
npx wrangler deploy
```

### 4. Add DNS Record

Cloudflare Dashboard → DNS:
- Type: `AAAA`
- Name: `degoosify`
- Content: `100::`
- Proxy: ✅ Enabled (orange cloud)

Worker will be accessible at: `https://degoosify.noodlings.ai`

## Monitoring

### View Logs

```bash
npx wrangler tail
```

This streams live Worker logs.

### Check KV Storage

```bash
# List all registered emails
npx wrangler kv:key list --binding GOOSE_USERS

# Get specific user data
npx wrangler kv:key get "email:test@example.com" --binding GOOSE_USERS
```

### Resend Dashboard

Monitor email delivery: https://resend.com/emails

## Security Checklist

- ✅ `.dev.vars` is gitignored (local secrets safe)
- ✅ Production secret stored via `wrangler secret put` (encrypted)
- ✅ API key NEVER in source code
- ✅ `.gitignore` includes `.env`, `.dev.vars`, `node_modules/`

## Troubleshooting

### "Error: No account_id specified"

Run: `npx wrangler login`

### "Error: A request to the Cloudflare API failed"

Check if you have a Cloudflare account and you're logged in.

### Emails not sending

1. Check Resend API key: `npx wrangler secret list`
2. Verify domain in Resend dashboard
3. Check email logs: https://resend.com/emails

### CORS errors from NoodleStudio

Worker includes CORS headers. If still blocked, try testing in browser console:

```javascript
fetch('https://your-worker.workers.dev/api/degoosify/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email: 'test@example.com' })
})
.then(r => r.json())
.then(console.log)
```

## Cost Estimate

**Free tier:** 3,000 emails/month = **$0**

Your Resend account: 3,000 emails/month free
Your Cloudflare: 100,000 requests/day free

You can handle thousands of users before any costs!

---

**Ordnung muss sein!** 🦆
