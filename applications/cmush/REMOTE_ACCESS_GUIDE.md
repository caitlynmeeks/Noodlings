# cMUSH Remote Access via Tailscale

**Purpose:** Enable remote users to connect to your cMUSH server using Tailscale VPN.

---

## Overview

Tailscale creates a secure peer-to-peer mesh network, allowing remote users to connect to your cMUSH server as if they were on the same local network. No port forwarding or firewall configuration required!

**Architecture:**
```
Remote User
    ↓ (Tailscale VPN)
    ↓ (100.x.x.x IP)
    ↓
Your Mac (Thistlequell)
    ├─ Tailscale IP: 100.x.x.x
    ├─ cMUSH HTTP: port 8080
    └─ cMUSH WebSocket: port 8765
```

---

## Prerequisites

### On Your Mac (Server Host)

1. **Tailscale installed and running**
   ```bash
   # Check if Tailscale is installed
   which tailscale

   # Check status
   tailscale status
   ```

   If not installed: https://tailscale.com/download

2. **Find your Tailscale IP**
   ```bash
   tailscale ip -4
   ```

   Example output: `100.64.123.45`

3. **Ensure cMUSH is running**
   ```bash
   cd /Users/thistlequell/git/consilience/applications/cmush
   ./start.sh
   ```

### On Remote User's Machine

1. **Tailscale installed and authenticated** (same account or shared network)
2. **Web browser** (Chrome, Firefox, Safari, etc.)

---

## Step-by-Step Setup

### 1. Configure cMUSH for Remote Access

By default, cMUSH binds to `0.0.0.0`, which allows connections from any network interface (including Tailscale).

**Verify server bindings:**

```bash
# Check server.py configuration
cd /Users/thistlequell/git/consilience/applications/cmush
grep "host.*port" server.py
```

Should show:
```python
# WebSocket server
host = "0.0.0.0"  # Accept connections from any interface
port = 8765

# HTTP server (in start.sh)
python3 -m http.server 8080 --bind 0.0.0.0
```

✅ No changes needed if already `0.0.0.0`

---

### 2. Get Your Tailscale IP

On your Mac:

```bash
tailscale ip -4
```

Example: `100.64.123.45`

**This is the IP remote users will use to connect.**

---

### 3. Start cMUSH Server

```bash
cd /Users/thistlequell/git/consilience/applications/cmush
source ~/git/consilience/venv/bin/activate
./start.sh
```

You should see:
```
Starting cMUSH...
Starting HTTP server on port 8080...
Starting WebSocket server on port 8765...
cMUSH server ready!
```

---

### 4. Test Local Access First

On your Mac, open browser to:
```
http://localhost:8080
```

- Login with your credentials
- Send a message to verify everything works
- Leave server running

---

### 5. Test Remote Access

**On remote user's machine:**

1. **Ensure Tailscale is connected**
   ```bash
   tailscale status
   ```

   Should show your Mac in the peer list.

2. **Open browser to your Tailscale IP**
   ```
   http://100.64.123.45:8080
   ```

   *(Replace with your actual Tailscale IP)*

3. **Register a new account or login**

4. **Send a test message**
   ```
   say Hello from remote!
   ```

5. **Verify server sees the connection**

   On your Mac, check logs:
   ```bash
   tail -f logs/cmush_2025-10-27.log | grep "connection\|User authenticated"
   ```

---

## Verification Checklist

Run these tests to ensure remote access works:

### Test 1: Basic Connectivity

**On remote machine:**
```bash
# Ping Tailscale IP
ping 100.64.123.45

# Test HTTP port (should return HTML)
curl -I http://100.64.123.45:8080

# Test WebSocket port (should connect)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  http://100.64.123.45:8765
```

**Expected:**
- Ping succeeds (low latency, <100ms typical)
- HTTP returns `200 OK`
- WebSocket returns `101 Switching Protocols`

### Test 2: Web Interface Loads

**On remote machine, open browser:**
```
http://100.64.123.45:8080
```

**Expected:**
- HTML page loads
- No JavaScript errors in console (F12)
- Login/register form visible

### Test 3: WebSocket Connection

**After login:**
- Browser console should show: `WebSocket connected`
- Server logs should show: `New connection from (100.x.x.x, port)`

### Test 4: Full Interaction

**Remote user sends messages:**
```
say Testing remote access!
@observe agent_desobelle
say How are you feeling?
```

**Expected:**
- Messages appear in chat
- Agents respond (if surprise > threshold)
- No lag or connection drops

---

## Troubleshooting

### Remote user can't load page

**Symptom:** `ERR_CONNECTION_REFUSED` or timeout

**Fixes:**
1. Verify Tailscale IP is correct:
   ```bash
   tailscale ip -4  # On server
   ```

2. Check firewall (macOS):
   ```bash
   # Allow ports 8080 and 8765
   # System Settings > Network > Firewall > Options
   # Add Python to allowed apps
   ```

3. Verify server is running:
   ```bash
   lsof -i :8080  # Should show python3
   lsof -i :8765  # Should show python3
   ```

4. Check server is bound to 0.0.0.0:
   ```bash
   netstat -an | grep -E "8080|8765"
   # Should show: *.8080  and  *.8765  (not 127.0.0.1)
   ```

### WebSocket connection fails

**Symptom:** Page loads, but can't connect to chat

**Fixes:**
1. Check browser console (F12) for errors

2. Verify WebSocket URL in client code:
   ```bash
   grep "ws://" /Users/thistlequell/git/consilience/applications/cmush/static/app.js
   ```

   Should dynamically use correct host:
   ```javascript
   const ws = new WebSocket(`ws://${window.location.hostname}:8765`);
   ```

3. Test WebSocket directly:
   ```bash
   websocat ws://100.64.123.45:8765
   ```

### High latency or lag

**Symptom:** Messages delayed, responses slow

**Fixes:**
1. Check Tailscale connection quality:
   ```bash
   tailscale status --peers
   # Look for "direct" connection (best)
   # vs "relay" connection (slower)
   ```

2. Improve connection:
   ```bash
   # Force direct connection
   tailscale configure --accept-routes
   ```

3. Check network:
   ```bash
   # Ping server
   ping -c 10 100.64.123.45
   # Should be <100ms, <1% loss
   ```

### Multiple users can't connect

**Symptom:** First user works, second user fails

**Fixes:**
1. Check server logs for errors:
   ```bash
   tail -100 logs/cmush_2025-10-27.log | grep ERROR
   ```

2. Verify no port conflicts:
   ```bash
   lsof -i :8080 | wc -l  # Should be 1
   lsof -i :8765 | wc -l  # Should be 1
   ```

3. Check user authentication:
   ```bash
   # View registered users
   cat world/users.json | python3 -m json.tool
   ```

---

## Security Considerations

### Tailscale Network Access

**Who can connect?**
- Only devices on your Tailscale network
- Must be authenticated with Tailscale account
- Optionally: Share devices with specific users via Tailscale admin panel

**To invite external users:**
1. Go to https://login.tailscale.com/admin/machines
2. Click "Share" on your Mac
3. Send invite link to user
4. They can access your Tailscale IP (but not your LAN)

### cMUSH Authentication

- Users still need cMUSH account (register first time)
- Passwords stored hashed (bcrypt)
- Session tokens expire after inactivity

**Create accounts for trusted users:**
```bash
# They register via web interface
# Or you can pre-create in Python:
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from auth import AuthManager

auth = AuthManager('world/users.json')
auth.register_user('alice', 'secure_password_here')
print("User created!")
EOF
```

### Firewall

**macOS Firewall:**
- No changes needed (Tailscale handles encryption)
- Optionally block external ports 8080/8765:
  ```bash
  # System Settings > Network > Firewall > Options
  # Block incoming connections for Python (if you want local-only + Tailscale)
  ```

**Tailscale ACLs:**
- Advanced: Restrict who can access which ports
- See: https://tailscale.com/kb/1018/acls/

---

## Performance Optimization

### For Multiple Remote Users

If you have many remote users, consider:

1. **Increase server limits** (config.yaml):
   ```yaml
   server:
     max_connections: 50  # Default: 10
     response_timeout: 30  # Seconds
   ```

2. **Monitor resource usage**:
   ```bash
   # CPU/RAM usage
   top -pid $(pgrep -f "python3.*server.py")

   # Connection count
   lsof -i :8765 | wc -l
   ```

3. **Log rotation** (prevent disk fill):
   ```bash
   # Add to crontab:
   0 0 * * * find /Users/thistlequell/git/consilience/applications/cmush/logs -mtime +7 -delete
   ```

### For Low-Bandwidth Users

1. **Reduce log verbosity** (fewer messages):
   - Edit logging config in server.py
   - Set level to `WARNING` instead of `INFO`

2. **Compress responses** (if implementing REST API):
   - Enable gzip in HTTP server

---

## Testing Remote Access

### Quick Test (5 minutes)

1. **On server Mac:**
   ```bash
   tailscale ip -4  # Get IP
   ./start.sh       # Start cMUSH
   ```

2. **On remote machine:**
   - Open browser to `http://YOUR_TAILSCALE_IP:8080`
   - Register account
   - Send message: `say Testing remote!`

3. **On server Mac:**
   ```bash
   tail logs/cmush_2025-10-27.log | grep "Testing remote"
   ```

**✅ Success:** You see the message in logs

---

### Full Test (15 minutes)

1. **Multiple remote users** connect simultaneously
2. **Send messages** back and forth
3. **Interact with agents**
4. **Check for lag** or disconnects
5. **Monitor server** resource usage

**✅ Success:** All users can chat smoothly, no errors

---

## Common Scenarios

### Scenario 1: Friend wants to try cMUSH

1. Share your Tailscale machine or add them to your Tailscale network
2. Send them the URL: `http://YOUR_TAILSCALE_IP:8080`
3. They register an account
4. They chat with agents

### Scenario 2: Collaborate on agent training

1. Multiple users connect to same cMUSH instance
2. All interactions logged to training data
3. Diverse conversations improve model
4. Export training data: already collected automatically

### Scenario 3: Remote development

1. Connect to cMUSH from laptop while server runs on desktop
2. Edit code, restart server via SSH
3. Test changes immediately via Tailscale IP
4. No need to sync code or run local server

---

## Advanced: Custom Domain

If you want a friendlier URL than `100.64.123.45`:

### Option 1: Tailscale MagicDNS

Tailscale automatically creates DNS names:

```
http://thistlequell.your-tailscale-network.ts.net:8080
```

Enable in Tailscale admin panel: https://login.tailscale.com/admin/dns

### Option 2: Local Hosts File

On remote machine, edit `/etc/hosts`:
```
100.64.123.45  cmush.local
```

Then access:
```
http://cmush.local:8080
```

---

## Summary

**To enable remote access:**
1. ✅ Tailscale installed and running
2. ✅ cMUSH server bound to `0.0.0.0`
3. ✅ Share Tailscale IP with remote users
4. ✅ Remote users connect via `http://TAILSCALE_IP:8080`

**Benefits:**
- ✅ Secure (encrypted by Tailscale)
- ✅ No port forwarding or router config
- ✅ Works across NATs and firewalls
- ✅ Low latency (peer-to-peer when possible)

**Limitations:**
- ⚠️ Requires Tailscale on all devices
- ⚠️ Not public (only invited Tailscale users)
- ⚠️ Server must stay running on your Mac

---

## Next Steps

After confirming remote access works:
1. ✅ Invite trusted users to test
2. ✅ Collect training data from diverse conversations
3. ✅ Monitor server performance under load
4. ✅ Consider deploying to cloud server for 24/7 availability

---

*For questions about Tailscale setup, see: https://tailscale.com/kb/*
*For questions about cMUSH server, see: CLAUDE.md or server.py*
