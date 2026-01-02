# Bug Reporting System

**Automatic crash detection and issue tracking for NoodleStudio**

---

## Overview

NoodleStudio includes a built-in bug reporting system that:

1. Catches unhandled exceptions and offers to report them
2. Provides a manual "Report a Bug" dialog
3. Submits reports to GitHub Issues via Cloudflare Worker proxy
4. Collects system info, logs, and stack traces automatically

---

## User Experience

### Manual Bug Report

**Help > Report a Bug...**

Opens a dialog to submit bug reports:

- Summary (required)
- Severity: Crash, Major, Minor, Cosmetic
- Steps to reproduce
- System info (auto-collected)
- Console logs (optional)

### Automatic Crash Reports

When an unhandled exception occurs:

1. Exception is caught by `sys.excepthook`
2. Crash dialog appears with pre-filled error details
3. User can add context and submit
4. Stack trace and system info included automatically

### View Known Issues

**Help > View Known Issues...**

Opens GitHub Issues in browser: `https://github.com/noodlings-ai/noodlings/issues`

---

## Architecture

```
NoodleStudio                    Cloudflare Worker              GitHub
┌────────────────┐              ┌─────────────────┐           ┌────────┐
│ Bug Report     │   POST       │ /api/bug-report │   API     │ Issues │
│ Dialog         │ ─────────────│                 │──────────▶│        │
└────────────────┘              │ - Validates     │           └────────┘
                                │ - Formats body  │
                                │ - Creates issue │
                                └─────────────────┘
```

**Why proxy through Cloudflare?**

- Users don't need GitHub accounts or tokens
- Bot token stored server-side (secure)
- Rate limiting and validation at edge
- Works offline (falls back to clipboard)

---

## Key Files

### NoodleStudio

| File | Purpose |
|------|---------|
| `dialogs/bug_report_dialog.py` | Dialog UI and submission |
| `main.py` | Crash reporter hook (`sys.excepthook`) |
| `core/main_window_settings_mixin.py` | Menu handlers |
| `core/main_window_menus_mixin.py` | Help menu items |

### Backend

| File | Purpose |
|------|---------|
| `backend/noodlings-api/src/routes/bugs.ts` | Cloudflare Worker route |
| `backend/noodlings-api/src/types.ts` | Env bindings (GITHUB_BOT_TOKEN) |

---

## Report Format

Reports submitted to GitHub Issues have this structure:

```markdown
## Steps to Reproduce
[User description]

## Error Details (for crashes)
```
ExceptionType: message
```

<details>
<summary>Full Traceback</summary>

```python
[stack trace]
```
</details>

## System Information

| Property | Value |
|----------|-------|
| NoodleStudio | 0.1.0-alpha |
| Platform | macOS-14.0-arm64 |
| Python | 3.12.0 |
| Qt | 6.6.1 |
| GPU | Apple M3 Ultra |

<details>
<summary>Console Logs</summary>

```
[recent log output]
```
</details>

---
*Submitted via NoodleStudio Bug Reporter*
```

---

## Labels

Issues are automatically labeled based on severity:

| Severity | Labels |
|----------|--------|
| Crash | `bug`, `severity:crash`, `priority:high` |
| Major | `bug`, `severity:major` |
| Minor | `bug`, `severity:minor` |
| Cosmetic | `bug`, `severity:cosmetic` |

---

## Backend Configuration

### Environment Variables

Add to `wrangler.toml` or Cloudflare dashboard:

```toml
[vars]
GITHUB_REPO_OWNER = "noodlings-ai"
GITHUB_REPO_NAME = "noodlings"

# In secrets (not vars):
# GITHUB_BOT_TOKEN = "ghp_xxx..."
```

### Creating the Bot Token

1. Go to GitHub > Settings > Developer settings > Personal access tokens
2. Create a **Fine-grained token** with:
   - Repository access: `noodlings-ai/noodlings`
   - Permissions: Issues (Read and write)
3. Add as secret: `npx wrangler secret put GITHUB_BOT_TOKEN`

---

## Fallback Behavior

If the backend is unavailable:

1. Report is formatted as GitHub-flavored markdown
2. Copied to clipboard
3. User prompted to paste into new GitHub issue manually

This ensures bug reports aren't lost even when offline.

---

## Claude Access

Claude can access bug reports via the `gh` CLI:

```bash
# List all bugs
gh issue list --label bug

# List crashes
gh issue list --label severity:crash

# View specific issue
gh issue view 42

# Search
gh issue list --search "crash facet"
```

---

## Collecting Console Logs

The bug reporter attempts to collect recent console logs:

```python
# In ConsolePanel, add this method:
def get_recent_logs(self, max_lines: int = 50) -> str:
    """Get recent console output for bug reports."""
    # Return last N lines from console buffer
```

If not available, logs are omitted from the report.

---

## Future Enhancements

- [ ] Screenshot capture option
- [ ] Automatic duplicate detection
- [ ] Issue templates in GitHub
- [ ] Rate limiting per IP
- [ ] Anonymous submission (no account needed - already done)

---

## See Also

- [CI/CD](ci-cd.md) - Continuous integration
- [Testing](../noodlestudio/testing.md) - Test suite
- [GitHub Issues](https://github.com/noodlings-ai/noodlings/issues) - Bug tracker
