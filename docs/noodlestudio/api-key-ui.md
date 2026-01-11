# API Key Settings UI

**Status**: COMPLETED (2026-01-10)
**Date**: 2026-01-09
**Authors**: Caity + Claude
**Priority**: High (users need this to use LLM features)

## Implementation Summary

**Files Created:**
- `panels/api_key_settings.py` - APIKeySettingsWidget with macOS Keychain integration
- `tests/test_api_key_settings.py` - 14 unit tests

**Files Modified:**
- `panels/settings_panel.py` - Added "Account" tab with API key widget

**Features Implemented:**
- Key display with copy-to-clipboard
- Secure storage via macOS Keychain (`security` command)
- Regenerate key with confirmation dialog
- Show/hide key toggle
- Clear key functionality

**Tests:** 14 passing

---

## Overview

Non-technical users need a dead-simple way to get and manage their API key for NoodleROUTER. The goal: **they shouldn't have to think about it**.

### Design Principles

1. **Zero friction** - Key auto-generates, auto-configures
2. **No jargon** - "API key" is already borderline; explain in plain terms
3. **Copy-friendly** - One click to clipboard
4. **Safe regeneration** - Easy to fix if compromised
5. **Usage visibility** - Show them what they're using (optional)

---

## First-Time Experience

When a user first opens NoodleStudio and hasn't set up an API key:

### Option A: Silent Auto-Setup (Recommended)

1. User logs in / creates account
2. Backend auto-generates API key
3. NoodleStudio fetches and stores it locally
4. User never sees a setup screen
5. Everything just works

They only see the key if they go looking in Settings.

### Option B: Gentle Welcome

If we want them to know the key exists:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ✨ You're all set!                                         │
│                                                             │
│  We've created an API key for you. This lets your          │
│  noodlings connect to language models.                      │
│                                                             │
│  You don't need to do anything - it's already configured.  │
│                                                             │
│  If you ever need to see or copy your key, you'll find     │
│  it in Settings > Account > API Key.                        │
│                                                             │
│                    [Got it!]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

One button. No choices to make. Just acknowledgment.

---

## Settings Panel

### Location

```
Settings
├── General
├── Appearance
├── Account
│   ├── Profile
│   └── API Key        ← Here
├── Editor
└── About
```

### API Key Panel Design

```
┌─────────────────────────────────────────────────────────────┐
│  API Key                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  This key lets your noodlings talk to AI models through     │
│  NoodleROUTER. It's already configured - you only need      │
│  this if you're using NoodleStudio on another device.       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ nood_k7Xm9pQr2sT4vW6xY8zA0bC1dE3fG5hI7jK9lM0nO1p │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [📋 Copy]                                                  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  🔄 Regenerate Key                                          │
│  If you think your key was exposed, generate a new one.     │
│  Your old key will stop working immediately.                │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  📊 This Month's Usage                                      │
│  Requests: 1,247 of 10,000                                  │
│  ████████░░░░░░░░░░░░  12%                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Display

- Show full key (it's theirs, they should see it)
- Monospace font for easy reading
- Select-all on click for easy copying
- Copy button with feedback ("Copied!")

### Copy Interaction

```
[📋 Copy]  →  click  →  [✓ Copied!]  →  2 sec  →  [📋 Copy]
```

Toast notification optional but nice:
```
┌────────────────────────┐
│ ✓ Key copied to clipboard │
└────────────────────────┘
```

---

## Regenerate Flow

When user clicks "Regenerate Key":

### Confirmation Dialog

```
┌─────────────────────────────────────────────────────────────┐
│  Regenerate API Key?                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  This will:                                                 │
│  • Create a new key                                         │
│  • Immediately disable your old key                         │
│  • Update NoodleStudio on this device automatically         │
│                                                             │
│  Any other devices or apps using your old key will need     │
│  to be updated with the new one.                            │
│                                                             │
│              [Cancel]    [Regenerate]                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### After Regeneration

```
┌─────────────────────────────────────────────────────────────┐
│  ✓ New Key Generated                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Your new API key:                                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ nood_NEW_KEY_HERE_abcdefghijklmnopqrstuvwxyz12345 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [📋 Copy]                                                  │
│                                                             │
│  Your old key has been disabled.                            │
│                                                             │
│                         [Done]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage Display (Optional)

If we're tracking usage:

```
📊 This Month's Usage

Requests: 1,247 of 10,000
████████░░░░░░░░░░░░  12%

Resets: January 31, 2026
```

Or for unlimited/pay-as-you-go:

```
📊 This Month's Usage

Requests: 1,247
Estimated cost: $0.47

View detailed usage →
```

### Why Show Usage?

1. Users understand they're consuming a resource
2. Helps them notice if something's wrong (sudden spike = leak?)
3. Transparency builds trust

---

## Local Storage

NoodleStudio stores the key locally so users don't have to paste it every time:

```python
# In NoodleStudio settings
class AppSettings:
    def __init__(self):
        self.api_key: Optional[str] = None

    def get_api_key(self) -> Optional[str]:
        """Get stored API key."""
        return self.api_key or self._load_from_keychain()

    def set_api_key(self, key: str):
        """Store API key securely."""
        self.api_key = key
        self._save_to_keychain(key)

    def _load_from_keychain(self) -> Optional[str]:
        """Load from OS keychain (macOS Keychain, Windows Credential Manager, etc.)"""
        # Platform-specific secure storage
        pass

    def _save_to_keychain(self, key: str):
        """Save to OS keychain."""
        pass
```

### Security Note

- Store in OS keychain, not plain text config
- macOS: Keychain
- Windows: Credential Manager
- Linux: Secret Service API / libsecret

---

## Auto-Configuration Flow

```
┌─────────────────┐
│  User logs in   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Check: has API key?    │
└────────┬────────────────┘
         │
    No   │   Yes
         │    └──────────────────────┐
         ▼                           ▼
┌─────────────────────────┐  ┌─────────────────┐
│  POST /api-keys         │  │  Fetch existing │
│  (create new key)       │  │  GET /api-keys  │
└────────┬────────────────┘  └────────┬────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────┐
│  Store key locally (keychain)               │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Configure LLM Router with key              │
│  (automatic, no user action needed)         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Ready! Noodlings can talk to LLMs          │
└─────────────────────────────────────────────┘
```

User does: nothing.
System does: everything.

---

## Error States

### No Internet

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️ Can't reach NoodleROUTER                                │
│                                                             │
│  Check your internet connection.                            │
│  Your noodlings can still work offline with local models.   │
│                                                             │
│                    [Try Again]                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Revoked/Invalid

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️ API Key Invalid                                         │
│                                                             │
│  Your API key isn't working. This can happen if:            │
│  • You regenerated it on another device                     │
│  • Your account was suspended                               │
│                                                             │
│  [Generate New Key]    [Contact Support]                    │
└─────────────────────────────────────────────────────────────┘
```

### Rate Limited

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️ Usage Limit Reached                                     │
│                                                             │
│  You've used all your requests for this month.              │
│  Resets: January 31, 2026                                   │
│                                                             │
│  You can still use local models, or upgrade your plan.      │
│                                                             │
│  [Use Local Models]    [View Plans]                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Settings Panel Component

```python
# panels/api_key_settings.py

class APIKeySettingsPanel(QWidget):
    """API Key management panel in Settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._api_key: Optional[str] = None
        self._build_ui()
        self._load_key()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Explanation
        explanation = QLabel(
            "This key lets your noodlings talk to AI models through "
            "NoodleROUTER. It's already configured - you only need "
            "this if you're using NoodleStudio on another device."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Key display
        self._key_display = QLineEdit()
        self._key_display.setReadOnly(True)
        self._key_display.setFont(QFont("SF Mono", 12))
        layout.addWidget(self._key_display)

        # Copy button
        self._copy_btn = QPushButton("📋 Copy")
        self._copy_btn.clicked.connect(self._copy_key)
        layout.addWidget(self._copy_btn)

        # Separator
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))

        # Regenerate section
        regen_label = QLabel("🔄 Regenerate Key")
        regen_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(regen_label)

        regen_explanation = QLabel(
            "If you think your key was exposed, generate a new one. "
            "Your old key will stop working immediately."
        )
        regen_explanation.setWordWrap(True)
        layout.addWidget(regen_explanation)

        self._regen_btn = QPushButton("Regenerate Key")
        self._regen_btn.clicked.connect(self._regenerate_key)
        layout.addWidget(self._regen_btn)

        # Usage (optional)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        self._usage_widget = UsageDisplayWidget()
        layout.addWidget(self._usage_widget)

        layout.addStretch()

    def _load_key(self):
        """Load API key from backend or local storage."""
        # Try local first
        key = AppSettings.instance().get_api_key()
        if key:
            self._display_key(key)
            return

        # Fetch from backend
        self._fetch_or_create_key()

    def _display_key(self, key: str):
        """Display the key in the UI."""
        self._api_key = key
        self._key_display.setText(key)

    def _copy_key(self):
        """Copy key to clipboard."""
        if self._api_key:
            QApplication.clipboard().setText(self._api_key)
            self._copy_btn.setText("✓ Copied!")
            QTimer.singleShot(2000, lambda: self._copy_btn.setText("📋 Copy"))

    def _regenerate_key(self):
        """Regenerate API key with confirmation."""
        dialog = RegenerateKeyDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._do_regenerate()

    def _do_regenerate(self):
        """Actually regenerate the key."""
        # Call backend DELETE then POST
        # Update local storage
        # Display new key
        pass
```

### Backend Integration

```python
# api/noodle_router_client.py

class NoodleRouterClient:
    """Client for NoodleROUTER API."""

    def __init__(self, base_url: str = "https://llm.noodlings.ai"):
        self.base_url = base_url
        self._session_token: Optional[str] = None

    async def get_or_create_api_key(self) -> str:
        """Get existing API key or create new one."""
        # Try to get existing
        keys = await self._request("GET", "/api-keys")
        if keys and len(keys) > 0:
            # Return first active key
            for key in keys:
                if not key.get('revoked_at'):
                    return key['key_prefix'] + "..."  # Backend doesn't return full key!

        # Create new
        result = await self._request("POST", "/api-keys", {"name": "NoodleStudio"})
        return result['key']  # Full key only returned on creation!

    async def regenerate_api_key(self) -> str:
        """Revoke all existing keys and create a new one."""
        # Get existing keys
        keys = await self._request("GET", "/api-keys")

        # Revoke all
        for key in keys:
            if not key.get('revoked_at'):
                await self._request("DELETE", f"/api-keys/{key['id']}")

        # Create new
        result = await self._request("POST", "/api-keys", {"name": "NoodleStudio"})
        return result['key']
```

---

## Implementation Checklist

- [ ] Create APIKeySettingsPanel widget
- [ ] Add to Settings dialog
- [ ] Implement secure local storage (OS keychain)
- [ ] Auto-fetch/create key on login
- [ ] Copy to clipboard functionality
- [ ] Regenerate key flow with confirmation
- [ ] Error state handling
- [ ] Usage display (if tracking)
- [ ] Test on macOS, Windows, Linux

---

## Copy for UI

### Explanation Text
> This key lets your noodlings talk to AI models through NoodleROUTER. It's already configured - you only need this if you're using NoodleStudio on another device.

### Regenerate Warning
> If you think your key was exposed, generate a new one. Your old key will stop working immediately.

### Regenerate Confirmation
> This will create a new key and immediately disable your old one. Any other devices or apps using your old key will need to be updated.

### Success Toast
> ✓ Key copied to clipboard

---

*"You don't need to understand it. It just works."*
