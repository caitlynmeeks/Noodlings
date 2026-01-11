# Build Settings

**Status**: Specification
**Date**: 2026-01-10
**Authors**: Caity + Claude
**Priority**: High (required for publishing apps)
**Inspiration**: Unity's File > Build Settings

---

## Overview

Build Settings is where creators configure how their NoodleStudio project exports to a standalone application. Like Unity's Build Settings panel - one place for all export configuration.

### Design Principles

1. **One place for everything** - Don't scatter settings across multiple panels
2. **Sensible defaults** - Should "just work" for simple projects
3. **Progressive disclosure** - Basic settings visible, advanced settings expandable
4. **Preview before build** - See what you'll get before committing

---

## Access

**Menu**: File > Build Settings... (Ctrl+Shift+B / Cmd+Shift+B)

Opens the Build Settings dialog.

---

## The Panel

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Build Settings                                                    [X]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ TARGET PLATFORM                                                  │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  ○ macOS Application (.app)                                      │   │
│  │  ○ Windows Application (.exe)                                    │   │
│  │  ○ Linux Application                                             │   │
│  │  ○ Web (coming soon)                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ APP IDENTITY                                                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  App Name:        [Let's Consciousness!              ]           │   │
│  │  Bundle ID:       [ai.noodlings.lets-consciousness   ]           │   │
│  │  Version:         [1.0.0                             ]           │   │
│  │  App Icon:        [icon.png                    ] [Browse...]     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ SPLASH SCREEN                                          [▼ Show] │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ☑ Show splash screen on launch                                 │   │
│  │                                                                  │   │
│  │  Splash Image:    [splash.png                  ] [Browse...]    │   │
│  │                   [Preview]                                      │   │
│  │                                                                  │   │
│  │  Duration:        [3.0    ] seconds                              │   │
│  │  ☑ Click/keypress to dismiss                                    │   │
│  │                                                                  │   │
│  │  Background:      [#1a1a1a] [■]                                  │   │
│  │  Fade In:         [0.3    ] seconds                              │   │
│  │  Fade Out:        [0.3    ] seconds                              │   │
│  │                                                                  │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  REQUIRED ATTRIBUTION (cannot be disabled)                       │   │
│  │  ☑ "Made with NoodleSTUDIO" badge                               │   │
│  │  ☑ Link to Noodling Ethical Covenant                            │   │
│  │  Position: [Bottom Right ▼]                                      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ EDITOR ACCESS                                          [▼ Show] │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ☑ Allow "View Project" (unfold to editor)                      │   │
│  │      Keyboard shortcut: [Ctrl+Shift+U    ]                       │   │
│  │                                                                  │   │
│  │  ☐ Require password to unfold                                   │   │
│  │      Password: [••••••••          ]                              │   │
│  │                                                                  │   │
│  │  ☐ Hide editor completely (app-only mode)                       │   │
│  │      ⚠ Users cannot inspect or modify the project               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLM PROVIDER                                           [▼ Show] │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ○ NoodleROUTER (recommended)                                   │   │
│  │      Uses noodlings.ai API. Users need account.                 │   │
│  │      Cost: Provider rate + 20% margin                           │   │
│  │                                                                  │   │
│  │  ○ User provides own API keys                                   │   │
│  │      Users enter their own Anthropic/OpenAI keys.               │   │
│  │      Settings panel will prompt for keys on first run.          │   │
│  │                                                                  │   │
│  │  ○ Local models only (Ollama)                                   │   │
│  │      Requires Ollama installed. No cloud dependency.            │   │
│  │      ⚠ Limited model selection, requires user setup             │   │
│  │                                                                  │   │
│  │  ○ Bundled API key (not recommended)                            │   │
│  │      ⚠ Your key is embedded in the app. You pay for all usage. │   │
│  │      Key: [sk-...                          ] [Browse...]         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ INCLUDED CONTENT                                       [▼ Show] │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ☑ All stages                                                   │   │
│  │  ☑ All noodlings                                                │   │
│  │  ☑ All UI layouts                                               │   │
│  │  ☑ All facet assemblies                                         │   │
│  │  ☑ All plays (.play.yaml)                                       │   │
│  │                                                                  │   │
│  │  ☐ Include unused assets                                        │   │
│  │  ☐ Include source facet code                                    │   │
│  │                                                                  │   │
│  │  Estimated size: 127 MB                                          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ADVANCED                                               [▼ Show] │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  Python Version:    [3.11 (bundled)        ▼]                   │   │
│  │  Qt Version:        [6.6.1 (bundled)       ▼]                   │   │
│  │                                                                  │   │
│  │  ☐ Strip debug symbols                                          │   │
│  │  ☐ Code signing (macOS)                                         │   │
│  │      Certificate: [                        ] [Select...]         │   │
│  │  ☐ Notarization (macOS)                                         │   │
│  │      Apple ID: [                           ]                     │   │
│  │                                                                  │   │
│  │  Build script hooks:                                             │   │
│  │      Pre-build:  [                         ] [Browse...]         │   │
│  │      Post-build: [                         ] [Browse...]         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Output Directory: [~/Desktop/builds           ] [Browse...]           │
│                                                                         │
│              [Cancel]    [Build and Run]    [Build]                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Sections

### Target Platform

| Platform | Status | Packager |
|----------|--------|----------|
| macOS (.app) | Supported | py2app |
| Windows (.exe) | Supported | PyInstaller |
| Linux | Supported | PyInstaller |
| Web | Future | Pyodide? |

Cross-compilation is limited - build on target platform for best results.

### App Identity

- **App Name** - Display name, can include spaces and special chars
- **Bundle ID** - Reverse domain notation (ai.noodlings.myapp)
- **Version** - Semantic versioning (1.0.0)
- **App Icon** - PNG/ICNS, will be converted to platform format

### Splash Screen

See full spec: `/docs/noodlestudio/splash-screen.md`

Key settings:
- Custom splash image
- Duration
- Click-to-dismiss
- Fade timing

**Required attribution cannot be disabled:**
- "Made with NoodleSTUDIO" badge
- Link to NEC (Noodling Ethical Covenant)

This is part of the covenant. If you use NoodleStudio, users know it.

### Editor Access

Controls whether the built app can "unfold" to reveal the editor.

| Option | Description | Use Case |
|--------|-------------|----------|
| **Allow unfold** | Users can View Project and see/edit everything | Educational, open-source, "the demo is the documentation" |
| **Password protected** | Unfold requires password | Limited access, beta testers |
| **Hidden completely** | No editor access, app-only | Commercial apps, locked experiences |

Default: **Allow unfold** (this is NoodleStudio's philosophy - transparency)

### LLM Provider

How the built app connects to language models.

| Option | Who Pays | Setup Required | Recommendation |
|--------|----------|----------------|----------------|
| **NoodleROUTER** | User (via noodlings.ai account) | User creates account | Default, simplest |
| **User's own keys** | User (direct to provider) | User enters API keys | Power users |
| **Local only (Ollama)** | User (electricity) | Install Ollama | Offline/privacy |
| **Bundled key** | Creator | None for user | ⚠ Dangerous - you pay |

### Included Content

What gets bundled into the app.

- **All stages** - 3D scenes, environments
- **All noodlings** - Characters, their assemblies
- **All UI layouts** - ui.yaml files
- **All facet assemblies** - Cognitive architectures
- **All plays** - .play.yaml scripts

Options:
- **Include unused assets** - Everything in project, even if not referenced
- **Include source facet code** - Python/JS source for custom facets (default: bytecode only)

### Advanced

For power users:
- Python/Qt version selection (if multiple bundled)
- Code signing for macOS distribution
- Notarization for Gatekeeper
- Build script hooks for custom steps

---

## Build Process

```
┌─────────────────┐
│  Click "Build"  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Validate settings      │
│  - Check required fields│
│  - Verify assets exist  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Run pre-build hook     │
│  (if configured)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Collect assets         │
│  - Copy stages          │
│  - Copy noodlings       │
│  - Copy UI layouts      │
│  - Copy assemblies      │
│  - Copy plays           │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Generate app config    │
│  - build.yaml           │
│  - Splash settings      │
│  - Permission flags     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Package with py2app/   │
│  PyInstaller            │
│  - Bundle Python        │
│  - Bundle Qt            │
│  - Bundle assets        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Code sign (if macOS    │
│  and configured)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Run post-build hook    │
│  (if configured)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Done!                  │
│  Output: MyApp.app      │
└─────────────────────────┘
```

---

## build.yaml

Settings are stored in the project as `build.yaml`:

```yaml
# build.yaml - Build configuration for NoodleStudio project

target: macos  # macos | windows | linux

identity:
  name: "Let's Consciousness!"
  bundle_id: ai.noodlings.lets-consciousness
  version: 1.0.0
  icon: assets/icon.png

splash:
  enabled: true
  image: assets/splash.png
  duration: 3.0
  click_to_dismiss: true
  background: "#1a1a1a"
  fade_in: 0.3
  fade_out: 0.3
  # Attribution is automatic and cannot be disabled

editor:
  allow_unfold: true
  require_password: false
  password_hash: null  # bcrypt hash if password required
  keyboard_shortcut: "Ctrl+Shift+U"

llm:
  provider: noodlerouter  # noodlerouter | user_keys | ollama | bundled
  # If bundled (not recommended):
  # bundled_key: "sk-..."

content:
  include_unused: false
  include_source: false

advanced:
  python_version: "3.11"
  strip_debug: false
  codesign:
    enabled: false
    certificate: null
  notarize:
    enabled: false
    apple_id: null
  hooks:
    pre_build: null
    post_build: null

output:
  directory: ~/Desktop/builds
```

---

## Runtime Behavior

The built app reads `build.yaml` at startup to determine:

1. **Show splash?** → Display splash screen with configured image/timing
2. **Show attribution** → Always (cannot disable)
3. **Allow unfold?** → Show/hide "View Project" button, enable/disable shortcut
4. **LLM provider?** → Configure appropriate API client

```python
# In built app startup
class BuiltApp:
    def __init__(self):
        self.build_config = load_build_yaml()

        if self.build_config.splash.enabled:
            self.show_splash()

        if not self.build_config.editor.allow_unfold:
            self.disable_editor_access()

        self.configure_llm_provider(self.build_config.llm.provider)
```

---

## Implementation Checklist

### Phase 1: Core Dialog - COMPLETE
- [x] Build Settings dialog (File > Build Settings, Ctrl+Shift+B)
- [x] Target platform selection (macOS/Windows/Linux radio buttons)
- [x] App identity fields (name, bundle ID, version, icon)
- [x] Save/load build.yaml
- [x] BuildConfig dataclass with full YAML serialization
- [x] 36 unit tests

### Phase 2: Splash Screen - COMPLETE
- [x] Splash screen section in dialog (collapsible)
- [x] Image picker with preview
- [x] Attribution (required, locked checkboxes)
- [x] SplashScreenWidget for runtime display
- [x] AttributionWidget with NEC link
- [x] LoadingIndicator (dots/bar/spinner styles)
- [x] Runtime integration - splash shows before main window
- [x] Fade in/out animations
- [x] Click-to-dismiss support
- [x] 35 unit tests

### Phase 3: Editor Access - UI COMPLETE
- [x] Unfold permission toggle (radio buttons)
- [x] Password protection option with password field
- [x] Hide editor completely option
- [ ] Runtime permission checking (pending)

### Phase 4: LLM Provider - UI COMPLETE
- [x] Provider selection radio buttons
- [x] NoodleROUTER configuration
- [x] User keys option
- [x] Ollama option
- [x] Bundled key warning (red text)
- [ ] Runtime provider switching (pending)

### Phase 5: Included Content - UI COMPLETE
- [x] Content checkboxes (stages, noodlings, UI, assemblies, plays)
- [x] Include unused assets option
- [x] Include source code option
- [ ] Size estimation (placeholder)

### Phase 6: Distribution - UI COMPLETE
- [x] Signing options (NoodleStudio/Own Cert/Unsigned)
- [x] Certificate field for own cert
- [x] Notarization checkbox
- [ ] Actual signing integration (pending)

### Phase 7: Build Process - PENDING
- [ ] Asset collection
- [ ] py2app integration (macOS)
- [ ] PyInstaller integration (Windows/Linux)
- [ ] Progress dialog during build
- [ ] Build and Run functionality

---

## Distribution & Signing

### The Runtime Model

NoodleStudio apps aren't truly "standalone" - they're the **NoodleStudio Runtime** bundled with **project data**.

```
MyApp.app/
├── NoodleStudio Runtime (signed, notarized)  ← The executable
└── Resources/
    └── project/                               ← User's data
        ├── stages/
        ├── noodlings/
        ├── ui.yaml
        ├── build.yaml
        └── assets/
```

The runtime is signed. The project data is just... data. YAML files, images, scripts - not executable code that requires signing.

### Signed Distribution Service

Because the runtime is signed under noodlings.ai's certificate:

| What | Status |
|------|--------|
| NoodleStudio Runtime | Signed + Notarized (by us) |
| Project data | Not code, doesn't need signing |
| Bundled app | Inherits runtime signature |

**Users get signed, notarized apps without:**
- Paying Apple $99/year for Developer ID
- Learning code signing
- Dealing with notarization
- Managing certificates

### Distribution Channels

| Channel | Signed Distribution Works? | Notes |
|---------|---------------------------|-------|
| **Direct download** | ✅ YES | DMG, ZIP, website |
| **itch.io** | ✅ YES | Indie standard |
| **Steam** | ✅ YES | Steam has own launcher |
| **GitHub Releases** | ✅ YES | Open source projects |
| **Mac App Store** | ❌ NO | Requires individual app review |
| **Microsoft Store** | ❌ NO | Requires individual submission |
| **iOS App Store** | ❌ NO | Different signing model entirely |

For most indie, educational, and open-source distribution - the signed runtime model works perfectly.

### Business Model Options

```
┌─────────────────────────────────────────────────────────────────┐
│  FREE TIER                                                       │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Signed distribution                                          │
│  ✓ Notarized (macOS)                                            │
│  ✓ "Made with NoodleSTUDIO" attribution (required)              │
│  ✓ NEC link (required)                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PAID TIER (future)                                              │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Everything in Free                                            │
│  ✓ Custom splash (no NoodleSTUDIO badge)                        │
│  ✓ Remove attribution requirement                                │
│  ✓ Priority signing queue                                        │
│  ✓ Analytics dashboard                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ENTERPRISE (future)                                             │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Everything in Paid                                            │
│  ✓ Use YOUR OWN certificate (we help configure)                 │
│  ✓ White-label runtime                                           │
│  ✓ Volume licensing                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Trust & Safety

If we're signing apps, we're vouching for them. Safeguards:

1. **Terms of Service**
   - No malware
   - No illegal content
   - No harassment tools
   - Violation = account termination + blacklist

2. **Automated Scanning**
   - Scan project data before signing
   - Check for known malicious patterns
   - Flag suspicious behavior for review

3. **Revocation Capability**
   - Track which apps were signed
   - Ability to revoke specific app hashes
   - Emergency certificate rotation if compromised

4. **Rate Limiting**
   - Limit builds per account per day
   - Prevent abuse of signing infrastructure

### The Attribution Trade

Free signed distribution in exchange for:
- "Made with NoodleSTUDIO" badge on splash
- Link to Noodling Ethical Covenant
- We can see what's being published (not content, just metadata)

The attribution isn't just branding - it's a **trust signal**. Users see that badge and know:
- This came through a vetted pipeline
- The creator agreed to the NEC
- There's accountability

### Build Settings UI Addition

```
┌─────────────────────────────────────────────────────────────────┐
│ DISTRIBUTION                                            [▼ Show] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Signing:                                                        │
│  ○ NoodleStudio Signed (recommended)                            │
│      Free signed distribution. Requires attribution.            │
│                                                                  │
│  ○ Your Own Certificate                                         │
│      Use your Apple Developer ID. You handle signing.           │
│      Certificate: [                        ] [Select...]         │
│                                                                  │
│  ○ Unsigned                                                      │
│      ⚠ Users will see security warnings on macOS                │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Notarization (macOS):                                           │
│  ☑ Submit for notarization after build                          │
│      Required for NoodleStudio Signed distribution.              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Philosophy Notes

### Attribution Cannot Be Disabled

This is intentional and non-negotiable. If you build with NoodleStudio:
- Users see "Made with NoodleSTUDIO"
- Users can find the NEC
- Transparency is maintained

This is part of the covenant. If someone wants to hide that they used NoodleStudio, they can't use NoodleStudio.

### Default to Openness

The default is "Allow unfold" - users CAN see the project, CAN learn from it, CAN modify it. This is the NoodleStudio philosophy: the demo is the documentation.

Hiding the editor is an option, but it's not the default and it's not encouraged.

### We Don't Recommend Bundling Keys

The "Bundled API key" option exists but is explicitly discouraged. If you bundle your key:
- You pay for all usage
- You can't revoke access without rebuilding
- Keys can be extracted from the binary

It's there for edge cases (kiosk installations, demos), not general use.

---

*"Build Settings: where transparency is configured, never disabled."*
