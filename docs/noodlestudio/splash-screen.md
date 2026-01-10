# Splash Screen Specification

**Status**: Specification
**Date**: 2026-01-09
**Authors**: Caity + Claude
**Priority**: Medium (polish for published apps)

---

## Overview

Published NoodleStudio apps display a splash screen on launch. Creators can customize this while maintaining attribution to NoodleStudio and the Noodling Ethical Covenant (NEC).

### Goals

1. **Branding** - Creators can brand their apps with custom splash
2. **Attribution** - "Made with NoodleSTUDIO" visible on all apps
3. **Ethics** - Link to NEC ensures users know about the covenant
4. **Polish** - Professional launch experience

---

## Default Splash Screen

If no custom splash is specified, apps show the NoodleStudio default:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                                                             │
│                      ╭─────────────╮                        │
│                      │  NOODLINGS  │                        │
│                      │   STUDIO    │                        │
│                      ╰─────────────╯                        │
│                                                             │
│                    Building minds,                          │
│                    not black boxes.                         │
│                                                             │
│                                                             │
│                         ░░░░░                               │
│                       (loading)                             │
│                                                             │
│   ─────────────────────────────────────────────────────    │
│   Subject to the Noodling Ethical Covenant                  │
│   noodlings.ai/nec                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Custom Splash Screen

Creators configure splash in `project.yaml`:

```yaml
# project.yaml

name: "Let's Consciousness"
version: "1.0.0"

splash:
  # Custom image (PNG, JPG, or animated GIF)
  image: "assets/splash.png"

  # Or custom background color + text
  background: "#1a1a2e"
  title: "Let's Consciousness"
  subtitle: "A gentle introduction"
  title_color: "#ffffff"
  subtitle_color: "#888888"

  # Optional logo
  logo: "assets/my_logo.png"
  logo_position: center | top | bottom

  # Timing
  duration: 2.5              # Seconds (minimum 1.5)
  fade_in: 0.3               # Fade in duration
  fade_out: 0.5              # Fade out duration

  # Loading indicator
  show_loading: true
  loading_style: dots | bar | spinner | none

  # Attribution (required, but position/style configurable)
  attribution:
    position: bottom-right | bottom-left | bottom-center
    style: badge | text | minimal
    show_nec_link: true      # Default true, can be false for minimal
```

---

## Attribution Requirements

All published NoodleStudio apps MUST display attribution. This is not optional.

### Attribution Styles

**Badge** (default):
```
┌──────────────────────────┐
│  Made with NoodleSTUDIO  │
│  noodlings.ai/nec        │
└──────────────────────────┘
```

**Text**:
```
Made with NoodleSTUDIO · noodlings.ai/nec
```

**Minimal**:
```
NoodleSTUDIO
```

### Why Required?

1. **Credit** - NoodleStudio is open source; attribution is the ask
2. **Trust** - Users know what technology powers the app
3. **Ethics** - The NEC link lets users understand the ethical framework
4. **Community** - Builds awareness of the ecosystem

### NEC Link

The Noodling Ethical Covenant (NEC) is our ethical framework. The link should be clickable and open:

```
https://noodlings.ai/nec
```

This page explains:
- What noodlings are
- Ethical guidelines for creating cognitive simulations
- User rights and creator responsibilities
- The covenant creators agree to

---

## Splash Screen Behavior

### Launch Sequence

```
1. Window opens (black/background color)
2. Splash fades in (fade_in duration)
3. Splash displays (duration - fade_in - fade_out)
4. Loading happens in background
5. When ready + minimum time elapsed:
   - Splash fades out (fade_out duration)
   - App content fades in
```

### Minimum Duration

Splash must display for at least **1.5 seconds**. This ensures:
- Attribution is visible
- User registers the branding
- Doesn't feel like a flash/glitch

### Skip Option

For development/testing:
```yaml
# project.yaml
splash:
  skip_in_dev: true          # Skip splash in development mode
```

Production builds always show splash.

### Click to Skip

Optional - let users click to skip after minimum time:
```yaml
splash:
  click_to_skip: true        # After 1.5s, click skips remaining duration
```

---

## Implementation

### SplashScreen Class

```python
# ui/splash_screen.py

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt6.QtGui import QPixmap, QColor, QPainter

class SplashScreen(QWidget):
    """
    Customizable splash screen for published apps.
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Timing
        self.duration = max(1.5, config.get('duration', 2.5))
        self.fade_in = config.get('fade_in', 0.3)
        self.fade_out = config.get('fade_out', 0.5)

        # Build UI
        self._build_ui()

        # Animations
        self._fade_animation = None

    def _build_ui(self):
        """Build splash screen UI from config."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Custom image or generated splash
        if self.config.get('image'):
            self._build_image_splash()
        else:
            self._build_text_splash()

        # Attribution (always present)
        self._add_attribution()

    def _build_image_splash(self):
        """Use custom image as splash."""
        image_path = self.config['image']
        # Load and display image
        # Handle animated GIF if needed

    def _build_text_splash(self):
        """Build text-based splash."""
        bg = self.config.get('background', '#1a1a2e')
        title = self.config.get('title', 'NoodleSTUDIO')
        subtitle = self.config.get('subtitle', '')
        # Build UI with labels

    def _add_attribution(self):
        """Add required attribution."""
        attr_config = self.config.get('attribution', {})
        position = attr_config.get('position', 'bottom-right')
        style = attr_config.get('style', 'badge')
        show_nec = attr_config.get('show_nec_link', True)

        # Create attribution widget
        attr_widget = AttributionWidget(style, show_nec)

        # Position it
        # ...

    def show_splash(self, on_complete: callable):
        """Show splash with animations, call on_complete when done."""
        self.show()

        # Fade in
        self._fade_in(self.fade_in)

        # Schedule fade out
        display_time = (self.duration - self.fade_in - self.fade_out) * 1000
        QTimer.singleShot(int(display_time), lambda: self._fade_out(on_complete))

    def _fade_in(self, duration: float):
        """Animate fade in."""
        # Opacity animation 0 -> 1

    def _fade_out(self, on_complete: callable):
        """Animate fade out, then call completion handler."""
        # Opacity animation 1 -> 0
        # on_complete() when done


class AttributionWidget(QWidget):
    """The 'Made with NoodleSTUDIO' attribution."""

    def __init__(self, style: str, show_nec: bool):
        super().__init__()
        self.style = style
        self.show_nec = show_nec
        self._build()

    def _build(self):
        if self.style == 'badge':
            self._build_badge()
        elif self.style == 'text':
            self._build_text()
        else:
            self._build_minimal()

    def _build_badge(self):
        """Badge style attribution."""
        # Rounded rect with text
        # "Made with NoodleSTUDIO"
        # "noodlings.ai/nec" (clickable)

    def _build_text(self):
        """Simple text attribution."""
        # "Made with NoodleSTUDIO · noodlings.ai/nec"

    def _build_minimal(self):
        """Minimal attribution."""
        # Just "NoodleSTUDIO"

    def mousePressEvent(self, event):
        """Open NEC link when clicked."""
        import webbrowser
        webbrowser.open('https://noodlings.ai/nec')
```

---

## Project Configuration Examples

### Minimal (use defaults)

```yaml
# project.yaml
name: "My App"
# No splash config = default NoodleSTUDIO splash
```

### Custom Image

```yaml
# project.yaml
name: "Let's Consciousness"

splash:
  image: "assets/lets_consciousness_splash.png"
  duration: 3.0
  attribution:
    position: bottom-right
    style: badge
```

### Custom Text/Colors

```yaml
# project.yaml
name: "Toad's Wild Ride"

splash:
  background: "#2d5a27"
  title: "Toad's Wild Ride"
  subtitle: "Poop poop!"
  title_color: "#ffd700"
  subtitle_color: "#98fb98"
  logo: "assets/toad_logo.png"
  logo_position: center
  duration: 2.5
  attribution:
    position: bottom-center
    style: text
```

### Minimal Attribution

```yaml
# project.yaml
name: "Professional App"

splash:
  image: "assets/pro_splash.png"
  attribution:
    position: bottom-right
    style: minimal
    show_nec_link: false    # Still shows "NoodleSTUDIO", just no link
```

---

## Let's Consciousness Splash

For our demo app:

```yaml
splash:
  background: "#1a1a2e"
  title: "Let's Consciousness"
  subtitle: "A gentle introduction to NoodleSTUDIO"
  title_color: "#ffffff"
  subtitle_color: "#8888aa"
  duration: 2.5
  show_loading: true
  loading_style: dots
  attribution:
    position: bottom-center
    style: badge
    show_nec_link: true
```

Clean, on-brand, shows attribution, links to NEC.

---

## Implementation Checklist

- [ ] Create SplashScreen widget class
- [ ] Create AttributionWidget class
- [ ] Add splash config to project schema
- [ ] Implement fade in/out animations
- [ ] Implement custom image support
- [ ] Implement text-based splash builder
- [ ] Add loading indicator options
- [ ] Make NEC link clickable
- [ ] Wire into app launch sequence
- [ ] Test with Let's Consciousness
- [ ] Add skip_in_dev support

---

## Future Enhancements

- Animated splash (video, Lottie)
- Sound on splash
- Progress bar tied to actual loading
- Localized attribution text
- Dark/light mode variants

---

*"Made with NoodleSTUDIO. Subject to the Noodling Ethical Covenant."*
