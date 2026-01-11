# The Unfold: App ↔ Studio Transition

**Status**: COMPLETED (2026-01-10)
**Date**: 2026-01-08
**Authors**: Caity + Claude
**Priority**: High (core UX for Let's Consciousness)

## Implementation Summary

**Files Created:**
- `core/panel_fold_manager.py` - PanelFoldManager with animated transitions
- `core/main_window_fold_mixin.py` - MainWindowFoldMixin integration
- `widgets/view_project_button.py` - ViewProjectButton with fade animations
- `tests/test_panel_fold.py` - 17 unit tests

**Files Modified:**
- `core/main_window.py` - Added MainWindowFoldMixin to inheritance, calls `_setup_fold()`

**Features Implemented:**
- 400ms ease-out unfold animation (fast start, gentle landing)
- 300ms ease-in fold animation (gentle start, fast finish)
- QSplitter-based panel management (left, center, right, bottom)
- Keyboard shortcut: Ctrl+Shift+U
- ViewProjectButton with opacity fade effects
- Signal-based state change notifications
- Saved sizes restoration on unfold

**Tests:** 17 passing (instant transitions + easing function tests)

---

## The Concept

When you publish a NoodleStudio app, the user sees "just the app" - your workspace filling the window, no editor chrome. But the studio isn't gone. It's *folded away*. The panel dividers are pushed to the edges, the panels not rendering, but the structure is there.

**Unfold** reveals the studio. The dividers animate inward to their default positions. The panels slide into view. The workspace shrinks to its proper proportion. The user realizes: *oh, it was always here*.

**Fold** hides it again. The panels slide away, the workspace expands, and you're back to "just the app."

---

## Why This Matters

1. **No mode switch** - Same app, same process, same state. Just revealing/hiding panels.
2. **The metaphor is literal** - "Unfold" means unfold. Paper folding. Origami. Opening a letter.
3. **Reinforces the message** - Your app IS a NoodleStudio project. The studio is always right there.
4. **Reversible** - Fold back up anytime. Play with it. Safe to explore.
5. **Technically elegant** - No launching a new process. Just animating panel positions.

---

## Visual Design

### Folded State (App Mode)

```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │                                                         │ │
│ │                                                         │ │
│ │                    YOUR APP                             │ │
│ │                  (Workspace)                            │ │
│ │                                                         │ │
│ │                                                         │ │
│ │                                                         │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                    [View Project]                           │
└─────────────────────────────────────────────────────────────┘

Panel dividers exist but are at window edges (0px).
Panels exist but are not rendering (width/height = 0).
```

### Unfolding Animation

```
User clicks [View Project]

Frame 1:  Dividers begin moving inward
Frame 2:  Panels start rendering as they gain size
Frame 3:  Workspace smoothly shrinks
Frame 4:  ...
Frame N:  Dividers reach default positions

Duration: ~400ms
Easing: ease-out (fast start, gentle landing)
```

### Unfolded State (Studio Mode)

```
┌─────────────────────────────────────────────────────────────┐
│  Project    │                              │   Inspector    │
│  Browser    │         Workspace            │                │
│             │        (Your App)            │   Properties   │
│  Noodlings  │                              │   Facets       │
│  Stages     │                              │   Channels     │
│  Assets     │                              │                │
│             ├──────────────────────────────┤                │
│             │        Timeline / Log        │                │
└─────────────┴──────────────────────────────┴────────────────┘

Standard NoodleStudio layout with your app in the workspace.
```

---

## Implementation

### Panel State

```python
class PanelLayout:
    """Manages panel divider positions and fold state."""

    def __init__(self):
        self.left_divider = 0      # 0 = folded, ~250 = unfolded
        self.right_divider = 0     # 0 = folded, ~300 = unfolded
        self.bottom_divider = 0    # 0 = folded, ~150 = unfolded

        self.default_left = 250
        self.default_right = 300
        self.default_bottom = 150

        self.is_folded = True

    def unfold(self, animated=True):
        """Animate dividers to default positions."""
        if animated:
            self._animate_to(
                left=self.default_left,
                right=self.default_right,
                bottom=self.default_bottom,
                duration_ms=400,
                easing='ease_out'
            )
        else:
            self.left_divider = self.default_left
            self.right_divider = self.default_right
            self.bottom_divider = self.default_bottom

        self.is_folded = False

    def fold(self, animated=True):
        """Animate dividers to edges (fold away)."""
        if animated:
            self._animate_to(
                left=0,
                right=0,
                bottom=0,
                duration_ms=300,
                easing='ease_in'
            )
        else:
            self.left_divider = 0
            self.right_divider = 0
            self.bottom_divider = 0

        self.is_folded = True
```

### Panel Rendering Optimization

```python
class Panel:
    def should_render(self) -> bool:
        """Only render if panel has meaningful size."""
        return self.width > 10 and self.height > 10

    def paint(self, painter):
        if not self.should_render():
            return  # Skip rendering when folded
        # ... normal rendering
```

### App Publish Settings

```yaml
# project.yaml
publish:
  mode: app                    # "app" = starts folded, "studio" = starts unfolded
  show_unfold_button: true     # Show [View Project] button
  allow_fold: true             # Allow folding back after unfold
  unfold_button_label: "View Project"  # Customizable text
```

---

## The Button

The **[View Project]** button appears in app mode. Subtle but discoverable.

Placement options:
- **Bottom center** - Unobtrusive, easy to find
- **Menu bar** - If app has a menu bar
- **Floating** - Small floating button in corner
- **Context menu** - Right-click anywhere

For Let's Consciousness, Guide points at it during the tutorial:
> "See that 'View Project' button? When you're ready to see how this whole thing is built, that's your door."

---

## Keyboard Shortcut

```
Cmd+Shift+U  /  Ctrl+Shift+U  →  Toggle fold/unfold
```

Power users can unfold without clicking.

---

## Edge Cases

### During Animation
- User input still works (don't block)
- If user clicks during unfold, complete the animation
- If user starts dragging a divider during animation, cancel animation and let them take over

### Minimum Sizes
- Panels have minimum sizes; dividers can't go past them when unfolded
- When folded, minimums are 0 (panels fully hidden)

### Workspace Content
- Workspace content should handle resize gracefully
- Use flex/responsive layout in workspace apps
- Large resize during unfold is a good stress test

### Persistence
- Remember fold state across sessions? Probably not for published apps (always start folded)
- In development, remember last state

---

## The Moment

In Let's Consciousness, the unfold is THE moment. Guide has been building to it:

> "Most apps are black boxes. You use them but you can't see inside."
>
> "NoodleStudio apps aren't like that. They're transparent. Tinkerable."
>
> "Want to see?"
>
> *clicks View Project*
>
> *the studio unfolds*
>
> "See? This is the project that makes this app. And look - there I am."

The medium is the message. The demo is the documentation.

---

## Implementation Checklist

- [ ] Add `is_folded` state to main window / panel layout
- [ ] Implement `unfold()` with animation
- [ ] Implement `fold()` with animation
- [ ] Add [View Project] button component
- [ ] Add publish settings for fold behavior
- [ ] Optimize panel rendering when folded (skip draw)
- [ ] Add keyboard shortcut
- [ ] Test with Let's Consciousness
- [ ] Make Guide point at the button during tutorial

---

*"The studio was always there. Just waiting for you to unfold it."*
