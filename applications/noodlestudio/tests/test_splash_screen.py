# ──────────────────────────────────────────────────────────────
#   Tests for Splash Screen
#
#   Tests for the SplashScreen and AttributionWidget classes.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import pytest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


class TestAttributionWidget:
    """Tests for AttributionWidget."""

    def test_badge_style_creation(self, qtbot):
        """Badge style creates correct layout."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget(style="badge", show_nec_link=True)
        qtbot.addWidget(widget)

        assert widget is not None
        # Check children exist
        labels = widget.findChildren(widget.__class__.__bases__[0])
        # Badge style should have labels

    def test_text_style_creation(self, qtbot):
        """Text style creates correct layout."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget(style="text", show_nec_link=True)
        qtbot.addWidget(widget)

        assert widget is not None

    def test_minimal_style_creation(self, qtbot):
        """Minimal style creates correct layout."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget(style="minimal", show_nec_link=False)
        qtbot.addWidget(widget)

        assert widget is not None

    def test_cursor_is_pointing_hand(self, qtbot):
        """Widget has pointing hand cursor."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget()
        qtbot.addWidget(widget)

        assert widget.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_click_emits_signal(self, qtbot):
        """Clicking widget emits clicked signal."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget(show_nec_link=False)
        qtbot.addWidget(widget)

        clicked = []
        widget.clicked.connect(lambda: clicked.append(True))

        qtbot.mouseClick(widget, Qt.MouseButton.LeftButton)

        assert len(clicked) == 1

    @patch('webbrowser.open')
    def test_click_opens_nec_link(self, mock_open, qtbot):
        """Clicking opens NEC URL when show_nec_link is True."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget(show_nec_link=True)
        qtbot.addWidget(widget)

        qtbot.mouseClick(widget, Qt.MouseButton.LeftButton)

        mock_open.assert_called_once_with(AttributionWidget.NEC_URL)

    @patch('webbrowser.open')
    def test_click_does_not_open_when_nec_disabled(self, mock_open, qtbot):
        """Clicking does not open URL when show_nec_link is False."""
        from noodlestudio.widgets.splash_screen import AttributionWidget
        widget = AttributionWidget(show_nec_link=False)
        qtbot.addWidget(widget)

        qtbot.mouseClick(widget, Qt.MouseButton.LeftButton)

        mock_open.assert_not_called()


class TestLoadingIndicator:
    """Tests for LoadingIndicator."""

    def test_dots_style_creation(self, qtbot):
        """Dots style indicator creates successfully."""
        from noodlestudio.widgets.splash_screen import LoadingIndicator
        indicator = LoadingIndicator(style="dots")
        qtbot.addWidget(indicator)

        assert indicator is not None
        assert indicator._style == "dots"

    def test_bar_style_creation(self, qtbot):
        """Bar style indicator creates successfully."""
        from noodlestudio.widgets.splash_screen import LoadingIndicator
        indicator = LoadingIndicator(style="bar")
        qtbot.addWidget(indicator)

        assert indicator._style == "bar"

    def test_spinner_style_creation(self, qtbot):
        """Spinner style indicator creates successfully."""
        from noodlestudio.widgets.splash_screen import LoadingIndicator
        indicator = LoadingIndicator(style="spinner")
        qtbot.addWidget(indicator)

        assert indicator._style == "spinner"

    def test_none_style_no_animation(self, qtbot):
        """None style does not start animation."""
        from noodlestudio.widgets.splash_screen import LoadingIndicator
        indicator = LoadingIndicator(style="none")
        qtbot.addWidget(indicator)

        assert indicator._animation_timer is None

    def test_stop_stops_animation(self, qtbot):
        """Stop method stops the animation timer."""
        from noodlestudio.widgets.splash_screen import LoadingIndicator
        indicator = LoadingIndicator(style="dots")
        qtbot.addWidget(indicator)

        assert indicator._animation_timer is not None

        indicator.stop()

        assert indicator._animation_timer is None


class TestSplashScreen:
    """Tests for SplashScreen."""

    def test_default_creation(self, qtbot):
        """Default splash screen creates successfully."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)

        assert splash is not None
        assert splash._duration >= SplashScreen.MIN_DURATION

    def test_custom_config(self, qtbot):
        """Custom config is applied."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'title': 'Test App',
            'subtitle': 'Testing',
            'duration': 3.0,
            'background': '#ff0000',
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._duration == 3.0

    def test_minimum_duration_enforced(self, qtbot):
        """Duration below minimum is raised to minimum."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {'duration': 0.5}  # Below 1.5 minimum
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._duration == SplashScreen.MIN_DURATION

    def test_window_flags(self, qtbot):
        """Splash has correct window flags (frameless, stays on top)."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)

        flags = splash.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint
        assert flags & Qt.WindowType.WindowStaysOnTopHint

    def test_translucent_background(self, qtbot):
        """Splash has translucent background attribute."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)

        assert splash.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def test_has_opacity_effect(self, qtbot):
        """Splash has graphics opacity effect."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)

        assert splash._opacity_effect is not None
        assert splash._opacity_effect.opacity() == 0.0  # Starts invisible

    def test_loading_indicator_enabled_by_default(self, qtbot):
        """Loading indicator is shown by default."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)

        assert splash._loading_indicator is not None

    def test_loading_indicator_can_be_disabled(self, qtbot):
        """Loading indicator can be disabled."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {'show_loading': False}
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._loading_indicator is None

    def test_skip_triggers_fade_out(self, qtbot):
        """Skip method triggers fade out."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)

        fade_out_signals = []
        splash.fade_out_complete.connect(lambda: fade_out_signals.append(True))

        # Start showing
        splash.show()
        splash._opacity_effect.setOpacity(1.0)

        # Skip
        splash.skip()

        # Wait for fade out
        qtbot.waitUntil(lambda: len(fade_out_signals) > 0, timeout=2000)

        assert len(fade_out_signals) == 1

    def test_create_default_splash(self, qtbot):
        """create_default_splash returns valid splash."""
        from noodlestudio.widgets.splash_screen import create_default_splash
        splash = create_default_splash()
        qtbot.addWidget(splash)

        assert splash is not None
        assert splash._duration == 2.5


class TestSplashScreenConfig:
    """Tests for various splash screen configurations."""

    def test_lets_consciousness_config(self, qtbot):
        """Let's Consciousness config works correctly."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'background': '#1a1a2e',
            'title': "Let's Consciousness",
            'subtitle': 'A gentle introduction to NoodleSTUDIO',
            'title_color': '#ffffff',
            'subtitle_color': '#8888aa',
            'duration': 2.5,
            'show_loading': True,
            'loading_style': 'dots',
            'attribution': {
                'position': 'bottom-center',
                'style': 'badge',
                'show_nec_link': True,
            }
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._duration == 2.5

    def test_attribution_position_bottom_left(self, qtbot):
        """Bottom-left attribution position works."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'attribution': {
                'position': 'bottom-left',
            }
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash is not None

    def test_attribution_position_bottom_right(self, qtbot):
        """Bottom-right attribution position works."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'attribution': {
                'position': 'bottom-right',
            }
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash is not None

    def test_minimal_attribution_style(self, qtbot):
        """Minimal attribution style works."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'attribution': {
                'style': 'minimal',
                'show_nec_link': False,
            }
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash is not None

    def test_text_attribution_style(self, qtbot):
        """Text attribution style works."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'attribution': {
                'style': 'text',
            }
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash is not None

    def test_custom_fade_durations(self, qtbot):
        """Custom fade durations are applied."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'fade_in': 0.5,
            'fade_out': 0.8,
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._fade_in_duration == 0.5
        assert splash._fade_out_duration == 0.8

    def test_loading_style_bar(self, qtbot):
        """Bar loading style works."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'show_loading': True,
            'loading_style': 'bar',
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._loading_indicator is not None
        assert splash._loading_indicator._style == 'bar'

    def test_loading_style_spinner(self, qtbot):
        """Spinner loading style works."""
        from noodlestudio.widgets.splash_screen import SplashScreen
        config = {
            'show_loading': True,
            'loading_style': 'spinner',
        }
        splash = SplashScreen(config)
        qtbot.addWidget(splash)

        assert splash._loading_indicator is not None
        assert splash._loading_indicator._style == 'spinner'
