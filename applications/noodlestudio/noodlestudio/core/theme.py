"""
Theme and styling for NoodleSTUDIO.

Defines colors, fonts, and QSS stylesheet.
"""

from PyQt6.QtGui import QFont

# Color palette (dark theme)
COLORS = {
    # Base
    'background': '#0a0e1a',
    'panel_bg': '#131824',
    'border': '#2a3f5f',
    'text': '#e0e0e0',
    'text_dim': '#808080',
    'accent': '#64b5f6',

    # Layers (for phenomenal state visualization)
    'fast': '#66bb6a',    # Green
    'medium': '#ffa726',  # Orange
    'slow': '#ba68c8',    # Purple

    # Affect
    'valence_pos': '#66bb6a',
    'valence_neg': '#ef5350',
    'arousal': '#ffa726',
    'fear': '#ef5350',
    'surprise': '#64b5f6',

    # Status
    'success': '#66bb6a',
    'warning': '#ffa726',
    'error': '#ef5350',
}

# Fonts
FONTS = {
    'body': QFont('Roboto', 14),
    'header': QFont('Roboto', 16, QFont.Weight.Bold),
    'caption': QFont('Roboto', 12, QFont.Weight.Light),
    'monospace': QFont('Source Code Pro', 13),
}

# Dark theme stylesheet
DARK_THEME = f"""
QMainWindow {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
}}

QDockWidget {{
    background-color: {COLORS['panel_bg']};
    color: {COLORS['text']};
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(float.png);
}}

QDockWidget::title {{
    background-color: {COLORS['border']};
    padding: 6px;
    color: {COLORS['text']};
}}

QMenuBar {{
    background-color: {COLORS['panel_bg']};
    color: {COLORS['text']};
    border-bottom: 1px solid {COLORS['border']};
}}

QMenuBar::item:selected {{
    background-color: {COLORS['accent']};
    color: #000;
}}

QMenu {{
    background-color: {COLORS['panel_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
}}

QMenu::item:selected {{
    background-color: {COLORS['accent']};
    color: #000;
}}

QToolBar {{
    background-color: {COLORS['panel_bg']};
    border-bottom: 1px solid {COLORS['border']};
    spacing: 3px;
    padding: 3px;
}}

QStatusBar {{
    background-color: {COLORS['panel_bg']};
    color: {COLORS['text']};
    border-top: 1px solid {COLORS['border']};
}}

QPushButton {{
    background-color: {COLORS['border']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    padding: 6px 12px;
    border-radius: 3px;
}}

QPushButton:hover {{
    background-color: {COLORS['accent']};
    color: #000;
}}

QPushButton:pressed {{
    background-color: {COLORS['border']};
}}

QLabel {{
    color: {COLORS['text']};
}}

QScrollArea {{
    background-color: {COLORS['background']};
    border: none;
}}

QScrollBar:vertical {{
    background-color: {COLORS['panel_bg']};
    width: 12px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border']};
    min-height: 20px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['accent']};
}}
"""
