"""
Login Dialog for NoodleStudio
=============================

OAuth login dialog with branded provider buttons.
"""

import webbrowser
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon

from ..core.account_manager import AccountManager, start_oauth_callback_server

# Icon paths
ICONS_DIR = Path(__file__).parent.parent / "resources" / "icons"


class LoginDialog(QDialog):
    """
    Login dialog with branded OAuth provider buttons.
    """

    login_successful = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Sign In")
        self.setFixedSize(420, 300)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
        """)

        self._setup_ui()
        self._connect_signals()
        start_oauth_callback_server()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(60, 50, 60, 40)

        # Title
        title = QLabel("Sign in to noodlings.ai")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("", 20, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        layout.addSpacing(20)

        # Google button with icon
        self.google_btn = QPushButton("Sign in with Google")
        self.google_btn.setMinimumHeight(50)
        self.google_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        google_icon = QIcon(str(ICONS_DIR / "google.svg"))
        self.google_btn.setIcon(google_icon)
        self.google_btn.setIconSize(QSize(20, 20))
        self.google_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #3c4043;
                border: 1px solid #dadce0;
                border-radius: 4px;
                padding: 12px 24px 12px 20px;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            QPushButton:pressed {
                background-color: #f1f3f4;
            }
        """)
        layout.addWidget(self.google_btn)

        # GitHub button with icon
        self.github_btn = QPushButton("Sign in with GitHub")
        self.github_btn.setMinimumHeight(50)
        self.github_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        github_icon = QIcon(str(ICONS_DIR / "github.svg"))
        self.github_btn.setIcon(github_icon)
        self.github_btn.setIconSize(QSize(20, 20))
        self.github_btn.setStyleSheet("""
            QPushButton {
                background-color: #24292f;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 12px 24px 12px 20px;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #32383f;
            }
            QPushButton:pressed {
                background-color: #1c2024;
            }
        """)
        layout.addWidget(self.github_btn)

        layout.addStretch()

        # Footer
        footer = QLabel("Manage and sync your projects, assets and services")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #888888; font-size: 14px;")
        layout.addWidget(footer)

    def _connect_signals(self):
        self.google_btn.clicked.connect(lambda: self._login_with("google"))
        self.github_btn.clicked.connect(lambda: self._login_with("github"))
        AccountManager.instance().logged_in.connect(self._on_login_success)
        AccountManager.instance().login_failed.connect(self._on_login_failed)

    def _login_with(self, provider: str):
        login_url = AccountManager.instance().get_login_url(provider)
        webbrowser.open(login_url)
        self.setWindowTitle("Completing sign in...")

    def _on_login_success(self, user):
        self.login_successful.emit()
        self.accept()

    def _on_login_failed(self, error: str):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Sign In Failed", error)
        self.setWindowTitle("Sign In")


class AccountInfoDialog(QDialog):
    """
    Dialog showing account info and credits balance.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Account")
        self.setFixedSize(400, 350)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
            }
            QLabel {
                color: #d2d2d2;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)

        account_manager = AccountManager.instance()
        user = account_manager.user

        if not user:
            layout.addWidget(QLabel("Not logged in"))
            return

        # Email
        email_label = QLabel(user.email)
        email_label.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        email_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(email_label)

        # Display name
        if user.display_name:
            name_label = QLabel(user.display_name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("color: #888;")
            layout.addWidget(name_label)

        layout.addSpacing(24)

        # Credits section
        credits_frame = QFrame()
        credits_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        credits_layout = QVBoxLayout(credits_frame)

        credits_title = QLabel("Credits Balance")
        credits_title.setStyleSheet("color: #888; font-size: 12px;")
        credits_layout.addWidget(credits_title)

        credits_value = QLabel(f"{user.credits_balance:,}")
        credits_value.setFont(QFont("SF Pro Display", 32, QFont.Weight.Bold))
        credits_value.setStyleSheet("color: #76AF6A;")
        credits_layout.addWidget(credits_value)

        layout.addWidget(credits_frame)

        # Buy credits button
        buy_btn = QPushButton("Buy Credits")
        buy_btn.setStyleSheet("""
            QPushButton {
                background-color: #76AF6A;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #8BC37E;
            }
        """)
        buy_btn.clicked.connect(self._buy_credits)
        layout.addWidget(buy_btn)

        layout.addStretch()

        # Linked providers
        providers_label = QLabel(f"Linked: {', '.join(user.providers)}")
        providers_label.setStyleSheet("color: #666; font-size: 11px;")
        providers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(providers_label)

        # Logout button
        logout_btn = QPushButton("Sign Out")
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #333;
                color: #d2d2d2;
            }
        """)
        logout_btn.clicked.connect(self._logout)
        layout.addWidget(logout_btn)

    def _buy_credits(self):
        """Open credits purchase flow."""
        import webbrowser
        # For now, just open the web page. Later could integrate Stripe checkout
        webbrowser.open("https://noodlings.ai/credits")

    def _logout(self):
        """Log out and close dialog."""
        AccountManager.instance().logout()
        self.accept()
