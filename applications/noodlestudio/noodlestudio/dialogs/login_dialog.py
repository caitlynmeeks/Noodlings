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
        self.setWindowTitle("Sign In to Noodlings")
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
        layout.setSpacing(12)
        layout.setContentsMargins(40, 30, 40, 20)

        # Title
        title = QLabel("Noodlings")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("", 18, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #76AF6A;")  # Green brand color
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Sign in to sync your Noodlings to the cloud")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888888; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # Google button
        self.google_btn = QPushButton("Continue with Google")
        self.google_btn.setMinimumHeight(44)
        self.google_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.google_btn.setStyleSheet("""
            QPushButton {
                background-color: #4285F4;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #5294F5;
            }
            QPushButton:pressed {
                background-color: #3275E4;
            }
        """)
        layout.addWidget(self.google_btn)

        # GitHub button
        self.github_btn = QPushButton("Continue with GitHub")
        self.github_btn.setMinimumHeight(44)
        self.github_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.github_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
            }
        """)
        layout.addWidget(self.github_btn)

        # Apple button
        self.apple_btn = QPushButton("Continue with Apple")
        self.apple_btn.setMinimumHeight(44)
        self.apple_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apple_btn.setStyleSheet("""
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1a1a1a;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        layout.addWidget(self.apple_btn)

        # Facebook button
        self.facebook_btn = QPushButton("Continue with Facebook")
        self.facebook_btn.setMinimumHeight(44)
        self.facebook_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.facebook_btn.setStyleSheet("""
            QPushButton {
                background-color: #1877F2;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2888F3;
            }
            QPushButton:pressed {
                background-color: #0866E2;
            }
        """)
        layout.addWidget(self.facebook_btn)

        layout.addSpacing(12)

        # Footer
        footer = QLabel("Your Noodlings, recipes, and facet assemblies will sync to the\ncloud. Use credits for routed LLM calls and future Asset Store\npurchases.")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(footer)

        layout.addSpacing(8)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #888888;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #404040;
                color: #aaaaaa;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

        # Adjust dialog size to fit content
        self.adjustSize()
        self.setMinimumWidth(380)

    def _connect_signals(self):
        self.google_btn.clicked.connect(lambda: self._login_with("google"))
        self.github_btn.clicked.connect(lambda: self._login_with("github"))
        self.apple_btn.clicked.connect(lambda: self._show_coming_soon("Apple"))
        self.facebook_btn.clicked.connect(lambda: self._show_coming_soon("Facebook"))
        AccountManager.instance().logged_in.connect(self._on_login_success)
        AccountManager.instance().login_failed.connect(self._on_login_failed)

    def _show_coming_soon(self, provider: str):
        """Show coming soon message for unsupported providers."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Coming Soon",
            f"{provider} sign-in is coming soon.\n\nPlease use Google or GitHub for now."
        )

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
