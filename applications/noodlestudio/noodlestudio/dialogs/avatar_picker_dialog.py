"""
Avatar Picker Dialog

Allows users to select an avatar before entering noodleMUSH world.
Displays list of user's avatars with preview and description.

Author: Caitlyn + Claude
Date: December 2025
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QWidget, QTextEdit, QLineEdit, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon

from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class AvatarPickerDialog(QDialog):
    """
    Dialog for selecting an avatar before entering the world.

    Shows user's avatars with:
    - Display name
    - Description
    - Tags
    - Thumbnail (if available)

    Also allows creating a quick temporary avatar for first-time users.
    """

    avatar_selected = pyqtSignal(object)  # Emits selected AvatarMetadata or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Your Avatar")
        self.setMinimumSize(500, 400)
        self.setModal(True)

        self.selected_avatar = None
        self._avatars = []

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("Select an avatar to enter the world")
        header.setFont(QFont("", 12))
        header.setStyleSheet("color: #D2D2D2;")
        layout.addWidget(header)

        # Main content area
        content_layout = QHBoxLayout()

        # Avatar list (left side)
        list_frame = QFrame()
        list_frame.setFrameShape(QFrame.Shape.StyledPanel)
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(0, 0, 0, 0)

        self.avatar_list = QListWidget()
        self.avatar_list.setMinimumWidth(200)
        self.avatar_list.currentItemChanged.connect(self._on_selection_changed)
        self.avatar_list.itemDoubleClicked.connect(self._on_double_click)
        list_layout.addWidget(self.avatar_list)

        content_layout.addWidget(list_frame)

        # Details panel (right side)
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_layout = QVBoxLayout(details_frame)

        # Avatar name
        self.name_label = QLabel("No avatar selected")
        self.name_label.setFont(QFont("", 14, QFont.Weight.Bold))
        self.name_label.setStyleSheet("color: #FFFFFF;")
        details_layout.addWidget(self.name_label)

        # Pronouns
        self.pronouns_label = QLabel("")
        self.pronouns_label.setStyleSheet("color: #888888; font-style: italic;")
        details_layout.addWidget(self.pronouns_label)

        # Description
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(100)
        self.description_text.setPlaceholderText("No description")
        details_layout.addWidget(self.description_text)

        # Tags
        self.tags_label = QLabel("")
        self.tags_label.setStyleSheet("color: #64B5F6;")
        self.tags_label.setWordWrap(True)
        details_layout.addWidget(self.tags_label)

        details_layout.addStretch()

        content_layout.addWidget(details_frame, stretch=1)

        layout.addLayout(content_layout)

        # Quick avatar creation for users with no avatars
        self.quick_create_frame = QFrame()
        self.quick_create_frame.setFrameShape(QFrame.Shape.StyledPanel)
        quick_layout = QVBoxLayout(self.quick_create_frame)

        quick_label = QLabel("Or enter as a new traveler:")
        quick_label.setStyleSheet("color: #888888;")
        quick_layout.addWidget(quick_label)

        quick_input_layout = QHBoxLayout()
        self.quick_name_input = QLineEdit()
        self.quick_name_input.setPlaceholderText("Enter display name...")
        self.quick_name_input.setMaxLength(30)
        quick_input_layout.addWidget(self.quick_name_input)

        self.quick_enter_btn = QPushButton("Enter World")
        self.quick_enter_btn.clicked.connect(self._on_quick_enter)
        quick_input_layout.addWidget(self.quick_enter_btn)

        quick_layout.addLayout(quick_input_layout)
        layout.addWidget(self.quick_create_frame)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.select_btn = QPushButton("Enter World")
        self.select_btn.setEnabled(False)
        self.select_btn.clicked.connect(self._on_select)
        self.select_btn.setDefault(True)
        button_layout.addWidget(self.select_btn)

        layout.addLayout(button_layout)

    def _apply_style(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2D2D2D;
            }
            QFrame {
                background-color: #3A3A3A;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QListWidget {
                background-color: #2A2A2A;
                color: #D2D2D2;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #4A7CBA;
                color: #FFFFFF;
            }
            QListWidget::item:hover {
                background-color: #3A3A3A;
            }
            QTextEdit {
                background-color: #2A2A2A;
                color: #D2D2D2;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QLineEdit {
                background-color: #2A2A2A;
                color: #D2D2D2;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border: 1px solid #4A7CBA;
            }
            QPushButton {
                background-color: #3A3A3A;
                color: #D2D2D2;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
            QPushButton:pressed {
                background-color: #2A2A2A;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                color: #666666;
            }
            QPushButton:default {
                background-color: #4A7CBA;
                border: 1px solid #5A8CCF;
            }
            QPushButton:default:hover {
                background-color: #5A8CCF;
            }
        """)

    def set_avatars(self, avatars: List):
        """
        Set the list of avatars to display.

        Args:
            avatars: List of AvatarMetadata objects
        """
        self._avatars = avatars
        self.avatar_list.clear()

        if not avatars:
            # Show quick create prominently
            self.quick_create_frame.show()
            self.name_label.setText("No avatars yet")
            self.description_text.setPlainText(
                "Create your first avatar or enter with a temporary name."
            )
            return

        # Hide quick create if user has avatars (but still show it)
        self.quick_create_frame.show()

        # Add avatars to list
        default_idx = 0
        for i, avatar in enumerate(avatars):
            item = QListWidgetItem(avatar.display_name or "Unnamed Avatar")
            item.setData(Qt.ItemDataRole.UserRole, avatar)

            # Mark default avatar
            if avatar.is_default:
                item.setText(f"{avatar.display_name} (default)")
                default_idx = i

            self.avatar_list.addItem(item)

        # Select default avatar
        if self.avatar_list.count() > 0:
            self.avatar_list.setCurrentRow(default_idx)

    def _on_selection_changed(self, current, previous):
        """Handle avatar selection change."""
        if current is None:
            self.selected_avatar = None
            self.select_btn.setEnabled(False)
            self.name_label.setText("No avatar selected")
            self.pronouns_label.setText("")
            self.description_text.setPlainText("")
            self.tags_label.setText("")
            return

        avatar = current.data(Qt.ItemDataRole.UserRole)
        self.selected_avatar = avatar
        self.select_btn.setEnabled(True)

        # Update details
        self.name_label.setText(avatar.display_name or "Unnamed")
        self.pronouns_label.setText(avatar.pronouns or "")
        self.description_text.setPlainText(avatar.description or "")

        if avatar.tags:
            self.tags_label.setText(" | ".join(avatar.tags))
        else:
            self.tags_label.setText("")

    def _on_double_click(self, item):
        """Handle double-click on avatar."""
        self._on_select()

    def _on_select(self):
        """Handle select button click."""
        if self.selected_avatar:
            self.avatar_selected.emit(self.selected_avatar)
            self.accept()

    def _on_quick_enter(self):
        """Handle quick enter with temporary name."""
        name = self.quick_name_input.text().strip()
        if not name:
            name = "Traveler"

        # Create a temporary avatar metadata object
        from ..core.account_manager import AvatarMetadata, AvatarAssetLocation

        temp_avatar = AvatarMetadata(
            display_name=name,
            description="A wandering soul",
            asset_location=AvatarAssetLocation.LOCAL,
            asset_ref=""
        )

        self.selected_avatar = temp_avatar
        self.avatar_selected.emit(temp_avatar)
        self.accept()

    def get_selected_avatar(self):
        """Get the selected avatar (call after dialog accepted)."""
        return self.selected_avatar
