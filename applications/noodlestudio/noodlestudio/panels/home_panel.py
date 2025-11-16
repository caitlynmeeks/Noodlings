"""
Home Panel - Discovery hub for Noodlings Books.

Shows featured books, user library, and quick access to marketplace.
"""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from pathlib import Path


class HomePanel(QDockWidget):
    """
    Home panel for NoodleSTUDIO.

    Shows:
    - Welcome message
    - Featured Noodlings Books
    - User's library
    - Community creations
    """

    book_selected = pyqtSignal(str)  # Emits book_id when user selects a book

    def __init__(self, parent: QWidget = None):
        super().__init__("Home", parent)

        # Allow moving and floating, but not closing
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._setup_ui()

    def _setup_ui(self):
        """Build UI components."""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Content widget
        content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Welcome section
        layout.addWidget(self._create_welcome_section())

        # Featured books section
        layout.addWidget(self._create_section_header("Featured This Week"))
        layout.addWidget(self._create_featured_section())

        # Your library section
        layout.addWidget(self._create_section_header("Your Library"))
        layout.addWidget(self._create_library_section())

        # Actions bar
        layout.addWidget(self._create_actions_bar())

        # Add stretch to push everything to top
        layout.addStretch()

        content.setLayout(layout)
        scroll.setWidget(content)
        self.setWidget(scroll)

    def _create_welcome_section(self) -> QWidget:
        """Create welcome banner."""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #1a2332;
                border-radius: 8px;
                padding: 20px;
            }
        """)

        layout = QVBoxLayout()

        # Title
        title = QLabel("🧠 Welcome to NoodleSTUDIO")
        title.setFont(QFont("Roboto", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #64b5f6;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("The professional IDE for Noodlings consciousness agents")
        subtitle.setFont(QFont("Roboto", 12))
        subtitle.setStyleSheet("color: #b0b0b0;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Quick links
        links = QHBoxLayout()
        links.setSpacing(10)
        links.addStretch()

        quick_start = QPushButton("📖 Quick Start Guide")
        tutorials = QPushButton("🎓 Video Tutorials")
        discord = QPushButton("💬 Join Discord")

        for btn in [quick_start, tutorials, discord]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a3f5f;
                    border: 1px solid #64b5f6;
                    padding: 8px 16px;
                    border-radius: 4px;
                    color: #64b5f6;
                }
                QPushButton:hover {
                    background-color: #64b5f6;
                    color: #000;
                }
            """)
            links.addWidget(btn)

        links.addStretch()
        layout.addLayout(links)

        widget.setLayout(layout)
        return widget

    def _create_section_header(self, title: str) -> QWidget:
        """Create a section header."""
        label = QLabel(title)
        label.setFont(QFont("Roboto", 16, QFont.Weight.Bold))
        label.setStyleSheet("color: #e0e0e0; padding: 10px 0;")
        return label

    def _create_featured_section(self) -> QWidget:
        """Create featured books grid."""
        widget = QWidget()
        grid = QGridLayout()
        grid.setSpacing(15)

        # Sample featured books
        featured_books = [
            {
                "title": "Gilligan's Island",
                "price": "$4.99",
                "rating": "⭐⭐⭐⭐⭐",
                "description": "7 iconic characters, tropical island"
            },
            {
                "title": "Shakespeare Collection",
                "price": "$9.99",
                "rating": "⭐⭐⭐⭐⭐",
                "description": "Hamlet, Ophelia, Macbeth as Noodlings"
            },
            {
                "title": "Therapy Companion",
                "price": "Free",
                "rating": "⭐⭐⭐⭐",
                "description": "Carl Rogers-inspired empathetic listener"
            },
            {
                "title": "D&D Campaign Gen",
                "price": "$14.99",
                "rating": "⭐⭐⭐⭐⭐",
                "description": "DM Noodling + procedural NPCs"
            }
        ]

        for i, book in enumerate(featured_books):
            card = self._create_book_card(book)
            grid.addWidget(card, i // 2, i % 2)

        widget.setLayout(grid)
        return widget

    def _create_library_section(self) -> QWidget:
        """Create user library grid."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Check if user has any books
        library_path = Path.home() / ".noodlestudio" / "library"

        if not library_path.exists() or not any(library_path.glob("*.noodling")):
            # Empty state
            empty = QLabel(
                "📚 Your library is empty\n\n"
                "Try the featured books above or create your own!"
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("""
                QLabel {
                    color: #808080;
                    padding: 40px;
                    font-size: 14px;
                }
            """)
            layout.addWidget(empty)
        else:
            # Show books from library
            grid = QGridLayout()
            grid.setSpacing(15)

            for i, book_file in enumerate(library_path.glob("*.noodling")):
                book = {
                    "title": book_file.stem.replace("_", " ").title(),
                    "price": "Owned",
                    "rating": "",
                    "description": "Your custom book"
                }
                card = self._create_book_card(book, owned=True)
                grid.addWidget(card, i // 2, i % 2)

            layout.addLayout(grid)

        widget.setLayout(layout)
        return widget

    def _create_book_card(self, book: dict, owned: bool = False) -> QWidget:
        """Create a book card."""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #1a2332;
                border-radius: 8px;
                padding: 15px;
            }
            QWidget:hover {
                background-color: #2a3f5f;
            }
        """)

        layout = QVBoxLayout()

        # Cover placeholder
        cover = QLabel("📖")
        cover.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cover.setFont(QFont("", 48))
        cover.setStyleSheet("background-color: #0a0e1a; border-radius: 4px; padding: 20px;")
        layout.addWidget(cover)

        # Title
        title = QLabel(book["title"])
        title.setFont(QFont("Roboto", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        title.setWordWrap(True)
        layout.addWidget(title)

        # Description
        desc = QLabel(book["description"])
        desc.setFont(QFont("Roboto", 11))
        desc.setStyleSheet("color: #b0b0b0;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Price and rating
        info = QHBoxLayout()
        price = QLabel(book["price"])
        price.setFont(QFont("Roboto", 12, QFont.Weight.Bold))
        price.setStyleSheet("color: #66bb6a;")
        info.addWidget(price)

        if book["rating"]:
            rating = QLabel(book["rating"])
            rating.setFont(QFont("Roboto", 10))
            info.addWidget(rating)

        info.addStretch()
        layout.addLayout(info)

        # Buttons
        buttons = QHBoxLayout()

        if owned:
            load_btn = QPushButton("Load in MUSH")
            edit_btn = QPushButton("Edit")
            buttons.addWidget(load_btn)
            buttons.addWidget(edit_btn)
        else:
            demo_btn = QPushButton("Try Demo")
            purchase_btn = QPushButton("Purchase")
            buttons.addWidget(demo_btn)
            buttons.addWidget(purchase_btn)

        layout.addLayout(buttons)

        card.setLayout(layout)
        return card

    def _create_actions_bar(self) -> QWidget:
        """Create bottom actions bar."""
        widget = QWidget()
        layout = QHBoxLayout()

        browse_btn = QPushButton("🛒 Browse Full Marketplace")
        create_btn = QPushButton("📖 Create New Book")

        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #64b5f6;
                color: #000;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #90caf9;
            }
        """)

        layout.addWidget(browse_btn)
        layout.addWidget(create_btn)
        layout.addStretch()

        widget.setLayout(layout)
        return widget
