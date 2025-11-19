"""
Unity Editor Dark Theme

Replicates Unity's professional dark mode color scheme.
Clean, minimal, purpose-built for game dev workflows.

Colors from Unity 2023.2 Dark Theme
"""

UNITY_DARK_THEME = """
/* ===== UNITY DARK THEME ===== */

QMainWindow, QWidget {
    background-color: #383838;  /* Unity background gray */
    color: #D2D2D2;  /* Unity text */
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* Fix ALL white artifacts - everything black! */
QMainWindow::separator {
    background: #000000;
    width: 4px;
    height: 4px;
}

QMainWindow::separator:hover {
    background: #2C5D87;
}

QDockWidget::title:pressed {
    background: #1E1E1E;
}

QSplitter {
    background: #000000;
}

QSplitter::handle {
    background: #000000;
}

QSplitter::handle:hover {
    background: #2C5D87;
}

/* Drag overlay */
QMainWindow {
    background: #000000;
}

QRubberBand {
    background: #2C5D87;
    border: 2px solid #4A9EFF;
}

/* ===== PANELS ===== */
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    background: #2B2B2B;
}

QDockWidget::title {
    background: #2B2B2B;
    color: #B4B4B4;
    padding: 6px;
    font-weight: bold;
    font-size: 12px;
    border-bottom: 1px solid #1E1E1E;
}

/* ===== TREE VIEWS (Scene Hierarchy) ===== */
QTreeWidget, QTreeView {
    background-color: #2B2B2B;
    color: #D2D2D2;
    border: none;
    outline: none;
    selection-background-color: #2C5D87;  /* Unity blue selection */
}

QTreeWidget::item, QTreeView::item {
    padding: 4px 2px;
    border: none;
}

QTreeWidget::item:selected, QTreeView::item:selected {
    background-color: #2C5D87;
    color: white;
}

QTreeWidget::item:hover, QTreeView::item:hover {
    background-color: #3E3E3E;
}

/* ===== INSPECTOR (Property Editor) ===== */
QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1E1E1E;
    color: #D2D2D2;
    border: 1px solid #1E1E1E;
    padding: 4px;
    selection-background-color: #2C5D87;
}

QTextEdit:focus, QLineEdit:focus {
    border: 1px solid: #4A9EFF;  /* Unity focus blue */
}

QLabel {
    background: transparent;
    color: #D2D2D2;
}

/* ===== SCROLL BARS (Unity style) ===== */
QScrollBar:vertical {
    background: #2B2B2B;
    width: 14px;
    border: none;
}

QScrollBar::handle:vertical {
    background: #5A5A5A;
    border-radius: 7px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #6A6A6A;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #2B2B2B;
    height: 14px;
    border: none;
}

QScrollBar::handle:horizontal {
    background: #5A5A5A;
    border-radius: 7px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background: #6A6A6A;
}

/* ===== BUTTONS ===== */
QPushButton {
    background-color: #5A5A5A;
    color: #D2D2D2;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
}

QPushButton:hover {
    background-color: #6A6A6A;
}

QPushButton:pressed {
    background-color: #4A4A4A;
}

/* ===== SPLITTERS ===== */
QSplitter::handle {
    background: #1E1E1E;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

/* ===== MENU BAR ===== */
QMenuBar {
    background-color: #2B2B2B;
    color: #D2D2D2;
    border-bottom: 1px solid #1E1E1E;
}

QMenuBar::item:selected {
    background: #3E3E3E;
}

QMenu {
    background-color: #2B2B2B;
    color: #D2D2D2;
    border: 1px solid #1E1E1E;
}

QMenu::item:selected {
    background-color: #2C5D87;
}
"""
