"""
Neural Network Export Dialog

Unified export interface for multiple neural network formats:
- .nncanvas (NoodleSTUDIO native)
- MLX Python (Apple Silicon)
- ONNX (universal interchange)
- PyTorch (.pt)
- CoreML (Apple ecosystem)

Author: Commander Spock + Captain Caity
Date: December 10, 2025
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt


class NeuralExportDialog(QDialog):
    """
    Export format selection dialog.

    Presents available export formats with descriptions and
    returns user's selected format.
    """

    # Export format definitions
    FORMATS = [
        {
            'id': 'nncanvas',
            'name': '.nncanvas',
            'title': 'NoodleSTUDIO Neural Canvas',
            'ext': '.nncanvas',
            'filter': 'Neural Canvas Files (*.nncanvas);;All Files (*)',
            'description': (
                'Native NoodleSTUDIO format. Saves complete topology as JSON:\n'
                '• Node definitions with all parameters\n'
                '• Port connections and data flow\n'
                '• Human-readable port labels\n'
                '• Metadata (author, version, description)\n\n'
                'Use this format to share networks within NoodleSTUDIO or\n'
                'version control your neural architectures.'
            ),
            'status': 'ready',
            'implemented': True
        },
        {
            'id': 'mlx',
            'name': 'MLX Python',
            'title': 'Apple MLX Framework',
            'ext': '.py',
            'filter': 'Python Files (*.py);;All Files (*)',
            'description': (
                'Apple Silicon optimized ML framework code generation.\n'
                'Generates executable Python code using mlx.nn:\n'
                '• Optimized for M1/M2/M3 chips\n'
                '• Metal GPU acceleration\n'
                '• Complete model definition with forward pass\n'
                '• Parameter count calculations\n\n'
                'Copy generated code into your MLX training pipeline.\n'
                'Requires: pip install mlx'
            ),
            'status': 'ready',
            'implemented': True
        },
        {
            'id': 'onnx',
            'name': 'ONNX',
            'title': 'Open Neural Network Exchange',
            'ext': '.onnx',
            'filter': 'ONNX Files (*.onnx);;All Files (*)',
            'description': (
                'Universal ML interchange format (Facebook/Microsoft).\n'
                'Industry standard for model portability:\n'
                '• Convert to TensorFlow, PyTorch, CoreML, TFLite\n'
                '• Deploy to ONNX Runtime (cross-platform)\n'
                '• Optimize with ONNX tools\n'
                '• Hardware accelerator support\n\n'
                'The "Rosetta Stone" of neural networks.\n'
                'Note: Requires trained weights for full export.'
            ),
            'status': 'planned',
            'implemented': False
        },
        {
            'id': 'pytorch',
            'name': 'PyTorch',
            'title': 'PyTorch Model',
            'ext': '.pt',
            'filter': 'PyTorch Files (*.pt *.pth);;All Files (*)',
            'description': (
                'Most popular deep learning framework.\n'
                'Exports torch.nn.Module definition:\n'
                '• Compatible with PyTorch training pipelines\n'
                '• Large ecosystem of tools and libraries\n'
                '• Easy conversion to ONNX\n'
                '• TorchScript compilation support\n\n'
                'Use for training or fine-tuning in PyTorch.\n'
                'Requires: pip install torch'
            ),
            'status': 'planned',
            'implemented': False
        },
        {
            'id': 'coreml',
            'name': 'CoreML',
            'title': 'Apple CoreML',
            'ext': '.mlmodel',
            'filter': 'CoreML Files (*.mlmodel);;All Files (*)',
            'description': (
                'Apple\'s native ML format for iOS/macOS deployment.\n'
                'Optimized for on-device inference:\n'
                '• Native iOS/macOS integration\n'
                '• Neural Engine acceleration\n'
                '• Low power consumption\n'
                '• Privacy-preserving (on-device)\n\n'
                'Perfect for shipping ML to Apple devices.\n'
                'Note: Requires trained weights.'
            ),
            'status': 'planned',
            'implemented': False
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Neural Network")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.selected_format = None
        self._init_ui()

    def _init_ui(self):
        """Initialize dialog UI."""
        layout = QVBoxLayout()

        # Header
        header = QLabel("Select Export Format:")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)

        # Format selection
        self.button_group = QButtonGroup(self)
        self.format_buttons = {}

        for fmt in self.FORMATS:
            # Radio button with format name
            radio = QRadioButton(fmt['title'])
            radio.setProperty('format_id', fmt['id'])
            self.button_group.addButton(radio)
            self.format_buttons[fmt['id']] = radio

            # Add status indicator
            status_text = ""
            if not fmt['implemented']:
                status_text = " [Not Yet Implemented]"
                radio.setEnabled(False)
                radio.setStyleSheet("color: #666666;")
            elif fmt['status'] == 'ready':
                status_text = " ✓"

            radio.setText(f"{fmt['title']}{status_text}")

            # Connect to description update
            radio.toggled.connect(self._on_format_selected)

            layout.addWidget(radio)

        # Default selection (first available format)
        for fmt in self.FORMATS:
            if fmt['implemented']:
                self.format_buttons[fmt['id']].setChecked(True)
                break

        layout.addSpacing(10)

        # Description area
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold; padding-top: 10px;")
        layout.addWidget(desc_label)

        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(200)
        self.description_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #e8e8e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.description_text)

        # Update description for default selection
        self._update_description()

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        btn_export = QPushButton("Export")
        btn_export.setDefault(True)
        btn_export.clicked.connect(self._on_export_clicked)
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #e8e8e0;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        button_layout.addWidget(btn_export)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_format_selected(self):
        """Handle format radio button selection."""
        self._update_description()

    def _update_description(self):
        """Update description text for selected format."""
        selected_button = self.button_group.checkedButton()
        if not selected_button:
            return

        format_id = selected_button.property('format_id')

        # Find format definition
        fmt = None
        for f in self.FORMATS:
            if f['id'] == format_id:
                fmt = f
                break

        if fmt:
            self.description_text.setText(fmt['description'])

    def _on_export_clicked(self):
        """Handle export button click."""
        selected_button = self.button_group.checkedButton()
        if not selected_button:
            QMessageBox.warning(self, "No Format Selected", "Please select an export format.")
            return

        format_id = selected_button.property('format_id')

        # Find format definition
        for fmt in self.FORMATS:
            if fmt['id'] == format_id:
                self.selected_format = fmt
                break

        if not self.selected_format:
            QMessageBox.warning(self, "Invalid Format", "Selected format is invalid.")
            return

        # Check if implemented
        if not self.selected_format['implemented']:
            QMessageBox.information(
                self,
                "Not Yet Implemented",
                f"{self.selected_format['title']} export is planned but not yet implemented.\n\n"
                "Available formats:\n"
                "• .nncanvas (native)\n"
                "• MLX Python\n\n"
                "Coming soon:\n"
                "• ONNX\n"
                "• PyTorch\n"
                "• CoreML"
            )
            return

        self.accept()

    def get_selected_format(self):
        """
        Get selected export format definition.

        Returns:
            Dict with format info or None if canceled
        """
        return self.selected_format
