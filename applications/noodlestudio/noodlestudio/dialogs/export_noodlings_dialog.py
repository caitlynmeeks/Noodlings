"""
Export Noodling(s) Dialog

Unified export interface for single Noodlings or ensembles.

Features:
- Export 1+ selected Noodlings
- Add metadata (name, description, tags)
- Optional: Generate ensemble (.ensemble file)
- Captures relationships between Noodlings if exporting ensemble

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QLineEdit, QTextEdit, QPushButton, QCheckBox,
                             QGroupBox, QListWidget, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from pathlib import Path
from typing import List, Dict


class ExportNoodlingsDialog(QDialog):
    """Dialog for exporting Noodling(s) with metadata."""

    def __init__(self, noodlings_data: List[Dict], parent=None):
        super().__init__(parent)
        self.noodlings_data = noodlings_data
        self.result_path = None

        self.setWindowTitle(f"Export {len(noodlings_data)} Noodling(s)")
        self.resize(600, 500)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        count = len(self.noodlings_data)
        header = QLabel(f"<h2>Export {count} Noodling{'s' if count > 1 else ''}</h2>")
        layout.addWidget(header)

        # Show which Noodlings are being exported
        noodlings_list = QLabel("Exporting: " + ", ".join([n.get('name', n.get('id')) for n in self.noodlings_data]))
        noodlings_list.setStyleSheet("color: #888; margin-bottom: 10px;")
        noodlings_list.setWordWrap(True)
        layout.addWidget(noodlings_list)

        # Metadata section
        meta_group = QGroupBox("Metadata")
        meta_layout = QFormLayout()

        self.name_field = QLineEdit()
        self.name_field.setPlaceholderText("My Custom Noodling" if count == 1 else "My Ensemble")
        meta_layout.addRow("Name:", self.name_field)

        self.description_field = QTextEdit()
        self.description_field.setPlaceholderText("Description of this character/ensemble...")
        self.description_field.setMaximumHeight(80)
        meta_layout.addRow("Description:", self.description_field)

        self.author_field = QLineEdit()
        self.author_field.setPlaceholderText("Your name")
        meta_layout.addRow("Author:", self.author_field)

        self.tags_field = QLineEdit()
        self.tags_field.setPlaceholderText("fantasy, warrior, brave")
        meta_layout.addRow("Tags:", self.tags_field)

        meta_group.setLayout(meta_layout)
        layout.addWidget(meta_group)

        # Ensemble option (only if multiple Noodlings)
        if count > 1:
            ensemble_group = QGroupBox("Ensemble Options")
            ensemble_layout = QVBoxLayout()

            self.ensemble_checkbox = QCheckBox("Generate Ensemble (.ensemble)")
            self.ensemble_checkbox.setChecked(True)
            ensemble_layout.addWidget(self.ensemble_checkbox)

            hint = QLabel("Ensemble includes relationship dynamics between Noodlings")
            hint.setStyleSheet("color: #888; font-size: 10px;")
            ensemble_layout.addWidget(hint)

            ensemble_group.setLayout(ensemble_layout)
            layout.addWidget(ensemble_group)
        else:
            self.ensemble_checkbox = None

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        export_btn = QPushButton("Export")
        export_btn.setDefault(True)
        export_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        export_btn.clicked.connect(self.do_export)
        button_layout.addWidget(export_btn)

        layout.addLayout(button_layout)

    def do_export(self):
        """Perform the export."""
        name = self.name_field.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Please enter a name.")
            return

        description = self.description_field.toPlainText().strip()
        author = self.author_field.text().strip()
        tags = [t.strip() for t in self.tags_field.text().split(',') if t.strip()]

        # Check if generating ensemble
        is_ensemble = self.ensemble_checkbox and self.ensemble_checkbox.isChecked()

        if is_ensemble:
            # Export as .ensemble
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Ensemble",
                str(Path.home() / ".noodlestudio" / "ensembles" / f"{name.lower().replace(' ', '_')}.ensemble"),
                "Ensemble Files (*.ensemble)"
            )

            if filename:
                success = self.export_as_ensemble(name, description, author, tags, Path(filename))
                if success:
                    self.result_path = filename
                    self.accept()
        else:
            # Export individual Noodlings as .nood files
            if len(self.noodlings_data) == 1:
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export Noodling",
                    str(Path.home() / ".noodlestudio" / "characters" / f"{name.lower().replace(' ', '_')}.nood"),
                    "Noodling Files (*.nood)"
                )

                if filename:
                    success = self.export_single_noodling(self.noodlings_data[0], Path(filename))
                    if success:
                        self.result_path = filename
                        self.accept()
            else:
                # Multiple .nood files
                directory = QFileDialog.getExistingDirectory(
                    self,
                    "Export Noodlings (choose directory)",
                    str(Path.home() / ".noodlestudio" / "characters")
                )

                if directory:
                    # Export each to separate .nood file
                    for noodling in self.noodlings_data:
                        nood_name = noodling.get('name', noodling.get('id'))
                        filepath = Path(directory) / f"{nood_name.lower().replace(' ', '_')}.nood"
                        self.export_single_noodling(noodling, filepath)

                    self.result_path = directory
                    self.accept()

    def export_as_ensemble(self, name: str, description: str, author: str, tags: List[str], output_path: Path) -> bool:
        """Export as .ensemble file."""
        try:
            from ..data.ensemble_exporter import EnsembleExporter

            exporter = EnsembleExporter()
            agent_ids = [n.get('id') for n in self.noodlings_data]

            success = exporter.export_from_noodlings(
                agent_ids,
                name,
                description,
                output_path
            )

            return success

        except Exception as e:
            import traceback
            print(f"Ensemble export error: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Export Failed", f"Error: {e}")
            return False

    def export_single_noodling(self, noodling: Dict, output_path: Path) -> bool:
        """Export single Noodling as .nood file."""
        try:
            import yaml

            # Get recipe data
            agent_id = noodling.get('id')
            recipe_name = agent_id.replace('agent_', '')

            recipe_paths = [
                Path(__file__).parent.parent.parent.parent / "cmush" / "recipes" / f"{recipe_name}.yaml",
                Path.home() / "git" / "noodlings_clean" / "applications" / "cmush" / "recipes" / f"{recipe_name}.yaml",
            ]

            recipe_data = None
            for path in recipe_paths:
                if path.exists():
                    with open(path, 'r') as f:
                        recipe_data = yaml.safe_load(f)
                    break

            if not recipe_data:
                print(f"Recipe not found for {agent_id}")
                return False

            # Save as .nood file (YAML format)
            with open(output_path, 'w') as f:
                yaml.dump(recipe_data, f, default_flow_style=False, sort_keys=False)

            print(f"Exported {recipe_name} to {output_path}")
            return True

        except Exception as e:
            import traceback
            print(f"Noodling export error: {e}")
            traceback.print_exc()
            return False
