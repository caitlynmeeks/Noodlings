# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Test Stage Demo
#
#   Test suite for stage demo.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.test_stage_demo
# PURPOSE:  Tests for stage demo
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   main()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

#!/usr/bin/env python3
"""
Test Demo: RadianceViewport

Demonstrates the clean viewport API - just load files and render.
The viewport doesn't know about noodlings, stages, or recipes.
It just renders RadianceComponents.

Usage:
    cd applications/noodlestudio
    PYTHONPATH=.:../.. python3 noodlestudio/runtime/ui/test_stage_demo.py
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "applications" / "noodlestudio"))


def main():
    """Run the viewport demo."""
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
    from PyQt6.QtCore import Qt, QTimer

    app = QApplication(sys.argv)

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("RadianceViewport Demo - Phase 3c")
    window.setGeometry(100, 100, 800, 600)

    # Central widget
    central = QWidget()
    window.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)

    # Create viewport
    from noodlestudio.runtime.ui.components.radiance_viewport import (
        RadianceViewport, RadianceViewportWidget
    )

    viewport_component = RadianceViewport(name="demo_viewport")
    viewport_component.geometry.width = 800
    viewport_component.geometry.height = 550
    viewport_component.background = "#1a1a1a"

    viewport_widget = RadianceViewportWidget(viewport_component)
    layout.addWidget(viewport_widget)

    # Status label
    status_label = QLabel("Loading...")
    status_label.setStyleSheet("background: #2a2a2a; color: #808080; padding: 5px;")
    status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(status_label)

    def on_component_loaded(entity_id, gaussian_count):
        status_label.setText(f"Loaded: {entity_id} ({gaussian_count:,} Gaussians)")

    def on_render_complete(info):
        visible = info.get('visible', 0)
        backend = info.get('backend', 'unknown')
        status_label.setText(
            f"Visible: {visible:,} | Backend: {backend}"
        )

    viewport_widget.componentLoaded.connect(on_component_loaded)
    viewport_widget.renderComplete.connect(on_render_complete)

    def load_demo_asset():
        # Load a radiance file directly - viewport doesn't care what it is
        radiance_path = project_root / "external" / "vrm_samples" / "alicia_densified_tuned.radiance"
        if radiance_path.exists():
            viewport_widget.load_file(str(radiance_path), "alicia")
        else:
            status_label.setText(f"Demo asset not found: {radiance_path}")

    QTimer.singleShot(100, load_demo_asset)

    window.show()

    print("=" * 60)
    print("RadianceViewport Demo - Phase 3c")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Left mouse:   Orbit camera")
    print("  Right mouse:  Pan camera")
    print("  Scroll:       Zoom in/out")
    print("  F:            Focus on model")
    print("  A:            Frame all")
    print()
    print("The viewport just renders RadianceComponents.")
    print("It doesn't know about noodlings, stages, or recipes.")
    print()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
