"""Ensemble noodling selector mixin for AssemblyEditorView.

Provides a dropdown (QComboBox) to switch between noodlings in ensemble
mode. Sets _selected_noodling_id which the execution mixin uses to
filter events.

Ported from facets_editor_panel.py (set_ensemble_noodlings,
clear_ensemble_noodlings, select_noodling, _on_noodling_selected).
"""

from typing import Optional, List, Dict

from PyQt6.QtWidgets import QComboBox


class AssemblyEnsembleMixin:
    """Ensemble noodling selector and event filtering."""

    # ================================================================
    # Initialization
    # ================================================================

    def _init_ensemble_state(self):
        """Initialize ensemble mode state."""
        self._ensemble_noodlings: List[Dict] = []
        self._selected_noodling_id: Optional[str] = None

        # Noodling selector dropdown (hidden until ensemble mode)
        self._noodling_selector = QComboBox()
        self._noodling_selector.setFixedWidth(160)
        self._noodling_selector.setToolTip("Select noodling to inspect")
        self._noodling_selector.currentIndexChanged.connect(
            self._on_noodling_selected
        )
        self._noodling_selector.setStyleSheet("""
            QComboBox {
                background-color: #2A2A2A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                color: #CCCCCC;
                selection-background-color: #444444;
            }
        """)
        self._noodling_selector.hide()

    # ================================================================
    # Public API (called by GuidePerformanceManager)
    # ================================================================

    def set_ensemble_noodlings(self, noodlings: list):
        """Set up the noodling selector for ensemble mode.

        Args:
            noodlings: List of dicts with 'id', 'name', 'assembly',
                       'assembly_path'.
        """
        self._ensemble_noodlings = noodlings
        self._noodling_selector.blockSignals(True)
        self._noodling_selector.clear()

        for entry in noodlings:
            self._noodling_selector.addItem(
                entry['name'], userData=entry['id']
            )

        self._noodling_selector.blockSignals(False)
        self._noodling_selector.show()

        if noodlings:
            self._noodling_selector.setCurrentIndex(0)
            self._on_noodling_selected(0)

    def clear_ensemble_noodlings(self):
        """Clear noodling selector and return to single mode."""
        self._ensemble_noodlings = []
        self._selected_noodling_id = None
        self._noodling_selector.blockSignals(True)
        self._noodling_selector.clear()
        self._noodling_selector.blockSignals(False)
        self._noodling_selector.hide()

    def select_noodling(self, noodling_id: str):
        """Select a noodling by ID (programmatic, e.g. turn-taking switch)."""
        for i, entry in enumerate(self._ensemble_noodlings):
            if entry['id'] == noodling_id:
                self._noodling_selector.blockSignals(True)
                self._noodling_selector.setCurrentIndex(i)
                self._noodling_selector.blockSignals(False)
                self._on_noodling_selected(i)
                return

    # ================================================================
    # Internal
    # ================================================================

    def _on_noodling_selected(self, index: int):
        """Handle noodling selector dropdown change.

        Loads the selected noodling's assembly and sets the event filter.
        """
        if index < 0 or index >= len(self._ensemble_noodlings):
            return

        entry = self._ensemble_noodlings[index]
        self._selected_noodling_id = entry['id']
        self.current_agent_id = entry['id']

        # Load this noodling's assembly
        assembly = entry.get('assembly')
        assembly_path = entry.get('assembly_path')
        if assembly and hasattr(self, 'load_assembly_from_data'):
            self.load_assembly_from_data(
                assembly, source_path=assembly_path, force_reload=True
            )
