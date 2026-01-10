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
#   Facets Editor Assembly Mixin - Assembly loading, saving, and validation
#
#   Contains assembly I/O operations: - load_assembly_from_da...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.facets_editor_assembly_mixin
# PURPOSE:  facets editor assembly mixin facet implementation
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetsEditorAssemblyMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Optional
from PyQt6.QtWidgets import QMessageBox, QFileDialog


class FacetsEditorAssemblyMixin:
    """Mixin providing assembly I/O for FacetsEditorPanel."""

    def load_assembly_from_data(self, assembly, force_reload: bool = False, source_path: Optional[str] = None):
        """
        Load a facet assembly into the editor.

        Args:
            assembly: FacetAssembly to load
            force_reload: If True, reload even if same assembly already loaded
            source_path: Optional path to source YAML file (for direct saves)
        """
        # Import here to avoid circular imports
        from .facets_editor_graphics import FacetNodeGraphics, ConnectionWire

        # CRITICAL: Prevent re-entrant calls during scene transition
        if self.scene_transition_lock:
            return

        # Check if this assembly is already loaded
        if not force_reload and self.current_assembly_name == assembly.name:
            return

        # CRITICAL: Lock BEFORE auto-save to prevent re-entrancy during YAML loading
        self.scene_transition_lock = True

        # Auto-save previous assembly before switching (if positions changed)
        if self.current_assembly and self.current_assembly_name:
            try:
                import os
                assembly_dir = os.path.join(os.path.dirname(__file__), '../facet_assemblies')
                for filename in os.listdir(assembly_dir):
                    if filename.endswith('.yaml'):
                        try:
                            from ..core.facet_system import FacetAssembly
                            test_path = os.path.join(assembly_dir, filename)
                            test_assembly = FacetAssembly.load_yaml(test_path)
                            if test_assembly.name == self.current_assembly_name:
                                self.current_assembly.save_yaml(test_path)
                                break
                        except:
                            pass
            except:
                pass  # Silent auto-save failure

        # Hide empty state message if showing
        self.hide_empty_state()

        self.current_assembly = assembly
        self.current_assembly_name = assembly.name
        self.current_assembly_path = source_path
        self.assembly_label.setText(assembly.name)

        # CRITICAL: Stop all animations before clearing scene to prevent segfault
        for node_gfx in self.node_graphics.values():
            if hasattr(node_gfx, 'animation_timer') and node_gfx.animation_timer:
                node_gfx.animation_timer.stop()
                node_gfx.animation_timer = None

        # Clear existing graphics
        self.scene.clear()
        self.node_graphics.clear()
        self.wire_graphics.clear()
        self.grid_lines.clear()  # Grid lines are also cleared by scene.clear()

        # Create node graphics for each facet
        for facet in assembly.facets:
            node = FacetNodeGraphics(facet, editor_panel=self)
            self.scene.addItem(node)
            self.node_graphics[facet.id] = node

        # Create connection wires
        for conn in assembly.connections:
            from_node = self.node_graphics.get(conn.from_facet)
            to_node = self.node_graphics.get(conn.to_facet)

            if from_node and to_node:
                from_pad = from_node.output_pads.get(conn.from_pad)
                to_pad = to_node.input_pads.get(conn.to_pad)

                if from_pad and to_pad:
                    wire = ConnectionWire(from_pad, to_pad)
                    self.scene.addItem(wire)
                    self.wire_graphics.append(wire)

        # Force scene update and ensure all items are visible
        self.scene.update()
        for node in self.node_graphics.values():
            node.update()

        # Restore grid if it was enabled
        if self.grid_visible:
            self._draw_grid_background()

        # Center view on content
        self.view.centerOn(500, 350)

        # Unlock scene - safe to process events now
        self.scene_transition_lock = False

    def save_assembly(self):
        """Save current assembly to YAML file."""
        if not self.current_assembly:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Facet Assembly",
            f"../facet_assemblies/{self.current_assembly.name}.yaml",
            "YAML Files (*.yaml *.yml)"
        )

        if filepath:
            try:
                self.current_assembly.save_yaml(filepath)
                QMessageBox.information(self, "Success", f"Assembly saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save assembly: {e}")

    def load_assembly(self):
        """Load assembly from YAML file."""
        from ..core.facet_system import FacetAssembly

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Facet Assembly",
            "../facet_assemblies/",
            "YAML Files (*.yaml *.yml)"
        )

        if filepath:
            try:
                assembly = FacetAssembly.load_yaml(filepath)
                self.load_assembly_from_data(assembly)
                QMessageBox.information(self, "Success", f"Loaded assembly: {assembly.name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load assembly: {e}")

    def validate_assembly(self):
        """Validate current assembly and show errors."""
        if not self.current_assembly:
            return

        errors = self.current_assembly.validate()

        if errors:
            error_text = "\n".join(f"- {e}" for e in errors)
            QMessageBox.warning(self, "Validation Errors", f"Assembly has errors:\n\n{error_text}")
        else:
            QMessageBox.information(self, "Validation Success", "Assembly is valid!")

    def save_current_assembly_positions(self):
        """
        Save current node positions to the assembly and disk.

        Called after layout changes (auto-arrange, manual drag, etc).
        """
        if not self.current_assembly:
            return

        # Update facet positions from graphics
        for facet_id, node_gfx in self.node_graphics.items():
            facet = self.current_assembly.get_facet(facet_id)
            if facet:
                facet.position = {
                    'x': node_gfx.pos().x(),
                    'y': node_gfx.pos().y()
                }

        # Save to disk
        self._save_assembly_to_disk()

    def _save_assembly_to_disk(self):
        """Save current assembly to disk (called by internal methods)."""
        if not self.current_assembly or not self.current_assembly_path:
            return

        try:
            import os
            if os.path.exists(self.current_assembly_path):
                self.current_assembly.save_yaml(self.current_assembly_path)
        except Exception:
            pass  # Silent save errors

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
