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
#   Facets Editor Selection Mixin - Selection, clipboard, and undo operations
#
#   Contains selection/editing operations: - copy_selection, ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.facets_editor_selection_mixin
# PURPOSE:  facets editor selection mixin facet implementation
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetsEditorSelectionMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import QGraphicsItem, QMessageBox
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QCursor, QColor

from .facets_editor_graphics import FacetNodeGraphics, ConnectionWire


class FacetsEditorSelectionMixin:
    """Mixin providing selection/clipboard/undo for FacetsEditorPanel."""

    def toggle_node_expansion(self):
        """Open field editor for selected node (E key - edits Processing Prompt)."""
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if len(selected_nodes) == 1:
            node = selected_nodes[0]
            # Open editor for Processing Prompt field (most common edit)
            fields = node.facet.get_editable_fields()
            prompt_field = next((f for f in fields if f['key'] == 'prompt'), None)
            if prompt_field:
                self.show_floating_editor(node.facet, prompt_field)

    def copy_selection(self):
        """Copy selected facets to clipboard."""
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
            return

        # Copy facet data (deep copy)
        self.clipboard = []
        for node in selected_nodes:
            # Skip special nodes
            if node.facet.facet_type == "SpecialNode":
                continue

            # Store facet copy
            import copy
            facet_copy = copy.deepcopy(node.facet)
            self.clipboard.append(facet_copy)

    def paste_selection(self):
        """Paste facets from clipboard with internal connections preserved."""
        from ..core.facet_system import Facet, FacetConnection

        if not self.clipboard or not self.current_assembly:
            return

        # Get current mouse position in scene coords
        cursor_pos = self.view.mapToScene(self.view.mapFromGlobal(QCursor.pos()))

        # Map old IDs to new IDs for connection rewiring
        id_mapping = {}

        # Calculate offset for group (preserve relative positions)
        if self.clipboard:
            # Find top-left corner of clipboard group
            min_x = min(f.position['x'] for f in self.clipboard)
            min_y = min(f.position['y'] for f in self.clipboard)

            # Offset entire group to cursor position
            group_offset_x = cursor_pos.x() - min_x
            group_offset_y = cursor_pos.y() - min_y

        # Paste each facet with new UUID and preserved relative position
        import copy
        for facet_template in self.clipboard:
            # Deep copy and generate new UUID
            new_facet = copy.deepcopy(facet_template)
            old_id = new_facet.id
            new_facet.id = Facet.generate_uuid()
            id_mapping[old_id] = new_facet.id

            # Preserve relative position within group
            new_facet.position = {
                'x': facet_template.position['x'] + group_offset_x + 50,  # +50 for slight offset
                'y': facet_template.position['y'] + group_offset_y + 50
            }

            # Add to assembly
            self.current_assembly.facets.append(new_facet)

            # Create graphics
            node = FacetNodeGraphics(new_facet, editor_panel=self)
            self.scene.addItem(node)
            self.node_graphics[new_facet.id] = node

        # Duplicate internal connections (connections between pasted nodes)
        clipboard_ids = set(f.id for f in self.clipboard)
        for conn in self.current_assembly.connections:
            # Check if this connection is entirely within the copied set
            if conn.from_facet in clipboard_ids and conn.to_facet in clipboard_ids:
                # Duplicate this connection with new IDs
                new_conn = FacetConnection(
                    from_facet=id_mapping[conn.from_facet],
                    from_pad=conn.from_pad,
                    to_facet=id_mapping[conn.to_facet],
                    to_pad=conn.to_pad
                )
                self.current_assembly.connections.append(new_conn)

                # Create visual wire
                from_node = self.node_graphics.get(new_conn.from_facet)
                to_node = self.node_graphics.get(new_conn.to_facet)
                if from_node and to_node:
                    from_pad = from_node.output_pads.get(new_conn.from_pad)
                    to_pad = to_node.input_pads.get(new_conn.to_pad)
                    if from_pad and to_pad:
                        wire = ConnectionWire(from_pad, to_pad)
                        self.scene.addItem(wire)
                        self.wire_graphics.append(wire)

        self.assemblyModified.emit()

    def duplicate_selection(self):
        """Duplicate selected facets in place (Cmd-D)."""
        # Copy selection to clipboard
        self.copy_selection()
        # Immediately paste
        if self.clipboard:
            self.paste_selection()

    def delete_selection(self):
        """Delete selected facets (with undo support)."""
        if not self.current_assembly:
            return

        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
            return

        # Filter out special nodes (can't delete)
        deletable_nodes = [
            node for node in selected_nodes
            if not node.is_special_node and node.facet.name not in ["INCOMING", "OUTGOING"]
        ]

        if not deletable_nodes:
            return

        # Push delete commands via UndoManager
        from ..core.undo_manager import undo_manager
        from ..core.commands import DeleteFacetCommand

        # Use macro for multiple deletions (single undo)
        if len(deletable_nodes) > 1:
            undo_manager.begin_group(f"Delete {len(deletable_nodes)} Facets")

        for node in deletable_nodes:
            # Collect connections involving this facet (for restoration on undo)
            connections_data = [
                c.to_dict() for c in self.current_assembly.connections
                if c.from_facet == node.facet.id or c.to_facet == node.facet.id
            ]

            # Push delete command
            cmd = DeleteFacetCommand(
                editor=self,
                facet_data=node.facet.to_dict(),
                connections_data=connections_data,
                facet_name=node.facet.name
            )
            undo_manager.push(cmd)

        if len(deletable_nodes) > 1:
            undo_manager.end_group()

    def invert_selection(self):
        """Invert selection (select unselected, deselect selected)."""
        for item in self.scene.items():
            if isinstance(item, FacetNodeGraphics):
                item.setSelected(not item.isSelected())

    def on_selection_changed(self):
        """Handle selection changes - emit signal for Inspector."""
        selected = self.scene.selectedItems()
        selected_facets = [
            item for item in selected
            if isinstance(item, FacetNodeGraphics)
        ]

        if len(selected_facets) == 1:
            # Single selection - emit for Inspector
            self.facetSelected.emit(selected_facets[0].facet)
        else:
            # Multi-selection or no selection
            self.facetSelected.emit(None)

    def collapse_all_nodes(self):
        """Collapse all expanded nodes (hide fields on all nodes)."""
        try:
            if not self.scene:
                return
            # Copy items list to avoid iteration issues
            items = list(self.scene.items())
            for item in items:
                if isinstance(item, FacetNodeGraphics):
                    try:
                        item.hide_fields()
                    except Exception:
                        pass
        except Exception:
            pass

    # ========== UNDO/REDO ==========

    def undo(self):
        """Undo last operation via UndoManager."""
        from ..core.undo_manager import undo_manager
        undo_manager.undo()

    def redo(self):
        """Redo last undone operation via UndoManager."""
        from ..core.undo_manager import undo_manager
        undo_manager.redo()

    # ========== INTERNAL METHODS FOR UNDO COMMANDS ==========
    # These methods perform direct state changes without pushing commands.
    # They are called by command classes in undo/redo operations.

    def _set_facet_position_internal(self, facet_id: str, position: tuple):
        """
        Set facet position without pushing undo command.

        Called by MoveFacetCommand during undo/redo.
        """
        # Update data model
        facet = self.current_assembly.get_facet(facet_id) if self.current_assembly else None
        if facet:
            facet.position = {'x': position[0], 'y': position[1]}

        # Update graphics
        node_gfx = self.node_graphics.get(facet_id)
        if node_gfx:
            # Block signals to prevent recursive position saving
            node_gfx.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
            node_gfx.setPos(position[0], position[1])
            node_gfx.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

            # Update connected wires
            for pad_dict in [node_gfx.input_pads, node_gfx.output_pads]:
                for pad_graphics in pad_dict.values():
                    for wire in pad_graphics.connections:
                        wire.update_path()

        # Save to disk
        self._save_assembly_to_disk()

    def _create_facet_internal(self, facet_data: dict):
        """
        Create facet from serialized data without pushing undo command.

        Called by CreateFacetCommand.redo() and DeleteFacetCommand.undo().
        """
        from ..core.facet_system import Facet

        if not self.current_assembly:
            return

        # Deserialize facet
        facet = Facet.from_dict(facet_data)

        # Add to assembly
        self.current_assembly.facets.append(facet)

        # Create graphics
        node = FacetNodeGraphics(facet, editor_panel=self)
        self.scene.addItem(node)
        self.node_graphics[facet.id] = node

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _delete_facet_internal(self, facet_id: str):
        """
        Delete facet by ID without pushing undo command.

        Called by DeleteFacetCommand.redo() and CreateFacetCommand.undo().
        """
        if not self.current_assembly:
            return

        # Remove from assembly
        self.current_assembly.facets = [
            f for f in self.current_assembly.facets if f.id != facet_id
        ]

        # Remove connections involving this facet
        self.current_assembly.connections = [
            c for c in self.current_assembly.connections
            if c.from_facet != facet_id and c.to_facet != facet_id
        ]

        # Remove wire graphics involving this facet
        wires_to_remove = []
        for wire in self.wire_graphics:
            if wire.from_pad.facet_node.facet.id == facet_id or \
               wire.to_pad.facet_node.facet.id == facet_id:
                wires_to_remove.append(wire)

        for wire in wires_to_remove:
            self.scene.removeItem(wire)
            self.wire_graphics.remove(wire)

        # Remove from scene
        node_gfx = self.node_graphics.get(facet_id)
        if node_gfx:
            self.scene.removeItem(node_gfx)
            del self.node_graphics[facet_id]

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _set_facet_property_internal(self, facet_id: str, prop_name: str, value):
        """
        Set facet property without pushing undo command.

        Called by EditFacetPropertyCommand and ToggleLockCommand.
        """
        facet = self.current_assembly.get_facet(facet_id) if self.current_assembly else None
        if not facet:
            return

        setattr(facet, prop_name, value)

        # Update graphics if needed (e.g., lock icon)
        node_gfx = self.node_graphics.get(facet_id)
        if node_gfx and prop_name == 'locked':
            node_gfx.lock_icon.setPlainText("[L]" if value else "")
            node_gfx.lock_icon.setDefaultTextColor(
                QColor("#CCAA00" if value else "#888888")
            )

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _create_connection_internal(self, conn_data: dict):
        """
        Create connection from serialized data without pushing undo command.

        Called by CreateConnectionCommand.redo() and DeleteConnectionCommand.undo().
        """
        from ..core.facet_system import FacetConnection

        if not self.current_assembly:
            return

        # Parse connection data
        from_parts = conn_data['from'].split('.')
        to_parts = conn_data['to'].split('.')
        from_facet = from_parts[0]
        from_pad = '.'.join(from_parts[1:])
        to_facet = to_parts[0]
        to_pad = '.'.join(to_parts[1:])

        # Create connection object
        conn = FacetConnection(from_facet, from_pad, to_facet, to_pad)

        # Add to assembly
        self.current_assembly.connections.append(conn)

        # Create wire graphics
        from_node = self.node_graphics.get(from_facet)
        to_node = self.node_graphics.get(to_facet)

        if from_node and to_node:
            from_pad_gfx = from_node.output_pads.get(from_pad)
            to_pad_gfx = to_node.input_pads.get(to_pad)

            if from_pad_gfx and to_pad_gfx:
                wire = ConnectionWire(from_pad_gfx, to_pad_gfx)
                self.scene.addItem(wire)
                self.wire_graphics.append(wire)

                # Register connections on pads
                from_pad_gfx.connections.append(wire)
                to_pad_gfx.connections.append(wire)
                to_pad_gfx.update_color_from_connection()

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _delete_connection_internal(self, from_facet: str, from_pad: str,
                                    to_facet: str, to_pad: str):
        """
        Delete connection without pushing undo command.

        Called by DeleteConnectionCommand.redo() and CreateConnectionCommand.undo().
        """
        if not self.current_assembly:
            return

        # Remove from assembly
        self.current_assembly.connections = [
            c for c in self.current_assembly.connections
            if not (c.from_facet == from_facet and c.from_pad == from_pad and
                    c.to_facet == to_facet and c.to_pad == to_pad)
        ]

        # Remove wire graphics
        wire_to_remove = None
        for wire in self.wire_graphics:
            if (wire.from_pad.facet_node.facet.id == from_facet and
                wire.from_pad.pad.name == from_pad and
                wire.to_pad.facet_node.facet.id == to_facet and
                wire.to_pad.pad.name == to_pad):
                wire_to_remove = wire
                break

        if wire_to_remove:
            # Unregister from pads
            if wire_to_remove in wire_to_remove.from_pad.connections:
                wire_to_remove.from_pad.connections.remove(wire_to_remove)
            if wire_to_remove in wire_to_remove.to_pad.connections:
                wire_to_remove.to_pad.connections.remove(wire_to_remove)
                wire_to_remove.to_pad.update_color_from_connection()

            self.scene.removeItem(wire_to_remove)
            self.wire_graphics.remove(wire_to_remove)

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _push_move_commands_if_needed(self):
        """
        Push move commands for nodes that have moved since drag started.

        Called from eventFilter on mouse release.
        """
        if not self.drag_start_positions:
            return

        from ..core.undo_manager import undo_manager
        from ..core.commands import MoveFacetCommand

        moved_nodes = []

        # Check which nodes actually moved
        for facet_id, old_pos in self.drag_start_positions.items():
            node_gfx = self.node_graphics.get(facet_id)
            if node_gfx:
                new_pos = (node_gfx.pos().x(), node_gfx.pos().y())

                # Only count as moved if position changed significantly
                if abs(new_pos[0] - old_pos[0]) > 1 or abs(new_pos[1] - old_pos[1]) > 1:
                    moved_nodes.append((facet_id, old_pos, new_pos, node_gfx.facet.name))

                    # Update facet data model
                    node_gfx.facet.position = {'x': new_pos[0], 'y': new_pos[1]}

        if not moved_nodes:
            return

        # Use macro for multiple moves
        if len(moved_nodes) > 1:
            undo_manager.begin_group(f"Move {len(moved_nodes)} Facets")

        for facet_id, old_pos, new_pos, facet_name in moved_nodes:
            cmd = MoveFacetCommand(
                editor=self,
                facet_id=facet_id,
                old_pos=old_pos,
                new_pos=new_pos,
                facet_name=facet_name
            )
            undo_manager.push(cmd)

        if len(moved_nodes) > 1:
            undo_manager.end_group()

    # ========== LAYOUT OPERATIONS ==========

    def auto_arrange_facets(self):
        """
        Auto-arrange facets using topological layering (circuit schematic style).

        Algorithm:
        1. Build dependency graph from connections
        2. Compute layers using topological sort (execution order)
        3. Position INCOMING at top, OUTGOING at bottom
        4. Distribute intermediate facets in layers
        5. Minimize wire crossings within each layer
        """
        if not self.current_assembly:
            return

        print("[Auto-Arrange] Starting topological layout...")

        # Build adjacency lists (who depends on whom)
        dependencies = {}  # facet_id -> list of facets it depends on (inputs from)
        dependents = {}    # facet_id -> list of facets that depend on it (outputs to)

        for facet in self.current_assembly.facets:
            dependencies[facet.id] = []
            dependents[facet.id] = []

        # Parse connections to build graph
        for conn in self.current_assembly.connections:
            from_id = conn.from_facet  # Source facet ID
            to_id = conn.to_facet      # Destination facet ID

            if from_id in dependencies and to_id in dependencies:
                if from_id not in dependencies[to_id]:  # Avoid duplicates
                    dependencies[to_id].append(from_id)
                if to_id not in dependents[from_id]:
                    dependents[from_id].append(to_id)

        # Topological sort to determine layers (Kahn's algorithm)
        layers = []
        in_degree = {fid: len(deps) for fid, deps in dependencies.items()}

        # Layer 0: Nodes with no dependencies (usually INCOMING)
        current_layer = [fid for fid, deg in in_degree.items() if deg == 0]

        while current_layer:
            layers.append(current_layer[:])
            next_layer = []

            for node_id in current_layer:
                # Remove this node from dependents' in-degree
                for dependent in dependents.get(node_id, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_layer.append(dependent)

            current_layer = next_layer

        # Handle cycles (shouldn't happen in well-formed assemblies)
        remaining = [fid for fid, deg in in_degree.items() if deg > 0]
        if remaining:
            layers.append(remaining)
            print(f"[Auto-Arrange] Warning: Circular dependencies detected: {remaining}")

        print(f"[Auto-Arrange] Computed {len(layers)} layers: {[len(l) for l in layers]} facets")

        # Layout parameters (HORIZONTAL FLOW - left to right like Neural Canvas)
        layer_spacing = 300  # Horizontal spacing between layers
        node_spacing = 180   # Vertical spacing within layer
        start_x = 100        # Left margin
        start_y = 100        # Top margin

        # Position facets layer by layer (HORIZONTAL FLOW)
        for layer_idx, layer_facets in enumerate(layers):
            x = start_x + (layer_idx * layer_spacing)  # Horizontal progression

            for facet_idx, facet_id in enumerate(sorted(layer_facets)):
                y = start_y + (facet_idx * node_spacing)  # Vertical stacking within layer

                # Find graphics node and move it
                if facet_id in self.node_graphics:
                    node_gfx = self.node_graphics[facet_id]
                    node_gfx.setPos(x, y)
                    print(f"[Auto-Arrange] {facet_id}: ({x}, {y}) [Layer {layer_idx}]")

        # Update all wire paths
        try:
            for wire in self.wire_graphics:
                wire.update_path()
        except Exception as e:
            print(f"[Auto-Arrange] Warning: Could not update wires: {e}")

        # Save new positions
        try:
            self.save_current_assembly_positions()
        except Exception as e:
            print(f"[Auto-Arrange] Warning: Could not save positions: {e}")

        print("[Auto-Arrange] Layout complete!")

        # Frame all nodes to show the result
        self.frame_all()

    def align_selected_horizontally(self):
        """Align selected facets to same Y coordinate (horizontal line)."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]
        if len(selected) < 2:
            return

        # Use average Y position
        avg_y = sum(node.pos().y() for node in selected) / len(selected)

        for node in selected:
            node.setPos(node.pos().x(), avg_y)

        # Update wires
        for wire in self.wire_graphics:
            wire.update_path()

        self.save_current_assembly_positions()
        print(f"[Align] Aligned {len(selected)} facets horizontally at y={avg_y:.0f}")

    def align_selected_vertically(self):
        """Align selected facets to same X coordinate (vertical line)."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]
        if len(selected) < 2:
            return

        # Use average X position
        avg_x = sum(node.pos().x() for node in selected) / len(selected)

        for node in selected:
            node.setPos(avg_x, node.pos().y())

        # Update wires
        for wire in self.wire_graphics:
            wire.update_path()

        self.save_current_assembly_positions()
        print(f"[Align] Aligned {len(selected)} facets vertically at x={avg_x:.0f}")

    def delete_selected_facets(self):
        """Delete selected facets from the assembly."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]
        if not selected:
            return

        # Confirm deletion
        facet_names = [node.facet.name for node in selected]
        reply = QMessageBox.question(
            self,
            "Delete Facets",
            f"Delete {len(selected)} facet(s)?\n\n" + "\n".join(f"- {name}" for name in facet_names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Remove from assembly and scene
        for node in selected:
            facet_id = node.facet.id

            # Remove connected wires
            wires_to_remove = [w for w in self.wire_graphics if w.from_pad.facet_node == node or w.to_pad.facet_node == node]
            for wire in wires_to_remove:
                # Remove wire from pad connection lists
                if wire in wire.from_pad.connections:
                    wire.from_pad.connections.remove(wire)
                if wire in wire.to_pad.connections:
                    wire.to_pad.connections.remove(wire)
                self.scene.removeItem(wire)
                self.wire_graphics.remove(wire)

            # Remove from assembly
            self.current_assembly.facets = [f for f in self.current_assembly.facets if f.id != facet_id]
            self.current_assembly.connections = [
                c for c in self.current_assembly.connections
                if c.from_facet != facet_id and c.to_facet != facet_id
            ]

            # Remove from scene
            self.scene.removeItem(node)
            del self.node_graphics[facet_id]

        self.save_current_assembly_positions()
        self.assemblyModified.emit()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
