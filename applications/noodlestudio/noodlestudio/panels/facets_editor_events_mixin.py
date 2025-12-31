"""
Facets Editor Events Mixin - WebSocket, execution events, sounds, and cognition

Contains event handling operations:
- WebSocket: _start_websocket_connection, _websocket_handler, etc.
- Execution events: _handle_execution_event, _process_event_queue
- Sound: toggle_sound, play_sound, _play_pachinko_sound
- Cognition: toggle_pause_cognition, set_current_agent
- Context menu: show_context_menu, add_facet
- Floating editor: show_floating_editor

Author: Noodlings Project
Date: December 2025
"""

import json
import asyncio
import requests
from typing import Optional

from PyQt6.QtWidgets import QMenu, QMessageBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor

from .facets_editor_graphics import FacetNodeGraphics, ConnectionWire, _log_facet
from ..core.facet_system import Facet


class FacetsEditorEventsMixin:
    """Mixin providing event handling for FacetsEditorPanel."""

    # ========== CONTEXT MENU ==========

    def show_context_menu(self, position):
        """Show right-click context menu for adding facets."""
        # Set flag to prevent selection changes during context menu
        self._in_right_click = True

        # Temporarily disconnect selection changed to prevent crashes
        if self._selection_signal_connected:
            try:
                self.scene.selectionChanged.disconnect(self.on_selection_changed)
                self._selection_signal_connected = False
            except:
                pass  # Already disconnected

        try:
            menu = QMenu(self)

            # Check if right-click is on a wire
            scene_pos = self.view.mapToScene(position)
            clicked_item = self.scene.itemAt(scene_pos, self.view.transform())

            # Check if we clicked on a ConnectionWire
            clicked_wire = None
            if isinstance(clicked_item, ConnectionWire):
                clicked_wire = clicked_item

            # Wire-specific context menu
            if clicked_wire:
                # Show wire disconnect menu
                from_facet = clicked_wire.from_pad.facet_node.facet
                to_facet = clicked_wire.to_pad.facet_node.facet
                from_pad_name = clicked_wire.from_pad.pad.name
                to_pad_name = clicked_wire.to_pad.pad.name

                wire_label = f"{from_facet.name}.{from_pad_name} -> {to_facet.name}.{to_pad_name}"
                info_action = menu.addAction(f"Connection: {wire_label}")
                info_action.setEnabled(False)  # Just a label

                menu.addSeparator()

                delete_action = menu.addAction("Delete Connection")
                delete_action.triggered.connect(
                    lambda: self.delete_connection_wire(clicked_wire)
                )

                # Also add "Delete" as keyboard shortcut hint
                delete_action.setShortcut("Delete")

                menu.exec(self.view.mapToGlobal(position))
                return  # Early return for wire context menu

            # Add facet submenu (excluding INCOMING/OUTGOING - those are auto-created)
            add_menu = menu.addMenu("Add Facet")

            facet_types = [
                ("Intuition Facet", "IntuitionFacet"),
                ("Emotion Facet", "EmotionFacet"),
                ("Social Context Facet", "SocialFacet"),
                ("Memory Recall Facet", "MemoryFacet"),
                ("Response Planning Facet", "PlanningFacet"),
                ("Convergence Facet", "ConvergenceFacet"),
                ("Scripted Facet (JavaScript)", "ScriptedFacet"),
                ("MCP Tool Facet", "MCPFacet"),
            ]

            # Math facets submenu
            math_menu = add_menu.addMenu("Math")
            math_types = [
                ("Add (a + b)", "MathAddFacet"),
                ("Subtract (a - b)", "MathSubtractFacet"),
                ("Multiply (a * b)", "MathMultiplyFacet"),
                ("Divide (a / b)", "MathDivideFacet"),
                ("Min", "MathMinFacet"),
                ("Max", "MathMaxFacet"),
                ("Clamp", "MathClampFacet"),
                ("Absolute Value", "MathAbsFacet"),
            ]
            for display_name, facet_type in math_types:
                action = math_menu.addAction(display_name)
                action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                        self.add_facet(ft, dn, position))

            # Logic facets submenu
            logic_menu = add_menu.addMenu("Logic")
            logic_types = [
                ("AND", "LogicAndFacet"),
                ("OR", "LogicOrFacet"),
                ("NOT", "LogicNotFacet"),
                ("Compare", "LogicCompareFacet"),
                ("Switch (If/Else)", "LogicSwitchFacet"),
            ]
            for display_name, facet_type in logic_types:
                action = logic_menu.addAction(display_name)
                action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                        self.add_facet(ft, dn, position))

            # String facets submenu
            string_menu = add_menu.addMenu("String")
            string_types = [
                ("Concat", "StringConcatFacet"),
                ("Split", "StringSplitFacet"),
                ("Replace", "StringReplaceFacet"),
                ("Format (Template)", "StringFormatFacet"),
                ("Length", "StringLengthFacet"),
                ("Contains", "StringContainsFacet"),
                ("Regex Match", "StringRegexFacet"),
            ]
            for display_name, facet_type in string_types:
                action = string_menu.addAction(display_name)
                action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                        self.add_facet(ft, dn, position))

            # Array facets submenu
            array_menu = add_menu.addMenu("Array")
            array_types = [
                ("Get Element", "ArrayGetFacet"),
                ("First", "ArrayFirstFacet"),
                ("Last", "ArrayLastFacet"),
                ("Join", "ArrayJoinFacet"),
                ("Length", "ArrayLengthFacet"),
            ]
            for display_name, facet_type in array_types:
                action = array_menu.addAction(display_name)
                action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                        self.add_facet(ft, dn, position))

            # Data/Control facets submenu
            data_menu = add_menu.addMenu("Data")
            data_types = [
                ("Pass Through", "PassThroughFacet"),
                ("Gate", "GateFacet"),
                ("Counter", "CounterFacet"),
                ("JSON Parse", "JSONParseFacet"),
                ("JSON Stringify", "JSONStringifyFacet"),
                ("Get Property", "GetPropertyFacet"),
                ("Set Property", "SetPropertyFacet"),
            ]
            for display_name, facet_type in data_types:
                action = data_menu.addAction(display_name)
                action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                        self.add_facet(ft, dn, position))

            for display_name, facet_type in facet_types:
                action = add_menu.addAction(display_name)
                action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                        self.add_facet(ft, dn, position))

            # Separator
            add_menu.addSeparator()

            # Custom/Empty facet at bottom
            custom_action = add_menu.addAction("Create empty facet")
            custom_action.triggered.connect(lambda: self.add_facet("CustomFacet", "Custom Facet", position))

            # Layout menu
            menu.addSeparator()
            layout_menu = menu.addMenu("Layout")

            auto_arrange_action = layout_menu.addAction("Auto-Arrange (Topological)")
            auto_arrange_action.triggered.connect(self.auto_arrange_facets)

            layout_menu.addSeparator()

            # Alignment (requires selection)
            selected_nodes = self.scene.selectedItems()
            selected_facets = [item for item in selected_nodes if isinstance(item, FacetNodeGraphics)]

            align_h_action = layout_menu.addAction(f"Align Horizontally ({len(selected_facets)} selected)")
            align_h_action.setEnabled(len(selected_facets) > 1)
            align_h_action.triggered.connect(self.align_selected_horizontally)

            align_v_action = layout_menu.addAction(f"Align Vertically ({len(selected_facets)} selected)")
            align_v_action.setEnabled(len(selected_facets) > 1)
            align_v_action.triggered.connect(self.align_selected_vertically)

            layout_menu.addSeparator()

            # Zoom (use zoom_view to respect limits)
            zoom_in_action = layout_menu.addAction("Zoom In (+)")
            zoom_in_action.triggered.connect(lambda: self.zoom_view(1.2))

            zoom_out_action = layout_menu.addAction("Zoom Out (-)")
            zoom_out_action.triggered.connect(lambda: self.zoom_view(1/1.2))

            reset_zoom_action = layout_menu.addAction("Reset View")
            reset_zoom_action.triggered.connect(self.reset_view)

            # Delete (requires selection)
            if selected_facets:
                menu.addSeparator()
                delete_action = menu.addAction(f"Delete {len(selected_facets)} facet(s)")
                delete_action.triggered.connect(self.delete_selected_facets)

            menu.exec(self.view.mapToGlobal(position))

        except Exception as e:
            pass  # Silent context menu errors
        finally:
            # Always clear the flag when menu closes
            self._in_right_click = False
            # Reconnect selection changed signal
            if not self._selection_signal_connected:
                try:
                    self.scene.selectionChanged.connect(self.on_selection_changed)
                    self._selection_signal_connected = True
                except:
                    pass

    def add_facet(self, facet_type: str, display_name: str, position):
        """Add a new facet to the assembly (with undo support)."""
        if not self.current_assembly:
            return

        # Convert view position to scene position
        scene_pos = self.view.mapToScene(position)

        # Create new facet data (not added to assembly yet - command will do that)
        facet_id = Facet.generate_uuid()
        facet = Facet(
            id=facet_id,
            name=display_name,
            facet_type=facet_type,
            prompt=f"TODO: Define prompt for {display_name}",
            position={'x': scene_pos.x(), 'y': scene_pos.y()}
        )

        # Add default pads based on type
        if facet_type == "ConvergenceFacet":
            facet.add_input_pad("input1", "First input")
            facet.add_input_pad("input2", "Second input")
            facet.add_output_pad("output", "Merged output")
        else:
            facet.add_input_pad("in", "Input")
            facet.add_output_pad("out", "Output")

        # Push create command via UndoManager (command will create the facet)
        from ..core.undo_manager import undo_manager
        from ..core.commands import CreateFacetCommand

        cmd = CreateFacetCommand(
            editor=self,
            facet_data=facet.to_dict(),
            facet_name=display_name
        )
        undo_manager.push(cmd)

    # ========== FLOATING EDITOR ==========

    def show_floating_editor(self, facet: Facet, field_data: dict):
        """
        Show floating text editor for a facet field.

        Args:
            facet: Facet being edited
            field_data: Field definition dict
        """
        from .floating_text_editor import FloatingTextEditor

        editor = FloatingTextEditor(
            field_name=field_data['name'],
            field_key=field_data['key'],
            initial_value=field_data['value'],
            read_only=field_data['read_only'],
            parent=self
        )

        # Connect apply signal
        def on_applied(key, value):
            # Update facet field
            if key == 'prompt':
                facet.prompt = value
            # Refresh field display if node currently showing fields
            for item in self.scene.items():
                if isinstance(item, FacetNodeGraphics) and item.facet.id == facet.id:
                    if item.field_widgets:  # If fields currently visible, refresh them
                        item.show_fields(force=True)

        editor.textApplied.connect(on_applied)

        # Position centered on screen
        editor.move(
            self.mapToGlobal(self.rect().center()).x() - 250,
            self.mapToGlobal(self.rect().center()).y() - 200
        )

        # Show as modal dialog
        editor.exec()

    # ========== COGNITION CONTROL ==========

    def toggle_pause_cognition(self, checked: bool):
        """Toggle cognitive processing pause for the current agent."""
        if not self.current_agent_id:
            QMessageBox.warning(self, "No Agent", "No agent is currently loaded in the Facets Editor.")
            self.pause_button.setChecked(False)
            return

        try:
            if checked:
                # PAUSING: Request immediate freeze (mid-cycle pause for debugging)
                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={
                    'paused': True,
                    'agent_id': self.current_agent_id,
                    'freeze_mode': 'immediate'  # Freeze mid-cycle for inspection
                }, timeout=5)

                if response.status_code == 200:
                    self.cognition_paused = True
                    self.pause_button.setText("Resume Cognition")
                    self.bottom_pause_btn.setText(">")
                    self.bottom_pause_btn.setChecked(True)

                    # Update Stage panel to reflect pause state
                    self._update_stage_pause_state(True)

                    # Refresh all visible fields to show output as editable
                    for node in self.node_graphics.values():
                        if node.field_widgets:  # If fields currently visible, refresh them
                            node.show_fields(force=True)
                else:
                    QMessageBox.warning(self, "Pause Failed", f"Failed to pause cognition: {response.status_code}")
                    self.pause_button.setChecked(False)
                    self.bottom_pause_btn.setChecked(False)

            else:
                # RESUMING: Apply edits and resume cognition
                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={'paused': False, 'agent_id': self.current_agent_id}, timeout=2)

                if response.status_code == 200:
                    self.cognition_paused = False
                    self.pause_button.setText("Pause Cognition")
                    self.bottom_pause_btn.setText("||")
                    self.bottom_pause_btn.setChecked(False)

                    # Update Stage panel to reflect resume state
                    self._update_stage_pause_state(False)

                    # Refresh all visible fields to show output as read-only again
                    for node in self.node_graphics.values():
                        if node.field_widgets:  # If fields currently visible, refresh them
                            node.show_fields(force=True)
                else:
                    QMessageBox.warning(self, "Resume Failed", f"Failed to resume cognition: {response.status_code}")
                    self.pause_button.setChecked(True)
                    self.bottom_pause_btn.setChecked(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pause/resume error: {str(e)}")
            self.pause_button.setChecked(not checked)  # Revert button state

    def set_current_agent(self, agent_id: str):
        """
        Set the current agent for the Facets Editor.

        Called when a noodling is selected in Stage View.
        Loads the agent's facet assembly and enables pause controls.

        Args:
            agent_id: Agent ID to set as current
        """
        import os
        from ..core.facet_system import FacetAssembly

        self.current_agent_id = agent_id

        # Enable pause button
        self.pause_button.setEnabled(True)
        self.bottom_pause_btn.setEnabled(True)

        # Load agent's facet assembly if available
        # Try to get assembly path from agent config
        if agent_id:
            try:
                # Query server for agent config
                response = requests.get(f"{self.api_base}/agents/{agent_id}", timeout=2)
                if response.status_code == 200:
                    agent_data = response.json()
                    assembly_ref = agent_data.get('config', {}).get('facet_assembly')
                    if assembly_ref:
                        # Find and load the assembly
                        assembly_dir = os.path.join(os.path.dirname(__file__), '../facet_assemblies')
                        assembly_path = os.path.join(assembly_dir, f"{assembly_ref}.yaml")
                        if os.path.exists(assembly_path):
                            assembly = FacetAssembly.load_yaml(assembly_path)
                            self.load_assembly_from_data(assembly, force_reload=True)
                        else:
                            print(f"[Facets Editor] Assembly file not found: {assembly_path}")
            except Exception as e:
                print(f"[Facets Editor] Failed to load agent assembly: {e}")

    def _update_stage_pause_state(self, paused: bool):
        """Notify Stage panel to update pause state for current agent."""
        if not self.current_agent_id:
            return

        # Find main window and update Stage panel
        widget = self.parent()
        while widget and not hasattr(widget, 'hierarchy'):
            widget = widget.parent() if hasattr(widget, 'parent') else None

        if widget and hasattr(widget, 'hierarchy'):
            stage_panel = widget.hierarchy
            # Update tracked pause state
            stage_panel.agent_pause_states[self.current_agent_id] = paused
            # Refresh Stage to update icon
            stage_panel.refresh_scene()

    # ========== SOUND ==========

    def toggle_sound(self, checked: bool):
        """Toggle execution sound effects."""
        self.sound_enabled = checked
        if checked:
            self.sound_button.setText("Sound On")
        else:
            self.sound_button.setText("Sound Off")

    def play_sound(self, sound_type: str):
        """
        Play sound effect for execution events.

        Args:
            sound_type: 'cycle_start', 'data_flow', 'cycle_complete'
        """
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            import os

            # Map sound types to terminal beep files (Kraftwerk aesthetic!)
            resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'terminal_beeps_hq')

            sound_files = {
                'cycle_start': 'termstart.ogg',          # High pitch - cycle begins!
                'data_flow': 'termkeypress.ogg',         # Quick click - data packet
                'cycle_complete': 'bell_vt100_250ms.ogg' # Bell chime - cycle ends
            }

            sound_file = sound_files.get(sound_type)
            if not sound_file:
                return

            sound_path = os.path.join(resources_dir, sound_file)
            if not os.path.exists(sound_path):
                return  # Sound file not found, silent fail

            # Create cached sound effect for this type
            cache_attr = f'_sound_{sound_type}'
            if not hasattr(self, cache_attr):
                sound_effect = QSoundEffect()
                sound_effect.setSource(QUrl.fromLocalFile(sound_path))

                # Volume settings - industrial precision
                volumes = {
                    'cycle_start': 0.5,    # Clear attention signal
                    'data_flow': 0.2,      # Quiet clicks (many events)
                    'cycle_complete': 0.4  # Satisfying closure
                }
                sound_effect.setVolume(volumes.get(sound_type, 0.3))
                setattr(self, cache_attr, sound_effect)

            # Play (non-blocking)
            sound_effect = getattr(self, cache_attr)
            sound_effect.play()

        except Exception:
            # Silent fail - don't break visualization if sound fails
            pass

    def _play_pachinko_sound(self):
        """Play termkeypress.ogg sound (Kraftwerk pachinko click)."""
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            import os

            # Get sound file path
            resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'terminal_beeps_hq')
            sound_path = os.path.join(resources_dir, 'termkeypress.ogg')

            if not os.path.exists(sound_path):
                return  # Sound file not found, silent fail

            # Create sound effect if not already created
            if not hasattr(self, '_pachinko_sound'):
                self._pachinko_sound = QSoundEffect()
                self._pachinko_sound.setSource(QUrl.fromLocalFile(sound_path))
                self._pachinko_sound.setVolume(0.3)  # 30% volume (not too loud!)

            # Play (non-blocking)
            self._pachinko_sound.play()

        except Exception:
            # Silent fail - don't break execution visualization if sound fails
            pass

    def _play_quantum_collapse_sound(self):
        """
        Play quantum collapse sound effect.

        Higher pitch than normal pachinko click to indicate quantum event.
        """
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            import os

            # Get sound file path (use termstart.ogg for higher pitch)
            resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'terminal_beeps_hq')
            sound_path = os.path.join(resources_dir, 'termstart.ogg')  # Higher pitch than keypress

            if not os.path.exists(sound_path):
                return  # Sound file not found, silent fail

            # Create sound effect if not already created
            if not hasattr(self, '_quantum_sound'):
                self._quantum_sound = QSoundEffect()
                self._quantum_sound.setSource(QUrl.fromLocalFile(sound_path))
                self._quantum_sound.setVolume(0.4)  # Slightly louder than pachinko

            # Play (non-blocking)
            self._quantum_sound.play()

        except Exception:
            # Silent fail - don't break execution visualization if sound fails
            pass

    # ========== WEBSOCKET - EXECUTION EVENT STREAMING ==========

    def _start_websocket_connection(self):
        """Start WebSocket connection to execution event stream."""
        try:
            import websockets
        except ImportError:
            return

        # Start event processing timer (polls queue from Qt thread)
        self.event_timer = QTimer()
        self.event_timer.timeout.connect(self._process_event_queue)
        self.event_timer.start(16)  # 60fps event processing

        # Start WebSocket task in separate thread
        import threading
        ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        ws_thread.start()

    def _run_websocket_loop(self):
        """Run WebSocket event loop in separate thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._websocket_handler())
        except Exception:
            pass  # WebSocket errors handled in _websocket_handler

    async def _websocket_handler(self):
        """Handle WebSocket connection and message receiving."""
        import websockets

        uri = "ws://localhost:8081/ws/execution_events"

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    self.ws_connection = websocket
                    self.ws_connected = True

                    async for message in websocket:
                        try:
                            event_data = json.loads(message)
                            # Add to queue for Qt thread processing
                            if self.event_queue:
                                await self.event_queue.put(event_data)
                        except json.JSONDecodeError:
                            pass  # Ignore malformed messages

            except Exception:
                self.ws_connected = False
                await asyncio.sleep(5)  # Reconnect delay

    def _process_event_queue(self):
        """Process execution events from queue (called from Qt timer)."""
        if not self.event_queue:
            return

        # Process up to 10 events per frame (prevent UI blocking)
        for _ in range(10):
            try:
                event = self.event_queue.get_nowait()
                self._handle_execution_event(event)
            except asyncio.QueueEmpty:
                break  # Queue empty - expected
            except Exception as e:
                _log_facet("EventProcessor", "ERROR", "", str(e))
                break

    def _handle_execution_event(self, event: dict):
        """
        Handle execution event and trigger appropriate animation.

        Event types:
        - facet_start: Node begins executing (yellow + pulse)
        - facet_complete: Node finishes (brief bright, then idle)
        - data_flow: Animate packet along connection wire
        - convergence_wait: (future: show waiting state)
        """
        # CRITICAL: Skip event processing during scene transitions
        if self.scene_transition_lock:
            return  # Scene is being cleared/rebuilt, ignore all events

        event_type = event.get('type')
        event_subtype = event.get('subtype')

        if event_type != 'facet_execution':
            return  # Ignore non-execution events

        # Handle cycle-level events (no facet_id)
        if event_subtype == 'cycle_start':
            self.play_sound('cycle_start')
            return
        elif event_subtype == 'cycle_complete':
            self.play_sound('cycle_complete')
            # Clean up cycle color for completed cycle (prevent memory leak)
            execution_id = event.get('data', {}).get('execution_id', '')
            if not execution_id:
                execution_id = event.get('execution_id', '')
            if execution_id and execution_id in self.cycle_colors:
                del self.cycle_colors[execution_id]
            return

        # Handle data_flow events separately (they have from_facet/to_facet, not source_id)
        if event_subtype == 'data_flow':
            from_facet = event.get('from_facet')
            to_facet = event.get('to_facet')

            if from_facet and to_facet:
                # Play data flow sound
                self.play_sound('data_flow')

                # Find connection wire between these facets
                try:
                    for wire in list(self.wire_graphics):
                        if not wire or not wire.scene():
                            continue  # Wire was deleted, skip
                        if not hasattr(wire, 'from_pad') or not hasattr(wire, 'to_pad'):
                            continue  # Invalid wire state
                        if not wire.from_pad or not wire.to_pad:
                            continue
                        if not hasattr(wire.from_pad, 'facet_node') or not hasattr(wire.to_pad, 'facet_node'):
                            continue
                        if (wire.from_pad.facet_node.facet.id == from_facet and
                            wire.to_pad.facet_node.facet.id == to_facet):
                            wire.animate_data_flow()
                            break
                except Exception:
                    pass  # Silent data flow animation errors
            return  # data_flow handled, exit

        facet_id = event.get('source_id')
        if not facet_id or facet_id not in self.node_graphics:
            return  # Facet not in current assembly (normal during transitions)

        node = self.node_graphics.get(facet_id)
        if not node:
            return  # Node was deleted (race condition during scene transition)

        # CRITICAL: Check if node is still in scene (not deleted)
        if not node.scene():
            return  # Node removed from scene, skip event

        # KRAFTWERK CLICK - Play terminal keypress sound for every event
        self._play_pachinko_sound()

        # Extract execution_id for cycle tracking
        execution_id = event.get('data', {}).get('execution_id', '')
        if not execution_id:
            execution_id = event.get('execution_id', '')

        try:
            # Get facet name for logging
            facet_name = node.facet.name if node.facet else facet_id

            if event_subtype == 'facet_start':
                # KRAFTWERK: Node begins processing
                # Assign cycle color if not already assigned
                if execution_id and execution_id not in self.cycle_colors:
                    self.cycle_colors[execution_id] = self.cycle_color_palette[
                        self.next_cycle_color_index % len(self.cycle_color_palette)
                    ]
                    self.next_cycle_color_index += 1

                # Capture inputs for inspection (debugging feature)
                event_data = event.get('data', {})
                inputs = event_data.get('inputs')
                if inputs:
                    node.last_inputs = inputs
                    # Store per-cycle inputs for inspection
                    if execution_id:
                        if execution_id not in node.cycle_data:
                            node.cycle_data[execution_id] = {}
                        node.cycle_data[execution_id]['inputs'] = inputs

                # Add this cycle to active_cycles list (supports stacking!)
                if execution_id:
                    cycle_color = self.cycle_colors.get(execution_id, QColor("#00BFFF"))
                    # Check if this cycle is already active on this node (avoid duplicates)
                    existing_ids = [c[0] for c in node.active_cycles]
                    if execution_id not in existing_ids:
                        node.active_cycles.append((execution_id, cycle_color, inputs))

                # Log to FACETS console
                input_keys = list(inputs.keys()) if inputs else []
                _log_facet(facet_name, "START", execution_id, f"inputs: {input_keys}")

                node.set_execution_state('processing')
                node.update()  # Force repaint to show cycle badge

            elif event_subtype == 'facet_complete':
                # KRAFTWERK: Node completes (brief satisfaction, then idle)
                # Capture outputs for inspection (debugging feature)
                event_data = event.get('data', {})
                outputs = event_data.get('outputs')
                if outputs:
                    node.last_outputs = outputs
                    # Store per-cycle outputs for inspection
                    if execution_id:
                        if execution_id not in node.cycle_data:
                            node.cycle_data[execution_id] = {}
                        node.cycle_data[execution_id]['outputs'] = outputs

                # Log to FACETS console
                output_keys = list(outputs.keys()) if outputs else []
                _log_facet(facet_name, "COMPLETE", execution_id, f"outputs: {output_keys}")

                node.set_execution_state('complete')
                node.update()

                # Remove this cycle from active_cycles after animation completes
                captured_exec_id = execution_id  # Capture for closure
                def clear_cycle_from_list():
                    if node and node.scene():
                        # Remove only the completed cycle from the list
                        node.active_cycles = [c for c in node.active_cycles if c[0] != captured_exec_id]
                        # Clean up cycle_data after a delay (keep for inspection during pause)
                        if not self.cognition_paused:
                            if captured_exec_id in node.cycle_data:
                                del node.cycle_data[captured_exec_id]
                        node.update()
                QTimer.singleShot(300, clear_cycle_from_list)

            elif event_subtype == 'facet_error':
                # ERROR: Something went wrong - flash red
                error_msg = event.get('data', {}).get('error', 'Unknown error')
                _log_facet(facet_name, "ERROR", execution_id, error_msg)
                node.set_execution_state('error')
                node.update()

            elif event_subtype == 'quantum_collapse':
                # QUANTUM: Orchestrated objective reduction event
                _log_facet(facet_name, "QUANTUM_COLLAPSE", execution_id)
                node.set_execution_state('quantum_collapse')
                self._play_quantum_collapse_sound()

        except Exception as e:
            _log_facet(facet_name if 'facet_name' in dir() else facet_id, "ANIMATION_ERROR", "", str(e))

    def _clear_right_click_flag(self):
        """Clear the right-click flag after context menu closes."""
        self._in_right_click = False

    def _reconnect_selection_signal(self):
        """Reconnect selection changed signal after context menu."""
        if not self._selection_signal_connected:
            try:
                self.scene.selectionChanged.connect(self.on_selection_changed)
                self._selection_signal_connected = True
            except:
                pass
