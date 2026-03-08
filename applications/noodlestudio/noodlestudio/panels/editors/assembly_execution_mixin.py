"""Execution visualization mixin for AssemblyEditorView.

Handles execution events delivered directly by GuidePerformanceManager
(no WebSocket -- events arrive synchronously on the Qt main thread via
_emit_execution_event -> _handle_execution_event).
"""

import os
from typing import Optional, Dict, List

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor

from .assembly_graphics_items import FacetNodeItem, FacetConnectionItem


def _log_facet(facet_name: str, event: str, execution_id: str, detail: str = ""):
    """Log execution event to console (FACETS prefix)."""
    eid = execution_id[:8] if execution_id else ""
    if detail:
        print(f"[FACETS] {facet_name} {event} [{eid}] {detail}")
    else:
        print(f"[FACETS] {facet_name} {event} [{eid}]")


class AssemblyExecutionMixin:
    """Execution visualization: node pulsing, wire packets, sound, pause."""

    # ================================================================
    # Initialization
    # ================================================================

    def _init_execution_state(self):
        """Initialize execution visualization state."""
        self.sound_enabled: bool = True
        self.cognition_paused: bool = False
        self.scene_transition_lock: bool = False
        self.current_agent_id: Optional[str] = None

        # Generation counter: incremented on scene transitions to
        # invalidate pending cleanup timers from a previous scene.
        self._scene_generation: int = 0

        # Cycle color tracking
        self.cycle_colors: Dict[str, QColor] = {}
        self.cycle_color_palette: List[QColor] = [
            QColor("#00BFFF"),  # Deep sky blue
            QColor("#32CD32"),  # Lime green
            QColor("#FFD700"),  # Gold
            QColor("#FF69B4"),  # Hot pink
            QColor("#00CED1"),  # Dark turquoise
            QColor("#FFA500"),  # Orange
            QColor("#9370DB"),  # Medium purple
            QColor("#20B2AA"),  # Light sea green
        ]
        self.next_cycle_color_index: int = 0

    # ================================================================
    # Event dispatch
    # ================================================================

    def _handle_execution_event(self, event: dict):
        """Process an execution event from the performance manager.

        Called synchronously on the Qt main thread by
        GuidePerformanceManager._emit_execution_event().

        Event protocol:
            type: 'facet_execution'
            subtype: cycle_start | cycle_complete | data_flow |
                     facet_start | facet_complete | facet_error |
                     quantum_collapse
            source_id: facet UUID (for facet-level events)
            from_facet / to_facet: facet IDs (for data_flow)
            execution_id: cycle UUID
            noodling_id: ensemble filtering tag
        """
        if self.scene_transition_lock:
            return

        # Ensemble filtering
        selected = getattr(self, '_selected_noodling_id', None)
        if selected is not None:
            event_nid = event.get('noodling_id')
            if event_nid is not None and event_nid != selected:
                return

        event_type = event.get('type')
        event_subtype = event.get('subtype')

        if event_type != 'facet_execution':
            return

        # -- Cycle-level events (no facet_id) --

        if event_subtype == 'cycle_start':
            self.play_sound('cycle_start')
            return

        if event_subtype == 'cycle_complete':
            self.play_sound('cycle_complete')
            execution_id = event.get('data', {}).get('execution_id', '')
            if execution_id and execution_id in self.cycle_colors:
                del self.cycle_colors[execution_id]
            return

        # -- Data flow events (wire animation) --

        if event_subtype == 'data_flow':
            from_facet = event.get('from_facet')
            to_facet = event.get('to_facet')

            if from_facet and to_facet:
                self.play_sound('data_flow')
                for wire in list(self._wire_items):
                    try:
                        if not wire or not wire.scene():
                            continue
                    except RuntimeError:
                        continue  # C++ object deleted
                    if (wire.from_port.facet_node.facet.id == from_facet and
                            wire.to_port.facet_node.facet.id == to_facet):
                        wire.animate_data_flow()
                        break
            return

        # -- Facet-level events --

        facet_id = event.get('source_id')
        if not facet_id or facet_id not in self._node_items:
            return

        node = self._node_items.get(facet_id)
        try:
            if not node or not node.scene():
                return
        except RuntimeError:
            return  # C++ object deleted

        self._play_pachinko_sound()

        execution_id = event.get('data', {}).get('execution_id', '')
        if not execution_id:
            execution_id = event.get('execution_id', '')

        try:
            facet_name = node.facet.name if node.facet else facet_id

            if event_subtype == 'facet_start':
                self._handle_facet_start(node, facet_name, execution_id, event)

            elif event_subtype == 'facet_complete':
                self._handle_facet_complete(node, facet_name, execution_id, event)

            elif event_subtype == 'facet_error':
                error_msg = event.get('data', {}).get('error', 'Unknown error')
                _log_facet(facet_name, "ERROR", execution_id, error_msg)
                node.set_execution_state('error')
                node.update()

            elif event_subtype == 'quantum_collapse':
                _log_facet(facet_name, "QUANTUM_COLLAPSE", execution_id)
                node.set_execution_state('quantum_collapse')
                self._play_quantum_collapse_sound()

        except Exception as e:
            name = facet_name if 'facet_name' in dir() else facet_id
            _log_facet(name, "ANIMATION_ERROR", "", str(e))

    def _handle_facet_start(self, node: FacetNodeItem, facet_name: str,
                            execution_id: str, event: dict):
        """Handle facet_start event: assign cycle color, track inputs."""
        # Assign cycle color
        if execution_id and execution_id not in self.cycle_colors:
            self.cycle_colors[execution_id] = self.cycle_color_palette[
                self.next_cycle_color_index % len(self.cycle_color_palette)
            ]
            self.next_cycle_color_index += 1

        # Capture inputs
        event_data = event.get('data', {})
        inputs = event_data.get('inputs')
        if inputs:
            node.last_inputs = inputs
            if execution_id:
                if execution_id not in node.cycle_data:
                    node.cycle_data[execution_id] = {}
                node.cycle_data[execution_id]['inputs'] = inputs

        # Add to active_cycles (avoid duplicates)
        if execution_id:
            cycle_color = self.cycle_colors.get(execution_id, QColor("#00BFFF"))
            existing_ids = [c[0] for c in node.active_cycles]
            if execution_id not in existing_ids:
                node.active_cycles.append((execution_id, cycle_color, inputs))

        input_keys = list(inputs.keys()) if inputs else []
        _log_facet(facet_name, "START", execution_id, f"inputs: {input_keys}")

        node.set_execution_state('processing')
        node.update()

    def _handle_facet_complete(self, node: FacetNodeItem, facet_name: str,
                               execution_id: str, event: dict):
        """Handle facet_complete event: capture outputs, schedule cleanup."""
        event_data = event.get('data', {})
        outputs = event_data.get('outputs')
        if outputs:
            node.last_outputs = outputs
            if execution_id:
                if execution_id not in node.cycle_data:
                    node.cycle_data[execution_id] = {}
                node.cycle_data[execution_id]['outputs'] = outputs

        output_keys = list(outputs.keys()) if outputs else []
        _log_facet(facet_name, "COMPLETE", execution_id, f"outputs: {output_keys}")

        node.set_execution_state('complete')
        node.update()

        # Remove cycle after 300ms animation.
        # Capture scene generation so the timer is a no-op if the scene
        # was cleared and rebuilt before the 300ms fires. Without this,
        # the closure holds a stale QGraphicsItem pointer and calling
        # node.scene() segfaults (EXC_BAD_ACCESS on freed C++ object).
        captured_exec_id = execution_id
        captured_generation = self._scene_generation

        def clear_cycle():
            if captured_generation != self._scene_generation:
                return  # Scene was rebuilt; node is stale
            try:
                if node and node.scene():
                    node.active_cycles = [
                        c for c in node.active_cycles
                        if c[0] != captured_exec_id
                    ]
                    if not self.cognition_paused:
                        if captured_exec_id in node.cycle_data:
                            del node.cycle_data[captured_exec_id]
                    node.update()
            except RuntimeError:
                pass  # C++ object already deleted

        QTimer.singleShot(300, clear_cycle)

    # ================================================================
    # Sound
    # ================================================================

    def play_sound(self, sound_type: str):
        """Play cached QSoundEffect for execution events."""
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl

            resources_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'resources',
                'terminal_beeps_hq'
            )

            sound_files = {
                'cycle_start': 'termstart.ogg',
                'data_flow': 'termkeypress.ogg',
                'cycle_complete': 'bell_vt100_250ms.ogg',
            }
            sound_file = sound_files.get(sound_type)
            if not sound_file:
                return

            sound_path = os.path.join(resources_dir, sound_file)
            if not os.path.exists(sound_path):
                return

            cache_attr = f'_sound_{sound_type}'
            if not hasattr(self, cache_attr):
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(sound_path))
                volumes = {
                    'cycle_start': 0.5,
                    'data_flow': 0.2,
                    'cycle_complete': 0.4,
                }
                effect.setVolume(volumes.get(sound_type, 0.3))
                setattr(self, cache_attr, effect)

            getattr(self, cache_attr).play()

        except Exception:
            pass

    def _play_pachinko_sound(self):
        """Play termkeypress.ogg (Kraftwerk pachinko click) for every facet event."""
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl

            resources_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'resources',
                'terminal_beeps_hq'
            )
            sound_path = os.path.join(resources_dir, 'termkeypress.ogg')
            if not os.path.exists(sound_path):
                return

            if not hasattr(self, '_pachinko_sound'):
                self._pachinko_sound = QSoundEffect()
                self._pachinko_sound.setSource(QUrl.fromLocalFile(sound_path))
                self._pachinko_sound.setVolume(0.3)

            self._pachinko_sound.play()

        except Exception:
            pass

    def _play_quantum_collapse_sound(self):
        """Play termstart.ogg (higher pitch) for quantum collapse events."""
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl

            resources_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'resources',
                'terminal_beeps_hq'
            )
            sound_path = os.path.join(resources_dir, 'termstart.ogg')
            if not os.path.exists(sound_path):
                return

            if not hasattr(self, '_quantum_sound'):
                self._quantum_sound = QSoundEffect()
                self._quantum_sound.setSource(QUrl.fromLocalFile(sound_path))
                self._quantum_sound.setVolume(0.4)

            self._quantum_sound.play()

        except Exception:
            pass

    def toggle_sound(self, checked: bool):
        """Toggle execution sound effects."""
        self.sound_enabled = checked

    # ================================================================
    # Cognition pause/resume
    # ================================================================

    def toggle_pause_cognition(self, checked: bool):
        """Toggle cognitive processing pause for the current agent.

        Sets the paused flag on the local NoodlingPerformer so that
        execute() becomes a no-op while paused.
        """
        if not self.current_agent_id:
            return

        performer = self._find_performer(self.current_agent_id)

        if checked:
            if performer:
                performer.set_paused(True)
            self.cognition_paused = True
            self._update_stage_pause_state(True)
        else:
            if performer:
                performer.set_paused(False)
            self.cognition_paused = False
            self._update_stage_pause_state(False)

    def _find_performer(self, agent_id: str):
        """Find the NoodlingPerformer for an agent through the main window."""
        widget = self.parent()
        while widget and not hasattr(widget, 'guide_performance_manager'):
            widget = widget.parent() if hasattr(widget, 'parent') else None

        if widget and hasattr(widget, 'guide_performance_manager'):
            manager = widget.guide_performance_manager
            if manager and hasattr(manager, '_performers'):
                noodling_id = agent_id.replace('agent_', '')
                return manager._performers.get(noodling_id)

        return None

    def _update_stage_pause_state(self, paused: bool):
        """Notify Stage panel to update pause state for current agent."""
        if not self.current_agent_id:
            return

        widget = self.parent()
        while widget and not hasattr(widget, 'hierarchy'):
            widget = widget.parent() if hasattr(widget, 'parent') else None

        if widget and hasattr(widget, 'hierarchy'):
            hierarchy = widget.hierarchy
            if hasattr(hierarchy, 'update_instance_pause_state'):
                hierarchy.update_instance_pause_state(
                    self.current_agent_id, paused
                )
