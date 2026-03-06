# ------------------------------------------------------------------
#   Set Dressing Inspector Mixin
#
#   Inspector views for StageSet and BlockingMark entities.
#   Provides editable properties for set descriptions, scene objects,
#   mark perspectives, and the can_see visibility checklist.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.panels.inspector_set_dressing
# PURPOSE:  Set Dressing Inspector
# LAYER:    Studio / Panels
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

from PyQt6.QtWidgets import (
    QLabel, QTextEdit, QPlainTextEdit, QPushButton, QCheckBox, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox,
)
from PyQt6.QtCore import Qt
import os
import yaml


class SetDressingInspectorMixin:
    """
    Mixin providing inspector views for set dressing entities.

    Requires host class to have:
    - self.properties_layout (QVBoxLayout)
    - self.property_fields (dict)
    - self.create_property_group(title)
    - self.add_text_field(group, label, value, read_only)
    - self.add_text_area(group, label, value)
    """

    # ========== SET PROPERTIES ==========

    def load_set_properties(self, entity_data):
        """Show StageSet properties -- name, description, scene objects."""
        from noodlestudio.core.set_dressing import StageSet, SetObject, save_set

        self.property_fields = {}
        data = entity_data.get('data', {})
        stage_path = entity_data.get('stage_path', '')
        stage_set = StageSet.from_dict(data)

        # ===== SET INFO =====
        set_group = self.create_property_group("Set")

        self.property_fields['set_name'] = self.add_text_field(
            set_group, "Name", stage_set.name)

        desc_edit = QTextEdit(stage_set.description)
        desc_edit.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        desc_edit.setMaximumHeight(120)
        desc_edit.setTabChangesFocus(True)
        set_group.content.layout().addRow("Description:", desc_edit)
        self.property_fields['set_description'] = desc_edit

        self.properties_layout.addWidget(set_group)

        # ===== SCENE OBJECTS =====
        objects_group = self.create_property_group("Scene Objects")

        for i, obj in enumerate(stage_set.objects):
            obj_label = QLabel(f"{obj.name} ({obj.id})")
            obj_label.setStyleSheet("color: #D2D2D2; font-weight: bold;")
            objects_group.content.layout().addRow(obj_label)

            obj_desc = QLabel(obj.description)
            obj_desc.setStyleSheet("color: #AAAAAA; padding-left: 8px;")
            obj_desc.setWordWrap(True)
            objects_group.content.layout().addRow(obj_desc)

        # Object count
        count_label = QLabel(f"{len(stage_set.objects)} objects")
        count_label.setStyleSheet("color: #666;")
        objects_group.content.layout().addRow(count_label)

        self.properties_layout.addWidget(objects_group)

        # ===== OPENING SCENE =====
        self._build_opening_scene_section(entity_data, stage_set, stage_path)

        self.properties_layout.addStretch()

    # ========== BLOCKING MARK PROPERTIES ==========

    def load_mark_properties(self, entity_data):
        """Show BlockingMark properties -- name, perspective, can_see checklist."""
        from noodlestudio.core.set_dressing import (
            BlockingMark, load_set, load_mark, save_mark,
        )

        self.property_fields = {}
        data = entity_data.get('data', {})
        mark_path = entity_data.get('path', '')
        stage_path = entity_data.get('stage_path', '')
        mark = BlockingMark.from_dict(data)

        # ===== MARK INFO =====
        mark_group = self.create_property_group("Blocking Mark")

        self.property_fields['mark_name'] = self.add_text_field(
            mark_group, "Name", mark.name)

        self.property_fields['mark_id'] = self.add_text_field(
            mark_group, "ID", mark.id, read_only=True)

        # Perspective (large text area)
        persp_edit = QTextEdit(mark.perspective)
        persp_edit.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        persp_edit.setMaximumHeight(160)
        persp_edit.setTabChangesFocus(True)
        mark_group.content.layout().addRow("Perspective:", persp_edit)
        self.property_fields['mark_perspective'] = persp_edit

        # Activity (what the noodling is doing at this mark)
        activity_edit = QTextEdit(mark.activity)
        activity_edit.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        activity_edit.setMaximumHeight(80)
        activity_edit.setTabChangesFocus(True)
        activity_edit.setPlaceholderText(
            "What is the noodling doing here? (used in opening scene)")
        activity_edit.textChanged.connect(
            lambda mp=mark_path, m=mark:
                self._on_mark_activity_changed(mp, m)
        )
        mark_group.content.layout().addRow("Activity:", activity_edit)
        self.property_fields['mark_activity'] = activity_edit

        self.properties_layout.addWidget(mark_group)

        # ===== VISIBLE FROM HERE (can_see checklist) =====
        parent_set = load_set(stage_path) if stage_path else None

        if parent_set and parent_set.objects:
            vis_group = self.create_property_group("Visible From Here")

            self._mark_can_see_checkboxes = []
            for obj in parent_set.objects:
                cb = QCheckBox(f"{obj.name} ({obj.id})")
                cb.setStyleSheet("QCheckBox { color: #D2D2D2; }")
                cb.setChecked(obj.id in mark.can_see)

                # Wire up save-on-toggle
                cb.stateChanged.connect(
                    lambda state, o=obj.id, mp=mark_path, m=mark:
                        self._on_can_see_toggled(o, state, mp, m)
                )
                vis_group.content.layout().addRow(cb)
                self._mark_can_see_checkboxes.append((obj.id, cb))

            self.properties_layout.addWidget(vis_group)

        self.properties_layout.addStretch()

    def _on_can_see_toggled(self, obj_id, state, mark_path, mark):
        """Handle can_see checkbox toggle -- update mark and save."""
        from noodlestudio.core.set_dressing import save_mark

        if self.is_loading:
            return

        checked = state == Qt.CheckState.Checked.value
        if checked and obj_id not in mark.can_see:
            mark.can_see.append(obj_id)
        elif not checked and obj_id in mark.can_see:
            mark.can_see.remove(obj_id)

        if mark_path and os.path.exists(mark_path):
            save_mark(mark_path, mark)

    def _on_mark_activity_changed(self, mark_path, mark):
        """Handle activity text change -- update mark and save."""
        from noodlestudio.core.set_dressing import save_mark

        if self.is_loading:
            return

        activity_edit = self.property_fields.get('mark_activity')
        if activity_edit:
            mark.activity = activity_edit.toPlainText()

        if mark_path and os.path.exists(mark_path):
            save_mark(mark_path, mark)

    # ========== OPENING SCENE SECTION (on Set inspector) ==========

    def _build_opening_scene_section(self, entity_data, stage_set, stage_path):
        """Build the Opening Scene inspector section on the Set view.

        Shows mode dropdown. In 'live' mode: beat list editor.
        In 'narrated' mode: narration text area.
        In 'silent' mode: informational label.
        """
        from noodlestudio.core.set_dressing import (
            OpeningScene, OpeningBeat, save_set,
        )

        opening = stage_set.opening or OpeningScene()

        opening_group = self.create_property_group("Opening Scene")

        # Mode dropdown
        mode_combo = QComboBox()
        mode_combo.addItems(['silent', 'live', 'narrated'])
        mode_combo.setCurrentText(opening.mode)
        mode_combo.setStyleSheet(
            "QComboBox { background-color: #1E1E1E; color: #D2D2D2; "
            "padding: 4px; border: 1px solid #3A3A3A; }")
        opening_group.content.layout().addRow("Mode:", mode_combo)
        self.property_fields['opening_mode'] = mode_combo

        # Container for mode-specific widgets
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        opening_group.content.layout().addRow(mode_container)

        # Build beat list widget (for live mode)
        beat_list_widget = self._build_beat_list_widget(
            opening, stage_set, stage_path)

        # Build narration widget (for narrated mode)
        narration_edit = QTextEdit(opening.narration)
        narration_edit.setStyleSheet(
            "background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        narration_edit.setMaximumHeight(120)
        narration_edit.setTabChangesFocus(True)
        narration_edit.setPlaceholderText("Narration text...")
        self.property_fields['opening_narration'] = narration_edit

        # Silent label
        silent_label = QLabel("No opening scene. Ensemble starts silent.")
        silent_label.setStyleSheet("color: #666; font-style: italic;")
        silent_label.setWordWrap(True)

        mode_layout.addWidget(beat_list_widget)
        mode_layout.addWidget(narration_edit)
        mode_layout.addWidget(silent_label)

        # Store refs for visibility toggling
        self._opening_beat_list = beat_list_widget
        self._opening_narration_edit = narration_edit
        self._opening_silent_label = silent_label

        def _update_mode_visibility(mode_text=None):
            if mode_text is None:
                mode_text = mode_combo.currentText()
            beat_list_widget.setVisible(mode_text == 'live')
            narration_edit.setVisible(mode_text == 'narrated')
            silent_label.setVisible(mode_text == 'silent')

        _update_mode_visibility(opening.mode)

        def _on_mode_changed(mode_text):
            if self.is_loading:
                return
            _update_mode_visibility(mode_text)
            opening.mode = mode_text
            if stage_set.opening is None:
                stage_set.opening = opening
            if stage_path and os.path.exists(stage_path):
                save_set(stage_path, stage_set)

        mode_combo.currentTextChanged.connect(_on_mode_changed)

        # Wire narration save
        def _on_narration_changed():
            if self.is_loading:
                return
            opening.narration = narration_edit.toPlainText()
            if stage_set.opening is None:
                stage_set.opening = opening
            if stage_path and os.path.exists(stage_path):
                save_set(stage_path, stage_set)

        narration_edit.textChanged.connect(_on_narration_changed)

        self.properties_layout.addWidget(opening_group)

    def _build_beat_list_widget(self, opening, stage_set, stage_path):
        """Build the beat list editor for live mode."""
        from noodlestudio.core.set_dressing import (
            OpeningBeat, save_set,
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # Discover noodling instance IDs from stage
        instance_ids = []
        if stage_path:
            instances_dir = os.path.join(stage_path, 'Instances')
            if os.path.isdir(instances_dir):
                for name in sorted(os.listdir(instances_dir)):
                    if os.path.isdir(os.path.join(instances_dir, name)):
                        instance_ids.append(name)

        self._opening_beat_rows = []

        def _rebuild_beats():
            """Refresh the beat list from opening.beats."""
            # Clear existing rows
            for row_widget in self._opening_beat_rows:
                row_widget.setParent(None)
            self._opening_beat_rows = []

            for i, beat in enumerate(opening.beats):
                row = self._build_beat_row(
                    i, beat, opening, instance_ids, stage_set, stage_path,
                    _rebuild_beats)
                layout.insertWidget(layout.count() - 1, row)
                self._opening_beat_rows.append(row)

        # Add beat button
        add_btn = QPushButton("+ Add Beat")
        add_btn.setStyleSheet(
            "QPushButton { background-color: #2A2A2A; color: #888; "
            "border: 1px solid #3A3A3A; padding: 4px 8px; } "
            "QPushButton:hover { color: #D2D2D2; }")
        add_btn.setFixedHeight(26)

        def _add_beat():
            from noodlestudio.core.set_dressing import save_set
            new_beat = OpeningBeat(
                beat_type='cue',
                noodling=instance_ids[0] if instance_ids else '',
                cue='',
            )
            opening.beats.append(new_beat)
            if stage_set.opening is None:
                stage_set.opening = opening
            if stage_path and os.path.exists(stage_path):
                save_set(stage_path, stage_set)
            _rebuild_beats()

        add_btn.clicked.connect(_add_beat)
        layout.addWidget(add_btn)

        # Build initial rows
        _rebuild_beats()

        return container

    def _build_beat_row(self, index, beat, opening, instance_ids,
                        stage_set, stage_path, rebuild_callback):
        """Build a single beat row in the beat list editor."""
        from noodlestudio.core.set_dressing import OpeningBeat, save_set

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        # Beat type selector (noodling dropdown or special types)
        type_combo = QComboBox()
        type_combo.setFixedWidth(120)
        type_combo.setStyleSheet(
            "QComboBox { background-color: #1E1E1E; color: #D2D2D2; "
            "padding: 2px; border: 1px solid #3A3A3A; font-size: 11px; }")

        # Populate: noodling IDs + special types
        for nid in instance_ids:
            type_combo.addItem(nid, nid)
        type_combo.addItem("-- pause --", "__pause__")
        type_combo.addItem("-- narration --", "__narration__")

        # Set current selection
        if beat.beat_type == 'cue':
            idx = type_combo.findData(beat.noodling)
            if idx >= 0:
                type_combo.setCurrentIndex(idx)
        elif beat.beat_type == 'pause':
            type_combo.setCurrentIndex(type_combo.findData("__pause__"))
        elif beat.beat_type == 'narration':
            type_combo.setCurrentIndex(type_combo.findData("__narration__"))

        row_layout.addWidget(type_combo)

        # Cue/text editor
        text_edit = QPlainTextEdit()
        text_edit.setStyleSheet(
            "QPlainTextEdit { background-color: #1E1E1E; color: #D2D2D2; "
            "padding: 2px; border: 1px solid #3A3A3A; font-size: 11px; }")
        text_edit.setMaximumHeight(50)
        text_edit.setTabChangesFocus(True)

        if beat.beat_type == 'cue':
            text_edit.setPlainText(beat.cue)
        elif beat.beat_type == 'narration':
            text_edit.setPlainText(beat.text)
        elif beat.beat_type == 'pause':
            text_edit.setPlainText(str(beat.duration))

        row_layout.addWidget(text_edit, stretch=1)

        # Move up / move down / delete buttons
        up_btn = QPushButton("^")
        up_btn.setFixedSize(22, 22)
        up_btn.setStyleSheet(
            "QPushButton { background: #2A2A2A; color: #888; border: none; } "
            "QPushButton:hover { color: #D2D2D2; }")
        down_btn = QPushButton("v")
        down_btn.setFixedSize(22, 22)
        down_btn.setStyleSheet(
            "QPushButton { background: #2A2A2A; color: #888; border: none; } "
            "QPushButton:hover { color: #D2D2D2; }")
        del_btn = QPushButton("x")
        del_btn.setFixedSize(22, 22)
        del_btn.setStyleSheet(
            "QPushButton { background: #2A2A2A; color: #888; border: none; } "
            "QPushButton:hover { color: #CC6666; }")

        row_layout.addWidget(up_btn)
        row_layout.addWidget(down_btn)
        row_layout.addWidget(del_btn)

        def _save():
            if stage_set.opening is None:
                stage_set.opening = opening
            if stage_path and os.path.exists(stage_path):
                save_set(stage_path, stage_set)

        def _on_type_changed(combo_index):
            if self.is_loading:
                return
            data = type_combo.currentData()
            if data == '__pause__':
                beat.beat_type = 'pause'
                beat.noodling = ''
                beat.cue = ''
                beat.text = ''
                beat.duration = 1.0
                text_edit.setPlainText('1.0')
            elif data == '__narration__':
                beat.beat_type = 'narration'
                beat.noodling = ''
                beat.cue = ''
                beat.duration = 1.0
                text_edit.setPlainText(beat.text)
            else:
                beat.beat_type = 'cue'
                beat.noodling = data
                beat.text = ''
                beat.duration = 1.0
            _save()

        def _on_text_changed():
            if self.is_loading:
                return
            txt = text_edit.toPlainText()
            if beat.beat_type == 'cue':
                beat.cue = txt
            elif beat.beat_type == 'narration':
                beat.text = txt
            elif beat.beat_type == 'pause':
                try:
                    beat.duration = float(txt)
                except ValueError:
                    pass
            _save()

        def _move_up():
            if index > 0:
                opening.beats[index], opening.beats[index - 1] = (
                    opening.beats[index - 1], opening.beats[index])
                _save()
                rebuild_callback()

        def _move_down():
            if index < len(opening.beats) - 1:
                opening.beats[index], opening.beats[index + 1] = (
                    opening.beats[index + 1], opening.beats[index])
                _save()
                rebuild_callback()

        def _delete():
            opening.beats.pop(index)
            _save()
            rebuild_callback()

        type_combo.currentIndexChanged.connect(_on_type_changed)
        text_edit.textChanged.connect(_on_text_changed)
        up_btn.clicked.connect(_move_up)
        down_btn.clicked.connect(_move_down)
        del_btn.clicked.connect(_delete)

        return row
