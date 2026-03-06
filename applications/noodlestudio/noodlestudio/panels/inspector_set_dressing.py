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
    QLabel, QTextEdit, QPushButton, QCheckBox, QWidget,
    QVBoxLayout, QHBoxLayout,
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
