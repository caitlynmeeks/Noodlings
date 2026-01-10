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
#   Component System Tests
#
#   Tests for NoodleStudio's component architecture: - Compon...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_component_system
# PURPOSE:  Component System Tests
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestComponentBase, TestComponentRegistry, TestComponentCollection, TestArtbookComponent
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


# =============================================================================
# ComponentBase Tests
# =============================================================================

class TestComponentBase:
    """Test ComponentBase abstract class."""

    def test_cannot_instantiate_abstract(self):
        """ComponentBase cannot be instantiated directly."""
        from noodlestudio.core.component_base import ComponentBase

        with pytest.raises(TypeError):
            ComponentBase()

    def test_component_has_unique_id(self):
        """Each component instance has unique ID."""
        from noodlestudio.core.components import ArtbookComponent

        c1 = ArtbookComponent(entity_id="test")
        c2 = ArtbookComponent(entity_id="test")

        assert c1.id != c2.id
        assert len(c1.id) == 36  # UUID format

    def test_component_tracks_entity_id(self):
        """Component stores entity ID."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent(entity_id="red_fire_anklebiter")
        assert comp.entity_id == "red_fire_anklebiter"

        comp.entity_id = "blue_fire_anklebiter"
        assert comp.entity_id == "blue_fire_anklebiter"

    def test_component_enabled_flag(self):
        """Component can be enabled/disabled."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        assert comp.enabled is True

        comp.enabled = False
        assert comp.enabled is False
        assert comp.is_dirty is True

    def test_component_dirty_tracking(self):
        """Component tracks dirty state."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        comp.clear_dirty()
        assert comp.is_dirty is False

        comp.enabled = False
        assert comp.is_dirty is True

        comp.clear_dirty()
        assert comp.is_dirty is False


# =============================================================================
# ComponentRegistry Tests
# =============================================================================

class TestComponentRegistry:
    """Test ComponentRegistry singleton."""

    def test_registry_is_singleton(self):
        """Registry is a singleton."""
        from noodlestudio.core.component_base import ComponentRegistry

        r1 = ComponentRegistry()
        r2 = ComponentRegistry()
        assert r1 is r2

    def test_artbook_is_registered(self):
        """ArtbookComponent is auto-registered via decorator."""
        from noodlestudio.core.component_base import component_registry
        from noodlestudio.core.components import ArtbookComponent  # triggers registration

        assert "artbook" in component_registry.get_all_types()

    def test_registry_create(self):
        """Registry can create component instances."""
        from noodlestudio.core.component_base import component_registry
        from noodlestudio.core.components import ArtbookComponent

        comp = component_registry.create("artbook", entity_id="test_entity")

        assert comp is not None
        assert isinstance(comp, ArtbookComponent)
        assert comp.entity_id == "test_entity"

    def test_registry_unknown_type_returns_none(self):
        """Creating unknown type returns None."""
        from noodlestudio.core.component_base import component_registry

        comp = component_registry.create("nonexistent_component")
        assert comp is None

    def test_registry_get_display_info(self):
        """Registry provides display info for Inspector."""
        from noodlestudio.core.component_base import component_registry
        from noodlestudio.core.components import ArtbookComponent

        info = component_registry.get_display_info("artbook")

        assert info['type'] == 'artbook'
        assert info['display_name'] == 'Artbook'
        assert info['category'] == 'art'
        assert '#' in info['border_color']  # Has valid color


# =============================================================================
# ComponentCollection Tests
# =============================================================================

class TestComponentCollection:
    """Test ComponentCollection entity management."""

    def test_empty_collection(self):
        """New collection is empty."""
        from noodlestudio.core.component_collection import ComponentCollection

        coll = ComponentCollection(entity_id="test")

        assert len(coll) == 0
        assert "artbook" not in coll
        assert coll.get("artbook") is None

    def test_add_component_by_type(self):
        """Can add component by type name."""
        from noodlestudio.core.component_collection import ComponentCollection
        from noodlestudio.core.components import ArtbookComponent

        coll = ComponentCollection(entity_id="red")
        comp = coll.add("artbook")

        assert comp is not None
        assert isinstance(comp, ArtbookComponent)
        assert "artbook" in coll
        assert len(coll) == 1
        assert comp.entity_id == "red"

    def test_singleton_constraint(self):
        """Singleton components can only be added once."""
        from noodlestudio.core.component_collection import ComponentCollection
        from noodlestudio.core.components import ArtbookComponent

        coll = ComponentCollection(entity_id="test")
        comp1 = coll.add("artbook")
        comp2 = coll.add("artbook")

        assert comp1 is comp2  # Returns existing instance
        assert len(coll) == 1

    def test_remove_component(self):
        """Can remove components."""
        from noodlestudio.core.component_collection import ComponentCollection

        coll = ComponentCollection(entity_id="test")
        coll.add("artbook")
        assert len(coll) == 1

        result = coll.remove("artbook")
        assert result is True
        assert len(coll) == 0

    def test_remove_nonexistent_returns_false(self):
        """Removing nonexistent component returns False."""
        from noodlestudio.core.component_collection import ComponentCollection

        coll = ComponentCollection(entity_id="test")
        result = coll.remove("artbook")
        assert result is False

    def test_iteration(self):
        """Can iterate over components."""
        from noodlestudio.core.component_collection import ComponentCollection

        coll = ComponentCollection(entity_id="test")
        coll.add("artbook")

        components = list(coll)
        assert len(components) == 1
        assert components[0].component_type == "artbook"

    def test_serialization_roundtrip(self):
        """Collection can be serialized and deserialized."""
        from noodlestudio.core.component_collection import ComponentCollection

        # Create and populate
        coll1 = ComponentCollection(entity_id="test")
        artbook = coll1.add("artbook")
        artbook.add_art("/path/to/concept.png")
        artbook.thumbnail_size = 100

        # Serialize
        data = coll1.to_dict()

        # Deserialize
        coll2 = ComponentCollection(entity_id="test")
        count = coll2.from_dict(data)

        assert count == 1
        assert "artbook" in coll2

        loaded = coll2.get("artbook")
        assert loaded.art_count == 1
        assert loaded.thumbnail_size == 100


# =============================================================================
# ArtbookComponent Tests
# =============================================================================

class TestArtbookComponent:
    """Test ArtbookComponent concrete implementation."""

    def test_component_metadata(self):
        """Artbook has correct metadata."""
        from noodlestudio.core.components import ArtbookComponent
        from noodlestudio.core.component_base import ComponentCategory

        comp = ArtbookComponent()

        assert comp.component_type == "artbook"
        assert comp.display_name == "Artbook"
        assert comp.category == ComponentCategory.ART_REFERENCE
        assert comp.singleton is True

    def test_add_art(self):
        """Can add art files."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        comp.clear_dirty()

        result = comp.add_art("/path/to/image.png", note="Main concept")

        assert result is True
        assert comp.art_count == 1
        assert "/path/to/image.png" in comp.art_files
        assert comp.is_dirty is True

    def test_add_duplicate_art(self):
        """Adding duplicate art returns False."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        comp.add_art("/path/to/image.png")
        result = comp.add_art("/path/to/image.png")

        assert result is False
        assert comp.art_count == 1

    def test_remove_art(self):
        """Can remove art files."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        comp.add_art("/path/to/image.png")
        assert comp.art_count == 1

        result = comp.remove_art("/path/to/image.png")

        assert result is True
        assert comp.art_count == 0

    def test_art_notes(self):
        """Can get/set notes for art files."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        comp.add_art("/path/to/image.png", note="Original concept")

        assert comp.get_note("/path/to/image.png") == "Original concept"

        comp.set_note("/path/to/image.png", "Updated concept")
        assert comp.get_note("/path/to/image.png") == "Updated concept"

    def test_reorder_art(self):
        """Can reorder art files."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent()
        comp.add_art("/path/a.png")
        comp.add_art("/path/b.png")
        comp.add_art("/path/c.png")

        comp.reorder_art(2, 0)  # Move c.png to front

        files = comp.art_files
        assert files[0].endswith("c.png")

    def test_serialization(self):
        """Artbook serializes correctly."""
        from noodlestudio.core.components import ArtbookComponent

        comp = ArtbookComponent(entity_id="test")
        comp.add_art("/path/to/concept.png", note="Main character")
        comp.thumbnail_size = 120

        data = comp.to_dict()

        assert data['type'] == 'artbook'
        assert data['art_files'] == ['/path/to/concept.png']
        assert data['art_notes'] == {'/path/to/concept.png': 'Main character'}
        assert data['thumbnail_size'] == 120

    def test_deserialization(self):
        """Artbook deserializes correctly."""
        from noodlestudio.core.components import ArtbookComponent

        data = {
            'type': 'artbook',
            'id': 'test-id-123',
            'enabled': True,
            'art_files': ['/path/to/concept.png'],
            'art_notes': {'/path/to/concept.png': 'Main character'},
            'thumbnail_size': 120,
            'columns': 5,
        }

        comp = ArtbookComponent.from_dict(data, entity_id="loaded_entity")

        assert comp.entity_id == "loaded_entity"
        assert comp.art_count == 1
        assert comp.thumbnail_size == 120
        assert comp.columns == 5
        assert comp.get_note('/path/to/concept.png') == 'Main character'


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
