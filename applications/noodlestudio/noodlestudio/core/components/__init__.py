"""
NoodleStudio Components Package

All component types are registered automatically when imported.

Built-in components:
- ArtbookComponent: Reference art collection
- (more to come: VoiceReferenceComponent, MoodBoardComponent, etc.)

Usage:
    from noodlestudio.core.components import ArtbookComponent
    from noodlestudio.core.component_base import component_registry

    # Get registered component types
    types = component_registry.get_all_types()

    # Create component
    artbook = component_registry.create("artbook", entity_id="red")

Author: Caitlyn + Claude
Date: January 2026
"""

# Import all components to trigger registration
from .artbook_component import ArtbookComponent

# Re-export
__all__ = [
    'ArtbookComponent',
]
