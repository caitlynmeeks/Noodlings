"""
Environment package for multi-agent experiments.

Contains:
- ResourceAllocationGame: Multi-timescale decision environment
"""

from .resource_game import ResourceAllocationGame, ResourceState

__all__ = ['ResourceAllocationGame', 'ResourceState']
