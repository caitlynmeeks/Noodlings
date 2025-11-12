"""
Agents package for multi-agent hierarchical collapse experiments.

Contains:
- NoodlingAgent: Multi-timescale agent with fast/medium/slow layers
- ObserverAgent: Meta-cognitive observer that prevents collapse
"""

from .noodling_agent import NoodlingAgent
from .observer_agent import ObserverAgent

__all__ = ['NoodlingAgent', 'ObserverAgent']
