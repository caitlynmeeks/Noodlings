"""
Ablation Study Architecture Variants

This module defines 6 architecture variants for scientific comparison:
1. Baseline: No temporal model (returns zeros)
2. Control: Random states (to prove structure matters)
3. SingleLayer: One LSTM (no hierarchy)
4. Hierarchical: Fast + Medium + Slow (no observers)
5. Phase4: Full architecture with 75 observer loops
6. DenseObservers: 150 observer loops (2x density)

All architectures share the same interface for fair comparison.
"""

from .base import AblationArchitecture
from .baseline import BaselineArchitecture
from .control import ControlArchitecture
from .single_layer import SingleLayerArchitecture
from .hierarchical import HierarchicalArchitecture
from .phase4_observers import Phase4Architecture
from .dense_observers import DenseObserversArchitecture

__all__ = [
    'AblationArchitecture',
    'BaselineArchitecture',
    'ControlArchitecture',
    'SingleLayerArchitecture',
    'HierarchicalArchitecture',
    'Phase4Architecture',
    'DenseObserversArchitecture'
]
