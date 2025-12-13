"""
Noodlings Scripting System - Unity-like API in Python

Instead of C#, we use Python with Unity-style API.
Same workflow, easier integration!

Example:
```python
class ClickableBox(NoodleScript):
    def OnClick(self, clicker):
        # Spawn an Anklebiter!
        Noodlings.Spawn("anklebiter.noo", self.transform.position)
        Debug.Log("Oops! You released an Anklebiter!")
```

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from .noodle_script import NoodleScript
from .noodlings_api import Noodlings, Debug, Transform, Prim
from .script_executor import ScriptExecutor
from .noodle_api import NoodleAPI, get_noodle_api
from .models_api import ModelsAPI
from .neural_api import NeuralAPI, NeuralNetworkProxy
from .agents_api import AgentsAPI, FacetAssemblyProxy, FacetProxy

__all__ = [
    'NoodleScript', 'Noodlings', 'Debug', 'Transform', 'Prim', 'ScriptExecutor',
    'NoodleAPI', 'get_noodle_api', 'ModelsAPI', 'NeuralAPI', 'AgentsAPI',
    'NeuralNetworkProxy', 'FacetAssemblyProxy', 'FacetProxy'
]
