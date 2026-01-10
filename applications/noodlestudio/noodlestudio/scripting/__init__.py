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
#   Noodlings Scripting System - Unity-like API in Python
#
#   Instead of C#, we use Python with Unity-style API. Same w...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.__init__
# PURPOSE:  Module exports
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ClickableBox
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
