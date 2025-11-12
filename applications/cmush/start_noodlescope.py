#!/usr/bin/env python3
"""
Start NoodleScope in live mode (no test data).
"""
from noodlescope import NoodleScope

if __name__ == '__main__':
    scope = NoodleScope()
    print("🧠 NoodleScope ready for live data...")
    scope.run(debug=False, port=8050)
