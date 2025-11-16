#!/usr/bin/env python3
"""
Convenience launcher for NoodleSTUDIO.

Usage:
    python run_studio.py

Or make executable and run directly:
    chmod +x run_studio.py
    ./run_studio.py
"""

import sys
from pathlib import Path

# Add noodlestudio to path
sys.path.insert(0, str(Path(__file__).parent))

from noodlestudio.main import main

if __name__ == '__main__':
    main()
