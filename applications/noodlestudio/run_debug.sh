#!/bin/bash
# Debug launcher for NoodleStudio - shows console output

cd "$(dirname "$0")"

echo "Activating venv..."
source venv/bin/activate

echo "Launching NoodleStudio with debug output..."
echo "Output will be saved to debug.log"
echo ""

python3 run_studio.py 2>&1 | tee debug.log
