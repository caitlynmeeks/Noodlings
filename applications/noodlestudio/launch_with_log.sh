#!/bin/bash
# Launch NoodleStudio with output captured to timestamped log file

clear

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/noodlestudio_${TIMESTAMP}.log"

echo "Launching NoodleStudio..."
echo "Output will be saved to: $LOGFILE"
echo "Press Ctrl+C to stop"
echo ""

# Launch with venv python and output to both terminal and log file
"$PROJECT_ROOT/venv/bin/python3" -m noodlestudio.main 2>&1 | tee "$LOGFILE"
