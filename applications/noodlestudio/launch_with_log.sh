#!/bin/bash
# Launch NoodleStudio with output captured to timestamped log file

clear
cd /Users/thistlequell/git/noodlings_clean/applications/noodlestudio

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
/Users/thistlequell/git/noodlings_clean/venv/bin/python3 -m noodlestudio.main 2>&1 | tee "$LOGFILE"
