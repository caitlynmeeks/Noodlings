#!/bin/bash

# noodleMUSH Startup Script
# Starts both WebSocket server and HTTP server for web client

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Starting noodleMUSH from $SCRIPT_DIR..."

# Use local noodlings venv python
PYTHON="$SCRIPT_DIR/../../venv/bin/python3"

# Check for PROJECT_PATH (new project system)
if [ -n "$PROJECT_PATH" ]; then
    echo "Using project: $PROJECT_PATH"
else
    # Legacy mode - check if world is initialized
    if [ ! -f "world/rooms.json" ]; then
        echo "Initializing legacy world..."
        $PYTHON init_world.py
    fi
fi

# Start HTTP server for web client in background (bind to all interfaces for network access)
echo "Starting HTTP server on port 8080 (accessible via network)..."
cd web
$PYTHON -m http.server 8080 --bind 0.0.0.0 &
HTTP_PID=$!
cd ..

# Start WebSocket server (with logging)
echo "Starting WebSocket server on port 8765..."
LOG_FILE="logs/server_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
$PYTHON server.py 2>&1 | tee "$LOG_FILE"

# Cleanup on exit
trap "kill $HTTP_PID 2>/dev/null" EXIT
