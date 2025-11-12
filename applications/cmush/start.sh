#!/bin/bash

# noodleMUSH Startup Script
# Starts both WebSocket server and HTTP server for web client

echo "Starting noodleMUSH..."

# Use consilience venv python
PYTHON=/Users/thistlequell/git/consilience/venv/bin/python3

# Check if world is initialized
if [ ! -f "world/rooms.json" ]; then
    echo "Initializing world..."
    $PYTHON init_world.py
fi

# Start HTTP server for web client in background
echo "Starting HTTP server on port 8080..."
cd web
$PYTHON -m http.server 8080 &
HTTP_PID=$!
cd ..

# Start WebSocket server
echo "Starting WebSocket server on port 8765..."
$PYTHON server.py

# Cleanup on exit
trap "kill $HTTP_PID 2>/dev/null" EXIT
