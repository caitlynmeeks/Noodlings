#!/bin/bash

# noodleMUSH Terminal UI Startup Script
# Starts WebSocket server, HTTP server, and Terminal Bridge

echo "Starting noodleMUSH Terminal UI..."

PYTHON=/Users/thistlequell/git/consilience/venv/bin/python3

# Start HTTP server for web client in background
echo "Starting HTTP server on port 8080..."
cd web
$PYTHON -m http.server 8080 &
HTTP_PID=$!
cd ..

# Start WebSocket server (main noodleMUSH server) in background
echo "Starting WebSocket server on port 8765..."
$PYTHON server.py &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Start Terminal Bridge (xterm.js ↔ Textual TUI)
echo "Starting Terminal Bridge on port 8766..."
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  noodleMUSH Terminal UI is ready!"
echo "  Open: http://localhost:8080/terminal.html"
echo "═══════════════════════════════════════════════════════════"
echo ""

$PYTHON terminal_bridge.py

# Cleanup on exit
trap "kill $HTTP_PID $SERVER_PID 2>/dev/null" EXIT
