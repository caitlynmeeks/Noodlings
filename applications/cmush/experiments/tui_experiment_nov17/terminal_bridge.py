#!/usr/bin/env python3
"""
Terminal Bridge - Serves Textual TUI to browser via xterm.js
Runs the TUI in a PTY and forwards I/O to browser WebSocket
"""

import asyncio
import websockets
import json
import pty
import os
import subprocess
import select
import sys


class TerminalBridge:
    def __init__(self, port=8766):
        self.port = port
        self.clients = set()
        self.master_fd = None
        self.slave_fd = None
        self.tui_process = None

    async def handle_client(self, websocket):
        """Handle a WebSocket client connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")

        try:
            # If this is the first client, start the TUI
            if len(self.clients) == 1:
                await self.start_tui()

            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'input':
                    # Forward keyboard input to PTY
                    input_data = data.get('data', '')
                    if self.master_fd:
                        os.write(self.master_fd, input_data.encode('utf-8'))

                elif msg_type == 'resize':
                    # Resize PTY
                    cols = data.get('cols', 80)
                    rows = data.get('rows', 24)
                    if self.master_fd:
                        import fcntl
                        import termios
                        import struct
                        size = struct.pack("HHHH", rows, cols, 0, 0)
                        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, size)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")

            # If no more clients, stop the TUI
            if len(self.clients) == 0:
                self.stop_tui()

    async def start_tui(self):
        """Start the Textual TUI in a PTY."""
        print("Starting TUI...")

        # Create PTY
        self.master_fd, self.slave_fd = pty.openpty()

        # Set terminal size
        import fcntl
        import termios
        import struct
        size = struct.pack("HHHH", 24, 80, 0, 0)
        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, size)

        # Set PTY to non-blocking
        import fcntl
        flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Start TUI process
        env = os.environ.copy()
        env['TERM'] = 'xterm-256color'

        self.tui_process = subprocess.Popen(
            [sys.executable, 'tui.py'],
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            env=env,
            close_fds=True
        )

        # Start reading from PTY
        asyncio.create_task(self.read_pty())

    def stop_tui(self):
        """Stop the TUI process."""
        print("Stopping TUI...")
        if self.tui_process:
            self.tui_process.terminate()
            self.tui_process.wait(timeout=5)
            self.tui_process = None

        if self.master_fd:
            os.close(self.master_fd)
            self.master_fd = None

        if self.slave_fd:
            os.close(self.slave_fd)
            self.slave_fd = None

    async def read_pty(self):
        """Read output from PTY and send to all WebSocket clients."""
        while self.master_fd and self.tui_process and self.tui_process.poll() is None:
            try:
                # Check if data is available
                readable, _, _ = select.select([self.master_fd], [], [], 0.1)

                if readable:
                    data = os.read(self.master_fd, 1024)
                    if data:
                        # Send to all connected clients
                        decoded = data.decode('utf-8', errors='ignore')
                        disconnected = set()

                        for client in self.clients:
                            try:
                                await client.send(decoded)
                            except:
                                disconnected.add(client)

                        # Remove disconnected clients
                        self.clients -= disconnected

                await asyncio.sleep(0.01)

            except OSError:
                break

        print("PTY read loop ended")

    async def start_server(self):
        """Start the WebSocket server."""
        print(f"Starting terminal bridge on port {self.port}...")
        server = await websockets.serve(self.handle_client, "localhost", self.port)
        print(f"Terminal bridge running on ws://localhost:{self.port}")
        print(f"Open http://localhost:8080/terminal.html in your browser")
        await asyncio.Future()  # Run forever


async def main():
    bridge = TerminalBridge(port=8766)
    await bridge.start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down terminal bridge...")
