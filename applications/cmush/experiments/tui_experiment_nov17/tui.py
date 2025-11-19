#!/usr/bin/env python3
"""
noodleMUSH TUI - Terminal User Interface
Beautiful WYSE amber terminal interface for consciousness architecture
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, TabbedContent, TabPane, RichLog, Input, Static, Button, Select, DataTable
from textual.binding import Binding
from rich.text import Text
import asyncio
import websockets
import json


class NoodleMUSHTUI(App):
    """A Textual app for noodleMUSH with WYSE amber aesthetic."""

    CSS = """
    /* WYSE Amber Theme */
    Screen {
        background: #000000;
    }

    Header {
        background: #000000;
        color: #ffd700;
        text-style: bold;
    }

    Footer {
        background: #000000;
        color: #cc8800;
    }

    TabbedContent {
        background: #000000;
        border: solid #ffb000;
    }

    TabPane {
        background: #000000;
        color: #ffb000;
    }

    RichLog {
        background: #000000;
        color: #ffb000;
        border: solid #ffb000;
    }

    Input {
        background: #1a0f00;
        color: #ffd700;
        border: solid #ffb000;
    }

    Input:focus {
        border: solid #ffd700;
        background: #2a1800;
    }

    Static {
        background: #000000;
        color: #ffb000;
    }

    .bright {
        color: #ffd700;
        text-style: bold;
    }

    .dim {
        color: #cc8800;
    }

    Button {
        background: #000000;
        color: #ffb000;
        border: solid #ffb000;
    }

    Button:hover {
        background: #2a1800;
        color: #ffd700;
        border: solid #ffd700;
    }

    DataTable {
        background: #000000;
        color: #ffb000;
    }

    DataTable > .datatable--cursor {
        background: #ffd700;
        color: #000000;
    }

    Select {
        background: #000000;
        color: #ffb000;
        border: solid #ffb000;
    }

    #config-panel {
        border: solid #ffd700;
        padding: 1;
    }

    .config-header {
        text-style: bold;
        color: #ffd700;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+t", "toggle_telepathy", "Toggle Telepathy"),
        ("tab", "focus_next", "Next"),
    ]

    TITLE = "noodleMUSH - Noodlings Multi-User Shared Hallucination"

    def __init__(self):
        super().__init__()
        self.ws = None
        self.authenticated = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with TabbedContent(initial="chat"):
            # Chat Tab
            with TabPane("Chat", id="chat"):
                yield RichLog(id="chat-log", highlight=True, markup=True)
                with Horizontal(id="input-area"):
                    yield Static(">", classes="bright")
                    yield Input(placeholder="Type a command...", id="chat-input")

            # Log Tab
            with TabPane("Log", id="log"):
                yield RichLog(id="log-output", highlight=True, markup=True)

            # Config Tab
            with TabPane("Config", id="config"):
                yield self.create_config_panel()

        yield Footer()

    def create_config_panel(self) -> Container:
        """Create the BIOS-style configuration panel."""
        container = Vertical(id="config-panel")

        with container:
            yield Static("⚙️  LLM CONFIGURATION ⚙️", classes="config-header")
            yield Static("Use arrow keys to navigate", classes="dim")
            yield Static("")

            # Global Config Section
            yield Static("═══ GLOBAL CONFIGURATION ═══", classes="bright")

            table = DataTable(id="config-table")
            table.add_column("Setting", width=20)
            table.add_column("Value", width=60)
            table.add_row("Provider", "local")
            table.add_row("API Base", "http://localhost:1234/v1")
            table.add_row("Model", "qwen/qwen3-4b-2507")
            table.add_row("Timeout", "60s")

            yield table

            yield Static("")
            yield Static("● System Status: [bold #ffd700]OPERATIONAL[/]", classes="dim")
            yield Static("Multi-model consciousness routing active", classes="dim")

        return container

    def on_mount(self) -> None:
        """Called when app starts."""
        self.chat_log = self.query_one("#chat-log", RichLog)
        self.chat_input = self.query_one("#chat-input", Input)
        self.log_output = self.query_one("#log-output", RichLog)

        # Connect to WebSocket server
        self.run_worker(self.connect_websocket())

        self.chat_log.write("Welcome to noodleMUSH!")
        self.chat_log.write("Connecting to server...")

    async def connect_websocket(self):
        """Connect to the noodleMUSH WebSocket server."""
        try:
            self.ws = await websockets.connect("ws://localhost:8765")
            self.chat_log.write("[bold #00ff00]✓[/] Connected to server!")

            # Auto-login if credentials exist
            # TODO: Load from cookies/config

            # Listen for messages
            async for message in self.ws:
                await self.handle_server_message(message)

        except Exception as e:
            self.chat_log.write(f"[bold #ff0000]✗[/] Connection failed: {e}")

    async def handle_server_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "message":
                # Agent speech or thought
                sender = data.get("sender", "Unknown")
                content = data.get("content", "")
                is_thought = data.get("is_thought", False)

                if is_thought:
                    self.chat_log.write(f"[dim]🧠 {sender} privately thinks: {content}[/]")
                else:
                    self.chat_log.write(f"[bold]{sender}[/]: {content}")

            elif msg_type == "log":
                # Log entry
                level = data.get("level", "INFO")
                log_message = data.get("message", "")
                self.log_output.write(f"[{level}] {log_message}")

            elif msg_type == "system":
                # System message
                self.chat_log.write(f"[dim italic]{data.get('content', '')}[/]")

        except json.JSONDecodeError:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chat input submission."""
        if event.input.id == "chat-input":
            message = event.value.strip()
            if not message:
                return

            # Clear input
            event.input.value = ""

            # Send to server
            if self.ws:
                await self.ws.send(json.dumps({
                    "type": "command",
                    "content": message
                }))

            # Echo locally
            self.chat_log.write(f"[dim]> {message}[/]")

    def action_toggle_telepathy(self) -> None:
        """Toggle telepathy mode (reading others' thoughts)."""
        self.chat_log.write("[italic]Telepathy toggled[/]")


def main():
    """Run the noodleMUSH TUI."""
    app = NoodleMUSHTUI()
    app.run()


if __name__ == "__main__":
    main()
