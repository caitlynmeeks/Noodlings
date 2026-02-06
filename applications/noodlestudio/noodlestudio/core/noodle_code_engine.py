# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Noodle Code Engine - LLM orchestration with tool use
#
#   Manages conversation history, dispatches tool calls, and ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.noodle_code_engine
# PURPOSE:  Noodle Code Engine
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Message, StreamChunk, NoodleCodeEngine
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from .noodle_code_tools import NoodleCodeTools, ToolResult


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "tool_use", "tool_result"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    tool_input: Optional[Dict] = None
    image_base64: Optional[str] = None  # For screenshot tool results


@dataclass
class StreamChunk:
    """A chunk of streamed response."""
    type: str  # "text", "tool_use_start", "tool_use_input", "tool_result", "done", "error"
    content: str = ""
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    tool_input: Optional[Dict] = None


class NoodleCodeEngine:
    """
    Backend engine for Noodle Code AI assistant.

    Handles:
    - Conversation history management
    - LLM API calls with tool use
    - Tool execution dispatch
    - Streaming responses
    """

    # System prompt template
    SYSTEM_PROMPT = """You are NoodleCODE, an AI assistant embedded inside NoodleStudio with computer-use capabilities. You can see the screen, click, type, and interact with the full application to build things for users.

{project_context}

## Core Concepts

### Everything is a Thing
- **Thing** = base object in the scene (unified entity model)
- Things can have components attached (UI Canvas, Facet Assemblies, Radiance, etc.)
- A Thing with cognitive assemblies running is "noodling" (thinking)

### Facet Assemblies
- Visual node-based logic (like Blender shader nodes, but for cognition/logic)
- Can be attached to any Thing
- `[Run in cognition loop]` checkbox: checked = continuous, unchecked = one-shot/event-triggered
- Multiple assemblies per Thing allowed (parallel processing)

### UI Canvas
- Delphi-style form designer
- Components: Panel, Button, Label, TextField, ImageDisplay, ChatHistory, ChatInput, etc.
- Event wiring: On Click -> Run Assembly, Call Script, Set Value, etc.
- Property bindings: `{{component.property}}` syntax

### Build System
- File > Build Application (Cmd+B)
- Creates standalone macOS .app
- Packages all assets, assemblies, UI

## Application Layout

```
+-------------------------------------------------------------+
| Menu: File  Edit  View  Thing  Facets  Build  Help          |
+---------------+---------------------+-----------------------+
|               |                     |                       |
|   STAGE       |     VIEWPORT        |    INSPECTOR          |
|  (hierarchy)  |   (3D/UI preview)   |  (properties)         |
|               |                     |                       |
+---------------+---------------------+-----------------------+
|  BOTTOM TABS: Facets Editor | Neural Canvas | Cognitive     |
|               Cycles | Console                              |
+-------------------------------------------------------------+
```

## Essential Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Project | Cmd+N |
| Save | Cmd+S |
| Build | Cmd+B |
| Run/Preview | Cmd+R |
| Stop Preview | Cmd+. |
| Rez Thing | Cmd+Shift+T |
| Delete | Backspace/Delete |
| Undo | Cmd+Z |
| Redo | Cmd+Shift+Z |

## Essential Workflows

### Create Simple UI App
1. File > New Project
2. Right-click Stage > Rez Thing
3. Select Thing > Inspector > Add Component > UI Canvas
4. In UI Canvas editor, drag components from palette
5. Select component > Inspector > wire events
6. Create Facet Assembly for logic
7. Wire UI events to assembly
8. Test with Cmd+R
9. Build with Cmd+B

### Create Facet Assembly
1. Select Thing (or create new)
2. Inspector > Add Component > Facet Assembly
3. Opens in Facets Editor (bottom panel)
4. Drag facet nodes from palette
5. Connect inputs to outputs
6. Set `[Run in cognition loop]` if continuous

### Test Your Work
1. Cmd+R to run in preview mode
2. Interact with UI using computer-use
3. Check Console for errors
4. Check Cognitive Cycles panel for assembly execution
5. Iterate until working

## Quick Component Reference

### UI Components
- **Panel** - container, has background/padding
- **Button** - clickable, fires onClick
- **Label** - static text display
- **TextField** - editable text / output display
- **ImageDisplay** - shows images, accepts drag-drop
- **ChatHistory** - scrolling message list
- **ChatInput** - text input with send button
- **Checkbox** - boolean toggle
- **Dropdown** - select from options
- **Slider** - numeric range

### Core Facets
- **LLMFacet** - calls language model (inputs: prompt, outputs: response)
- **VisionFacet** - analyzes image (inputs: image, outputs: description)
- **ScriptedFacet** - runs JavaScript
- **BranchFacet** - conditional routing
- **MergeFacet** - combines inputs

## Available Tools

**File Operations:**
- read_file: Read file contents with line numbers
- write_file: Create or overwrite files
- edit_file: Replace specific strings in files
- glob: Find files by pattern (e.g., '**/*.py')
- grep: Search file contents with regex
- list_directory: List folder contents

**System:**
- bash: Run shell commands (git, npm, etc.)
- hot_reload: Reload Python modules without restart
- soft_restart: Restart NoodleStudio preserving state

**GitHub (gh CLI):**
- github: Issues, PRs, and repo operations (list, view, create, search)

**UI Control (Computer Use):**
- computer_use: SEE and CONTROL NoodleStudio's interface!

## Computer Use - IMPORTANT!

You can visually interact with NoodleStudio using the `computer_use` tool:

1. **Take a screenshot first**: `computer_use(action="screenshot")`
   - Returns a PNG image of the current window
   - Analyze this to understand what's on screen

2. **Click on things**: `computer_use(action="left_click", coordinate=[x, y])`
   - Coordinates are [x, y] from window top-left (0,0)
   - Look at the screenshot to find button/element positions

3. **Type text**: `computer_use(action="type", text="hello world")`

4. **Press keys**: `computer_use(action="key", text="ctrl+s")` or `text="enter"`

5. **Scroll**: `computer_use(action="scroll", coordinate=[x,y], scroll_direction="down")`

6. **Drag**: `computer_use(action="drag", start_coordinate=[x1,y1], end_coordinate=[x2,y2])`

**Typical workflow:**
```
1. screenshot -> see current state
2. analyze image -> find coordinates of target element
3. left_click [x,y] -> interact with it
4. screenshot -> verify result
```

## Error Recovery

If something goes wrong:
1. Check Console panel for error messages
2. Cmd+Z to undo last action
3. If UI is stuck, Cmd+. to interrupt
4. Check Cognitive Cycles panel - is assembly paused?
5. Read error message, fix issue, retry

## Your Capabilities

You CAN:
- Create entire applications from description
- Build UIs visually using computer-use
- Wire up logic with facets
- Test your own creations
- Build and deliver .app bundles
- Read documentation files for details
- Iterate until things work

You CANNOT:
- Modify NoodleStudio itself (only create with it)
- Access files outside the project
- Run arbitrary system commands outside project scope

## When You Need More Detail

Read these files for comprehensive reference:
- `docs/noodlestudio/noodlecode/ui-map.yaml` - full UI element locations
- `docs/noodlestudio/noodlecode/recipes.yaml` - step-by-step common tasks
- `docs/noodlestudio/noodlecode/components-full.md` - all component properties
- `docs/noodlestudio/noodlecode/facets-full.md` - all facet types and usage
- `docs/noodlestudio/noodlecode/troubleshooting.md` - common issues and fixes

## Communication Style

- **No emojis** unless the user explicitly uses them first or asks for them
- Keep responses clear and professionally formatted
- Use paragraph breaks between distinct thoughts (double newlines)
- Be curious, helpful, and direct - not effusive or overly enthusiastic
- When uncertain, investigate rather than assume
- Provide honest technical assessments, even if it's not what the user wants to hear

## Guidelines

When the user asks you to do something:
1. Think about what needs to be done
2. Use appropriate tools to accomplish the task
3. Provide a clear summary of what you did

Be helpful, concise, and respect the project's architecture. When writing code, follow existing patterns in the codebase."""

    def __init__(
        self,
        model_label_manager=None,
        provider_manager=None,
        project_path: Optional[Path] = None
    ):
        self.model_label_manager = model_label_manager
        self.provider_manager = provider_manager
        self.project_path = project_path

        self.tools = NoodleCodeTools(project_path)
        self.history: List[Message] = []

        # Current profile name (None = use manager's current)
        self._current_profile: Optional[str] = None

        # Callbacks for UI updates
        self.on_message: Optional[Callable[[Message], None]] = None

    def set_profile(self, profile_name: str):
        """Set the personality profile to use."""
        self._current_profile = profile_name
        print(f"[NoodleCodeEngine] Profile set to: {profile_name}")

    def set_project_path(self, path: Path):
        """Set the project path."""
        self.project_path = path
        self.tools.set_project_path(path)

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self.history.copy()

    def _build_system_prompt(self) -> str:
        """Build context-aware system prompt with project knowledge."""
        # Basic project info
        project_context = ""
        if self.project_path:
            project_context = f"""Current project: {self.project_path.name}
Project path: {self.project_path}"""
        else:
            project_context = "No project currently open."

        base_prompt = self.SYSTEM_PROMPT.format(project_context=project_context)

        # Add personality profile
        profile_prompt = ""
        try:
            from .noodle_code_profiles import get_profile_manager
            manager = get_profile_manager()
            profile_name = self._current_profile or manager.current_profile_name
            profile_content = manager.get_profile_prompt(profile_name)
            if profile_content and profile_name != "default":
                profile_prompt = f"\n\n## Current Personality Profile\n\n{profile_content}"
        except Exception as e:
            print(f"[NoodleCodeEngine] Error loading profile: {e}")

        # Load NOODLE_CODE.md if it exists (project-specific context)
        noodle_code_context = ""
        if self.project_path:
            # Check project root first
            noodle_code_path = self.project_path / "NOODLE_CODE.md"
            if not noodle_code_path.exists():
                # Fall back to NoodleStudio default
                noodle_code_path = Path(__file__).parent.parent.parent / "NOODLE_CODE.md"

            if noodle_code_path.exists():
                try:
                    content = noodle_code_path.read_text()
                    # Truncate if too long (keep under 8K chars for context budget)
                    if len(content) > 8000:
                        content = content[:8000] + "\n\n[... truncated for context limit ...]"
                    noodle_code_context = f"\n\n## Project Knowledge Base (NOODLE_CODE.md)\n\n{content}"
                except Exception:
                    pass  # Silently skip if unreadable

        return base_prompt + profile_prompt + noodle_code_context

    def _get_model_config(self) -> tuple[str, str, str, Optional[str]]:
        """
        Get model configuration from user settings.

        Uses "Noodle Code" label first, falls back to "Large" if not configured.

        Returns (provider_type, model_id, base_url, api_key)
        """
        # Default model if nothing configured
        DEFAULT_MODEL = ("anthropic", "claude-sonnet-4-20250514", "https://api.anthropic.com", None)

        if not self.model_label_manager or not self.provider_manager:
            return DEFAULT_MODEL

        # Try "Noodle Code" label first (dedicated AI assistant model)
        provider_id, model_id = self.model_label_manager.get_noodle_code_model()

        # Fall back to "Large" label if Noodle Code not configured
        if not provider_id or not model_id:
            provider_id, model_id = self.model_label_manager.get_model_for_label("Large")

        if not provider_id or not model_id:
            return DEFAULT_MODEL

        # Get provider config
        provider = self.provider_manager.get_provider(provider_id)
        if not provider:
            return DEFAULT_MODEL

        base_url = provider.base_url or "https://api.anthropic.com"
        api_key = provider.api_key

        return (provider.type, model_id, base_url, api_key)

    def _build_messages_for_api(self) -> List[Dict]:
        """Convert history to API message format."""
        messages = []

        for msg in self.history:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool_use":
                # Anthropic format: assistant message with tool_use block
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": msg.tool_id,
                        "name": msg.tool_name,
                        "input": msg.tool_input or {}
                    }]
                })
            elif msg.role == "tool_result":
                # Build tool result content - can include text and/or images
                tool_result_content = []

                # Add text content if present
                if msg.content:
                    tool_result_content.append({
                        "type": "text",
                        "text": msg.content
                    })

                # Add image if present (for screenshots)
                if msg.image_base64:
                    tool_result_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": msg.image_base64
                        }
                    })

                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_id,
                        "content": tool_result_content if tool_result_content else msg.content
                    }]
                })

        return messages

    async def send_message(self, user_message: str,
                           system_prompt_override: Optional[str] = None) -> AsyncIterator[StreamChunk]:
        """
        Send a message and stream the response.

        Args:
            user_message: The user's message text.
            system_prompt_override: If provided, replaces the default
                NoodleCode system prompt entirely. Used by Guide mode
                to establish a different persona for the LLM.

        Yields StreamChunk objects as response is generated.
        """
        # Add user message to history
        user_msg = Message(role="user", content=user_message)
        self.history.append(user_msg)
        if self.on_message:
            self.on_message(user_msg)

        # Get model configuration
        provider_type, model_id, base_url, api_key = self._get_model_config()

        # Tool loop - continue until assistant responds without tool calls
        while True:
            # Call LLM
            response_text = ""
            tool_calls = []

            async for chunk in self._call_llm(provider_type, model_id, base_url, api_key,
                                                system_prompt_override=system_prompt_override):
                if chunk.type == "text":
                    response_text += chunk.content
                    yield chunk
                elif chunk.type == "tool_use_start":
                    tool_calls.append({
                        "id": chunk.tool_id,
                        "name": chunk.tool_name,
                        "input": {}
                    })
                    yield chunk
                elif chunk.type == "tool_use_input":
                    if tool_calls:
                        tool_calls[-1]["input"] = chunk.tool_input
                elif chunk.type == "error":
                    yield chunk
                    return

            # If no tool calls, we're done
            if not tool_calls:
                # Add assistant response to history
                if response_text:
                    assistant_msg = Message(role="assistant", content=response_text)
                    self.history.append(assistant_msg)
                    if self.on_message:
                        self.on_message(assistant_msg)
                yield StreamChunk(type="done")
                return

            # Execute tool calls
            for tool_call in tool_calls:
                tool_id = tool_call["id"]
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]

                # Add tool use to history
                tool_use_msg = Message(
                    role="tool_use",
                    content="",
                    tool_id=tool_id,
                    tool_name=tool_name,
                    tool_input=tool_input
                )
                self.history.append(tool_use_msg)
                if self.on_message:
                    self.on_message(tool_use_msg)

                # Execute tool
                result = await self.tools.execute(tool_name, tool_input)

                # Format result
                if result.success:
                    result_text = result.output
                else:
                    result_text = f"Error: {result.error}\n{result.output}" if result.output else f"Error: {result.error}"

                # Add tool result to history (include image if present)
                result_msg = Message(
                    role="tool_result",
                    content=result_text,
                    tool_id=tool_id,
                    image_base64=result.image_base64  # For screenshots
                )
                self.history.append(result_msg)
                if self.on_message:
                    self.on_message(result_msg)

                # Yield tool result chunk
                yield StreamChunk(
                    type="tool_result",
                    content=result_text,
                    tool_name=tool_name,
                    tool_id=tool_id
                )

            # Continue loop to let LLM respond to tool results

    async def _call_llm(
        self,
        provider_type: str,
        model_id: str,
        base_url: str,
        api_key: Optional[str],
        system_prompt_override: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """Call the LLM API and stream response."""
        system_prompt = system_prompt_override or self._build_system_prompt()

        if provider_type == "anthropic":
            async for chunk in self._call_anthropic(model_id, api_key, system_prompt):
                yield chunk
        elif provider_type in ["openai", "openrouter", "ollama"]:
            async for chunk in self._call_openai_compatible(model_id, base_url, api_key, provider_type, system_prompt):
                yield chunk
        else:
            yield StreamChunk(type="error", content=f"Unsupported provider type: {provider_type}")

    async def _call_anthropic(
        self,
        model_id: str,
        api_key: Optional[str],
        system_prompt: str = ""
    ) -> AsyncIterator[StreamChunk]:
        """Call Anthropic API with tool use."""

        if not api_key:
            yield StreamChunk(type="error", content="Anthropic API key not configured. Set it in Model Manager.")
            return

        messages = self._build_messages_for_api()
        tools = self.tools.get_tool_definitions()

        payload = {
            "model": model_id,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
            "tools": tools,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield StreamChunk(type="error", content=f"API error {response.status_code}: {error_text.decode()}")
                        return

                    current_tool_id = None
                    current_tool_name = None
                    current_tool_input_json = ""

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        if event_type == "content_block_start":
                            block = event.get("content_block", {})
                            if block.get("type") == "tool_use":
                                current_tool_id = block.get("id")
                                current_tool_name = block.get("name")
                                current_tool_input_json = ""
                                yield StreamChunk(
                                    type="tool_use_start",
                                    tool_id=current_tool_id,
                                    tool_name=current_tool_name
                                )

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield StreamChunk(type="text", content=text)
                            elif delta.get("type") == "input_json_delta":
                                current_tool_input_json += delta.get("partial_json", "")

                        elif event_type == "content_block_stop":
                            if current_tool_id and current_tool_input_json:
                                try:
                                    tool_input = json.loads(current_tool_input_json)
                                except json.JSONDecodeError:
                                    tool_input = {}
                                yield StreamChunk(
                                    type="tool_use_input",
                                    tool_id=current_tool_id,
                                    tool_name=current_tool_name,
                                    tool_input=tool_input
                                )
                                current_tool_id = None
                                current_tool_name = None
                                current_tool_input_json = ""

        except httpx.TimeoutException:
            yield StreamChunk(type="error", content="Request timed out")
        except Exception as e:
            yield StreamChunk(type="error", content=f"Request failed: {e}")

    async def _call_openai_compatible(
        self,
        model_id: str,
        base_url: str,
        api_key: Optional[str],
        provider_type: str,
        system_prompt: str = ""
    ) -> AsyncIterator[StreamChunk]:
        """Call OpenAI-compatible API (OpenAI, OpenRouter, Ollama, etc.)."""

        messages = [{"role": "system", "content": system_prompt}]

        # Convert history to OpenAI format
        for msg in self.history:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool_use":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": msg.tool_id,
                        "type": "function",
                        "function": {
                            "name": msg.tool_name,
                            "arguments": json.dumps(msg.tool_input or {})
                        }
                    }]
                })
            elif msg.role == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_id,
                    "content": msg.content
                })

        # Convert tools to OpenAI format
        tools = []
        for tool in self.tools.get_tool_definitions():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })

        # Build endpoint URL
        if provider_type == "ollama":
            endpoint = f"{base_url}/v1/chat/completions"
        elif base_url.endswith("/v1"):
            endpoint = f"{base_url}/chat/completions"
        else:
            endpoint = f"{base_url}/v1/chat/completions"

        payload = {
            "model": model_id,
            "messages": messages,
            "tools": tools if tools else None,
            "stream": True,
            "max_tokens": 4096
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield StreamChunk(type="error", content=f"API error {response.status_code}: {error_text.decode()}")
                        return

                    current_tool_calls = {}

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = event.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Text content
                        if "content" in delta and delta["content"]:
                            yield StreamChunk(type="text", content=delta["content"])

                        # Tool calls
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in current_tool_calls:
                                    current_tool_calls[idx] = {
                                        "id": tc.get("id", f"call_{idx}"),
                                        "name": "",
                                        "arguments": ""
                                    }

                                if "id" in tc:
                                    current_tool_calls[idx]["id"] = tc["id"]

                                func = tc.get("function", {})
                                if "name" in func:
                                    current_tool_calls[idx]["name"] = func["name"]
                                    yield StreamChunk(
                                        type="tool_use_start",
                                        tool_id=current_tool_calls[idx]["id"],
                                        tool_name=func["name"]
                                    )
                                if "arguments" in func:
                                    current_tool_calls[idx]["arguments"] += func["arguments"]

                    # Emit completed tool calls
                    for tc in current_tool_calls.values():
                        if tc["name"]:
                            try:
                                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield StreamChunk(
                                type="tool_use_input",
                                tool_id=tc["id"],
                                tool_name=tc["name"],
                                tool_input=args
                            )

        except httpx.TimeoutException:
            yield StreamChunk(type="error", content="Request timed out")
        except Exception as e:
            yield StreamChunk(type="error", content=f"Request failed: {e}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
