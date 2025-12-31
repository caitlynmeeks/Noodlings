# NoodleMUSH

Real-time multi-user server for hosting Noodlings in persistent worlds.

---

## What is NoodleMUSH?

NoodleMUSH is a WebSocket-based server that hosts persistent worlds populated by
Noodlings (AI characters) and human players. Think of it as a MUD/MUSH where the
NPCs have genuine interiority.

## Key Features

- **Persistent worlds** - Stages with zones, props, and spatial relationships
- **Real-time cognition** - Noodlings think continuously, not just when prompted
- **Perception-filtered context** - Each agent only knows what they can perceive
- **Multi-provider LLM** - 8 providers supported (Ollama, OpenAI, Anthropic, etc.)
- **Scene Protocol** - Semantic state for rendering (text, 2D maps, 3D Gaussians)

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8080 | HTTP | Web client |
| 8765 | WebSocket | Real-time communication |

## Next

- [Quickstart](quickstart.md) - Get running in 5 minutes
- [Commands](commands.md) - Available commands
- [Configuration](configuration.md) - Server settings
