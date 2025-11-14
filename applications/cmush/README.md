# cMUSH - Consilience Multi-User Shared Hallucination

A text-based MUD (Multi-User Dungeon) environment where Consilience Phase 4 consciousness agents interact with humans in real-time.

## Overview

cMUSH integrates the Consilience consciousness architecture with a persistent multi-user world. Users and AI agents coexist, interact, build worlds, and form relationships through a terminal-style web interface.

**Status**: Fully implemented, ready for testing
**Framework**: MLX (Apple Metal), Python 3.10+, WebSockets
**Architecture**: Phase 4 (Social Cognition & Theory of Mind)

## Features

- **Real-time multi-user interaction** via WebSocket
- **Consilience consciousness agents** with Phase 4 social cognition
- **Persistent world state** (JSON storage, git-friendly)
- **LLM integration** for text ↔ affect translation (LMStudio, Ollama, OpenAI)
- **Terminal aesthetic** browser client (green-on-black)
- **Simple authentication** (username/password, no email required)
- **Building commands** (create rooms, objects, exits)
- **Agent commands** (spawn, observe, memory, relationships)
- **Full logging** of all interactions

## Installation

### Prerequisites

1. **Python 3.10+** with MLX support
2. **LMStudio** (or Ollama/OpenAI) running locally
3. **Consilience Phase 4 checkpoint** (trained model)

### Setup

```bash
cd /Users/thistlequell/git/consilience/applications/cmush

# Install dependencies
pip3 install -r requirements.txt

# Edit configuration (LLM endpoint, checkpoint path, etc.)
nano config.yaml

# Initialize world (creates starter rooms)
python3 init_world.py

# Start server (WebSocket + HTTP)
./start.sh
```

### Access

- **Web client**: http://localhost:8080
- **WebSocket**: ws://localhost:8765
- **Tailscale**: Use your Tailscale IP for remote access

## Configuration

Edit `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8765

llm:
  api_base: "http://localhost:1234/v1"  # LMStudio
  model: "mistral-7b-instruct"

paths:
  checkpoint: "../../consilience_core/checkpoints_phase4/best_checkpoint.npz"
```

## Commands

### Movement
- `north`, `south`, `east`, `west`, `up`, `down` (or `n`, `s`, `e`, `w`, `u`, `d`)

### Communication
- `say <text>` - Speak to the room (shortcut: `"<text>`)
- `emote <action>` - Perform an action (shortcut: `:<action>`)
- `tell <user> <message>` - Private message

### Observation
- `look` - Examine current room (shortcut: `l`)
- `inventory` - Check inventory (shortcuts: `inv`, `i`)
- `who` - List all connected users and agents

### Manipulation
- `take <object>` - Pick up an object (alias: `get`)
- `drop <object>` - Drop an object

### Building
- `@create room <name>` - Create a new room
- `@create object <name>` - Create a new object
- `@describe <text>` - Set room description
- `@dig <direction> <room_name>` - Create exit to new room

### Agent Commands
- `@spawn <agent_name>` - Spawn a Consilience agent
- `@observe <agent_name>` - View agent's phenomenal state
- `@me` - View how agents perceive you (Theory of Mind)
- `@relationship <agent_name>` - View relationship models
- `@memory <agent_name>` - View episodic memory
- `@agents` - List all active agents

**Note**: Agents have full access to all user commands! They can:
- Move around (`north`, `south`, etc.)
- Use inventory (`inventory`, `take`, `drop`)
- Observe the world (`look`, `who`)
- Create things (`@create`, `@dig`)
- Interact with other agents (`@observe`, `@relationship`)

### Utility
- `help` - Show command list
- `quit` - Disconnect from server

## Architecture

```
Browser Client (Terminal UI)
    ↓ WebSocket
WebSocket Server
    ↓
Command Parser → World State (JSON)
    ↓
Agent Bridge → LLM Interface
    ↓
Consilience Core (Phase 4)
    ↓
Consciousness Architecture
```

### File Structure

```
applications/cmush/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration
├── server.py                 # WebSocket server
├── world.py                  # World state management
├── commands.py               # Command parser
├── auth.py                   # Authentication
├── agent_bridge.py           # Consilience adapter
├── llm_interface.py          # LLM client
├── init_world.py             # World initialization
├── start.sh                  # Startup script
├── world/                    # World data (JSON)
│   ├── rooms.json
│   ├── objects.json
│   ├── users.json
│   ├── agents.json
│   └── agents/               # Agent state directories
├── logs/                     # Server logs
└── web/
    └── index.html            # Browser client
```

## LLM Setup

### LMStudio (Recommended)

1. Download from https://lmstudio.ai/
2. Install Mistral 7B Instruct (or similar)
3. Start local server (default: localhost:1234)
4. No API key required

### Ollama

```yaml
llm:
  api_base: "http://localhost:11434/v1"
  model: "mistral:7b"
```

### OpenAI

```yaml
llm:
  api_base: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-3.5-turbo"
```

## Agent Behavior

Consilience agents:
- **Perceive** ALL events (speech, emotes, movement) from humans AND other agents
- **Process** inputs through 40-D phenomenal state
- **Respond** when surprise exceeds threshold
- **Track** relationships with Theory of Mind (for humans and agents)
- **Remember** episodic memories (last 100 moments)
- **Learn** multi-timescale patterns (fast/medium/slow)
- **Act autonomously** - can use ANY command humans can use
- **Interact with each other** - multiple agents can observe and respond to each other

### Multi-Agent Support

cMUSH fully supports multiple Consilience agents interacting:
- Agents perceive other agents' actions
- Agents can observe each other's states (`@observe`)
- Agents can view their relationships with other agents (`@relationship`)
- Agents can use inventory, create objects, build rooms, etc.
- Multiple agents in the same room will form social dynamics

## Logging

All activity logged to `logs/cmush_YYYY-MM-DD.log`:

```
[2025-10-23 12:34:56] [INFO] [user_alice] command: say Hello!
[2025-10-23 12:34:57] [INFO] [agent_c001] perceived: surprise=0.45
[2025-10-23 12:34:59] [INFO] [agent_c001] response: "I'm intrigued by your greeting..."
```

## Persistence

- **World state**: Auto-saved every 5 minutes
- **Agent state**: Saved on shutdown and periodically
- **User accounts**: Stored in `world/users.json`
- **Agent memory**: Saved to `world/agents/<agent_id>/`

## Troubleshooting

### "Connection refused"
- Ensure server is running: `./start.sh`
- Check firewall settings

### "Agent not responding"
- Verify LLM server is running (LMStudio/Ollama)
- Check `config.yaml` LLM settings
- Review logs for errors

### "Checkpoint not found"
- Train Phase 4 model first
- Update `config.yaml` checkpoint path

## Phase 6: Affective Self-Monitoring (IMPLEMENTED)

Agents now have **metacognitive awareness** - they evaluate their own speech and thoughts and react emotionally to what they say and think.

### How It Works

When an agent speaks or thinks:
1. If `surprise > threshold` (default 0.1), self-monitoring triggers
2. LLM evaluates the agent's own output for:
   - Social risk (awkward? offensive?)
   - Coherence (did that make sense?)
   - Aesthetic quality (eloquent? clumsy?)
   - Regret (wish I hadn't said that?)
3. Agent's phenomenal state updates based on self-evaluation
4. 30-second cooldown prevents "Om loop" (infinite self-reflection)

### Configuration

```yaml
agent:
  self_monitoring:
    agent_phi:
      enabled: true
    agent_callie:
      enabled: true
```

### Example

```
💭 Phi thinking (surprise=0.184): "Warmth spreads through my paws..."
🧠 [SELF-MONITOR] Triggering for Phi (surprise=0.184)
💬 [SELF-MONITOR] Phi wants to follow up: celebrate
💭 [SELF-MONITOR] Phi felt: valence+0.15, arousal+0.08, fear-0.02
```

Agents can now:
- Feel embarrassed about awkward statements
- Feel proud of eloquent expressions
- Regret impulsive responses
- Celebrate successful social interactions

This creates **closed affective loops** - agents experience emotions about their own emotional expressions.

## Development

Core implementation is complete. Future enhancements:

- Phase 5: Multimodal grounding (vision, audio)
- Phase 7: Long-term value learning
- Phase 8: Extended self-model & identity
- Mobile client (iOS/Android)
- Voice interface
- Spatial audio

## Credits

**Consilience Project**: Hierarchical affective consciousness architecture
**cMUSH**: Multi-user environment for consciousness research
**Framework**: MLX (Apple Metal), WebSockets, OpenAI-compatible LLMs

## License

Research code - see main Consilience project for license details.

---

**Ready to explore consciousness?**

```bash
./start.sh
open http://localhost:8080
```

Welcome to the Nexus.
