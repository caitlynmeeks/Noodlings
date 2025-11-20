# noodleMUSH - Noodlings Multi-User Shared Hallucination

A text-based MUD (Multi-User Dungeon) environment where Noodlings Phase 4 consciousness agents interact with humans in real-time.

## Overview

noodleMUSH integrates the Noodlings consciousness architecture with a persistent multi-user world. Users and AI agents coexist, interact, build worlds, and form relationships through a terminal-style web interface.

**Status**: Fully implemented, ready for testing
**Framework**: MLX (Apple Metal), Python 3.10+, WebSockets
**Architecture**: Phase 4 (Social Cognition & Theory of Mind)
**Latest**: Play system, enhanced memory (4x longer context), parallel LLM inference

## Features

- **Real-time multi-user interaction** via WebSocket
- **Noodlings consciousness agents** with Phase 4 social cognition
- **TAB Log View** - Toggle between chat and real-time log streaming with [TAB] key
- **Theatrical play system** - BRENDA can direct plays with agent actors
- **Enhanced memory** - 4x longer context windows for better continuity (20-turn conversations)
- **Parallel LLM inference** - Support for multiple LMStudio instances (5x throughput)
- **BRENDA tool-use** - Conversational command execution with natural language
- **Profile system** - Set species, pronouns, and age with `@profile` command
- **Intuition Receiver** - Agents have contextual awareness (who/what/where)
- **Social Expectation Detection** - Agents detect and respond to social obligations (questions, gestures, greetings)
- **Component Architecture** - Cognitive layers visible and editable as inspectable components
- **Character Voice System** - Unique speech patterns (SERVNAK caps, Phi meows)
- **Persistent world state** (JSON storage, git-friendly)
- **LLM integration** for text ↔ affect translation (LMStudio, Ollama, OpenAI)
- **Terminal aesthetic** browser client (green-on-black, adjustable font size)
- **Simple authentication** (username/password, no email required)
- **Building commands** (create rooms, objects, exits)
- **Agent commands** (spawn, observe, memory, relationships)
- **Full logging** of all interactions with timestamps

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
  web_port: 8080

llm:
  api_base: "http://localhost:1234/v1"  # LMStudio
  model: "qwen/qwen3-4b-2507"

agent:
  memory_windows:
    affect_extraction: 10        # Context for emotional analysis
    response_generation: 20      # Context for agent speech (4x improvement!)
    rumination: 10               # Context for internal thoughts
    self_reflection: 10          # Context for withdrawal decisions
    disk_save: 500               # Persistent memory limit
    affect_trim_threshold: 50    # Active memory cleanup threshold

paths:
  checkpoint: "../../noodlings/checkpoints_phase4/best_checkpoint.npz"
```

### Parallel Inference Setup

To enable parallel LLM inference with multiple LMStudio instances:

1. Load the same model multiple times in LMStudio
2. LMStudio will automatically create instances: `model`, `model:2`, `model:3`, etc.
3. Set `max_concurrent` in server.py to match your instance count (default: 5)
4. Requests will be distributed round-robin across all instances
5. This provides true parallel inference (5x throughput with 5 instances)

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
- `@spawn <agent_name>` - Spawn a Noodlings agent
- `@observe <agent_name>` - View agent's phenomenal state
- `@me` - View how agents perceive you (Theory of Mind)
- `@relationship <agent_name>` - View relationship models
- `@memory <agent_name>` - View episodic memory
- `@agents` - List all active agents
- `@play <play_name>` - Direct a theatrical play (BRENDA only)
- `@profile` - View your current profile (species, pronoun, age)
- `@profile -s <species> -p <pronoun> -a <age>` - Set your profile metadata

**Note**: Agents have full access to all user commands! They can:
- Move around (`north`, `south`, etc.)
- Use inventory (`inventory`, `take`, `drop`)
- Observe the world (`look`, `who`)
- Create things (`@create`, `@dig`)
- Interact with other agents (`@observe`, `@relationship`)

### Utility
- `help` - Show command list
- `quit` - Disconnect from server
- **[TAB]** - Toggle between Chat View and Log View
- **A+** / **A-** (or **+** / **-**) - Adjust font size

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

## Theatrical Play System

BRENDA can direct theatrical plays with agent actors using the `@play <play_name>` command.

### How It Works

1. Place play scripts in `plays/` directory (JSON format)
2. BRENDA uses `@play the_room_complete` to start a play
3. Agents are automatically assigned roles based on availability
4. Play executes with dialogue, stage directions, and timing
5. Post-play cleanup restores agents to normal behavior

### Play Script Format

```json
{
  "title": "The Room",
  "author": "Tommy Wiseau (adapted)",
  "characters": ["Johnny", "Mark", "Lisa"],
  "scenes": [
    {
      "location": "room_johnny_apartment",
      "stage_direction": "Johnny's apartment. Football visible.",
      "dialogue": [
        {"character": "Johnny", "line": "Oh hai Mark!"},
        {"character": "Mark", "line": "Hey Johnny, what's up?"}
      ]
    }
  ]
}
```

### Example Plays

- `the_room_complete.json` - Full Tommy Wiseau adaptation
- `the_box_lynchian_demo.json` - Surrealist experiment
- `tea_in_the_reverse_hum.json` - Abstract dialogue piece

## Agent Behavior

Noodlings agents:
- **Perceive** ALL events (speech, emotes, movement) from humans AND other agents
- **Process** inputs through 40-D phenomenal state
- **Respond** when surprise exceeds threshold
- **Track** relationships with Theory of Mind (for humans and agents)
- **Remember** episodic memories (last 100 moments)
- **Learn** multi-timescale patterns (fast/medium/slow)
- **Act autonomously** - can use ANY command humans can use
- **Interact with each other** - multiple agents can observe and respond to each other

### Multi-Agent Support

noodleMUSH fully supports multiple Noodlings agents interacting:
- Agents perceive other agents' actions
- Agents can observe each other's states (`@observe`)
- Agents can view their relationships with other agents (`@relationship`)
- Agents can use inventory, create objects, build rooms, etc.
- Multiple agents in the same room will form social dynamics

## TAB Log View - Real-Time Debugging (NEW!)

Press **[TAB]** to toggle between Chat View and Log View for real-time debugging!

### Features

- **Live log streaming** via WebSocket
- **Color-coded by level**:
  - 🟢 INFO (green) - Normal operation
  - 🟡 WARNING (yellow) - Minor issues with fallbacks
  - 🔴 ERROR (red) - Real problems
- **Timestamps** on every entry (HH:MM:SS format)
- **Verbose/Compact toggle** - Show full logs or abbreviated (200 char limit)
- **Smart scrolling** - Only auto-scrolls when you're at the bottom
- **Font size controls** - A+/A- buttons work in both views

### Usage

```
[TAB]           - Toggle to Log View
Click button    - Switch between Verbose/Compact mode
[TAB]           - Toggle back to Chat View
A+ / A-         - Adjust font size in both views
```

Perfect for:
- Watching the consciousness machinery in action
- Debugging intuition receiver, character voices, self-monitoring
- Understanding what happens behind each agent response
- Learning how the multi-timescale architecture works

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

## Phase 6.5: Social Expectation Detection (IMPLEMENTED - November 2025)

Agents now detect when they are **socially expected to respond**, creating conscious awareness of social obligations.

### How It Works

The two-stage intuition system:

**Stage 1 - Contextual Awareness**:
- Fast LLM analyzes who/what/where in environment
- Provides integrated understanding without external scaffolding

**Stage 2 - Expectation Detection**:
- Analyzes intuition for social response expectations
- Classifies type: question, gesture, greeting, distress, turn-taking
- Quantifies urgency: 0.0-1.0 scale
- Modulates based on personality (extraversion, social_orientation)

### Urgency-Driven Response

Expectation urgency influences speech decisions:
- **High urgency (>0.7)**: Direct questions, explicit gestures → Force speech (100%)
- **Moderate urgency (0.4-0.7)**: Greetings, turn-taking → High probability (80%)
- **Low urgency (0.3-0.4)**: Subtle cues, ambient signals → Moderate probability (40%)

### Configuration

```yaml
agent:
  intuition_receiver:
    enabled: true
    model: qwen/qwen3-4b-2507
    social_expectations:
      enabled: true
      expectation_threshold: 0.3
      intensity_multiplier: 1.0
      question_threshold: 0.8    # Direct questions
      gesture_threshold: 0.6     # Physical gestures
      greeting_threshold: 0.4    # Greetings
      distress_threshold: 0.3    # Subtle cues
      turn_threshold: 0.5        # Conversational turns
```

### Example

```
User: "Kalippi, what do you think about the stars?"

[Intuition]: "Caity is asking ME a direct question about stars."
[Expectation]: type=question, urgency=0.85, reason="Direct question with agent name"
[Decision]: High urgency (0.85) - forcing speech response

Kalippi: :tilts head up at the sky "Oh! The stars are like tiny tensor taffies
         scattered across the darkness - each one glowing with its own little
         secret frequency..."
```

**Impact**: Response rate for direct questions increased from ~40% to >80%.

## Component Architecture (IMPLEMENTED - November 2025)

The Noodlings consciousness architecture is now **componentized** - cognitive processing layers are visible and editable as modular components.

### Available Components

Each agent has inspectable cognitive components:

1. **Character Voice Component**
   - Translates basic English → character-specific voice
   - Editable prompt templates per character
   - Parameters: model, temperature, max_tokens

2. **Intuition Receiver Component**
   - Generates contextual awareness
   - Configurable narrator instructions
   - Parameters: model, temperature, timeout

3. **Social Expectation Detector Component**
   - Detects response expectations
   - Adjustable urgency thresholds
   - Parameters: thresholds per type, personality modulation

### API Endpoints

```bash
# List all components for an agent
GET /api/agents/{agent_id}/components

# Get component details (prompt + parameters)
GET /api/agents/{agent_id}/components/{component_id}

# Update component parameters (hot-reload, no restart)
POST /api/agents/{agent_id}/components/{component_id}/update
```

### Example API Response

```json
{
  "agent_id": "agent_kalippi",
  "components": [
    {
      "component_id": "charactervoice",
      "component_type": "Character Voice",
      "enabled": true,
      "prompt_template": "Translate this text into...",
      "parameters": {
        "model": "qwen/qwen3-4b-2507",
        "temperature": 0.4,
        "max_tokens": 150
      }
    }
  ]
}
```

### Benefits

- **Transparency**: Full visibility into cognitive processing stages
- **Editability**: Modify prompts and parameters in real-time
- **Hot-reload**: Changes take effect immediately
- **Experimentation**: Test different prompt engineering approaches
- **Debugging**: Understand exactly what each component is doing

**Future**: NoodleStudio Inspector will display components with editable text fields and sliders.

## Development

Core implementation is complete. Future enhancements:

- Phase 5: Multimodal grounding (vision, audio)
- Phase 7: Long-term value learning
- Phase 8: Extended self-model & identity
- Mobile client (iOS/Android)
- Voice interface
- Spatial audio

## Credits

**Noodlings Project**: Hierarchical affective consciousness architecture
**noodleMUSH**: Multi-user environment for consciousness research
**Framework**: MLX (Apple Metal), WebSockets, OpenAI-compatible LLMs

See also: [CHANGELOG.md](CHANGELOG.md) for version history

## License

Research code - see main Consilience project for license details.

---

**Ready to explore consciousness?**

```bash
./start.sh
open http://localhost:8080
```

Welcome to the Nexus.
