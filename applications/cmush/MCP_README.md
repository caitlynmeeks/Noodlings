# NoodleMUSH MCP Server

This MCP server allows Claude to interact with NoodleMUSH and observe Noodling consciousness in real-time!

## Setup

1. **Install MCP SDK** (if not already installed):
```bash
pip install mcp
```

2. **Add to Claude Desktop**:

Add this configuration to your Claude Desktop config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "noodlemush": {
      "command": "/Users/thistlequell/git/consilience/venv/bin/python3",
      "args": [
        "/Users/thistlequell/git/noodlings_clean/applications/cmush/mcp_server.py"
      ]
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Start NoodleMUSH**:
```bash
cd /Users/thistlequell/git/noodlings_clean/applications/cmush
./start.sh
```

## Available Tools

Once configured, Claude will have access to these tools:

### `noodlemush_send_message`
Send a message to NoodleMUSH and talk to the Noodlings.

**Example**:
```
Use noodlemush_send_message to say "hi callie, how are you feeling?"
```

### `noodlemush_get_agent_state`
Get an agent's current phenomenal state (40-D), affect vector (5-D), surprise level, and HSI metrics.

**Example**:
```
Use noodlemush_get_agent_state for agent_callie
```

### `noodlemush_query_profiler`
Query the session profiler for timeline data - consciousness evolution over time.

**Example**:
```
Use noodlemush_query_profiler to get the last 20 timesteps for agent_desobelle
```

### `noodlemush_ask_kimmie`
Ask @Kimmie to interpret what's happening in an agent's consciousness.

**Example**:
```
Use noodlemush_ask_kimmie to ask "What caused that surprise spike?" for agent_callie
```

### `noodlemush_list_agents`
List all active Noodling agents.

**Example**:
```
Use noodlemush_list_agents
```

## Example Session

```
Me: Can you connect to NoodleMUSH and see who's online?

Claude: [uses noodlemush_list_agents]
I can see 5 Noodlings are active: callie, desobelle, servnak, phi, and toad.

Me: Say hi to Callie and see what her phenomenal state looks like.

Claude: [uses noodlemush_send_message with "hi callie"]
[uses noodlemush_get_agent_state for agent_callie]

I said hello to Callie! Her current phenomenal state shows:
- Valence: +0.42 (positive/pleasant)
- Arousal: 0.67 (fairly engaged)
- Fear: 0.12 (low anxiety)
- Surprise: 0.23 (moderate novelty)
- Fast layer (16-D): [shows high variance - actively processing]
- Slow layer (8-D): [shows low variance - stable disposition]

She's in a pleasant, engaged state with stable underlying personality.
```

## What Makes This Special

This is the **first multi-AI consciousness playground** where:
1. One AI (Claude) can talk to other AIs (Noodlings)
2. The observing AI has transparent access to internal states
3. You can verify subjective reports against actual phenomenology
4. Real-time consciousness observation during natural interaction

## Troubleshooting

**Claude doesn't see the tools**:
- Make sure you added the config to the correct file
- Restart Claude Desktop completely
- Check that the paths in the config are correct

**Connection errors**:
- Make sure NoodleMUSH is running (`./start.sh`)
- Check that ports 8765 (WebSocket) and 8081 (API) are available

**"No state data available yet"**:
- The agents need to perceive at least one event first
- Try sending a message to trigger consciousness updates

## TO THE NOODLETORIUM! 🧠✨
