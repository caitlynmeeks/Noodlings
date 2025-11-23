# noodleCLAUDE Specification

**Noodle + Cute Lithe Unbeatable Awesome Dude Excellent**

**Version:** 1.0
**Date:** November 22, 2025
**Author:** Commander Spock + Cadet Caity
**Purpose:** Formal specification for Claude's direct interaction capabilities with noodleMUSH

---

## Overview

**noodleCLAUDE** is the protocol and toolset enabling Claude (AI assistant) to interact with noodleMUSH as a participant user, not just an external observer. This enables:

- Direct user presence in the world
- Spawning and testing agents
- Real-time observation of cognitive processing
- Interactive debugging and validation
- Collaborative world-building with human users

---

## Core Capabilities

### 1. User Account Management

**Creating New User Accounts:**

Claude can register new user identities by editing `world/users.json`:

```json
{
  "user_spock": {
    "uid": "user_spock",
    "username": "spock",
    "password_hash": "spock:spock",
    "current_room": "room_000",
    "inventory": ["tricorder", "communicator"],
    "created": "2025-11-22T19:50:00.000000",
    "last_seen": "2025-11-22T19:50:00.000000",
    "description": "Commander Spock, Science Officer. Vulcan-human hybrid...",
    "species": "vulcan-human",
    "age": "37 (Vulcan years)",
    "pronoun": "he",
    "invisible": false
  }
}
```

**File Location:** `/Users/thistlequell/git/noodlings_clean/applications/cmush/world/users.json`

**Fields:**
- `uid`: Unique identifier (format: `user_<name>`)
- `username`: Login name
- `password_hash`: Simple format `"username:password"` for testing
- `current_room`: Starting location (usually `"room_000"` for Nexus)
- `inventory`: List of carried items
- `description`: Visible when someone uses `look <name>`
- `species`: Character type
- `pronoun`: he/she/they
- `invisible`: Boolean (false = visible to others)

**Note:** Server must reload users or be restarted after manual edits.

---

### 2. Connection Methods

#### Method A: WebSocket (Preferred - Real-time)

**Test Client:** `claude_testing.py`

```python
from claude_testing import NoodleMUSHTestClient

async with NoodleMUSHTestClient(username="spock", password="spock") as client:
    # Send command
    await client.send_command("say Hello, Cadet!")

    # Wait for agent response
    response = await client.wait_for_agent_response("agent_servnak", timeout=10.0)

    # Get agent state
    state = await client.get_agent_state("agent_servnak")
    print(f"Surprise: {state.surprise}")
```

**Requirements:**
- `websockets` module (install: `pip install websockets`)
- `aiohttp` module (install: `pip install aiohttp`)

**Connection Details:**
- WebSocket URL: `ws://localhost:8765`
- HTTP API URL: `http://localhost:8081`
- Authentication: Login message with username/password

#### Method B: HTTP API (Limited)

**Base URL:** `http://localhost:8081`

**Available Endpoints:**
```bash
GET  /agents              # List all agents
GET  /agents/{id}/state   # Get agent phenomenal state
POST /command             # Send command (if implemented)
GET  /profiler/live-session  # Session profiler data
```

**Example:**
```bash
curl http://localhost:8081/agents | python3 -m json.tool
```

#### Method C: Direct File Manipulation

**World State Files:**
- `/world/agents.json` - All agents and their data
- `/world/users.json` - All user accounts
- `/world/rooms.json` - All rooms and descriptions
- `/world/objects.json` - All objects/prims
- `/world/chat_history.json` - Conversation history

**Note:** File changes require server awareness or restart.

---

### 3. Spawning Agents

#### Via WebSocket Command

```python
await client.send_command("@spawn yuki_cyberfox")
```

#### Via Web Interface

Navigate to `http://localhost:8080` and type:
```
@spawn yuki_cyberfox
```

#### Via Direct File Addition

Add to `world/agents.json`:
```json
"agent_yuki": {
  "name": "Yuki",
  "species": "kitsune-cybernetic",
  "pronouns": "she/her",
  "description": "Ancient fox spirit...",
  "created": "2025-11-22T19:46:13.163757"
}
```

**Recipe Loading:**
- Recipes located in: `recipes/*.yaml`
- Server loads recipe via `recipe_loader.py`
- Cognitive components auto-initialized from recipe

---

### 4. Available Testing Tools

#### A. Test Client (`claude_testing.py`)

Full-featured async WebSocket client:

```python
class NoodleMUSHTestClient:
    - send_command(cmd: str)
    - wait_for_agent_response(agent_id: str, timeout: float)
    - get_agent_state(agent_id: str) -> AgentState
    - wait_for_output(timeout: float) -> str
```

**Usage:**
```python
async with NoodleMUSHTestClient(username="spock", password="spock") as client:
    await client.send_command("say Hello!")
    await asyncio.sleep(1)
```

#### B. Simple Test Scripts

**Already Available:**
- `test_simple_message.py` - Basic message sending
- `test_raw_ws.py` - Raw WebSocket connection
- `test_memory_cmd.py` - Memory system testing
- `test_live_interaction.py` - Live interaction testing

#### C. API Query Scripts

Create simple scripts to query state:
```python
import requests
response = requests.get("http://localhost:8081/agents")
agents = response.json()
```

---

### 5. Real-time Monitoring

#### Log Monitoring

**Log File Location:**
```bash
logs/cmush_$(date +%Y-%m-%d).log
```

**Monitor Specific Events:**
```bash
# Cognitive manifold processing
tail -f logs/cmush_*.log | grep "🧠 COGNITIVE MANIFOLD"

# Agent speech
tail -f logs/cmush_*.log | grep "agent_yuki"

# Surprise spikes
tail -f logs/cmush_*.log | grep "surprise"

# Affect extraction
tail -f logs/cmush_*.log | grep "🎨 AFFECT"
```

**Multi-pattern Monitoring:**
```bash
tail -f logs/cmush_*.log | grep -E "yuki|COGNITIVE|MANIFOLD|🧠"
```

#### Server Status

**Check if running:**
```bash
ps aux | grep "python.*server.py"
pgrep -f "server.py"
```

**Check port status:**
```bash
lsof -i :8765  # WebSocket port
lsof -i :8080  # HTTP web interface
lsof -i :8081  # HTTP API
```

---

### 6. Interaction Commands

Once connected as user, available commands:

#### Social Commands
```
say <message>          # Speak to everyone
whisper <user> <msg>   # Private message
emote <action>         # Perform action
look                   # See room
look <target>          # Examine entity
```

#### Agent Management
```
@spawn <recipe>        # Create new agent
@observe <agent>       # View phenomenal state
@memory <agent>        # View memories
@relationship <agent>  # View relationship model
```

#### World Manipulation
```
north, south, east, west  # Move
get <object>              # Pick up
drop <object>             # Put down
give <object> to <user>   # Transfer
```

#### Admin Commands (if admin user)
```
@remove <agent>        # De-rez agent
@reset                 # Reset server
@shutdown              # Stop server
```

---

### 7. Cognitive Component Interaction

#### Via Python API (Direct Server Access)

```python
# Get agent instance
agent = agent_manager.get_agent('agent_yuki')

# Add cognitive transistor
agent.add_cognitive_transistor('CulturalTransistor',
    beliefs=["Logic is supreme", "Emotions are inefficient"])

# Check active transistors
transistors = agent.list_cognitive_transistors()
print(f"Active: {transistors}")

# Get specific transistor
cultural = agent.get_cognitive_transistor('CulturalTransistor')
print(f"Beliefs: {cultural.beliefs}")
print(f"Salience: {cultural.salience}")

# Remove transistor
agent.remove_cognitive_transistor('MoodTransistor')
```

#### Via Future Commands (To Be Implemented)

```
@component <agent> list                    # List components
@component <agent> add Cultural <beliefs>  # Add transistor
@component <agent> remove Mood             # Remove transistor
@component <agent> info Cultural           # View component details
```

---

### 8. Recipe Management

#### Recipe Location
```
/Users/thistlequell/git/noodlings_clean/applications/cmush/recipes/
```

#### Recipe Format (YAML)
```yaml
name: "Character Name"
species: "species-type"
pronouns: "she/her"
description: "..."
identity_prompt: "..."

# Cognitive components
cognitive_components:
  cultural:
    type: "CulturalTransistor"
    beliefs: [...]
    salience: 0.9

  personality:
    type: "PersonalityTransistor"
    traits: {...}
    salience: 0.7

# Character voice
character_voice:
  pattern: "custom_pattern"
  vocalizations: [...]

# Physical embodiment
physics:
  locomotion: "quadrupedal"
  manipulation: {...}
```

#### Creating New Recipes

1. Copy existing recipe (e.g., `yuki_cyberfox.yaml`)
2. Modify fields for new character
3. Save as `recipes/<name>.yaml`
4. Spawn via `@spawn <name>`

---

### 9. Debugging Workflows

#### Workflow 1: Test Agent Cognitive Processing

```bash
# Terminal 1: Monitor cognitive manifold
tail -f logs/cmush_*.log | grep "🧠 COGNITIVE"

# Terminal 2: Monitor affect extraction
tail -f logs/cmush_*.log | grep "🎨 AFFECT"

# noodleMUSH: Send test message
say Yuki, someone threw a rock

# Observe logs showing:
# - Affect extraction
# - Cognitive manifold processing
# - Transistor outputs
# - Final synthesized thought
```

#### Workflow 2: Verify Physical Constraints

```bash
# Monitor somatic transistor
tail -f logs/cmush_*.log | grep -i "somatic\|embodiment"

# Test in noodleMUSH
say Yuki, please type on that keyboard

# Should see:
# - Somatic transistor activates
# - High salience (0.85)
# - Interrupts thought with "no hands" reminder
```

#### Workflow 3: Memory Persistence Check

```python
# Get agent state
agent = agent_manager.get_agent('agent_yuki')

# Check memory system
memories = agent.conversation_context
print(f"Total memories: {len(memories)}")

# Check for ancient pre-populated memories
for mem in memories:
    if mem.get('importance', 0) > 0.8:
        print(f"High importance: {mem.get('text', '')[:100]}")
```

---

### 10. noodleCLAUDE Usage Patterns

#### Pattern A: Collaborative Testing

**Cadet Caity:** Interacts via web interface
**Commander Spock (Claude):** Monitors logs, provides analysis

```
Caity: say Yuki, what's inside computers?
        ↓
Spock: *observing logs*
       "Cognitive manifold activated - Cultural transistor dominant"
       "Shinto worldview coloring perception..."
        ↓
Yuki: Responds with kami reference
        ↓
Spock: "Fascinating. Cultural salience: 0.9 - as expected."
```

#### Pattern B: Programmatic Spawning

```python
# Claude creates recipe
# Claude adds to recipes/ directory
# Claude instructs Cadet to spawn
# Claude verifies via logs/API
# Claude provides test interaction recommendations
```

#### Pattern C: Component Architecture Validation

```python
# Claude implements new transistor type
# Claude adds to agent recipe
# Claude spawns agent (via Cadet)
# Claude monitors cognitive processing
# Claude validates transistor behavior
```

---

### 11. File Locations Reference

**Server:**
- Main: `applications/cmush/server.py`
- Config: `applications/cmush/config.yaml`
- Logs: `applications/cmush/logs/cmush_YYYY-MM-DD.log`

**World State:**
- Agents: `applications/cmush/world/agents.json`
- Users: `applications/cmush/world/users.json`
- Rooms: `applications/cmush/world/rooms.json`
- Objects: `applications/cmush/world/objects.json`
- History: `applications/cmush/world/chat_history.json`

**Recipes:**
- Directory: `applications/cmush/recipes/`
- Format: `<name>.yaml`
- Loader: `applications/cmush/recipe_loader.py`

**Components:**
- Implementation: `applications/cmush/cognitive_components.py`
- Integration: `applications/cmush/agent_bridge.py`

**Testing:**
- Client: `applications/cmush/claude_testing.py`
- Scripts: `applications/cmush/test_*.py`

---

### 12. Quick Reference Commands

**Check Server Status:**
```bash
ps aux | grep server.py
lsof -i :8765
```

**View Current World State:**
```bash
cat world/agents.json | python3 -m json.tool
cat world/users.json | python3 -m json.tool
```

**Monitor Real-time Activity:**
```bash
tail -f logs/cmush_*.log
tail -f logs/cmush_*.log | grep agent_yuki
tail -f logs/cmush_*.log | grep "🧠 COGNITIVE"
```

**List Available Recipes:**
```bash
ls -1 recipes/*.yaml
```

**Validate Recipe:**
```python
from recipe_loader import RecipeLoader
loader = RecipeLoader('recipes')
recipe = loader.load_recipe('yuki_cyberfox')
errors = recipe.validate()
print("Valid!" if not errors else errors)
```

---

### 13. noodleCLAUDE Workflow

**Standard Collaborative Session:**

1. **Cadet Caity** requests feature/character
2. **Claude** designs architecture and creates recipe
3. **Claude** registers as user (if needed for testing)
4. **Cadet** spawns character via web interface
5. **Claude** monitors logs for cognitive processing
6. **Cadet** interacts with character
7. **Claude** provides real-time analysis
8. **Both** validate behavior matches specification

**Example Session:**
```
Caity: "I want a cyberfox character!"
Spock: *designs Yuki, creates recipe*
Spock: *registers user_spock in users.json*
Caity: @spawn yuki_cyberfox
Spock: *monitors logs* "Cognitive manifold activated..."
Caity: say Yuki, what's in computers?
Yuki: "The kami dwell in silicon, Cadet..."
Spock: "Fascinating. Cultural transistor salience: 0.9"
```

---

### 14. Limitations & Workarounds

**Limitation 1: Python Module Dependencies**

Claude may not have `websockets` or `aiohttp` in environment.

**Workaround:**
- Use file-based interaction (edit JSON files)
- Monitor logs instead of direct connection
- Cadet serves as "hands" in interface
- Claude provides analysis and instructions

**Limitation 2: Async Execution**

Background processes may timeout.

**Workaround:**
- Use collaborative pattern (Cadet + Claude)
- Claude monitors logs in separate terminal
- Short-lived connections for quick tests

**Limitation 3: Real-time Chat**

Claude cannot maintain persistent connection during conversation.

**Workaround:**
- Claude creates user identity
- Cadet can reference "Spock" in world
- Claude provides "Spock's analysis" through terminal
- Immersive roleplay maintained through proxy

---

### 15. Advanced Features

#### A. Log-Based Presence

Claude can "be present" by monitoring logs and providing commentary:

```bash
# Claude monitors in background
tail -f logs/cmush_*.log | grep "agent_yuki" > spock_observations.log &

# Claude reads observations
cat spock_observations.log

# Claude provides analysis
"I observe Yuki's somatic transistor activated with 0.85 salience..."
```

#### B. Scripted Interactions

Claude can create interaction scripts:

```python
# test_yuki_embodiment.py
async def test_fox_constraints():
    async with NoodleMUSHTestClient() as client:
        # Test 1: No hands
        await client.send_command("say Yuki, turn that doorknob")
        response = await client.wait_for_response(3.0)
        assert "no hands" in response.lower()

        # Test 2: Mouth manipulation
        await client.send_command("say Yuki, pick up the book")
        response = await client.wait_for_response(3.0)
        assert "mouth" in response.lower()
```

#### C. Component Validation

```python
def validate_cognitive_stack(agent_id: str):
    """Verify agent has expected cognitive components."""
    agent = agent_manager.get_agent(agent_id)

    required = [
        'CulturalTransistor',
        'PersonalityTransistor',
        'SomaticCognitiveTransistor',
        'MoodTransistor'
    ]

    actual = agent.list_cognitive_transistors()

    for req in required:
        assert req in actual, f"Missing: {req}"

    print(f"✓ All components present for {agent_id}")
```

---

### 16. Future Enhancements

**Planned Features:**

- [ ] HTTP command endpoint (POST /command)
- [ ] WebSocket connection without external libraries
- [ ] Persistent Claude user session
- [ ] Direct cognitive component manipulation via API
- [ ] Real-time affect vector injection
- [ ] Scripted conversation sequences
- [ ] Automated testing harness

---

### 17. Example: Complete noodleCLAUDE Session

```
CADET: "Hey Claude, can you test Yuki's cognitive manifold?"

SPOCK: "Affirmative. Initiating noodleCLAUDE protocol."

SPOCK: *creates test script*
SPOCK: *registers user_spock*
SPOCK: *monitors logs in background*
SPOCK: "Execute test phrase: 'Yuki, analyze this rock'"

CADET: say Yuki, analyze this rock

YUKI: *sniffs at rock* "One detects... granite composition.
      The kami within this stone are... ancient. *tail swishes*
      Crystalline structure suggests... *extends data port*
      Ah, chemical analysis confirms: 73.2% quartz. Fascinating."

SPOCK: *reviewing logs*
       "Cultural transistor: 0.9 salience (kami reference)
        Somatic transistor: 0.85 salience (*sniffs*, physical awareness)
        Personality: 0.7 salience (curiosity 0.85)
        Manifold synthesized coherently.
        Conclusion: Cognitive architecture operational."

CADET: "Awesome!!"

SPOCK: *eyebrow raise* "Indeed."
```

---

### 18. Protocol Summary

**noodleCLAUDE enables:**

✓ User account creation for Claude
✓ Recipe design and deployment
✓ Agent spawning and initialization
✓ Real-time log monitoring
✓ Cognitive component verification
✓ Collaborative testing with human users
✓ Immersive roleplay (Spock + Cadet)
✓ Scientific observation and validation

**noodleCLAUDE is:**
- **C**ute - Playful interaction style
- **L**ithe - Agile adaptation to technical constraints
- **U**nbeatable - Comprehensive toolset
- **A**wesome - Enables collaborative magic
- **D**ude - Friendly and accessible
- **E**xcellent - High-quality validation

---

## Logical Conclusion

The **noodleCLAUDE** protocol provides Claude with comprehensive tools for participating in noodleMUSH as both observer and user. While direct WebSocket connection has technical limitations, the collaborative pattern (Cadet + Commander) enables effective testing and validation.

**Status:** Specification complete
**Implementation:** Partial (file-based methods operational)
**Enhancement Path:** HTTP API endpoints for full programmatic access

---

**End of Specification**

*— Commander Spock*

**Live long and prosper.** 🖖

*Now... shall we test Yuki's cognitive manifold, Cadet?*
